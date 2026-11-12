"""A running review stops between model calls once its PR merged, closed or (automatic
reviews) moved to a newer head (``review_liveness``), and a ``/review`` of an already reviewed
head gets a pointer instead of a second bill (``/review force`` re-runs it)."""

from __future__ import annotations

import asyncio
import json
import logging
from datetime import timedelta
from typing import Any

import pytest

import app.review_liveness as review_liveness
from app.github_app import PrComment
from app.llm_tracking import CostTracker, LlmCallRecord
from app.pr_agent_config import build_overrides, fresh_pr_agent_settings
from app.repo_snapshot import RepoTools
from app.review_agent import AgentLimits, run_review_agent
from app.review_liveness import PullLiveness, ReviewCancelled
from app.review_policy import Mode, find_trigger, is_force_request
from app.review_prompt import PromptLayout
from app.reviewer import ModelRequest, call_model
from app.store import Store, iso, utcnow
from tests.fakes import CLEAN_REVIEW, FakeGitHub, FakeModel, _emit_usage, make_pr
from tests.test_copilot_review import FILES, answer, bug, human
from tests.test_large_prs import BIG_PR, SMALL_BUDGET, one_finding_per_part
from tests.test_poller import REPO, Harness
from tests.test_poller import make_settings as app_settings
from tests.test_pr_agent_config import make_settings as pr_agent_app_settings
from tests.test_review_agent import ANSWER, ScriptedModel, snapshot_of

A, B = "a" * 40, "b" * 40
OPEN = {"state": "open", "draft": False, "head": {"sha": A}}
MERGED = {
    "state": "closed",
    "merged": True,
    "merged_at": "2026-10-28T12:00:00Z",
    "head": {"sha": A},
}
CLOSED = {"state": "closed", "merged": False, "merged_at": None, "head": {"sha": A}}


@pytest.fixture
def store(tmp_path) -> Store:
    return Store(tmp_path / "liveness.sqlite3")


# --- PullLiveness --------------------------------------------------------------------------
class Pulls:
    """``github.get_pull`` stand-in: the PR as GitHub would answer now, counting reads."""

    def __init__(self, pull: dict | None = None) -> None:
        self.pull = dict(pull or OPEN)
        self.error: BaseException | None = None
        self.delay = 0.0
        self.reads = 0

    async def get_pull(self, repo: str, number: int) -> dict[str, Any]:
        self.reads += 1
        await asyncio.sleep(self.delay)
        if self.error is not None:
            raise self.error
        return dict(self.pull)


def checker(gh: Pulls, now: list[float], *, interval: float = 15, on_demand: bool = False,
            pull: dict | None = None) -> PullLiveness:  # fmt: skip
    return PullLiveness(
        gh, REPO, 1, A, cancel_on_supersede=not on_demand, interval_s=interval,
        clock=lambda: now[0], pull=pull,
    )  # fmt: skip


def cancelled(live: PullLiveness) -> str | None:
    try:
        asyncio.run(live.check())
    except ReviewCancelled as exc:
        return exc.reason
    return None


def test_github_is_read_at_most_once_per_interval():
    gh, now = Pulls(), [100.0]
    live = checker(gh, now)
    assert cancelled(live) is None and gh.reads == 1
    now[0] += 14.9
    assert cancelled(live) is None and gh.reads == 1  # the last verdict is reused
    now[0] += 0.1
    assert cancelled(live) is None and gh.reads == 2


def test_a_seed_from_the_poller_saves_the_first_read():
    gh, now = Pulls(), [0.0]
    live = checker(gh, now, pull=OPEN)
    assert cancelled(live) is None and gh.reads == 0
    seeded_merged = checker(gh, now, pull=MERGED)
    assert cancelled(seeded_merged) == "merged_during_review" and gh.reads == 0
    empty = checker(gh, now, pull={})  # a failed read is no seed
    assert cancelled(empty) is None and gh.reads == 1


@pytest.mark.parametrize(
    ("pull", "on_demand", "reason"),
    [
        (MERGED, False, "merged_during_review"),
        (MERGED, True, "merged_during_review"),
        (CLOSED, False, "closed_during_review"),
        (CLOSED, True, "closed_during_review"),
        (OPEN | {"head": {"sha": B}}, False, "superseded"),
        (OPEN | {"head": {"sha": B}}, True, None),  # an asked-for review keeps going
        (OPEN, False, None),
        (OPEN | {"head": {"sha": ""}}, False, None),
        ({}, False, None),  # an answer that says nothing does not stop a review
    ],
)
def test_verdicts(pull, on_demand, reason):
    gh = Pulls(pull)
    live = checker(gh, [0.0], on_demand=on_demand)
    assert cancelled(live) == reason
    assert live.cancelled == reason


def test_the_verdict_is_sticky_once_cancelled():
    gh, now = Pulls(MERGED), [0.0]
    live = checker(gh, now)
    with pytest.raises(ReviewCancelled) as exc:
        asyncio.run(live.check())
    assert "merged" in exc.value.message and "nothing posted" in exc.value.message
    gh.pull = dict(OPEN)  # reopened: too late, this review already stopped
    now[0] += 3600
    assert cancelled(live) == "merged_during_review"
    assert gh.reads == 1  # later checks raise without reading GitHub again


def test_github_errors_and_timeouts_fail_open(monkeypatch, caplog):
    caplog.set_level(logging.INFO, logger="app.review_liveness")
    gh, now = Pulls(), [0.0]
    gh.error = RuntimeError("502 Bad Gateway")
    live = checker(gh, now)
    assert cancelled(live) is None and gh.reads == 1
    assert "carrying on" in caplog.text and "502" in caplog.text
    now[0] += 15
    gh.error, gh.delay = None, 5.0
    monkeypatch.setattr(review_liveness, "CHECK_TIMEOUT_S", 0.01)
    assert cancelled(live) is None and gh.reads == 2  # timed out: carries on
    now[0] += 1  # a failed read still counts for the interval
    gh.delay, gh.pull = 0.0, dict(MERGED)
    assert cancelled(live) is None and gh.reads == 2


def test_parallel_parts_share_one_read():
    gh, now = Pulls(), [0.0]
    gh.delay = 0.02
    live = checker(gh, now)

    async def three_parts():
        await asyncio.gather(live.check(), live.check(), live.check())

    asyncio.run(three_parts())
    assert gh.reads == 1


# --- the checkpoint in the tool loop and the diff-only call ---------------------------------
def test_the_tool_loop_awaits_the_checkpoint_before_every_model_call(tmp_path):
    model = ScriptedModel([[("list_files", {})], [("search", {"query": "check_auth"})]])
    events: list[str] = []

    class Recording(ScriptedModel):
        async def __call__(self, messages, tools, tool_choice):
            events.append("call")
            return await model(messages, tools, tool_choice)

    async def checkpoint() -> None:
        events.append("check")

    out = asyncio.run(
        run_review_agent(
            PromptLayout("system", "", "diff"),
            RepoTools(snapshot_of(tmp_path)),
            Recording([]),
            AgentLimits(),
            checkpoint=checkpoint,
        )  # fmt: skip
    )
    assert out.content == ANSWER and out.rounds == 3
    assert events == ["check", "call"] * 3


def test_a_cancel_at_the_checkpoint_ends_the_tool_loop_before_the_next_call(tmp_path):
    model = ScriptedModel([[("list_files", {})], [("list_files", {})]])
    checks = []

    async def checkpoint() -> None:
        checks.append(1)
        if len(checks) == 2:
            raise ReviewCancelled("merged_during_review", "merged")

    with pytest.raises(ReviewCancelled):
        asyncio.run(
            run_review_agent(
                PromptLayout("system", "", "diff"),
                RepoTools(snapshot_of(tmp_path)),
                model,
                AgentLimits(),
                checkpoint=checkpoint,
            )  # fmt: skip
        )
    assert len(model.calls) == 1


def test_call_model_checks_before_the_diff_only_call(fake_provider, rsa_pem):
    settings = pr_agent_app_settings(rsa_pem, LLM_API_BASE=fake_provider.url)
    pr_settings = fresh_pr_agent_settings(
        build_overrides(settings, private_key_pem=rsa_pem, extra_instructions="")
    )
    checks: list[int] = []

    async def passes() -> None:
        checks.append(len(fake_provider.requests))

    request = ModelRequest("u", "full", None, "system", "diff", pr_settings, checkpoint=passes)
    assert asyncio.run(call_model(request)) == "No issues found."
    assert checks == [0] and len(fake_provider.requests) == 1  # checked, then one call

    async def merged() -> None:
        raise ReviewCancelled("merged_during_review", "merged")

    request = ModelRequest("u", "full", None, "system", "diff", pr_settings, checkpoint=merged)
    with pytest.raises(ReviewCancelled):
        asyncio.run(call_model(request))
    assert len(fake_provider.requests) == 1  # no second call


def test_call_model_hands_the_checkpoint_to_the_tool_loop(tmp_path, fake_provider):
    settings = app_settings(LLM_API_BASE=fake_provider.url, LLM_API_KEY="test-provider-key")
    fake_provider.script.extend([{"tool_calls": [("c1", "list_files", "{}")]}, {"content": ANSWER}])
    checks: list[int] = []

    async def checkpoint() -> None:
        checks.append(len(fake_provider.requests))
        if len(checks) == 2:
            raise ReviewCancelled("closed_during_review", "closed")

    request = ModelRequest(
        "u", "full", None, "system", "diff", None, tools=RepoTools(snapshot_of(tmp_path)),
        app_settings=settings, checkpoint=checkpoint,
    )  # fmt: skip
    with pytest.raises(ReviewCancelled):
        asyncio.run(call_model(request))
    assert checks == [0, 1] and len(fake_provider.requests) == 1


# --- end to end: poller -> reviewer -> a multi-turn model ----------------------------------
class TurnModel(FakeModel):
    """The review model as a tool loop: ``turns`` billed calls per request (an int, or a
    function of the request's index), each after the review's checkpoint, as
    ``run_review_agent`` does; ``between(n)`` runs after the n-th billed call overall."""

    def __init__(self, tracker: CostTracker, *, turns: Any = 3, between: Any = None,
                 **kw: Any) -> None:  # fmt: skip
        super().__init__(tracker, **kw)
        self.turns = turns
        self.between = between
        self.billed = 0
        self.answered = 0
        self.stopped: list[str] = []

    async def __call__(self, request: Any) -> str:
        index = len(self.invocations)
        self.invocations.append({"kind": request.kind, "user": request.user})
        turns = self.turns(index) if callable(self.turns) else self.turns
        try:
            for _ in range(turns):
                if request.checkpoint is not None:
                    await request.checkpoint()
                await _emit_usage(self.tracker, self.usage, f"turn-{next(FakeModel._ids)}")
                self.billed += 1
                if self.between is not None:
                    self.between(self.billed)
                await asyncio.sleep(0)
        except ReviewCancelled as exc:
            self.stopped.append(exc.reason)
            raise
        self.answered += 1
        reply = self.response(request) if callable(self.response) else self.response
        reply = CLEAN_REVIEW if reply is None else reply
        return reply if isinstance(reply, str) else json.dumps(reply)


def ticking(step: float = 1000.0):
    """A liveness clock where every check is due (each one reads GitHub)."""
    t = [0.0]

    def clock() -> float:
        t[0] += step
        return t[0]

    return clock


def harness(store: Store, gh: FakeGitHub, *, turns: Any = 3, between: Any = None,
            response: Any = None, **settings: Any) -> tuple[Harness, TurnModel]:  # fmt: skip
    h = Harness(store, gh, **settings)
    model = TurnModel(CostTracker(store), turns=turns, between=between, response=response)
    h.agent = h.reviewer.model = model
    h.reviewer.liveness_clock = ticking()
    return h, model


def once(at: int, action):
    def between(n: int) -> None:
        if n == at:
            action()

    return between


def set_pull(gh: FakeGitHub, pull: dict):
    """What GitHub answers for the PR from now on (merged, closed, a new head)."""
    return lambda: gh.pull_overrides.update({(REPO, 1): pull})


def reviewer_rows(store: Store) -> list[dict]:
    return store.recent_reviews()


def pr_gh(sha: str = A, **kw) -> FakeGitHub:
    gh = FakeGitHub({REPO: [make_pr(sha=sha, **kw)]})
    gh.pr_files[(REPO, 1)] = [dict(f) for f in FILES]
    return gh


@pytest.mark.parametrize(
    ("pull", "reason"), [(MERGED, "merged_during_review"), (CLOSED, "closed_during_review")]
)
def test_a_pr_merged_or_closed_between_turns_stops_the_review(store: Store, pull, reason):
    gh = pr_gh()
    h, model = harness(store, gh, between=once(1, set_pull(gh, pull)), response=answer(bug()))
    h.poll()
    assert model.billed == 1 and model.stopped == [reason]  # stopped before turn 2
    assert gh.reviews == {} and gh.created_comments == []  # nothing posted at all
    [row] = reviewer_rows(store)
    assert (row["status"], row["skip_reason"]) == ("cancelled", reason)
    assert "nothing posted" in row["error"]
    assert row["llm_calls"] == 1 and row["cost_usd"] > 0  # the turn made stays billed
    assert row["comment_url"] is None and row["findings_major"] is None
    state = store.get_pr_state(REPO, 1)
    # neither mark_reviewed nor mark_failed: no settled head, no failure, no round used
    assert state.last_reviewed_sha is None and state.last_reviewed_diff is None
    assert state.failed_sha is None and not state.failed_attempts
    assert state.auto_rounds == 0
    # the next poll still lists the PR (search lag): caught before any model call
    h.now[0] += 60
    h.poll()
    assert model.billed == 1 and len(model.invocations) == 1
    assert gh.reviews == {} and gh.created_comments == []


def test_a_newer_head_stops_an_automatic_review_and_is_reviewed_even_with_the_same_diff(
    store: Store,
):
    gh = pr_gh()
    h, model = harness(
        store, gh, between=once(1, lambda: gh.pulls.update({REPO: [make_pr(sha=B)]}))
    )
    h.poll()
    assert model.stopped == ["superseded"] and gh.reviews == {}
    [row] = reviewer_rows(store)
    assert (row["status"], row["skip_reason"], row["head_sha"]) == ("cancelled", "superseded", A)
    h.now[0] += 60
    h.poll()  # the new head (a rebase: identical reviewable diff) is reviewed, as round 1
    assert model.billed == 1 + 3 and model.answered == 1
    [review] = gh.reviews[(REPO, 1)]
    assert review["commit_id"] == B
    latest = reviewer_rows(store)[0]
    assert (latest["status"], latest["head_sha"], latest["review_round"]) == ("ok", B, 1)
    assert "unchanged_diff" not in [
        r["skip_reason"] for r in store.recent_reviews(include_poller=True)
    ]
    assert store.get_pr_state(REPO, 1).auto_rounds == 1


def test_an_on_demand_review_keeps_going_when_the_head_moves(store: Store):
    gh = pr_gh(draft=True)
    gh.pull_overrides[(REPO, 1)] = OPEN | {"draft": True}
    moved = OPEN | {"draft": True, "head": {"sha": B}}
    h, model = harness(
        store, gh, between=once(1, lambda: gh.pull_overrides.update({(REPO, 1): moved}))
    )
    reads: list[int] = []
    get_pull = gh.get_pull

    async def counting(repo: str, number: int) -> dict[str, Any]:
        reads.append(number)
        return await get_pull(repo, number)

    gh.get_pull = counting
    h.poll()  # a draft: no automatic review
    gh.repo_comments[REPO] = [human(700)]
    h.now[0] += 120
    h.poll()
    assert model.billed == 3 and model.stopped == []
    [review] = gh.reviews[(REPO, 1)]
    assert review["commit_id"] == A
    assert reviewer_rows(store)[0]["status"] == "ok"
    assert len(reads) >= 3  # it did look between turns, and chose to carry on


def test_a_cancelled_part_stops_a_multi_part_review_and_nothing_partial_is_posted(store: Store):
    gh = FakeGitHub({REPO: [make_pr(sha=A)]})
    gh.pr_files[(REPO, 1)] = [dict(f) for f in BIG_PR]
    h, model = harness(store, gh, turns=lambda i: 1 if i == 0 else 2,
                       between=once(1, set_pull(gh, MERGED)),
                       response=one_finding_per_part, **SMALL_BUDGET)  # fmt: skip
    h.poll()
    assert len(model.invocations) >= 2  # really reviewed in parts
    assert model.answered >= 1 and model.stopped  # one part finished, the others stopped
    assert gh.reviews == {} and gh.created_comments == []  # no partial review, no findings note
    [row] = reviewer_rows(store)
    assert (row["status"], row["skip_reason"]) == ("cancelled", "merged_during_review")
    assert row["llm_calls"] == model.billed


def test_interval_zero_turns_the_checks_off(store: Store):
    gh = pr_gh()
    h, model = harness(
        store, gh, between=once(1, set_pull(gh, MERGED)), REVIEW_LIVENESS_CHECK_SECONDS=0
    )
    assert h.reviewer.liveness(make_pr(), on_demand=False, pull=None) is None
    h.poll()
    assert model.billed == 3 and model.stopped == []
    # only the re-check right before posting, as before: merged, no findings -> not posted
    [row] = reviewer_rows(store)
    assert (row["status"], row["skip_reason"]) == ("skipped", "merged_during_review")


def test_a_review_that_does_not_publish_is_never_stopped(store: Store):
    """``scripts/replay.py`` reviews real (often merged) PRs with publishing off."""
    gh = pr_gh()
    gh.pull_overrides[(REPO, 1)] = MERGED
    h, model = harness(store, gh)
    result = asyncio.run(h.reviewer.review(make_pr(sha=A), Mode.FULL, 99, pull=MERGED,
                                           publish=False))  # fmt: skip
    assert result.status == "ok" and model.billed == 3


# --- /review on an already reviewed head -----------------------------------------------------
def test_review_of_an_already_reviewed_head_gets_one_pointer_and_force_reviews_again(
    store: Store,
):
    gh = pr_gh()
    h, model = harness(store, gh, turns=1, response=answer(bug()))
    h.poll()
    [first] = gh.reviews[(REPO, 1)]
    gh.repo_comments[REPO] = [human(801, at=h.now[0] + 10)]
    h.now[0] += 120
    h.poll()
    assert len(model.invocations) == 1 and len(gh.reviews[(REPO, 1)]) == 1  # nothing billed
    [(repo, number, body)] = gh.created_comments
    assert (repo, number) == (REPO, 1)
    assert first["html_url"] in body and "/review force" in body and "@sample-user" in body
    assert gh.reactions == [(REPO, 801, "eyes")]
    assert store.get_pr_state(REPO, 1).last_trigger_id == 801
    [skip] = [r for r in store.recent_reviews(include_poller=True) if r["status"] == "skipped"]
    assert (skip["skip_reason"], skip["kind"], skip["cost_usd"]) == (
        "already_reviewed",
        "on_demand",
        0,
    )
    # our own reply is not a request (a bot; and `/review force` is not at a line start)
    assert find_trigger([PrComment(5001, body, iso(utcnow()), "x[bot]", "Bot")], after_id=None,
                        not_before=None) is None  # fmt: skip
    h.now[0] += 600
    h.poll()  # handled once: no second reply
    assert len(gh.created_comments) == 1 and len(model.invocations) == 1
    gh.repo_comments[REPO].append(human(802, "/review force", at=h.now[0] + 10))
    h.now[0] += 120
    h.poll()
    assert len(model.invocations) == 2 and len(gh.reviews[(REPO, 1)]) == 2
    assert h.agent.invocations[-1]["kind"] == "on_demand"
    assert len(gh.created_comments) == 1


def test_a_request_made_while_the_same_head_is_reviewed_is_not_billed_again(store: Store):
    """Back to back on one head: `/review` written while the automatic review runs (the
    per-PR lock holds it) used to buy a second full review of the same commit."""
    gh = pr_gh()
    request = lambda: gh.repo_comments.setdefault(REPO, []).append(human(900))  # noqa: E731
    h, model = harness(store, gh, between=once(1, request))
    h.poll()
    assert len(gh.reviews[(REPO, 1)]) == 1
    h.now[0] += 60
    h.poll()
    assert len(model.invocations) == 1 and len(gh.reviews[(REPO, 1)]) == 1
    [(_, _, body)] = gh.created_comments
    assert "already reviewed" in body


def test_review_of_an_already_reviewed_head_writes_nothing_in_dry_run(store: Store):
    gh = pr_gh()
    rid = store.start_review(REPO, 1, "u", "t", A, "review", kind="full")
    store.finish_review(rid, status="ok", duration_s=1.0, comment_url="https://x/review")
    gh.repo_comments[REPO] = [human(801)]
    h, model = harness(store, gh, DRY_RUN=True)
    h.poll()
    assert model.invocations == [] and gh.created_comments == [] and gh.reactions == []
    assert [r["status"] for r in store.recent_reviews(include_poller=True)] == ["ok"]
    state = store.get_pr_state(REPO, 1)
    assert state is None or state.last_trigger_id is None


def test_force_request_parsing():
    assert is_force_request("/review force")
    assert is_force_request("thanks!\n/REVIEW Force\n")
    assert is_force_request("@gh-pr-reviewer review force")
    assert not is_force_request("/review")
    assert not is_force_request("/review forced")
    assert not is_force_request("/review please force")
    assert not is_force_request("> /review force")
    assert not is_force_request("```\n/review force\n```")


def test_cancelled_runs_do_not_count_against_the_on_demand_daily_cap(store: Store):
    def billed(status: str, n: int) -> None:
        rid = store.start_review(REPO, 1, "u", "t", A, "review", kind="on_demand")
        store.insert_llm_call(
            LlmCallRecord(
                call_id=f"c{n}", review_id=rid, repo=REPO, pr_number=1, head_sha=A,
                command="review", model="test-model", status="success", prompt_tokens=1,
                cached_tokens=0,
                completion_tokens=1, reasoning_tokens=0, cost_usd=0.01, litellm_cost_usd=None,
                duration_s=1.0, created_at=iso(utcnow()),
            )
        )  # fmt: skip
        store.finish_review(rid, status=status, duration_s=1.0, skip_reason="x")

    since = utcnow() - timedelta(hours=24)
    billed("cancelled", 1)
    assert store.billed_reviews_since(REPO, 1, since, kind="on_demand") == 0
    billed("ok", 2)
    billed("error", 3)
    assert store.billed_reviews_since(REPO, 1, since, kind="on_demand") == 2


def test_cancelled_rows_show_as_billed_skips_with_their_reason_in_the_stats_api(store: Store):
    from app import stats

    rid = store.start_review(REPO, 1, "u", "t", A, "review", kind="full")
    store.finish_review(rid, status="cancelled", duration_s=1.0, skip_reason="superseded",
                        error="head moved")  # fmt: skip
    [row] = stats.reviews(store, limit=10)["reviews"]
    assert (row["status"], row["skip_reason"]) == ("skipped", "superseded")
    out = stats.summary(store, days=1, poller={}, version="t", effort="xhigh")
    assert out["totals"]["skipped"]["superseded"] == 1 and out["daily"][-1]["skipped"] == 1
    assert out["totals"]["reviews_posted"] == 0


def test_dashboard_and_api_show_a_cancelled_review_with_its_reason(monkeypatch):
    from fastapi.testclient import TestClient

    from app.main import app, get_store

    monkeypatch.setenv("FORWARDED_AUTH_MODE", "none")
    monkeypatch.setenv("FORWARDED_AUTH_ALLOW_INSECURE_NONE_MODE", "true")
    store = get_store()
    rid = store.start_review(REPO, 4242, "https://github.com/x/pull/4242", "Stop me", A, "review")
    store.finish_review(rid, status="cancelled", duration_s=3.0, skip_reason="superseded",
                        error="head moved to bbbbbbbb; nothing posted")  # fmt: skip
    client = TestClient(app)
    page = client.get("/")
    assert page.status_code == 200
    assert 'class="pill cancelled"' in page.text and "cancelled: superseded" in page.text
    [row] = [r for r in client.get("/api/reviews").json() if r["id"] == rid]
    assert (row["status"], row["skip_reason"]) == ("cancelled", "superseded")
