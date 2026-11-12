"""Poller scheduling (dedupe, incremental, drafts, bots, concurrency) and the review run."""

from __future__ import annotations

import asyncio

import pytest

from app.config import Settings
from app.llm_tracking import CostTracker
from app.poller import Poller
from app.review_policy import retry_delay_seconds
from app.reviewer import Reviewer
from app.stats import reviews as stats_reviews
from app.stats import summary as stats_summary
from app.store import Store
from tests.fakes import FakeGitHub, FakeModel, FakePrAgent, make_pr

REPO = "sample-org/example-project"


@pytest.fixture
def store(tmp_path) -> Store:
    return Store(tmp_path / "poller.sqlite3")


def make_settings(**kw) -> Settings:
    values = dict(
        REVIEW_REPOS="sample-org/example-project,sample-org/example-tools",
        LLM_API_BASE="https://api.example.test/v1",
        LLM_API_KEY="test-provider-key",
        LLM_MODEL="test-model",
        LLM_REASONING_EFFORT="xhigh",
        LLM_PROMPT_CACHE_LAYOUT=True,
        GITHUB_APP_ID="1",
        GITHUB_APP_PRIVATE_KEY="unused-by-fakes",
        RETRY_BACKOFF_SECONDS=60,
        REVIEW_SETTLE_SECONDS=0,  # tests of the settle wait opt in
    )
    return Settings(_env_file=None, **(values | kw))


class Harness:
    def __init__(self, store: Store, github: FakeGitHub, *, clock=None, **settings_kw) -> None:
        self.settings = make_settings(**settings_kw)
        self.github = github
        self.store = store
        self.agent = FakeModel(CostTracker(store))  # the review model
        self.improver = FakePrAgent(CostTracker(store), github)  # PR-Agent `/improve`
        self.now = [1_800_000_000.0]
        self.reviewer = Reviewer(
            self.settings, store, github, model=self.agent, invoker=self.improver
        )
        self.poller = Poller(
            self.settings,
            store,
            github=github,
            reviewer=self.reviewer,
            clock=lambda: self.now[0],
        )

    def poll(self, times: int = 1):
        async def run():
            await self.poller.setup()
            out = []
            for _ in range(times):
                out.append(await self.poller.poll_once())
                await self.poller.wait_idle()
            return out

        return asyncio.run(run())


def requests(h: Harness) -> list[str]:
    """Review modes the model was asked for, in order: "review" (full) / "incremental"."""
    return [inv["mode"] for inv in h.agent.invocations]


def test_first_review_then_skip_same_sha_then_one_full_rereview_then_max_rounds(store: Store):
    gh = FakeGitHub({REPO: [make_pr(sha="a" * 40)]})
    h = Harness(store, gh)
    h.poll(times=2)
    assert requests(h) == ["review"]

    state = store.get_pr_state(REPO, 1)
    assert state.last_reviewed_sha == "a" * 40 and state.last_review_mode == "review"
    assert state.auto_rounds == 1

    gh.pulls[REPO] = [make_pr(sha="b" * 40)]
    h.poll()
    assert requests(h) == ["review", "review"]  # round 2: a full re-review, not incremental
    assert [inv["kind"] for inv in h.agent.invocations] == ["full", "full"]
    assert "Full re-review of the whole PR" in h.agent.invocations[1]["user"]
    assert store.get_pr_state(REPO, 1).auto_rounds == 2
    assert [r["review_round"] for r in store.recent_reviews()] == [2, 1]

    gh.pulls[REPO] = [make_pr(sha="c" * 40)]
    h.now[0] += 3600
    h.poll(times=2)
    assert len(h.agent.invocations) == 2  # capped: never reviewed automatically again
    skips = [r for r in store.recent_reviews(include_poller=True) if r["status"] == "skipped"]
    assert [(r["skip_reason"], r["head_sha"]) for r in skips] == [("max_rounds", "c" * 40)]


def test_round_count_survives_a_restart(store: Store):
    gh = FakeGitHub({REPO: [make_pr(sha="a" * 40)]})
    Harness(store, gh).poll()
    gh.pulls[REPO] = [make_pr(sha="b" * 40)]
    second = Harness(store, gh)  # new process: round 2 because the log says one round ran
    second.poll()
    assert [inv["kind"] for inv in second.agent.invocations] == ["full"]
    assert "Full re-review" in second.agent.invocations[0]["user"]
    gh.pulls[REPO] = [make_pr(sha="c" * 40)]
    third = Harness(store, gh)  # and the cap holds after another restart
    third.poll()
    assert third.agent.invocations == []
    assert store.get_pr_state(REPO, 1).auto_rounds == 2


def test_skipped_and_failed_runs_do_not_use_up_a_round(store: Store):
    gh = FakeGitHub({REPO: [make_pr(sha="a" * 40)]})
    gh.pr_files[(REPO, 1)] = [{"filename": "uv.lock", "status": "modified", "patch": "+x"}]
    h = Harness(store, gh)
    h.poll()  # lockfile only: skipped, no round
    assert h.agent.invocations == [] and store.get_pr_state(REPO, 1).auto_rounds == 0
    gh.pulls[REPO] = [make_pr(sha="b" * 40)]
    gh.pr_files[(REPO, 1)] = [{"filename": "a.py", "status": "modified", "patch": "+x"}]
    h.agent.ok = False
    h.poll()  # failed: no round
    assert store.get_pr_state(REPO, 1).auto_rounds == 0
    h.agent.ok = True
    gh.pulls[REPO] = [make_pr(sha="c" * 40)]
    h.poll()
    assert "Full re-review" not in h.agent.invocations[-1]["user"]  # still round 1
    assert store.get_pr_state(REPO, 1).auto_rounds == 1


def test_failed_review_retries_after_jitter_and_success_clears_error(store: Store):
    gh = FakeGitHub({REPO: [make_pr(sha="a" * 40)]})
    h = Harness(store, gh)
    h.agent.ok = False
    h.poll()

    state = store.get_pr_state(REPO, 1)
    assert state.failed_attempts == 1 and state.failed_sha == "a" * 40
    assert h.poller.status()["last_error"].startswith("Review attempt 1/4 failed:")
    assert h.poller.status()["last_error_at"]
    delay = retry_delay_seconds(REPO, 1, "a" * 40, 1, base_seconds=60, max_seconds=3600)

    async def poll_once():
        await h.poller.poll_once()
        await h.poller.wait_idle()

    h.now[0] += delay - 1
    asyncio.run(poll_once())
    assert len(h.agent.invocations) == 1  # still in backoff

    h.now[0] += 2
    h.agent.ok = True
    asyncio.run(poll_once())
    assert len(h.agent.invocations) == 2
    state = store.get_pr_state(REPO, 1)
    assert state.last_reviewed_sha == "a" * 40 and state.failed_attempts == 0
    assert h.poller.status()["last_error"] is None and h.poller.status()["last_error_at"] is None


def test_successful_review_does_not_clear_a_non_review_error(store: Store):
    gh = FakeGitHub({REPO: [make_pr(sha="a" * 40)]})
    h = Harness(store, gh)
    h.poller._set(
        last_error="org discovery failed",
        last_error_at="2026-11-10T00:00:00+00:00",
        last_error_source="poll",
    )
    h.poll()
    status = h.poller.status()
    assert status["last_error"] == "org discovery failed"
    assert status["last_error_at"] == "2026-11-10T00:00:00+00:00"


def test_failed_review_exhaustion_stays_visible_in_stats(store: Store):
    gh = FakeGitHub({REPO: [make_pr(sha="a" * 40)]})
    h = Harness(store, gh)
    h.agent.ok = False
    h.poll()

    async def poll_once():
        await h.poller.poll_once()
        await h.poller.wait_idle()

    for attempt in range(1, 4):
        delay = retry_delay_seconds(REPO, 1, "a" * 40, attempt, base_seconds=60, max_seconds=3600)
        h.now[0] += delay + 1
        asyncio.run(poll_once())

    state = store.get_pr_state(REPO, 1)
    assert state.failed_attempts == 4 and state.failed_sha == "a" * 40
    assert len(h.agent.invocations) == 4
    assert "retries exhausted" in h.poller.status()["last_error"]
    summary = stats_summary(store, days=7, poller=h.poller.status(), version="test", effort="xhigh")
    assert "retries exhausted" in summary["poller"]["last_error"]
    assert summary["poller"]["last_error_at"]
    assert [r["status"] for r in stats_reviews(store, limit=10)["reviews"]] == ["failed"] * 4
    h.now[0] += 10_000
    asyncio.run(poll_once())
    assert len(h.agent.invocations) == 4  # no endless automatic retry after the cap


def test_max_review_rounds_is_configurable(store: Store):
    gh = FakeGitHub({REPO: [make_pr(sha="a" * 40)]})
    h = Harness(store, gh, MAX_REVIEW_ROUNDS=1)
    h.poll()
    gh.pulls[REPO] = [make_pr(sha="b" * 40)]
    h.poll()
    assert len(h.agent.invocations) == 1
    assert "last automatic review" in gh.reviews[(REPO, 1)][0]["body"]
    unlimited = Harness(store, gh, MAX_REVIEW_ROUNDS=0)
    unlimited.poll()
    assert len(unlimited.agent.invocations) == 1


def test_restart_does_not_re_review(store: Store):
    gh = FakeGitHub({REPO: [make_pr()]})
    Harness(store, gh).poll()
    restarted = Harness(store, gh)  # new process, same SQLite file
    restarted.poll()
    assert restarted.agent.invocations == []


def test_review_records_cost_and_appends_footer(store: Store):
    gh = FakeGitHub({REPO: [make_pr()]})
    h = Harness(store, gh)
    h.poll()
    review = store.recent_reviews()[0]
    assert review["status"] == "ok"
    assert review["command"] == "review"
    assert review["llm_calls"] == 1
    assert (review["prompt_tokens"], review["cached_tokens"], review["completion_tokens"]) == (
        10_000,
        1_000,
        2_000,
    )
    assert review["reasoning_tokens"] == 1_500
    assert review["cost_usd"] == pytest.approx(0.0067)
    assert review["litellm_cost_usd"] == pytest.approx(0.0067)
    posted = gh.reviews[(REPO, 1)]
    assert len(posted) == 1 and review["comment_url"] == posted[0]["html_url"]
    assert "#pullrequestreview-" in review["comment_url"]
    assert gh.updated == [] and gh.created_comments == []  # no issue comment any more
    body = posted[0]["body"]
    assert posted[0]["commit_id"] == "a" * 40
    assert "## Pull request overview" in body and "No issues found." in body
    assert "gh-pr-reviewer · gpt-4.1 xhigh · 10.0k in / 2.0k out · $0.0067" in body


def test_only_allowlisted_repos_are_polled(store: Store):
    gh = FakeGitHub({"sample-org/other": [make_pr(repo="sample-org/other")]})
    h = Harness(store, gh, REVIEW_REPOS="sample-org/example-project")
    h.poll()
    assert gh.listed == ["sample-org/example-project"]
    assert h.agent.invocations == []


def test_drafts_and_bots_are_not_reviewed(store: Store):
    gh = FakeGitHub(
        {
            REPO: [
                make_pr(number=1, draft=True),
                make_pr(number=2, author_login="dependabot[bot]", author_type="Bot"),
                make_pr(number=3, author_login="renovate[bot]", author_type="Bot"),
                make_pr(number=4, author_login="ci-app", author_type="Bot"),
                make_pr(number=5),
            ]
        }
    )
    h = Harness(store, gh)
    h.poll()
    assert [inv["pr_url"].rsplit("/", 1)[1] for inv in h.agent.invocations] == ["5"]


def test_leaving_draft_after_new_commits_gets_full_review(store: Store):
    gh = FakeGitHub({REPO: [make_pr(sha="a" * 40)]})
    h = Harness(store, gh)
    h.poll()
    gh.pulls[REPO] = [make_pr(sha="b" * 40, draft=True)]
    h.poll()
    assert store.get_pr_state(REPO, 1).draft_since_review is True
    gh.pulls[REPO] = [make_pr(sha="c" * 40)]
    h.poll()
    assert requests(h) == ["review", "review"]
    assert store.get_pr_state(REPO, 1).draft_since_review is False


def test_concurrency_cap_and_per_pr_lock(store: Store):
    gh = FakeGitHub({REPO: [make_pr(number=n, sha=f"{n:040d}") for n in range(1, 6)]})
    h = Harness(store, gh)
    h.agent.delay = 0.05

    async def run():
        await h.poller.setup()
        await h.poller.poll_once()
        # second poll while reviews are in flight must not double-schedule any PR
        await h.poller.poll_once()
        assert len(h.poller.status()["in_flight"]) == 5
        await h.poller.wait_idle()

    asyncio.run(run())
    assert h.agent.max_active == 2
    urls = [inv["pr_url"] for inv in h.agent.invocations]
    assert len(urls) == len(set(urls)) == 5
    assert h.poller.status()["in_flight"] == []


def test_failed_review_backs_off_and_is_retried_later(store: Store):
    gh = FakeGitHub({REPO: [make_pr()]})
    h = Harness(store, gh)
    h.agent.ok = False
    h.poll()
    state = store.get_pr_state(REPO, 1)
    assert state.failed_attempts == 1 and state.last_reviewed_sha is None
    assert store.recent_reviews()[0]["status"] == "error"
    assert gh.updated == []  # no footer on failed runs
    h.poll()  # inside backoff window
    assert len(h.agent.invocations) == 1
    h.now[0] += 601
    h.agent.ok = True
    h.poll()
    assert len(h.agent.invocations) == 2
    assert store.get_pr_state(REPO, 1).last_reviewed_sha == "a" * 40


def test_dry_run_makes_no_llm_or_github_writes(store: Store):
    gh = FakeGitHub({REPO: [make_pr()]})
    h = Harness(store, gh, DRY_RUN=True)
    decisions = h.poll(times=2)
    assert [d[0][1].mode.value for d in decisions] == ["review", "review"]
    assert h.agent.invocations == []
    assert gh.updated == []
    assert store.get_pr_state(REPO, 1) is None
    assert h.poller.status()["recent_decisions"][0]["mode"] == "review"


def test_repo_conventions_and_pip_compile_outputs_reach_pr_agent(store: Store):
    gh = FakeGitHub({REPO: [make_pr(base_ref="main")]})
    gh.files[(REPO, "CLAUDE.md", "main")] = "Always use uv run. " * 600  # > 6k chars
    gh.files[(REPO, "AGENTS.md", "main")] = "agents file"
    gh.pr_files[(REPO, 1)] = [
        {"filename": "requirements.txt", "status": "modified", "patch": "+fastapi"},
        {
            "filename": "svc/requirements-lock.txt",
            "status": "modified",
            "patch": "@@ -1 +1 @@\n-# This file is autogenerated by pip-compile\n",
        },
    ]
    gh.files[(REPO, "requirements.txt", "a" * 40)] = "fastapi\n"
    h = Harness(store, gh)
    h.poll()
    inv = h.agent.invocations[0]
    assert inv["reasoning_effort"] == "xhigh"
    assert "Review rubric" in inv["user"]
    assert "Repository-specific conventions (from `main`" in inv["user"]
    assert "Always use uv run." in inv["user"]
    assert "[... truncated ...]" in inv["user"]
    # pip-compile output is excluded from the diff (and named); a hand-written one is not
    assert "## File: 'requirements.txt'" in inv["user"]
    assert "## File: 'svc/requirements-lock.txt'" not in inv["user"]
    assert "- `svc/requirements-lock.txt`" in inv["user"]
    assert (REPO, "CLAUDE.md", "main") in gh.file_requests


def test_enable_improve_runs_improve_and_attributes_cost(store: Store):
    gh = FakeGitHub({REPO: [make_pr()]})
    h = Harness(store, gh, ENABLE_IMPROVE=True)
    h.poll()
    assert requests(h) == ["review"]
    assert [inv["request"] for inv in h.improver.invocations] == [["improve"]]
    review = store.recent_reviews()[0]
    commands = [c["command"] for c in store.llm_calls_for_review(review["id"])]
    assert commands == ["review", "improve"]


def test_not_configured_poller_does_not_start(store: Store):
    settings = Settings(_env_file=None, GITHUB_APP_ID="", GITHUB_APP_PRIVATE_KEY="", DRY_RUN=True)
    poller = Poller(settings, store)
    poller.start()
    status = poller.status()
    assert status["state"] == "not_configured"
    assert "GITHUB_APP_ID" in status["reason"]
    poller.stop()


def test_rereview_covers_the_whole_pr_and_names_what_changed_since_round_one(store: Store):
    gh = FakeGitHub({REPO: [make_pr(sha="a" * 40)]})
    h = Harness(store, gh)
    h.poll()
    gh.pulls[REPO] = [make_pr(sha="b" * 40)]
    gh.compares[(REPO, "a" * 40, "b" * 40)] = [{"filename": "new_module.py"}]
    h.poll()
    anchors = [inv["since_sha"] for inv in h.agent.invocations]
    assert anchors == [None, None]  # never an incremental-only review
    prompt = h.agent.invocations[1]["user"]
    assert "Incremental review" not in prompt
    assert f"since the last review (`{'a' * 12}`) touched: `new_module.py`" in prompt


def test_lockfile_only_pr_is_skipped_without_llm_or_retries(store: Store):
    # A synthetic empty-diff failure is retried after the configured backoff.
    gh = FakeGitHub({REPO: [make_pr()]})
    gh.pr_files[(REPO, 1)] = [
        {"filename": "frontend/package-lock.json", "status": "modified", "patch": "+x"},
        {"filename": "backend/uv.lock", "status": "modified", "patch": "+y"},
    ]
    h = Harness(store, gh)
    h.poll()
    assert h.agent.invocations == []
    review = store.recent_reviews()[0]
    assert review["status"] == "skipped" and review["llm_calls"] == 0
    assert store.get_pr_state(REPO, 1).last_reviewed_sha == "a" * 40
    h.now[0] += 3600
    h.poll()
    assert h.agent.invocations == [] and len(store.recent_reviews()) == 1


def test_one_reviewable_file_is_enough_and_hidden_lockfiles_are_named(store: Store):
    gh = FakeGitHub({REPO: [make_pr()]})
    gh.pr_files[(REPO, 1)] = [
        {"filename": "frontend/package-lock.json", "status": "modified", "patch": "+x"},
        {"filename": "frontend/package.json", "status": "modified", "patch": "+y"},
    ]
    h = Harness(store, gh)
    h.poll()
    assert requests(h) == ["review"]
    text = h.agent.invocations[0]["user"]
    assert "- `frontend/package-lock.json`" in text
    assert "- `frontend/package.json`" not in text


@pytest.mark.parametrize(
    ("fresh", "why", "skip_reason"),
    [
        ({"state": "closed", "draft": False, "head": {"sha": "a" * 40}}, "PR is closed", "closed"),
        (
            {"state": "closed", "merged": True, "draft": False, "head": {"sha": "a" * 40}},
            "PR was merged",
            "merged_during_wait",
        ),
        ({"state": "open", "draft": True, "head": {"sha": "a" * 40}}, "back to draft", "draft"),
        (
            {"state": "open", "draft": False, "head": {"sha": "c" * 40}},
            "head moved to cccccccc",
            None,
        ),
    ],
)
def test_queued_review_is_dropped_when_the_pr_moved_on(
    store: Store, fresh: dict, why: str, skip_reason: str | None
):
    # A synthetic pull request closes while it is queued for review.
    gh = FakeGitHub({REPO: [make_pr()]})
    gh.pull_overrides[(REPO, 1)] = fresh
    h = Harness(store, gh)
    h.poll()
    assert h.agent.invocations == []
    state = store.get_pr_state(REPO, 1)  # nothing reviewed: the next poll re-decides
    assert state.last_reviewed_sha is None and state.failed_attempts == 0
    assert why in h.poller.status()["recent_decisions"][0]["reason"]
    rows = store.recent_reviews(include_poller=True)
    assert [r["skip_reason"] for r in rows] == ([skip_reason] if skip_reason else [])


def test_recheck_failure_does_not_block_the_review(store: Store):
    class Flaky(FakeGitHub):
        async def get_pull(self, repo, number):
            raise RuntimeError("GitHub 502")

    h = Harness(store, Flaky({REPO: [make_pr()]}))
    h.poll()
    assert requests(h) == ["review"]


def test_codebase_context_reaches_pr_agent(store: Store):
    sha = "a" * 40
    gh = FakeGitHub({REPO: [make_pr(sha=sha)]})
    gh.pr_files[(REPO, 1)] = [
        {"filename": "backend/src/app/services/people.py", "status": "modified", "patch": "+q"},
    ]
    gh.files[(REPO, "backend/src/app/services/people.py", sha)] = 'Person.status == "ACTIVE"'
    h = Harness(store, gh)
    h.poll()
    text = h.agent.invocations[0]["user"]
    assert "## Codebase context" in text and 'Person.status == "ACTIVE"' in text


def test_rebase_with_identical_diff_is_not_re_reviewed(store: Store):
    # Identical synthetic changes after a rebase should not be billed twice.
    files = [{"filename": "app.py", "status": "modified", "patch": "@@ -1 +1 @@\n-a\n+b"}]
    gh = FakeGitHub({REPO: [make_pr(sha="a" * 40)]})
    gh.pr_files[(REPO, 1)] = files
    h = Harness(store, gh)
    h.poll()
    assert requests(h) == ["review"]
    assert store.get_pr_state(REPO, 1).last_reviewed_diff

    gh.pulls[REPO] = [make_pr(sha="b" * 40)]  # rebased: same change, new position
    gh.pr_files[(REPO, 1)] = [dict(files[0], patch="@@ -90 +90 @@\n-a\n+b")]
    h.poll()
    assert requests(h) == ["review"]
    assert store.recent_reviews()[0]["status"] == "skipped"
    assert store.get_pr_state(REPO, 1).last_reviewed_sha == "b" * 40

    gh.pulls[REPO] = [make_pr(sha="c" * 40)]  # a real change: round 2
    gh.pr_files[(REPO, 1)] = [dict(files[0], patch="@@ -1 +1 @@\n-a\n+c")]
    h.poll()
    assert requests(h) == ["review", "review"]
    assert "Full re-review" in h.agent.invocations[1]["user"]


def test_rereview_context_covers_the_whole_pr_again(store: Store):
    # Round 2 is a full re-review: every PR file gets its codebase context again.
    files = [
        {"filename": "a.py", "status": "modified", "patch": "@@ -1 +1 @@\n-a\n+b"},
        {"filename": "b.py", "status": "modified", "patch": "@@ -1 +1 @@\n-c\n+d"},
    ]
    gh = FakeGitHub({REPO: [make_pr(sha="a" * 40)]})
    gh.pr_files[(REPO, 1)] = files
    for sha in ("a" * 40, "b" * 40):
        gh.files[(REPO, "a.py", sha)] = "AAA_FULL_FILE"
        gh.files[(REPO, "b.py", sha)] = "BBB_FULL_FILE"
    h = Harness(store, gh)
    h.poll()
    first = h.agent.invocations[0]["user"]
    assert "AAA_FULL_FILE" in first and "BBB_FULL_FILE" in first

    gh.pulls[REPO] = [make_pr(sha="b" * 40)]
    gh.pr_files[(REPO, 1)] = [files[0], dict(files[1], patch="@@ -1 +1 @@\n-c\n+e")]
    gh.compares[(REPO, "a" * 40, "b" * 40)] = [{"filename": "b.py"}]
    h.poll()
    second = h.agent.invocations[1]["user"]
    assert "BBB_FULL_FILE" in second and "AAA_FULL_FILE" in second


def test_incremental_context_covers_only_files_the_new_commits_touched(store: Store):
    # A synthetic incremental review is exercised by its dedicated unit test.
    from app.review_policy import Mode

    files = [
        {"filename": "a.py", "status": "modified", "patch": "@@ -1 +1 @@\n-a\n+b"},
        {"filename": "b.py", "status": "modified", "patch": "@@ -1 +1 @@\n-c\n+e"},
    ]
    gh = FakeGitHub({REPO: [make_pr(sha="b" * 40)]})
    gh.pr_files[(REPO, 1)] = files
    for name in ("a.py", "b.py"):
        gh.files[(REPO, name, "b" * 40)] = name[0].upper() * 3 + "_FULL_FILE"
    gh.compares[(REPO, "a" * 40, "b" * 40)] = [{"filename": "b.py"}, {"filename": "main.py"}]
    h = Harness(store, gh)

    async def run():
        await h.poller.setup()
        return await h.reviewer.review(
            make_pr(sha="b" * 40), Mode.INCREMENTAL, 1, last_reviewed_sha="a" * 40,
            kind="incremental", publish=False,
        )  # fmt: skip

    asyncio.run(run())
    prompt = h.agent.invocations[0]["user"]
    assert "BBB_FULL_FILE" in prompt and "AAA_FULL_FILE" not in prompt
