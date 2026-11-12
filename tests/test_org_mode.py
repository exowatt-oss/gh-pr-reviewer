"""Org-wide mode: discovery by search, go-live cutoff, opt-outs, cost guards, queue, drain."""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta

import pytest

from app.github_app import PrComment, PullRequest
from app.poller import ORG_MODE_SINCE_KEY
from app.review_policy import (
    Mode,
    PrState,
    is_bot_author,
    parse_review_targets,
    post_cutoff_reason,
    repo_config_enabled,
    repo_opt_out_reason,
)
from app.store import LlmCallRecord, Store, iso, utcnow
from tests.fakes import FakeGitHub, make_pr
from tests.test_poller import Harness

ORG = "sample-org"
NEW = "sample-org/new-thing"
PILOT = "sample-org/example-project"
NOW = 1_800_000_000.0  # the Harness clock; also the default cutoff (first org-mode start)


def t(offset_s: float) -> str:
    """GitHub-style timestamp relative to the harness clock."""
    return datetime.fromtimestamp(NOW + offset_s, tz=UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def org_pr(number: int = 1, sha: str = "a" * 40, repo: str = NEW, **kw) -> PullRequest:
    base = dict(created_at=t(60), updated_at=t(60), head_committed_at=t(30))
    return make_pr(repo=repo, number=number, sha=sha, **(base | kw))


@pytest.fixture
def store(tmp_path) -> Store:
    return Store(tmp_path / "org.sqlite3")


def org_harness(store: Store, gh: FakeGitHub, **kw) -> Harness:
    settings = dict(REVIEW_REPOS=f"{ORG}/*", PILOT_REPOS=PILOT, MAX_CONCURRENT_REVIEWS=6)
    return Harness(store, gh, **(settings | kw))


def reviewed_urls(h: Harness) -> list[str]:
    return [inv["pr_url"] for inv in h.agent.invocations]


def billed_call(store: Store, repo: str, number: int, review_id: int, n: int) -> None:
    store.insert_llm_call(
        LlmCallRecord(
            call_id=f"billed-{repo}-{number}-{n}",
            review_id=review_id,
            repo=repo,
            pr_number=number,
            head_sha="x" * 40,
            command="review",
            model="test-model",
            status="success",
            prompt_tokens=1000,
            cached_tokens=0,
            completion_tokens=100,
            reasoning_tokens=50,
            cost_usd=0.01,
            litellm_cost_usd=None,
            duration_s=1.0,
            created_at=iso(utcnow()),
        )
    )


# --- targets & policy ------------------------------------------------------------------
def test_review_targets_mix_orgs_and_explicit_repos():
    targets = parse_review_targets("sample-org/*, sample-automation/example-project, sample-org/*")
    assert targets.orgs == ("sample-org",)
    assert targets.repos == ("sample-automation/example-project",)
    assert targets.org_mode and targets.covers("sample-org/anything")
    assert targets.covers("sample-automation/example-project") and not targets.covers(
        "sample-automation/x"
    )
    assert not parse_review_targets("sample-org/example-project").org_mode


@pytest.mark.parametrize(
    ("login", "user_type"),
    [("github-actions[bot]", "Bot"), ("github-actions", "User"), ("renovate[bot]", "User")],
)
def test_more_bots_are_skipped(login: str, user_type: str):
    assert is_bot_author(login, user_type)


@pytest.mark.parametrize(
    ("text", "enabled"),
    [
        ("enabled: false\n", False),
        ("enabled: no\n", False),
        ("enabled: 'off'\n", False),
        ("enabled: true\n", True),
        ("other: 1\n", True),
        ("::: not yaml", True),
        ("- a list\n", True),
    ],
)
def test_repo_config_enabled(text: str, enabled: bool):
    assert repo_config_enabled(text) is enabled


def test_repo_opt_out_reason():
    assert repo_opt_out_reason(("python", "No-AI-Review"), None) == "repo topic no-ai-review"
    assert "enabled: false" in repo_opt_out_reason((), "enabled: false")
    assert repo_opt_out_reason(("python",), "enabled: true") is None


def test_post_cutoff_reason_rules():
    cutoff = NOW
    old = dict(created_at=t(-86400), updated_at=t(10), head_committed_at=t(-3600))
    backlog = make_pr(**old)
    assert post_cutoff_reason(backlog, None, cutoff) is None
    assert post_cutoff_reason(make_pr(**(old | {"created_at": t(5)})), None, cutoff)
    assert post_cutoff_reason(make_pr(**(old | {"head_committed_at": t(5)})), None, cutoff)
    assert post_cutoff_reason(make_pr(**(old | {"ready_at": t(5)})), None, cutoff)
    assert post_cutoff_reason(make_pr(**(old | {"force_pushed_at": t(5)})), None, cutoff)
    # an old commit pushed late: commit dates say "backlog", the observed head change says no
    seen = PrState(observed_sha="b" * 40, observed_draft=False, observed_at=cutoff + 1)
    assert post_cutoff_reason(backlog, seen, cutoff) == "changed after the cutoff"
    # draft at an observation after the cutoff, now ready
    seen_draft = PrState(observed_sha="a" * 40, observed_draft=True, observed_at=cutoff + 1)
    assert post_cutoff_reason(backlog, seen_draft, cutoff) == "changed after the cutoff"
    # an observation from before the cutoff proves nothing
    early = PrState(observed_sha="b" * 40, observed_at=cutoff - 1)
    assert post_cutoff_reason(backlog, early, cutoff) is None
    assert post_cutoff_reason(backlog, PrState(org_live=True), cutoff) == "already live"


# --- discovery & cutoff ------------------------------------------------------------------
def test_org_mode_reconciles_first_then_searches_incrementally(store: Store):
    gh = FakeGitHub()
    gh.add_repo(NEW)
    h = org_harness(store, gh, RECONCILE_INTERVAL_SECONDS=900, SEARCH_OVERLAP_SECONDS=180)
    h.poll()
    assert gh.searches == [f"org:{ORG} is:pr is:open archived:false sort:updated-desc"]
    h.now[0] += 60
    h.poll()
    assert gh.searches[-1].endswith(f"updated:>={t(-180)}")  # watermark (NOW) - 180 s
    h.now[0] += 900
    h.poll()
    assert "updated:" not in gh.searches[-1]  # periodic reconcile
    status = h.poller.status()
    assert status["mode"] == "org" and status["repos"] == [f"{ORG}/*"]
    assert status["org_mode_since"] == iso(datetime.fromtimestamp(NOW, tz=UTC))
    assert status["repo_count"] == 2  # new-thing + the directly listed pilot repo
    assert status["discovery"][ORG]["last_reconcile"]["kind"] == "reconcile"


def test_backlog_is_never_reviewed_but_new_work_is(store: Store):
    gh = FakeGitHub()
    gh.add_repo(NEW)
    old = dict(created_at=t(-86400), updated_at=t(-60), head_committed_at=t(-90000))
    gh.add_hit(org_pr(1, **old))  # existing open PR: backlog
    gh.add_hit(org_pr(2, draft=True, **old))  # existing draft
    gh.add_hit(org_pr(3))  # opened after the cutoff
    h = org_harness(store, gh)
    h.poll()
    assert reviewed_urls(h) == [f"https://github.com/{NEW}/pull/3"]
    outcomes = h.poller.status()["discovery"][ORG]["last_reconcile"]["outcomes"]
    assert outcomes["backlog"] == 2 and outcomes["review"] == 1

    h.now[0] += 60
    h.poll()  # nothing changed: still nothing for 1 and 2
    assert len(h.agent.invocations) == 1

    # PR 1 gets an old-dated commit pushed late; PR 2 is marked ready (no event in the fake)
    h.now[0] += 60
    gh.add_hit(org_pr(1, sha="b" * 40, **(old | {"updated_at": t(120)})))
    gh.add_hit(org_pr(2, draft=False, **(old | {"updated_at": t(120)})))
    h.poll()
    assert sorted(reviewed_urls(h)[1:]) == [
        f"https://github.com/{NEW}/pull/1",
        f"https://github.com/{NEW}/pull/2",
    ]
    assert all(inv["mode"] == "review" for inv in h.agent.invocations)
    assert store.get_pr_state(NEW, 1).org_live


def test_pilot_repos_keep_their_behaviour_without_cutoff(store: Store):
    old = dict(created_at=t(-86400), updated_at=t(-60), head_committed_at=t(-90000))
    gh = FakeGitHub({PILOT: [make_pr(repo=PILOT, **old)]})
    gh.add_repo(PILOT)
    gh.add_hit(make_pr(repo=PILOT, **old))  # also found by search: handled once, directly
    h = org_harness(store, gh)
    h.poll()
    assert reviewed_urls(h) == [f"https://github.com/{PILOT}/pull/1"]
    assert gh.listed == [PILOT]
    outcomes = h.poller.status()["discovery"][ORG]["last_reconcile"]["outcomes"]
    assert outcomes == {"direct_repo": 1}


def test_cutoff_is_persisted_and_env_overrides_it(store: Store):
    gh = FakeGitHub()
    h = org_harness(store, gh)
    h.poll()
    key = f"{ORG_MODE_SINCE_KEY}:{ORG.lower()}"
    assert store.get_meta(key) == iso(datetime.fromtimestamp(NOW, tz=UTC))

    later = org_harness(store, gh)  # restart a day later: same cutoff
    later.now[0] += 86400
    later.poll()
    assert later.poller.cutoff == NOW

    override = org_harness(store, gh, ORG_MODE_SINCE="2027-01-01T00:00:00Z")
    override.poll()
    assert override.poller.cutoff == datetime(2027, 1, 1, tzinfo=UTC).timestamp()
    assert store.get_meta(key) == iso(datetime.fromtimestamp(NOW, tz=UTC))


def test_inaccessible_archived_and_forked_repos_are_skipped(store: Store):
    gh = FakeGitHub()
    gh.add_repo(NEW)
    gh.add_repo("sample-org/old", archived=True)
    gh.add_repo("sample-org/forked", fork=True)
    gh.add_hit(org_pr(1, repo="sample-org/not-installed"))
    gh.add_hit(org_pr(2, repo="sample-org/old"))
    gh.add_hit(org_pr(3, repo="sample-org/forked"))
    gh.add_hit(org_pr(4, repo="Other-Org/x"))  # not in this org's search
    gh.add_hit(org_pr(5))
    h = org_harness(store, gh)
    h.poll()
    assert reviewed_urls(h) == [f"https://github.com/{NEW}/pull/5"]
    outcomes = h.poller.status()["discovery"][ORG]["last"]["outcomes"]
    assert outcomes == {
        "not_accessible": 1,
        "archived": 1,
        "fork": 1,
        "review": 1,
    }
    assert h.poller.status()["repo_count"] == 2  # new-thing + pilot (archived/fork excluded)


def test_opt_outs(store: Store):
    gh = FakeGitHub()
    gh.add_repo(NEW)
    gh.add_repo("sample-org/topic-out", topics=("no-ai-review",))
    gh.add_repo("sample-org/file-out", default_branch="develop")
    gh.files[("sample-org/file-out", ".github/gh-pr-reviewer.yml", "develop")] = "enabled: false\n"
    gh.add_hit(org_pr(1, labels=("No-AI-Review",)))
    gh.add_hit(org_pr(2, title="chore: bump [Skip Review]"))
    gh.add_hit(org_pr(3, repo="sample-org/topic-out"))
    gh.add_hit(org_pr(4, repo="sample-org/file-out"))
    gh.add_hit(org_pr(5, author_login="github-actions[bot]", author_type="Bot"))
    gh.add_hit(org_pr(6, draft=True))
    gh.add_hit(org_pr(7))
    h = org_harness(store, gh)
    h.poll()
    assert reviewed_urls(h) == [f"https://github.com/{NEW}/pull/7"]
    outcomes = h.poller.status()["discovery"][ORG]["last"]["outcomes"]
    assert outcomes == {"opted_out": 2, "repo_opted_out": 2, "bot": 1, "draft": 1, "review": 1}


def test_repo_opt_out_is_cached(store: Store):
    calls = []

    class Counting(FakeGitHub):
        async def get_repo(self, repo):
            calls.append(repo)
            return await super().get_repo(repo)

    gh = Counting()
    gh.add_repo(NEW)
    for n in range(1, 4):
        gh.add_hit(org_pr(n, sha=f"{n:040d}"))
    h = org_harness(store, gh)
    h.poll()
    assert len(h.agent.invocations) == 3 and calls == [NEW]


# --- cost guards ---------------------------------------------------------------------------
def test_budget_at_90_percent_pauses_first_reviews_but_not_rereviews(store: Store):
    gh = FakeGitHub()
    gh.add_repo(NEW)
    gh.add_hit(org_pr(1))
    spend = [0.0]
    store.cost_since = lambda _start: {"cost": spend[0]}
    h = org_harness(store, gh, LLM_MONTHLY_BUDGET_USD=2000.0)
    h.poll()
    assert len(h.agent.invocations) == 1

    spend[0] = 1820.0
    gh.add_hit(org_pr(1, sha="b" * 40, updated_at=t(100)))  # round 2: still runs
    gh.add_hit(org_pr(2, updated_at=t(100)))  # first review: paused
    h.now[0] += 60
    h.poll()
    assert reviewed_urls(h) == [org_pr(1).html_url] * 2
    budget = h.poller.status()["budget"]
    assert budget["state"] == "paused_full" and budget["fraction"] == pytest.approx(0.91)
    assert gh.created_comments == []  # nothing posted about the pause

    spend[0] = 1990.0
    gh.add_hit(org_pr(3, updated_at=t(200)))
    h.now[0] += 60
    h.poll()
    assert len(h.agent.invocations) == 2
    assert h.poller.status()["budget"]["state"] == "stopped"

    spend[0] = 10.0  # simulated monthly quota reset
    h.now[0] += 900  # next reconcile sees the held PRs again
    h.poll()
    assert len(h.agent.invocations) == 4


def test_budget_falls_back_to_the_local_cost_log(store: Store):
    gh = FakeGitHub()
    gh.add_repo(NEW)
    gh.add_hit(org_pr(1))
    h = org_harness(store, gh, LLM_MONTHLY_BUDGET_USD=0.01)
    rid = store.start_review(NEW, 9, "", "", "x" * 40, "review")
    billed_call(store, NEW, 9, rid, 1)
    h.poll()
    assert h.agent.invocations == []
    budget = h.poller.status()["budget"]
    assert budget["source"] == "local" and budget["state"] == "stopped"


def posted_round(store: Store, number: int, sha: str, n: int, kind: str = "full") -> None:
    rid = store.start_review(NEW, number, "", "", sha, "review", kind=kind)
    billed_call(store, NEW, number, rid, n)
    store.finish_review(rid, status="ok", duration_s=1.0)


def test_max_rounds_cap_is_recorded_once_per_head_and_counted_in_stats(store: Store):
    from app import stats

    gh = FakeGitHub()
    gh.add_repo(NEW)
    posted_round(store, 1, "1" * 40, 1)
    posted_round(store, 1, "2" * 40, 2)
    posted_round(store, 1, "3" * 40, 3, kind="on_demand")  # on-demand: not a round
    store.mark_reviewed(NEW, 1, "2" * 40, "review")
    store.set_org_live(NEW, 1)
    gh.add_hit(org_pr(1, sha="e" * 40))
    h = org_harness(store, gh)
    h.poll()
    h.now[0] += 900
    h.poll()
    assert h.agent.invocations == []
    skips = [r for r in store.recent_reviews(include_poller=True) if r["status"] == "skipped"]
    assert [(r["skip_reason"], r["kind"]) for r in skips] == [("max_rounds", "full")]
    assert "max rounds: 2 automatic reviews" in skips[0]["error"]
    summary = stats.summary(store, days=1, poller=h.poller.status(), version="v", effort="x")
    assert summary["totals"]["skipped"]["max_rounds"] == 1
    rows = stats.reviews(store, limit=10)["reviews"]
    assert rows[0]["skip_reason"] == "max_rounds" and rows[0]["status"] == "skipped"


def test_on_demand_review_bypasses_the_round_cap_with_its_own_daily_cap(store: Store):
    gh = FakeGitHub()
    gh.add_repo(NEW)
    posted_round(store, 1, "1" * 40, 1)
    posted_round(store, 1, "2" * 40, 2)
    store.mark_reviewed(NEW, 1, "2" * 40, "review")
    store.set_org_live(NEW, 1)
    h = org_harness(store, gh, MAX_ON_DEMAND_REVIEWS_PER_PR_PER_DAY=2)
    for n, cid in enumerate((901, 902, 903)):
        h.now[0] += 600
        at = t(600 * (n + 1))
        # same head every time: `force` asks again although it was reviewed (#7b)
        comment = PrComment(cid, "/review force", at, "sample-user", "User")
        gh.add_hit(org_pr(1, sha="e" * 40, updated_at=at, comments=(comment,)))
        h.poll()
    kinds = [inv["kind"] for inv in h.agent.invocations]
    assert kinds == ["on_demand", "on_demand"]  # the third request waits for the 24 h window
    rows = store.recent_reviews(include_poller=True)
    assert rows[0]["skip_reason"] == "daily_cap"
    assert "on-demand" in rows[0]["error"]
    ok = [r for r in rows if r["status"] == "ok" and r["kind"] == "on_demand"]
    assert len(ok) == 2 and all(r["review_round"] is None for r in ok)
    assert store.get_pr_state(NEW, 1).auto_rounds == 2  # on-demand reviews are not rounds
    assert store.get_pr_state(NEW, 1).last_trigger_id == 902


def test_a_diff_larger_than_the_model_context_is_still_reviewed(store: Store):
    gh = FakeGitHub()
    gh.add_repo(NEW)
    gh.add_hit(org_pr(1))
    patch = "@@ -0,0 +1,400000 @@\n" + "+x\n" * 400_000  # ~1.2M chars: no size cap
    gh.pr_files[(NEW, 1)] = [
        {"filename": "big.py", "status": "added", "patch": patch, "additions": 400_000}
    ]
    h = org_harness(store, gh)
    h.poll()
    assert len(h.agent.invocations) >= 3 and gh.created_comments == []
    [review] = gh.reviews[(NEW, 1)]
    assert "reviewed in" in review["body"] and "parts</sub>" in review["body"]
    row = store.recent_reviews()[0]
    assert row["status"] == "ok" and row["llm_calls"] == len(h.agent.invocations)


def test_llm_timeout_counts_as_one_retryable_attempt(store: Store):
    gh = FakeGitHub()
    gh.add_repo(NEW)
    gh.add_hit(org_pr(1))
    h = org_harness(store, gh)

    h.agent.error = TimeoutError("litellm.Timeout: Request timed out after 900s")
    h.poll()
    state = store.get_pr_state(NEW, 1)
    assert state.failed_attempts == 1  # timeout consumes one persisted attempt
    h.now[0] += 3600
    h.poll()
    assert len(h.agent.invocations) == 2
    assert store.get_pr_state(NEW, 1).failed_attempts == 2


# --- queue -----------------------------------------------------------------------------------
def test_small_prs_are_not_stuck_behind_large_ones(store: Store):
    gh = FakeGitHub()
    gh.add_repo(NEW)
    for n in range(1, 5):  # four giant PRs in four repos, queued first
        repo = f"sample-org/big-{n}"
        gh.add_repo(repo)
        gh.add_hit(org_pr(n, repo=repo))
        gh.pr_files[(repo, n)] = [
            {"filename": "a.py", "status": "modified", "patch": "+x", "additions": 5000}
        ]
    gh.add_hit(org_pr(9))  # one small PR
    gh.pr_files[(NEW, 9)] = [
        {"filename": "a.py", "status": "modified", "patch": "+x", "additions": 3}
    ]
    h = org_harness(store, gh, MAX_CONCURRENT_REVIEWS=2, MAX_CONCURRENT_LARGE_REVIEWS=1)
    h.agent.delay = 0.02
    h.poll()
    order = reviewed_urls(h)
    assert order[0] == f"https://github.com/{NEW}/pull/9"  # small first, despite queue order
    assert len(order) == 5


def test_queue_holds_large_lane_and_per_repo_caps():
    from app.review_queue import ReviewQueue

    q = ReviewQueue(
        max_concurrent=6, max_large=2, max_per_repo=2, large_lines=400, aging_seconds=300
    )
    for n in range(4):
        q.push(("big", n), f"big-repo-{n}", 10_000, now=0)
    for n in range(4):
        q.push(("burst", n), "burst-repo", 10, now=0)
    q.push(("other", 1), "other-repo", 50, now=0)
    started = [q.next_job(1) for _ in range(6)]
    started = [j for j in started if j]
    assert sum(j.large for j in started) == 2  # large lane capped
    assert sum(j.repo == "burst-repo" for j in started) == 2  # per-repo cap
    assert ("other", 1) in {j.key for j in started}
    assert len(started) == 5  # the 6th slot only fits a large job, and that lane is full


def test_queue_aging_eventually_runs_a_large_job_first():
    from app.review_queue import ReviewQueue

    q = ReviewQueue(
        max_concurrent=1, max_large=1, max_per_repo=1, large_lines=400, aging_seconds=300
    )
    q.push(("big", 1), "a", 20_000, now=0)
    q.push(("small", 1), "b", 10, now=3600)  # big has waited an hour
    assert q.next_job(3600).key == ("big", 1)


def test_requeue_keeps_wait_time_and_running_prs_are_locked():
    from app.review_queue import ReviewQueue

    q = ReviewQueue(
        max_concurrent=1, max_large=1, max_per_repo=1, large_lines=400, aging_seconds=300
    )
    q.push(("r", 1), "r", 10, now=0)
    assert q.push(("r", 1), "r", 20, now=100).enqueued_at == 0
    job = q.next_job(100)
    assert q.push(job.key, "r", 10, now=200) is None  # running: not queued twice
    q.finish(job.key)
    assert q.snapshot(200)["running"] == 0


# --- graceful shutdown & recovery --------------------------------------------------------------
def test_drain_lets_running_reviews_finish_and_drops_the_queue(store: Store):
    gh = FakeGitHub()
    gh.add_repo(NEW)
    for n in range(1, 4):
        gh.add_hit(org_pr(n, sha=f"{n:040d}"))
    h = org_harness(store, gh, MAX_CONCURRENT_REVIEWS=1)
    h.agent.delay = 0.2

    async def run():
        await h.poller.setup()
        h.poller._stop = asyncio.Event()
        await h.poller.poll_once()
        assert h.poller.status()["queue"]["queued"] == 2
        h.poller._stop.set()
        await h.poller.drain(grace=5)

    asyncio.run(run())
    assert len(h.agent.invocations) == 1  # the running one finished; queued ones dropped
    assert store.recent_reviews()[0]["status"] == "ok"
    assert h.poller.status()["state"] == "draining"


def test_drain_cancels_reviews_that_outlive_the_grace_period(store: Store):
    gh = FakeGitHub()
    gh.add_repo(NEW)
    gh.add_hit(org_pr(1))
    h = org_harness(store, gh)
    h.agent.delay = 5

    async def run():
        await h.poller.setup()
        h.poller._stop = asyncio.Event()
        await h.poller.poll_once()
        await asyncio.sleep(0.05)
        await h.poller.drain(grace=0.1)

    asyncio.run(run())
    assert store.recent_reviews()[0]["status"] == "running"  # restart marks it interrupted
    assert store.get_pr_state(NEW, 1).last_reviewed_sha is None


def test_interrupted_review_whose_comment_landed_is_recovered_not_rebilled(store: Store):
    gh = FakeGitHub()
    gh.add_repo(NEW)
    gh.add_hit(org_pr(1))
    store.set_org_live(NEW, 1)
    rid = store.start_review(NEW, 1, f"https://github.com/{NEW}/pull/1", "t", "a" * 40, "review")
    billed_call(store, NEW, 1, rid, 1)
    gh.comments[(NEW, 1)] = [
        {
            "id": 77,
            "user": {"login": "gh-pr-reviewer[bot]"},
            "body": "<!-- pr-agent:review:full -->\n## PR Reviewer Guide\nok",
            "updated_at": (utcnow() + timedelta(seconds=90)).isoformat(),
            "html_url": f"https://github.com/{NEW}/pull/1#issuecomment-77",
        }
    ]
    assert store.interrupt_running() == 1  # the restart
    assert store.get_review(rid)["llm_calls"] == 1  # cost rolled up on interrupt
    h = org_harness(store, gh)
    h.poll()
    assert h.agent.invocations == []  # not re-billed
    assert store.get_pr_state(NEW, 1).last_reviewed_sha == "a" * 40
    review = store.get_review(rid)
    assert review["status"] == "ok" and review["comment_url"].endswith("issuecomment-77")
    assert "gh-pr-reviewer · gpt-4.1" in gh.updated[-1][2]  # footer attached


@pytest.mark.parametrize("previous_footer", [False, True])
def test_interrupted_review_without_comment_is_reviewed_again(store: Store, previous_footer):
    gh = FakeGitHub()
    gh.add_repo(NEW)
    gh.add_hit(org_pr(1))
    store.set_org_live(NEW, 1)
    rid = store.start_review(NEW, 1, "", "t", "a" * 40, "review")
    billed_call(store, NEW, 1, rid, 1)
    if previous_footer:  # the previous run's comment, last edited just before this run began
        gh.comments[(NEW, 1)] = [
            {
                "id": 76,
                "user": {"login": "gh-pr-reviewer[bot]"},
                "body": "<!-- pr-agent:review:full -->\nold review",
                "updated_at": (utcnow() - timedelta(seconds=1)).isoformat(),
                "html_url": "x",
            }
        ]
    store.interrupt_running()
    h = org_harness(store, gh)
    h.poll()
    assert len(h.agent.invocations) == 1
    assert store.get_review(rid)["status"] == "interrupted"


def test_dry_run_org_mode_queues_nothing(store: Store):
    gh = FakeGitHub()
    gh.add_repo(NEW)
    gh.add_hit(org_pr(1))
    h = org_harness(store, gh, DRY_RUN=True)
    h.poll()
    assert h.agent.invocations == []
    assert h.poller.status()["recent_decisions"][0]["mode"] == Mode.FULL.value


def test_sigterm_path_waits_for_the_running_review_in_the_poller_thread(store: Store):
    import time

    gh = FakeGitHub()
    gh.add_repo(NEW)
    gh.add_hit(org_pr(1))
    h = org_harness(store, gh, POLL_INTERVAL_SECONDS=3600)
    h.agent.delay = 0.5
    h.poller.start()
    deadline = time.monotonic() + 10
    while not h.agent.invocations and time.monotonic() < deadline:
        time.sleep(0.01)
    assert h.agent.invocations, "review never started"
    h.poller.stop(timeout=10)  # what the FastAPI lifespan does on SIGTERM
    assert store.recent_reviews()[0]["status"] == "ok"
    assert store.get_pr_state(NEW, 1).last_reviewed_sha == "a" * 40
    assert h.poller.status()["state"] == "stopped"


@pytest.mark.parametrize(
    ("fraction", "state"),
    [(None, "ok"), (0.5, "ok"), (0.9, "paused_full"), (0.989, "paused_full"), (0.99, "stopped")],
)
def test_budget_classify(fraction, state):
    from app.budget import classify

    assert classify(fraction, 0.9, 0.99) == state


def test_discovery_outcomes_count_running_and_queued_reviews_separately(store: Store):
    gh = FakeGitHub()
    gh.add_repo(NEW)
    for n in range(1, 4):
        gh.add_hit(org_pr(n, sha=f"{n:040d}"))
    h = org_harness(store, gh, MAX_CONCURRENT_REVIEWS=1)
    h.agent.delay = 0.2

    async def run():
        await h.poller.setup()
        await h.poller.poll_once()  # 1 starts, 2 wait
        await asyncio.sleep(0.05)
        h.now[0] += 60
        await h.poller.poll_once()  # same heads again: nothing new is queued
        outcomes = h.poller.status()["discovery"][ORG]["last"]["outcomes"]
        await h.poller.wait_idle()
        return outcomes

    assert asyncio.run(run()) == {"running": 1, "queued": 2}
    assert len(h.agent.invocations) == 3
