"""native inline reviews end to end (poller -> reviewer -> GitHub review), plus on-demand
`/review` requests and restart recovery, against the in-memory fakes."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from app.github_app import PrComment
from app.review_policy import find_trigger, retry_delay_seconds
from app.store import Store
from tests.fakes import FakeGitHub, make_pr
from tests.test_org_mode import NEW, NOW, org_harness, org_pr, t
from tests.test_poller import REPO, Harness

PATCH = "@@ -1,4 +1,5 @@\n import os\n-x = 1\n+x = 2\n+y = x + 1\n def f():\n     return x"
FILES = [{"filename": "a.py", "status": "modified", "patch": PATCH, "additions": 2}]


@pytest.fixture
def store(tmp_path) -> Store:
    return Store(tmp_path / "copilot.sqlite3")


def answer(*findings: dict, overview: str = "Changes `x`.") -> dict:
    return {
        "overview": overview,
        "files": [{"path": "a.py", "summary": "Bumps x."}],
        "findings": list(findings),
    }


def bug(start: int = 2, end: int = 2, **kw) -> dict:
    return {
        "path": "a.py",
        "start_line": start,
        "end_line": end,
        "severity": "major",
        "title": "x must stay one",
        "body": "Callers divide by `x`; changing it breaks the ratio.",
        "suggestion": "x = 1",
    } | kw


def setup(store: Store, *findings: dict, **settings) -> tuple[FakeGitHub, Harness]:
    gh = FakeGitHub({REPO: [make_pr(sha="a" * 40)]})
    gh.pr_files[(REPO, 1)] = [dict(f) for f in FILES]
    h = Harness(store, gh, **settings)
    h.agent.response = answer(*findings)
    return gh, h


def human(comment_id: int, body: str = "/review", *, at: float = NOW + 60, **kw) -> dict:
    return {
        "id": comment_id,
        "body": body,
        "created_at": datetime.fromtimestamp(at, UTC).isoformat(),
        "updated_at": datetime.fromtimestamp(at, UTC).isoformat(),
        "issue_url": f"https://api.github.com/repos/{REPO}/issues/1",
        "user": kw.get("user", {"login": "sample-user", "type": "User"}),
    }


# --- the review ---------------------------------------------------------------------------
def test_review_is_a_native_pr_review_with_inline_comments_and_a_suggestion(store: Store):
    gh, h = setup(store, bug(), bug(3, 3, severity="minor", title="Name", suggestion=None,
                                    body="`y` duplicates `x + 1` from `f`."))  # fmt: skip
    h.poll()
    [review] = gh.reviews[(REPO, 1)]
    assert review["commit_id"] == "a" * 40
    major, minor = review["comments"]
    assert (major["path"], major["line"], major["side"]) == ("a.py", 2, "RIGHT")
    assert major["body"].startswith("**[major]** Callers divide by `x`")
    assert "```suggestion\nx = 1\n```" in major["body"]
    assert not minor["body"].startswith("**[") and "```suggestion" not in minor["body"]
    text = review["body"]
    assert "## Pull request overview\n\nChanges `x`." in text
    assert "| `a.py` | Bumps x. |" in text
    assert "**2 comments:** 2 issues, 0 suggestions" in text
    assert gh.created_comments == [] and gh.updated == []
    row = store.recent_reviews()[0]
    assert row["status"] == "ok" and row["comment_url"] == review["html_url"]
    assert (row["findings_major"], row["findings_minor"], row["inline_comments"]) == (1, 1, 2)
    prompt = h.agent.invocations[0]
    assert "3 +y = x + 1" in prompt["user"] and "Description of PR 1" in prompt["user"]
    assert prompt["reasoning_effort"] == "xhigh" and '"findings"' in prompt["system"]


def test_clean_pr_gets_a_short_review_without_inline_comments(store: Store):
    gh, h = setup(store)
    h.poll()
    [review] = gh.reviews[(REPO, 1)]
    assert review["comments"] == [] and "No issues found." in review["body"]


def test_finding_outside_the_diff_is_folded_into_the_body(store: Store):
    gh, h = setup(store, bug(40, 41, title="Stale cache", suggestion="x"))
    h.poll()
    [review] = gh.reviews[(REPO, 1)]
    assert review["comments"] == []
    assert "<summary>1 comment outside the diff</summary>" in review["body"]
    assert "- `a.py:40-41` — **[major]** **Stale cache.**" in review["body"]


def test_github_refusing_an_anchor_folds_the_comments_into_the_body(store: Store):
    gh, h = setup(store, bug())
    gh.reject_inline = True
    h.poll()
    [review] = gh.reviews[(REPO, 1)]
    assert review["comments"] == [] and "Callers divide by `x`" in review["body"]
    assert store.recent_reviews()[0]["status"] == "ok"


def test_a_transient_posting_error_is_retried_once_without_a_second_model_call(store: Store):
    gh, h = setup(store, bug())
    h.reviewer.POST_RETRY_DELAY_S = 0
    real = gh.create_review
    failures = [RuntimeError("502 Bad Gateway")]

    async def flaky(*args, **kw):
        if failures:
            raise failures.pop()
        return await real(*args, **kw)

    gh.create_review = flaky
    h.poll()
    assert len(gh.reviews[(REPO, 1)]) == 1 and len(h.agent.invocations) == 1
    assert store.recent_reviews()[0]["status"] == "ok"


def test_unusable_model_output_fails_the_review_without_posting(store: Store):
    gh, h = setup(store)
    h.agent.response = "Sorry, I cannot help with that."
    h.poll()
    assert gh.reviews == {}
    row = store.recent_reviews()[0]
    assert row["status"] == "error" and "ModelOutputError" in row["error"]
    assert store.get_pr_state(REPO, 1).failed_attempts == 1


def test_incremental_review_comments_only_on_lines_of_the_new_commits(store: Store):
    # The reviewer's incremental mode (no longer chosen by the poller; round 2 is full).
    import asyncio

    from app.review_policy import Mode

    gh, h = setup(store)
    new_patch = PATCH + "\n+z = 3"
    gh.pulls[REPO] = [make_pr(sha="b" * 40)]
    gh.pr_files[(REPO, 1)] = [dict(FILES[0], patch=new_patch)]
    gh.compares[(REPO, "a" * 40, "b" * 40)] = [
        {"filename": "a.py", "status": "modified", "patch": "@@ -5,0 +6 @@\n+z = 3"}
    ]
    gh.compare_commits[(REPO, "a" * 40, "b" * 40)] = 2
    h.agent.response = answer(
        bug(6, 6, title="z unused", suggestion=None, body="`z` is never read."),
        bug(2, 2),  # already-reviewed code: not commented inline
    )

    async def run():
        await h.poller.setup()
        await h.reviewer.review(make_pr(sha="b" * 40), Mode.INCREMENTAL, 1,
                                last_reviewed_sha="a" * 40, kind="incremental")  # fmt: skip

    asyncio.run(run())
    [second] = gh.reviews[(REPO, 1)]
    assert [c["line"] for c in second["comments"]] == [6]
    assert "Reviewed 2 new commits since `aaaaaaa`." in second["body"]
    assert "1 comment outside the diff" in second["body"]
    prompt = h.agent.invocations[0]["user"]
    assert "Incremental review" in prompt and "+z = 3" in prompt and "-x = 1" not in prompt


# --- round 2: full re-review, only new findings -------------------------------------------
PATCH2 = (
    "@@ -1,4 +1,7 @@\n import os\n+import sys\n+\n-x = 1\n+x = 2\n+y = x + 1\n+w = y * 2"
    "\n def f():\n     return x"
)


def push_round_two(gh: FakeGitHub, h: Harness) -> None:
    gh.pulls[REPO] = [make_pr(sha="b" * 40)]
    gh.pr_files[(REPO, 1)] = [dict(FILES[0], patch=PATCH2)]
    h.now[0] += 600
    h.poll()


def test_round_two_posts_only_findings_not_raised_before(store: Store):
    gh, h = setup(
        store,
        bug(),  # inline at line 2
        bug(3, 3, severity="minor", title="Shadowed name", suggestion=None,
            body="`y` shadows the module-level `y` used by `report()`."),
        bug(9, 9, title="Missing env check", suggestion=None,  # outside the diff: in the body
            body="`DATABASE_URL` is read without a default in `settings.load()`."),
    )  # fmt: skip
    h.poll()
    [first] = gh.reviews[(REPO, 1)]
    assert len(first["comments"]) == 2 and "Missing env check" in first["body"]
    shadow = next(c for c in gh.review_comments[(REPO, 1)] if "shadows" in c["body"])
    gh.resolved_comments.add(shadow["id"])  # a human resolved that thread
    h.agent.response = answer(
        # same issue, moved two lines down by the push and reworded (title changed)
        bug(4, 4, title="x changed from one", suggestion=None,
            body="Callers divide by `x`, so changing it from 1 breaks the ratio they compute."),
        # the resolved thread's issue, loosely reworded
        bug(5, 5, severity="minor", title="Name clash", suggestion=None,
            body="`y` clashes with the module-level `y` that `report()` reads."),
        # the body finding again, different line
        bug(12, 12, title="Missing env check", suggestion=None, body="Still unchecked."),
        # genuinely new
        bug(6, 6, title="w is never used", suggestion=None, body="`w` is assigned but unused."),
    )  # fmt: skip
    push_round_two(gh, h)
    first, second = gh.reviews[(REPO, 1)]
    assert [c["line"] for c in second["comments"]] == [6]
    assert (
        "Re-reviewed the whole PR at `bbbbbbb` after new commits (automatic review 2 of 2)"
        in (second["body"])
    )
    assert "3 findings already raised in an earlier review are not repeated." in second["body"]
    assert "This was the last automatic review of this PR" in second["body"]
    assert "round=2 -->" in second["body"]
    prompt = h.agent.invocations[1]["user"]
    assert "Full re-review of the whole PR" in prompt and "(the last automatic one)" in prompt
    assert "Already reported by earlier reviews" in prompt
    assert "`a.py:9` — Missing env check." in prompt  # the body finding is listed too
    assert "(thread resolved)" in prompt
    row = store.recent_reviews()[0]
    assert (row["review_round"], row["kind"], row["inline_comments"]) == (2, "full", 1)


def test_round_two_with_nothing_new_posts_a_short_note(store: Store):
    gh, h = setup(store, bug())
    h.poll()
    h.agent.response = answer(bug(4, 4, suggestion=None))  # the same finding, moved
    push_round_two(gh, h)
    first, second = gh.reviews[(REPO, 1)]
    assert second["comments"] == []
    body = second["body"]
    assert "## Pull request overview" not in body and "Reviewed changes" not in body
    assert "nothing new beyond the earlier review. 1 finding already raised is not repeated." in (
        body
    )
    assert "comment `/review` to ask for another" in body
    assert len(body.splitlines()) <= 7


def test_on_demand_review_still_works_after_the_round_cap(store: Store):
    gh, h = setup(store)
    h.poll()
    push_round_two(gh, h)
    gh.pulls[REPO] = [make_pr(sha="c" * 40)]
    h.now[0] += 600
    h.poll()
    assert len(gh.reviews[(REPO, 1)]) == 2  # capped
    gh.repo_comments[REPO] = [human(990, at=h.now[0] + 10)]
    h.now[0] += 60
    h.poll()
    assert len(gh.reviews[(REPO, 1)]) == 3
    assert h.agent.invocations[-1]["kind"] == "on_demand"
    third = gh.reviews[(REPO, 1)][2]["body"]
    assert "## Pull request overview" in third and "round=" not in third
    assert store.get_pr_state(REPO, 1).auto_rounds == 2


def test_failed_forced_on_demand_review_retries_once_under_its_daily_cap(store: Store):
    gh, h = setup(store, MAX_ON_DEMAND_REVIEWS_PER_PR_PER_DAY=1)
    h.poll()
    push_round_two(gh, h)
    gh.pulls[REPO] = [make_pr(sha="b" * 40, draft=True)]
    gh.repo_comments[REPO] = [human(990, "/review force", at=h.now[0] + 10)]
    h.now[0] += 60

    original_create_review = gh.create_review

    async def fail_post(*args, **kwargs):
        raise RuntimeError("temporary GitHub post failure")

    gh.create_review = fail_post
    h.reviewer.POST_RETRY_DELAY_S = 0
    h.poll()
    state = store.get_pr_state(REPO, 1)
    assert state.last_reviewed_sha == "b" * 40 and state.failed_sha == "b" * 40
    assert state.failed_trigger == "on_demand" and state.failed_force
    assert state.failed_requested_by == "sample-user" and state.last_trigger_id == 990
    assert (
        store.billed_reviews_since(
            REPO, 1, datetime.now(UTC) - timedelta(hours=24), kind="on_demand"
        )
        == 1
    )

    gh.create_review = original_create_review
    delay = retry_delay_seconds(REPO, 1, "b" * 40, 1, base_seconds=60, max_seconds=3600)
    h.now[0] += delay + 1
    h.poll()  # retry even though the request is a force-review of a settled draft at round cap

    state = store.get_pr_state(REPO, 1)
    assert state.last_reviewed_sha == "b" * 40 and state.failed_trigger is None
    attempts = [r for r in store.recent_reviews() if r["kind"] == "on_demand"]
    assert [r["status"] for r in attempts] == ["ok", "error"]  # newest first
    assert all(r["on_demand_request_id"] == 990 for r in attempts)
    assert (
        store.billed_reviews_since(
            REPO, 1, datetime.now(UTC) - timedelta(hours=24), kind="on_demand"
        )
        == 1
    )
    assert len(gh.reviews[(REPO, 1)]) == 3


def test_findings_already_posted_are_not_repeated(store: Store):
    gh, h = setup(store, bug())
    h.poll()
    gh.repo_comments[REPO] = [human(501, "/review force", at=NOW + 120)]
    h.now[0] += 300
    h.poll()  # forced on-demand re-review of the same head: same finding again
    first, second = gh.reviews[(REPO, 1)]
    assert len(first["comments"]) == 1 and second["comments"] == []
    assert "No new issues found." in second["body"]
    assert "1 finding already raised in an earlier review is not repeated." in second["body"]
    assert "x must stay one" not in h.agent.invocations[0]["user"]
    assert "Already reported by earlier reviews" in h.agent.invocations[1]["user"]


# --- on-demand requests -----------------------------------------------------------------
def test_find_trigger_rules():
    def c(cid, body="/review", login="sample-user", typ="User", at=NOW + 10):
        return PrComment(cid, body, datetime.fromtimestamp(at, UTC).isoformat(), login, typ)

    pick = lambda cs, **kw: find_trigger(cs, after_id=kw.get("after"), not_before=NOW)  # noqa: E731
    assert pick([c(1)]).id == 1
    assert pick([c(1, "@gh-pr-reviewer review please")]).id == 1
    assert pick([c(1, "LGTM\n/review")]).id == 1
    assert pick([c(1, "/reviewed it, thanks")]) is None
    assert pick([c(1, "> /review\nwhy did it say that?")]) is None  # quoted
    assert pick([c(1, "```\n/review\n```")]) is None
    assert pick([c(1, login="copilot[bot]", typ="Bot")]) is None
    assert pick([c(1, at=NOW - 10)]) is None  # before the feature existed
    assert pick([c(1), c(5), c(3)]).id == 5
    assert pick([c(1), c(5)], after=5) is None


def test_review_comment_requests_a_fresh_full_review_of_a_reviewed_head(store: Store):
    gh, h = setup(store, REVIEW_SETTLE_SECONDS=300)
    h.poll()  # first sight starts the settle wait
    assert gh.reviews == {}
    h.now[0] += 400
    h.poll()
    assert len(gh.reviews[(REPO, 1)]) == 1
    gh.repo_comments[REPO] = [
        human(600, "/review", user={"login": "github-actions[bot]", "type": "Bot"}),
        human(601, "please @gh-pr-reviewer review", at=NOW + 450),  # not at line start
    ]
    h.now[0] += 60
    h.poll()
    assert len(gh.reviews[(REPO, 1)]) == 1  # bots and non-commands are ignored
    gh.repo_comments[REPO].append(human(602, "@gh-pr-reviewer review force", at=NOW + 470))
    h.now[0] += 60
    h.poll()  # no settle wait for an explicit request
    assert len(gh.reviews[(REPO, 1)]) == 2
    assert [inv["kind"] for inv in h.agent.invocations] == ["full", "on_demand"]
    assert "requested by @sample-user" in h.agent.invocations[1]["user"]
    assert gh.reactions == [(REPO, 602, "eyes")]
    row = store.recent_reviews()[0]
    assert row["kind"] == "on_demand" and row["status"] == "ok"
    assert store.get_pr_state(REPO, 1).last_trigger_id == 602
    h.now[0] += 600
    h.poll()  # the request is handled once
    assert len(gh.reviews[(REPO, 1)]) == 2


def test_review_request_on_a_draft_is_honoured(store: Store):
    gh, h = setup(store)
    gh.pulls[REPO] = [make_pr(sha="a" * 40, draft=True)]
    h.poll()
    assert gh.reviews == {}
    gh.repo_comments[REPO] = [human(700)]
    gh.pull_overrides[(REPO, 1)] = {"state": "open", "draft": True, "head": {"sha": "a" * 40}}
    h.now[0] += 120
    h.poll()
    assert len(gh.reviews[(REPO, 1)]) == 1


def test_review_requests_have_their_own_daily_cap(store: Store):
    gh, h = setup(store, MAX_ON_DEMAND_REVIEWS_PER_PR_PER_DAY=1)
    h.poll()  # the automatic review does not count against it
    gh.repo_comments[REPO] = [human(800, "/review force", at=h.now[0] + 10)]
    h.now[0] += 120
    h.poll()
    assert len(gh.reviews[(REPO, 1)]) == 2
    gh.repo_comments[REPO].append(human(801, "/review force", at=h.now[0] + 10))  # still capped
    h.now[0] += 120
    h.poll()
    assert len(gh.reviews[(REPO, 1)]) == 2  # held by the cap; the request stays pending
    assert store.get_pr_state(REPO, 1).last_trigger_id == 800
    assert "daily_cap" in [r["skip_reason"] for r in store.recent_reviews(include_poller=True)]


def test_review_request_on_an_org_backlog_pr_reviews_it(store: Store):
    gh = FakeGitHub()
    gh.add_repo(NEW)
    backlog = org_pr(1, created_at=t(-86400), updated_at=t(-86400), head_committed_at=t(-86400))
    gh.add_hit(backlog)
    h = org_harness(store, gh)
    h.poll()
    assert h.agent.invocations == []  # backlog: not reviewed by itself
    comment = PrComment(901, "/review", t(120), "sample-user", "User")
    gh.add_hit(org_pr(1, created_at=t(-86400), updated_at=t(120), head_committed_at=t(-86400),
                      comments=(comment,)))  # fmt: skip
    h.now[0] += 180
    h.poll()
    assert [inv["kind"] for inv in h.agent.invocations] == ["on_demand"]
    assert store.get_pr_state(NEW, 1).org_live is True


# --- restart recovery -----------------------------------------------------------------------
def test_interrupted_review_whose_pr_review_landed_is_adopted(store: Store):
    gh, h = setup(store)
    review_id = store.start_review(REPO, 1, "u", "t", "a" * 40, "review")
    from tests.test_org_mode import billed_call

    billed_call(store, REPO, 1, review_id, 1)
    store.interrupt_running()
    gh.reviews[(REPO, 1)] = [
        {
            "id": 1,
            "user": {"login": "gh-pr-reviewer[bot]"},
            "commit_id": "a" * 40,
            "body": f"<!-- gh-pr-reviewer:review sha={'a' * 40} kind=full -->\nok",
            "submitted_at": datetime.now(UTC).isoformat(),
            "html_url": "https://github.com/x/pull/1#pullrequestreview-1",
        }
    ]
    h.poll()
    assert h.agent.invocations == []  # not re-billed
    row = store.get_review(review_id)
    assert row["status"] == "ok" and row["comment_url"].endswith("pullrequestreview-1")
    assert store.get_pr_state(REPO, 1).last_reviewed_sha == "a" * 40


# --- the PR merged or closed while the review ran ------------------------------------------
def _ends_during_review(gh: FakeGitHub, h: Harness, pull: dict) -> None:
    """The PR's state changes while the (slow) model call runs."""
    model = h.reviewer.model

    async def slow_model(request):
        out = await model(request)
        gh.pull_overrides[(REPO, 1)] = pull
        return out

    h.reviewer.model = slow_model


MERGED = {"state": "closed", "merged": True, "merged_at": "2026-10-08T19:52:00Z",
          "draft": False, "head": {"sha": "a" * 40}}  # fmt: skip
CLOSED = {"state": "closed", "merged": False, "merged_at": None,
          "draft": False, "head": {"sha": "a" * 40}}  # fmt: skip


def test_merged_during_review_posts_one_findings_comment_not_the_review(store: Store):
    gh, h = setup(store, bug(), bug(3, 3, severity="minor", title="Name", suggestion=None))
    _ends_during_review(gh, h, MERGED)
    h.poll()
    assert gh.reviews == {}  # no inline review on a merged PR
    [(repo, number, body)] = gh.created_comments
    assert (repo, number) == (REPO, 1)
    sha = "a" * 40
    assert f"<!-- gh-pr-reviewer:post-merge sha={sha} -->" in body
    assert "**Merged before review finished** — 2 findings at" in body
    assert f"[`aaaaaaa`](https://github.com/{REPO}/commit/{sha})" in body
    assert "follow up on `main`" in body
    major_line = f"- **[major]** [`a.py:2`](https://github.com/{REPO}/blob/{sha}/a.py#L2)"
    assert major_line + " — x must stay one" in body
    assert body.index("[major]") < body.index("a.py:3")
    assert "```suggestion" not in body and "gh-pr-reviewer · " in body
    row = store.recent_reviews()[0]
    assert row["status"] == "posted_post_merge" and row["skip_reason"] is None
    assert row["comment_url"] and row["cost_usd"] > 0 and row["llm_calls"] == 1
    assert (row["findings_major"], row["findings_minor"], row["inline_comments"]) == (1, 1, 0)
    assert store.get_pr_state(REPO, 1).last_reviewed_sha == sha  # settled: never retried
    assert store.auto_rounds(REPO, 1) == 0  # not a round (the PR is done anyway)

    from app import stats

    feed = stats.reviews(store, limit=10)["reviews"][0]
    assert feed["status"] == "posted_post_merge" and feed["review_url"] == row["comment_url"]
    assert feed["findings"]["major"] == 1 and feed["cost_usd"] > 0
    totals = stats.summary(store, days=1, poller={}, version="t", effort="xhigh")["totals"]
    assert totals["posted_post_merge"] == 1 and totals["reviews_posted"] == 0
    assert totals["billed_reviews"] == 1 and totals["cost_usd"] > 0


def test_closed_unmerged_during_review_posts_nothing(store: Store):
    gh, h = setup(store, bug())
    _ends_during_review(gh, h, CLOSED)
    h.poll()
    assert gh.reviews == {} and gh.created_comments == []
    row = store.recent_reviews()[0]
    assert (row["status"], row["skip_reason"]) == ("skipped", "closed_during_review")
    assert row["cost_usd"] > 0  # the model call was paid and stays counted
    assert store.get_pr_state(REPO, 1).last_reviewed_sha == "a" * 40

    from app import stats

    totals = stats.summary(store, days=1, poller={}, version="t", effort="xhigh")["totals"]
    assert totals["skipped"]["closed_during_review"] == 1 and totals["cost_usd"] > 0
    assert stats.reviews(store, limit=10)["reviews"][0]["skip_reason"] == "closed_during_review"


def test_merged_during_review_with_no_findings_posts_nothing(store: Store):
    gh, h = setup(store)
    _ends_during_review(gh, h, MERGED)
    h.poll()
    assert gh.reviews == {} and gh.created_comments == []
    row = store.recent_reviews()[0]
    assert (row["status"], row["skip_reason"]) == ("skipped", "merged_during_review")


def test_post_time_recheck_failing_posts_the_review_as_usual(store: Store):
    gh, h = setup(store, bug())
    model = h.reviewer.model

    async def slow_model(request):
        out = await model(request)

        async def broken(repo, number):
            raise RuntimeError("502 Bad Gateway")

        gh.get_pull = broken
        return out

    h.reviewer.model = slow_model
    h.poll()
    assert len(gh.reviews[(REPO, 1)]) == 1 and gh.created_comments == []
    assert store.recent_reviews()[0]["status"] == "ok"
