"""No size cap: a diff too large for one model call is reviewed in parts (one call each) and
posted as one review; PRs the old cap skipped are picked up again."""

from __future__ import annotations

import re
import sqlite3

import pytest

from app.pr_agent_config import reviewable_filter
from app.repo_context import diff_fingerprint
from app.reviewer import TOO_LARGE_MARKER
from app.store import RELEASE_TOO_LARGE_KEY, Store
from tests.fakes import BOT_LOGIN, FakeGitHub, make_pr
from tests.test_poller import REPO, Harness

# 8k tokens = 28k chars of rendered diff per model call (the tests' stand-in for ~160k).
SMALL_BUDGET = dict(MAX_REVIEW_DIFF_TOKENS=8_000)
FILE_HEADER = re.compile(r"^## File: '([^']+)'", re.MULTILINE)


@pytest.fixture
def store(tmp_path) -> Store:
    return Store(tmp_path / "large.sqlite3")


def added(path: str, lines: int = 200, width: int = 40) -> dict:
    body = "\n".join("+" + f"{path} line {n} ".ljust(width, "x") for n in range(1, lines + 1))
    return {
        "filename": path,
        "status": "added",
        "patch": f"@@ -0,0 +1,{lines} @@\n{body}",
        "additions": lines,
        "deletions": 0,
    }


BIG_PR = [added(f"{d}/m{n}.py") for d in ("api", "core", "tests") for n in (1, 2)]


def files_in(user_prompt: str) -> list[str]:
    diff = user_prompt.split("## Diff\n\n", 1)[1]
    return list(dict.fromkeys(FILE_HEADER.findall(diff)))


def one_finding_per_part(request) -> dict:
    """The model's answer for a part: one finding on the first file of its diff."""
    path = files_in(request.user)[0]
    return {
        "overview": f"Changes {path} and its neighbours.",
        "files": [{"path": p, "summary": f"Adds {p}."} for p in files_in(request.user)],
        "findings": [
            {
                "path": path,
                "start_line": 3,
                "end_line": 3,
                "severity": "major",
                "title": f"Unchecked input in {path}",
                "body": f"`{path}` trusts its input; a bad value crashes the handler.",
                "suggestion": None,
            }
        ],
    }


def big_pr(store: Store, files: list[dict] | None = None, **settings) -> tuple[FakeGitHub, Harness]:
    gh = FakeGitHub({REPO: [make_pr(sha="a" * 40)]})
    gh.pr_files[(REPO, 1)] = [dict(f) for f in (files or BIG_PR)]
    h = Harness(store, gh, **(SMALL_BUDGET | settings))
    h.agent.response = one_finding_per_part
    return gh, h


def test_a_pr_that_fits_is_one_call_as_before(store: Store):
    gh, h = big_pr(store, [added("api/m1.py", 20)])
    h.poll()
    [call] = h.agent.invocations
    assert "This review is in parts" not in call["user"]
    [review] = gh.reviews[(REPO, 1)]
    assert "Reviewed in" not in review["body"] and "parts</sub>" not in review["body"]
    assert len(review["comments"]) == 1


def test_an_oversized_pr_is_reviewed_in_parts_and_posted_as_one_review(store: Store):
    gh, h = big_pr(store)
    h.agent.delay = 0.02
    h.poll()
    calls = h.agent.invocations
    n = len(calls)
    assert n >= 2 and h.agent.max_active == 2  # parts run two at a time
    # every file in exactly one part, in path order; each part knows the whole PR
    seen = [p for c in calls for p in files_in(c["user"])]
    assert sorted(seen) == sorted(f["filename"] for f in BIG_PR) and len(set(seen)) == len(seen)
    for i, call in enumerate(calls, 1):
        assert f"## This review is in parts: part {i} of {n}" in call["user"]
        assert all(f"- `{f['filename']}` +200/-0" in call["user"] for f in BIG_PR)
        assert call["kind"] == "full" and call["reasoning_effort"] == "xhigh"
    # one review: every part's finding inline, one merged overview, footer says parts
    [review] = gh.reviews[(REPO, 1)]
    assert len(review["comments"]) == n
    assert {c["path"] for c in review["comments"]} == {files_in(c["user"])[0] for c in calls}
    body = review["body"]
    assert f"Reviewed in {n} parts (the diff is too large for one model call)" in body
    assert "- **Part 1** (" in body and f"- **Part {n}** (" in body
    assert f"Reviewed 6 of 6 changed files and generated {n} comments." in body
    assert body.endswith(f" · reviewed in {n} parts</sub>")
    # every call billed to this one review, and the footer adds them all up
    row = store.recent_reviews()[0]
    assert row["status"] == "ok" and row["llm_calls"] == n
    assert row["prompt_tokens"] == n * 10_000 and row["findings_major"] == n
    assert f"{n * 10:.1f}k in / {n * 2:.1f}k out" in body
    assert len(store.llm_calls_for_review(row["id"])) == n
    assert store.get_pr_state(REPO, 1).auto_rounds == 1


def test_a_single_file_larger_than_a_call_is_split_and_fully_reviewed(store: Store):
    giant = added("core/generated_but_reviewed.py", lines=2_000)
    gh, h = big_pr(store, [giant, added("api/small.py", 10)])

    def flag_line_1500(request):
        answer = one_finding_per_part(request)
        if "1500 +core/generated_but_reviewed.py line 1500" in request.user:
            answer["findings"] = [
                answer["findings"][0] | {"path": giant["filename"], "start_line": 1500,
                                         "end_line": 1500, "title": "Off by one at 1500"}
            ]  # fmt: skip
        return answer

    h.agent.response = flag_line_1500
    h.poll()
    calls = h.agent.invocations
    assert len(calls) >= 3
    shown = "\n".join(c["user"] for c in calls)
    for n in (1, 999, 1500, 2000):  # every line, with its real line number
        assert f"\n{n} +core/generated_but_reviewed.py line {n} " in shown
    assert "(part 2 of " in shown and "this file's diff" in shown
    [review] = gh.reviews[(REPO, 1)]
    lines = {(c["path"], c["line"]) for c in review["comments"]}
    assert (giant["filename"], 1500) in lines  # anchored on GitHub's own hunk


def test_a_failed_part_fails_the_review_without_posting_partial_results(store: Store):
    gh, h = big_pr(store)
    real = h.agent.response

    def second_part_breaks(request):
        if "part 2 of" in request.user:
            return "not json at all"
        return real(request)

    h.agent.response = second_part_breaks
    h.poll()
    assert gh.reviews == {}  # never settle/post an incomplete review
    row = store.recent_reviews()[0]
    assert row["status"] == "error" and "ModelOutputError" in row["error"]
    state = store.get_pr_state(REPO, 1)
    assert state.failed_sha == "a" * 40 and state.failed_attempts == 1
    assert state.last_reviewed_sha is None


def test_every_part_failing_fails_the_review_like_one_call(store: Store):
    gh, h = big_pr(store)
    h.agent.error = TimeoutError("litellm.Timeout: Request timed out after 900s")
    h.poll()
    assert gh.reviews == {}
    row = store.recent_reviews()[0]
    assert row["status"] == "error" and "Timeout" in row["error"]
    assert store.get_pr_state(REPO, 1).failed_attempts == 1  # timeout consumes one attempt


def test_round_two_of_a_big_pr_is_a_full_re_review_in_parts_with_only_new_findings(
    store: Store,
):
    gh, h = big_pr(store)
    h.poll()
    first = len(h.agent.invocations)
    gh.pulls[REPO] = [make_pr(sha="b" * 40)]
    gh.pr_files[(REPO, 1)][0] = added("api/m1.py", lines=201)  # the push grows one file
    gh.compares[(REPO, "a" * 40, "b" * 40)] = [gh.pr_files[(REPO, 1)][0]]
    real = h.agent.response

    def same_plus_one_new(request):
        answer = real(request)  # the round-1 findings again
        if "part 1 of" in request.user:
            answer["findings"].append(
                answer["findings"][0] | {"start_line": 7, "end_line": 7,
                                         "title": "Lock never released",
                                         "body": "The lock leaks when `save` raises."}
            )  # fmt: skip
        return answer

    h.agent.response = same_plus_one_new
    h.poll()
    second = h.agent.invocations[first:]
    assert len(second) == first
    for call in second:
        assert "Full re-review of the whole PR" in call["user"]
        assert "This review is in parts" in call["user"]
        assert "api/m1.py" in call["user"].split("Files the commits since", 1)[1][:200]
    _, rereview = gh.reviews[(REPO, 1)]
    assert [c["line"] for c in rereview["comments"]] == [7]  # repeats dropped
    assert "Re-reviewed the whole PR at `bbbbbbb`" in rereview["body"]
    assert f"reviewed in {first} parts</sub>" in rereview["body"]
    assert store.get_pr_state(REPO, 1).auto_rounds == 2


def legacy_too_large_skip(store: Store, number: int, sha: str, files: list[dict]) -> None:
    """What releases with a size cap left behind: a too_large skip row, the head settled as
    if reviewed (with its diff fingerprint), and the one-line note on the PR."""
    rid = store.start_review(REPO, number, "", "", sha, "review", kind="full", review_round=1)
    store.finish_review(
        rid, status="skipped", duration_s=0.1, skip_reason="too_large",
        error="reviewable diff too large (~296000 tokens > 230000)",
    )  # fmt: skip
    fingerprint = diff_fingerprint(files, reviewable_filter())
    store.mark_reviewed(REPO, number, sha, "review", diff_fingerprint=fingerprint)


def upgrade(path) -> Store:
    """Open the database as the first start of this release does (never released yet)."""
    with sqlite3.connect(path) as conn:
        conn.execute("DELETE FROM meta WHERE key = ?", (RELEASE_TOO_LARGE_KEY,))
    return Store(path)


def test_a_pr_the_old_size_cap_skipped_is_reviewed_on_the_next_poll(store: Store):
    gh = FakeGitHub({REPO: [make_pr(sha="a" * 40)]})
    gh.pr_files[(REPO, 1)] = [dict(f) for f in BIG_PR]
    legacy_too_large_skip(store, 1, "a" * 40, BIG_PR)
    note = (
        f"{TOO_LARGE_MARKER}\n<sub>gh-pr-reviewer: not reviewed — the reviewable diff "
        "(~296k tokens) is larger than the model can take in one review (230k).</sub>"
    )
    gh.comments[(REPO, 1)] = [{"id": 31, "user": {"login": BOT_LOGIN}, "body": note}]
    old = Harness(store, gh, **SMALL_BUDGET)
    old.poll()
    assert old.agent.invocations == []  # the settled head blocks it until released

    store = upgrade(store.path)
    state = store.get_pr_state(REPO, 1)
    assert state.last_reviewed_sha is None and state.last_reviewed_diff is None
    h = Harness(store, gh, **SMALL_BUDGET)
    h.agent.response = one_finding_per_part
    h.poll()
    assert len(h.agent.invocations) >= 2
    [review] = gh.reviews[(REPO, 1)]
    assert "parts</sub>" in review["body"]
    assert store.get_pr_state(REPO, 1).last_reviewed_sha == "a" * 40
    assert store.get_pr_state(REPO, 1).auto_rounds == 1  # its first round
    # the old note now says the PR was reviewed (edited once, not re-posted)
    assert gh.created_comments == []
    [(_, comment_id, edited)] = gh.updated
    assert comment_id == 31 and TOO_LARGE_MARKER in edited and "not reviewed" not in edited
    assert "reviewed at `aaaaaaa`" in edited and review["html_url"] in edited
    h.poll()
    assert len(gh.updated) == 1


def test_release_restores_the_last_posted_head_and_runs_once(store: Store):
    # PR 2: round 1 posted at 1111, then a push too large for the old cap at 2222
    rid = store.start_review(REPO, 2, "", "", "1" * 40, "review", kind="full", review_round=1)
    store.finish_review(rid, status="ok", duration_s=1.0)
    legacy_too_large_skip(store, 2, "2" * 40, BIG_PR)
    # PR 3: too large once, then reviewed fine after a smaller push: nothing to release
    legacy_too_large_skip(store, 3, "3" * 40, BIG_PR)
    rid = store.start_review(REPO, 3, "", "", "4" * 40, "review", kind="full", review_round=1)
    store.finish_review(rid, status="ok", duration_s=1.0)
    store.mark_reviewed(REPO, 3, "4" * 40, "review", diff_fingerprint="f" * 64)

    store = upgrade(store.path)
    two = store.get_pr_state(REPO, 2)
    assert (two.last_reviewed_sha, two.last_reviewed_diff, two.org_live) == ("1" * 40, None, True)
    assert two.auto_rounds == 1  # the next review is round 2, told what changed since 1111
    three = store.get_pr_state(REPO, 3)
    assert (three.last_reviewed_sha, three.last_reviewed_diff) == ("4" * 40, "f" * 64)
    store.mark_reviewed(REPO, 2, "2" * 40, "review", diff_fingerprint="e" * 64)
    assert Store(store.path).get_pr_state(REPO, 2).last_reviewed_sha == "2" * 40  # once only
