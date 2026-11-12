"""Allowlist, bot/draft filtering, and the full/incremental/skip decision."""

from __future__ import annotations

import pytest

from app.github_app import PullRequest
from app.review_policy import (
    Mode,
    PrState,
    decide,
    is_allowed_repo,
    is_bot_author,
    parse_repo_allowlist,
    retry_delay_seconds,
)

NOW = 1_800_000_000.0


def pr(**kw) -> PullRequest:
    base = dict(
        repo="sample-org/example-project",
        number=7,
        title="t",
        draft=False,
        head_sha="bbbb",
        base_ref="main",
        author_login="sample-user",
        author_type="User",
        html_url="https://github.com/sample-org/example-project/pull/7",
        api_url="",
    )
    return PullRequest(**(base | kw))


def test_allowlist_parsing():
    raw = " sample-org/example-project, ,sample-org/example-project,sample-org/example-project,bad entry,https://github.com/o/r/"
    assert parse_repo_allowlist(raw) == ["sample-org/example-project", "o/r"]
    assert parse_repo_allowlist("") == []


def test_allowlist_membership_is_case_insensitive():
    allow = parse_repo_allowlist("sample-org/example-project")
    assert is_allowed_repo("sample-org/example-project", allow)
    assert not is_allowed_repo("sample-org/other", allow)


@pytest.mark.parametrize(
    "login,user_type,expected",
    [
        ("dependabot[bot]", "Bot", True),
        ("renovate[bot]", "Bot", True),
        ("some-app[bot]", "Bot", True),
        ("weird-name", "Bot", True),  # any user.type == Bot
        ("Dependabot[bot]", "User", True),  # listed login regardless of type casing
        ("sample-user", "User", False),
        ("robot-person", "User", False),
    ],
)
def test_bot_authors(login, user_type, expected):
    assert is_bot_author(login, user_type) is expected


def test_draft_and_bot_prs_are_skipped():
    assert decide(pr(draft=True), None, now=NOW).mode is Mode.SKIP
    d = decide(pr(author_login="dependabot[bot]", author_type="Bot"), None, now=NOW)
    assert d.mode is Mode.SKIP and "bot" in d.reason


def test_first_time_seen_is_full_review():
    assert decide(pr(), None, now=NOW).mode is Mode.FULL
    assert decide(pr(), PrState(), now=NOW).mode is Mode.FULL


def test_same_sha_is_skipped_so_restarts_do_not_re_review():
    state = PrState(last_reviewed_sha="bbbb", last_review_mode="review")
    d = decide(pr(head_sha="bbbb"), state, now=NOW)
    assert d.mode is Mode.SKIP and d.reason == "already reviewed at this sha"


def test_new_commits_after_round_one_get_a_full_rereview():
    state = PrState(last_reviewed_sha="aaaa", last_review_mode="review", auto_rounds=1)
    d = decide(pr(head_sha="bbbb"), state, now=NOW)
    assert d.mode is Mode.FULL and d.round == 2 and d.kind == "full"
    assert "round 2: full re-review" in d.reason and not d.needs_full_budget


def test_reviewed_but_no_round_posted_yet_is_still_round_one():
    # lockfile-only / unchanged-diff skips mark a SHA reviewed but post no review.
    state = PrState(last_reviewed_sha="aaaa", last_review_mode="review", auto_rounds=0)
    d = decide(pr(head_sha="bbbb"), state, now=NOW)
    assert (d.mode, d.round, d.reason) == (Mode.FULL, 1, "first review")
    assert d.needs_full_budget


def test_max_rounds_caps_automatic_reviews():
    state = PrState(last_reviewed_sha="aaaa", auto_rounds=2)
    d = decide(pr(head_sha="bbbb"), state, now=NOW)
    assert d.mode is Mode.SKIP and d.category == "max_rounds"
    # the reviewed head itself stays "reviewed" (not a max_rounds skip on every poll)
    assert decide(pr(head_sha="aaaa"), state, now=NOW).category == "reviewed"
    assert decide(pr(head_sha="bbbb"), state, now=NOW, max_rounds=3).round == 3
    assert decide(pr(head_sha="bbbb"), state, now=NOW, max_rounds=0).mode is Mode.FULL
    assert decide(pr(draft=True, head_sha="bbbb"), state, now=NOW).category == "draft"


def test_leaving_draft_with_new_commits_is_full_review():
    state = PrState(last_reviewed_sha="aaaa", draft_since_review=True, auto_rounds=1)
    d = decide(pr(head_sha="bbbb"), state, now=NOW)
    assert d.mode is Mode.FULL and "draft" in d.reason and d.round == 2


def test_leaving_draft_without_new_commits_is_skipped():
    state = PrState(last_reviewed_sha="bbbb", draft_since_review=True)
    assert decide(pr(head_sha="bbbb"), state, now=NOW).mode is Mode.SKIP


def test_failed_sha_retries_with_exponential_jitter_then_gives_up():
    first = retry_delay_seconds(
        "sample-org/example-project", 7, "bbbb", 1, base_seconds=60, max_seconds=3600
    )
    second = retry_delay_seconds(
        "sample-org/example-project", 7, "bbbb", 2, base_seconds=60, max_seconds=3600
    )
    other = retry_delay_seconds(
        "sample-org/example-project", 8, "bbbb", 1, base_seconds=60, max_seconds=3600
    )
    assert 48 <= first <= 72 and 96 <= second <= 144
    assert first != other  # deterministic across polls, but spreads different PRs
    assert (
        retry_delay_seconds(
            "sample-org/example-project", 7, "bbbb", 10, base_seconds=60, max_seconds=90
        )
        <= 90
    )
    failed = PrState(failed_sha="bbbb", failed_attempts=1, last_failure_at=NOW - first / 2)
    assert decide(pr(), failed, now=NOW, retry_backoff_seconds=60).reason == "retry backoff"
    later = PrState(failed_sha="bbbb", failed_attempts=1, last_failure_at=NOW - first - 1)
    assert decide(pr(), later, now=NOW, retry_backoff_seconds=60).mode is Mode.FULL
    exhausted = PrState(failed_sha="bbbb", failed_attempts=4, last_failure_at=NOW - 10_000)
    assert decide(pr(), exhausted, now=NOW, max_attempts=4).mode is Mode.SKIP
    # a new head SHA resets the failure budget
    assert decide(pr(head_sha="cccc"), exhausted, now=NOW).mode is Mode.FULL
