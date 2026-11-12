"""Incremental reviews are anchored on the last SHA we reviewed, not on comment timestamps."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

import pytest
from pr_agent.git_providers.git_provider import IncrementalPR
from pr_agent.git_providers.github_provider import GithubProvider

from app.incremental import anchored_commit_range, incremental_anchor, install_incremental_anchor

REVIEWED_AT = datetime(2026, 10, 7, 14, 48, 50, tzinfo=UTC)


def commit(sha: str, minutes_before_review: float, *files: str) -> SimpleNamespace:
    date = REVIEWED_AT - timedelta(minutes=minutes_before_review)
    return SimpleNamespace(
        sha=sha,
        commit=SimpleNamespace(
            message=f"commit {sha}",
            committer=SimpleNamespace(date=date),
            author=SimpleNamespace(date=date),
        ),
        files=[SimpleNamespace(filename=f) for f in files],
    )


def provider(commits: list) -> GithubProvider:
    """A GithubProvider without network: just the state _get_incremental_commits reads."""
    p = object.__new__(GithubProvider)
    p.pr_commits = commits
    p.incremental = IncrementalPR(True)
    p.unreviewed_files_map = {}
    p.get_previous_review = lambda **_: SimpleNamespace(created_at=REVIEWED_AT)
    p._get_repo = lambda: SimpleNamespace(default_branch="main")
    return p


@pytest.fixture(autouse=True)
def patched():
    install_incremental_anchor()
    install_incremental_anchor()  # idempotent
    yield


def run(p: GithubProvider, anchor: str | None) -> None:
    token = incremental_anchor.set(anchor)
    try:
        p._get_incremental_commits()
    finally:
        incremental_anchor.reset(token)


# Synthetic history: the second commit is added while the first review is running.
COMMITS = [commit("a" * 40, 30, "x.py"), commit("b" * 40, 4, "y.py"),
           commit("c" * 40, 2, "z.py")]  # fmt: skip


def test_commit_made_during_the_previous_review_is_reviewed():
    p = provider(COMMITS)
    run(p, COMMITS[1].sha)
    assert p.incremental.is_incremental
    assert p.incremental.first_new_commit_sha == COMMITS[2].sha
    assert p.incremental.last_seen_commit_sha == COMMITS[1].sha
    assert set(p.unreviewed_files_map) == {"z.py"}


def test_without_anchor_pr_agent_dates_miss_that_commit():
    p = provider(COMMITS)
    run(p, None)  # PR-Agent's own behaviour: every commit predates the review comment
    assert p.incremental.first_new_commit_sha is None
    assert p.unreviewed_files_map == {}


def test_rewritten_history_falls_back_to_full_review():
    # Rebased synthetic commits keep old dates and the previously reviewed SHA is gone.
    p = provider(COMMITS)
    run(p, "dead" * 10)
    assert p.incremental.is_incremental is False


def test_anchor_at_head_means_nothing_new():
    p = provider(COMMITS)
    calls = []
    out = anchored_commit_range(p, lambda prov: calls.append(prov) or [])
    assert calls == [p] and out == []  # no anchor: PR-Agent's original range
    token = incremental_anchor.set(COMMITS[2].sha)
    try:
        assert anchored_commit_range(p, lambda prov: []) == []
    finally:
        incremental_anchor.reset(token)
    assert p.incremental.first_new_commit is None
