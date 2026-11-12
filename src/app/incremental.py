"""Anchor incremental review on the last head SHA this app actually reviewed.

PR-Agent decides which commits are new by comparing each commit's committer date with the
creation time of its previous review comment. Commits made while a review was running (or
queued) but pushed afterwards, and rebased commits that keep older dates, can therefore look
"already reviewed": no model call is made and the new head is never reviewed.

We know exactly which SHA we reviewed last, so the commit range is cut there instead. When
that SHA is no longer in the PR's history (force-push / rebase), there is no safe baseline and
PR-Agent's own fallback runs a full review.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from contextvars import ContextVar
from typing import Any

logger = logging.getLogger(__name__)

# The last reviewed head SHA for the PR under review (set by Reviewer for incremental runs).
incremental_anchor: ContextVar[str | None] = ContextVar("incremental_anchor", default=None)

_MARK = "_reviewer_anchored"


def anchored_commit_range(provider: Any, original: Callable[[Any], list]) -> list:
    anchor = incremental_anchor.get()
    if not anchor:
        return original(provider)
    commits = list(provider.pr_commits or [])
    shas = [getattr(c, "sha", None) for c in commits]
    if anchor not in shas:
        logger.info("Last reviewed %s is not in the PR history; full review instead", anchor[:12])
        # A range with no last-seen commit makes PR-Agent fall back to a full review.
        provider.incremental.first_new_commit = commits[0] if commits else None
        provider.incremental.last_seen_commit = None
        return commits
    index = shas.index(anchor)
    new = commits[index + 1 :]
    provider.incremental.last_seen_commit = commits[index]
    provider.incremental.first_new_commit = new[0] if new else None
    return new


def install_incremental_anchor() -> None:
    """Patch ``GithubProvider.get_commit_range`` once (idempotent)."""
    from pr_agent.git_providers.github_provider import GithubProvider

    current = GithubProvider.get_commit_range
    if getattr(current, _MARK, False):
        return

    def get_commit_range(self: Any) -> list:
        return anchored_commit_range(self, current)

    setattr(get_commit_range, _MARK, True)
    GithubProvider.get_commit_range = get_commit_range
