"""Stop a running review between model calls once its PR merged, closed or moved on.

A review is a chain of model calls: the tool loop's turns, and the parts of a diff too large
for one call. Each call is paid in full, so the review re-reads its PR (``GET /pulls/N``)
before every call (``PullLiveness.check``, the ``checkpoint`` of ``review_agent`` and
``reviewer.call_model``) and stops with ``ReviewCancelled`` when:

- the PR was merged (``merged_during_review``) or closed unmerged (``closed_during_review``);
- a newer head was pushed (``superseded``), for automatic reviews only: the new head gets its
  own review, while an on-demand ``/review`` was asked for and keeps going.

A call already running is never aborted (its tokens are billed either way and the cost ledger
must match the provider). GitHub is read at most once per ``interval_s`` for all the parts of
one review; any read error or timeout fails open (the review carries on). Once a review is
cancelled the verdict is final: later checks raise at once, without reading GitHub.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Callable
from typing import Any

logger = logging.getLogger(__name__)

MERGED = "merged_during_review"
CLOSED = "closed_during_review"
SUPERSEDED = "superseded"
CANCEL_REASONS = (MERGED, CLOSED, SUPERSEDED)
# One PR read, at most (it never holds up the review longer than this).
CHECK_TIMEOUT_S = 10.0


class ReviewCancelled(Exception):
    """The review stopped before its next model call; nothing is posted."""

    def __init__(self, reason: str, message: str) -> None:
        super().__init__(message)
        self.reason = reason  # merged_during_review | closed_during_review | superseded
        self.message = message


def cancel_reason(
    pull: dict[str, Any], head_sha: str, *, cancel_on_supersede: bool
) -> tuple[str, str] | None:
    """``(reason, message)`` when a review of ``head_sha`` should stop, given the PR as
    GitHub returns it; ``None`` to carry on (also when the answer says nothing useful)."""
    state = str(pull.get("state") or "")
    if not state:
        return None
    stopped = "stopped before the next model call; nothing posted"
    if state != "open":
        if pull.get("merged") or pull.get("merged_at"):
            return MERGED, f"PR was merged while it was reviewed; {stopped}"
        return CLOSED, f"PR was closed while it was reviewed; {stopped}"
    head = str((pull.get("head") or {}).get("sha") or "")
    if cancel_on_supersede and head and head != head_sha:
        return SUPERSEDED, (
            f"head moved to {head[:8]} while {head_sha[:8]} was reviewed; {stopped} "
            "(the new head gets its own review)"
        )
    return None


class PullLiveness:
    """Re-reads one PR between the model calls of one review (shared by all its parts)."""

    def __init__(
        self,
        github: Any,
        repo: str,
        number: int,
        head_sha: str,
        *,
        cancel_on_supersede: bool,
        interval_s: float,
        clock: Callable[[], float] = time.monotonic,
        pull: dict[str, Any] | None = None,
    ) -> None:
        self.github = github
        self.repo = repo
        self.number = number
        self.head_sha = head_sha
        self.cancel_on_supersede = cancel_on_supersede
        self.interval_s = interval_s
        self.clock = clock
        self.reads = 0  # GitHub reads made (tests, logs)
        self._read_at: float | None = None
        self._verdict: tuple[str, str] | None = None
        self._lock = asyncio.Lock()
        if pull:
            self.seed(pull)

    @property
    def cancelled(self) -> str | None:
        """The cancel reason, once the review is cancelled."""
        return self._verdict[0] if self._verdict else None

    def seed(self, pull: dict[str, Any] | None) -> None:
        """Use a PR read the caller just made (no extra GET for the first check)."""
        if pull and self._verdict is None:
            self._read_at = self.clock()
            self._verdict = cancel_reason(
                pull, self.head_sha, cancel_on_supersede=self.cancel_on_supersede
            )

    def _raise_if_cancelled(self) -> None:
        if self._verdict is not None:
            raise ReviewCancelled(*self._verdict)

    async def check(self) -> None:
        """Raise ``ReviewCancelled`` when the review should stop before its next model call."""
        self._raise_if_cancelled()
        async with self._lock:  # parallel parts: one GET, the others reuse its verdict
            self._raise_if_cancelled()
            due = self._read_at is None or self.clock() - self._read_at >= self.interval_s
            if due:
                await self._read()
        self._raise_if_cancelled()

    async def _read(self) -> None:
        self.reads += 1
        try:
            pull = await asyncio.wait_for(
                self.github.get_pull(self.repo, self.number), CHECK_TIMEOUT_S
            )
        except Exception as exc:  # fail open: GitHub trouble never stops a review
            logger.info("Could not re-check %s#%s between model calls (carrying on): %s",
                        self.repo, self.number, exc or type(exc).__name__)  # fmt: skip
            pull = None
        self._read_at = self.clock()
        if isinstance(pull, dict):
            self._verdict = cancel_reason(
                pull, self.head_sha, cancel_on_supersede=self.cancel_on_supersede
            )
