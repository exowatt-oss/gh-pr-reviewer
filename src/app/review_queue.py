"""Size-aware, fair review queue.

Org mode turns a trickle of PRs into bursts across hundreds of repos, and one giant PR takes
minutes of model time. A FIFO behind a semaphore would let a burst of large PRs (or one busy
repo) hold every slot while two-line fixes wait. Picking rules, in order:

1. **Lanes.** A review is *large* above ``LARGE_REVIEW_LINES`` reviewable changed lines. At most
   ``max_large`` large reviews run at once, so small PRs always have free slots.
2. **Per-repo cap.** At most ``max_per_repo`` reviews of one repo run at once.
3. **Shortest job first, with aging.** Among what may start, the lowest score wins:
   ``log2(1 + lines) - waited / aging_seconds + 2 * running_in_same_repo``. Every
   ``aging_seconds`` waited counts like halving the PR's size, so a large PR is never starved,
   and a repo that already has reviews running yields to one that has none.

One job per PR: re-queuing a PR replaces its queued job (newer head wins); a PR that is
running is never queued a second time (the poller's per-PR lock).

**Settle wait.** A job may carry ``not_before`` (its head's event time plus
``REVIEW_SETTLE_SECONDS``); it is not started earlier, and its aging only starts then. A newer
push replaces the job with a later ``not_before`` (debounce).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

REPO_RUNNING_PENALTY = 2.0


@dataclass
class Job:
    key: tuple[str, int]
    repo: str
    size_lines: int
    enqueued_at: float
    large: bool
    payload: Any = None
    started_at: float | None = None
    not_before: float = 0.0

    def ready(self, now: float) -> bool:
        return now >= self.not_before

    def score(self, now: float, aging_seconds: float, running_in_repo: int) -> float:
        waited = max(0.0, now - max(self.enqueued_at, self.not_before))
        aging = waited / aging_seconds if aging_seconds > 0 else 0.0
        return (
            math.log2(1 + max(self.size_lines, 0)) - aging + REPO_RUNNING_PENALTY * running_in_repo
        )


@dataclass
class ReviewQueue:
    max_concurrent: int
    max_large: int
    max_per_repo: int
    large_lines: int
    aging_seconds: float
    queued: dict[tuple[str, int], Job] = field(default_factory=dict)
    running: dict[tuple[str, int], Job] = field(default_factory=dict)

    def is_large(self, size_lines: int) -> bool:
        return size_lines > self.large_lines

    def push(
        self,
        key: tuple[str, int],
        repo: str,
        size_lines: int,
        now: float,
        payload: Any = None,
        *,
        not_before: float = 0.0,
    ) -> Job | None:
        """Queue (or refresh) a PR's job; ``None`` when that PR is already running."""
        if key in self.running:
            return None
        previous = self.queued.get(key)
        job = Job(
            key=key,
            repo=repo.lower(),
            size_lines=size_lines,
            # keep the original wait time: a new push must not send a PR to the back
            enqueued_at=previous.enqueued_at if previous else now,
            large=self.is_large(size_lines),
            payload=payload,
            not_before=not_before,
        )
        self.queued[key] = job
        return job

    def drop(self, key: tuple[str, int]) -> None:
        self.queued.pop(key, None)

    def _running_in(self, repo: str) -> int:
        return sum(1 for j in self.running.values() if j.repo == repo)

    def _large_running(self) -> int:
        return sum(1 for j in self.running.values() if j.large)

    def next_job(self, now: float) -> Job | None:
        """Move the best startable job from queued to running, or return ``None``."""
        if len(self.running) >= self.max_concurrent or not self.queued:
            return None
        large_full = self._large_running() >= self.max_large
        best: tuple[float, float, Job] | None = None
        for job in self.queued.values():
            if not job.ready(now):
                continue  # still settling
            in_repo = self._running_in(job.repo)
            if in_repo >= self.max_per_repo or (job.large and large_full):
                continue
            candidate = (job.score(now, self.aging_seconds, in_repo), job.enqueued_at, job)
            if best is None or candidate[:2] < best[:2]:
                best = candidate
        if best is None:
            return None
        job = best[2]
        del self.queued[job.key]
        job.started_at = now
        self.running[job.key] = job
        return job

    def finish(self, key: tuple[str, int]) -> None:
        self.running.pop(key, None)

    def snapshot(self, now: float) -> dict[str, Any]:
        ready = [j for j in self.queued.values() if j.ready(now)]
        oldest = min((max(j.enqueued_at, j.not_before) for j in ready), default=None)
        return {
            "queued": len(self.queued),
            "settling": len(self.queued) - len(ready),
            "queued_large": sum(1 for j in self.queued.values() if j.large),
            "running": len(self.running),
            "running_large": self._large_running(),
            "oldest_wait_s": round(now - oldest, 1) if oldest is not None else None,
            "slots": self.max_concurrent,
            "large_slots": self.max_large,
            "per_repo_cap": self.max_per_repo,
        }
