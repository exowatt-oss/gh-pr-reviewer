"""Background worker: discovers pull requests and runs reviews.

Signed GitHub webhook events wake the worker for prompt reconciliation. Periodic polling remains
an independent recovery path and looks for work every ``POLL_INTERVAL_SECONDS``:

- **Explicit repos** (``owner/repo`` in ``REVIEW_REPOS``, and ``PILOT_REPOS``) are listed
  directly: one ``GET /pulls`` per repo per poll.
- **Org mode** (``owner/*``) never lists hundreds of repos. One GraphQL search per org per poll
  returns open PRs updated since the last search (minus an overlap for search-index lag), with
  head SHA, draft flag, labels and dates inline; a full open-PR sweep (reconcile) runs every
  ``RECONCILE_INTERVAL_SECONDS`` and at start. Only repos the installation can access, not
  archived and not forks, are considered, and only PRs that are new work after the go-live
  cutoff (``ORG_MODE_SINCE``) — the existing backlog is never reviewed.

Candidates go through ``review_policy.decide`` (at most ``MAX_REVIEW_ROUNDS`` automatic review
rounds per PR: the first review, then one full re-review on the next push) plus the budget
guard and the repo opt-out, then into a size-aware fair queue (``review_queue.py``). A review waits
``REVIEW_SETTLE_SECONDS`` after the head's push / ready / opened event (a newer push restarts
the wait); a PR merged or closed meanwhile is recorded as skipped and never billed.

A human can ask for a fresh full review with a PR comment ``/review`` or
``@gh-pr-reviewer review``: org search carries each PR's latest comments, explicitly
listed repos read their recent issue comments once per poll. The request skips the settle wait
and PR-level skips (draft, already reviewed, backlog, max rounds) but not the repo opt-out, the
budget or its own per-PR daily cap (``MAX_ON_DEMAND_REVIEWS_PER_PR_PER_DAY``).

It runs on its own thread with its own event loop. PR-Agent does blocking PyGithub I/O
inside coroutines; keeping it off the web server's loop keeps the UI and health probes
responsive, and gives litellm's background logging worker a single, stable loop. On shutdown
it stops taking work and lets in-flight reviews finish for up to ``SHUTDOWN_GRACE_SECONDS``.
"""

from __future__ import annotations

import asyncio
import logging
import threading
import time
from collections import Counter, deque
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime, timedelta
from typing import Any

from app.budget import BudgetGuard, BudgetStatus
from app.config import Settings
from app.github_app import GitHubApp, PrComment, PullRequest, RepoInfo
from app.incremental import install_incremental_anchor
from app.llm_tracking import install_cost_tracker
from app.pr_agent_config import reviewable_filter
from app.repo_context import reviewable_lines
from app.review_policy import (
    REPO_CONFIG_PATH,
    Decision,
    Mode,
    PrState,
    decide,
    find_trigger,
    is_force_request,
    parse_repo_allowlist,
    parse_review_targets,
    parse_time,
    post_cutoff_reason,
    repo_opt_out_reason,
    retry_delay_seconds,
    settle_update,
)
from app.review_queue import ReviewQueue
from app.reviewer import CANCELLED, SETTLED_STATUSES, Reviewer
from app.store import Store, epoch_iso, iso, utcnow

logger = logging.getLogger(__name__)

ORG_MODE_SINCE_KEY = "org_mode_since"  # legacy single-org key; per org: "<key>:<org>"
# On-demand ``/review`` comments older than this (first start of the feature) are ignored.
TRIGGERS_SINCE_KEY = "triggers_since"
REPO_POLICY_ERROR_TTL = 60.0
# Poller decisions recorded as skipped review rows (stats API), by decision category.
RECORDED_SKIPS = {
    "draft": "draft",
    "bot": "bot",
    "opted_out": "opted_out",
    "repo_opted_out": "opted_out",
    "daily_cap": "daily_cap",
    "max_rounds": "max_rounds",
    "already_reviewed": "already_reviewed",
}
# The one reply to a ``/review`` of a head that already has a posted review.
ALREADY_REVIEWED_MARKER = "<!-- gh-pr-reviewer:already-reviewed -->"


@dataclass
class PollerStatus:
    state: str = "starting"  # starting | not_configured | running | draining | stopped
    reason: str | None = None
    dry_run: bool = False
    mode: str = "repos"  # repos | org
    repos: list[str] = field(default_factory=list)
    direct_repos: list[str] = field(default_factory=list)
    pilot_repos: list[str] = field(default_factory=list)
    org_mode_since: str | None = None
    org_cutoffs: dict[str, str] = field(default_factory=dict)
    orgs: list[dict[str, Any]] = field(default_factory=list)
    settle_seconds: int = 0
    repo_count: int | None = None
    interval_seconds: int = 60
    polls: int = 0
    last_poll_at: str | None = None
    last_poll_duration_s: float | None = None
    last_poll_errors: list[str] = field(default_factory=list)
    last_error: str | None = None
    last_error_at: str | None = None
    last_error_source: str | None = None
    in_flight: list[str] = field(default_factory=list)
    running: list[str] = field(default_factory=list)
    running_jobs: list[dict[str, Any]] = field(default_factory=list)
    queued_jobs: list[dict[str, Any]] = field(default_factory=list)
    queue: dict[str, Any] = field(default_factory=dict)
    budget: dict[str, Any] = field(default_factory=dict)
    discovery: dict[str, Any] = field(default_factory=dict)
    rate_limits: dict[str, Any] = field(default_factory=dict)
    reviews_started: int = 0
    recent_decisions: deque = field(default_factory=lambda: deque(maxlen=100))

    def snapshot(self) -> dict[str, Any]:
        data = {k: v for k, v in self.__dict__.items() if k != "recent_decisions"}
        for key in ("in_flight", "running", "last_poll_errors", "repos", "direct_repos"):
            data[key] = list(data[key])
        for key in (
            "queue",
            "budget",
            "discovery",
            "rate_limits",
            "org_cutoffs",
            "orgs",
            "running_jobs",
            "queued_jobs",
        ):
            data[key] = _copy(data[key])
        data["recent_decisions"] = list(self.recent_decisions)
        return data


def _copy(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: _copy(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_copy(v) for v in value]
    return value


@dataclass
class JobPayload:
    pr: PullRequest
    decision: Decision
    files: list[dict] | None
    ready_at: float | None = None
    requested_by: str | None = None  # on-demand: who asked
    on_demand_request_id: int | None = None
    force: bool = False


@dataclass
class OrgState:
    org: str
    cutoff: float | None = None
    state: str = "pending"  # pending | ok | not_installed | error
    installation_id: int | None = None
    repos: dict[str, RepoInfo] | None = None
    repos_fetched_at: float | None = None
    last_reconcile: float | None = None
    watermark: float | None = None


def _gh_time(epoch: float) -> str:
    return datetime.fromtimestamp(epoch, tz=UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


class Poller:
    def __init__(
        self,
        settings: Settings,
        store: Store,
        *,
        github: GitHubApp | None = None,
        reviewer: Reviewer | None = None,
        clock: Any = time.time,
    ) -> None:
        self.settings = settings
        self.store = store
        self.github = github
        self.reviewer = reviewer
        self.clock = clock
        self.targets = parse_review_targets(settings.REVIEW_REPOS)
        self.pilot = [
            r for r in parse_repo_allowlist(settings.PILOT_REPOS) if self.targets.covers(r)
        ]
        direct: list[str] = []
        for repo in [*self.targets.repos, *self.pilot]:
            if repo.lower() not in {d.lower() for d in direct}:
                direct.append(repo)
        self.direct_repos = direct
        self._direct_set = {r.lower() for r in direct}
        self._pilot_set = {r.lower() for r in self.pilot}
        self.repos = self.direct_repos  # back-compat name
        self.budget = BudgetGuard(settings, store)
        self.queue = ReviewQueue(
            max_concurrent=settings.max_concurrent_reviews,
            max_large=settings.max_concurrent_large_reviews,
            max_per_repo=settings.max_concurrent_per_repo,
            large_lines=settings.LARGE_REVIEW_LINES,
            aging_seconds=settings.QUEUE_AGING_SECONDS,
        )
        self._orgs = {org: OrgState(org) for org in self.targets.orgs}
        self._orgs_by_key = {org.lower(): st for org, st in self._orgs.items()}
        self._repo_policy: dict[str, tuple[float, str | None]] = {}
        self._status = PollerStatus(
            dry_run=settings.DRY_RUN,
            mode="org" if self.targets.org_mode else "repos",
            repos=self.targets.describe(),
            direct_repos=list(self.direct_repos),
            pilot_repos=list(self.pilot),
            interval_seconds=settings.POLL_INTERVAL_SECONDS,
            settle_seconds=settings.REVIEW_SETTLE_SECONDS,
            queue=self.queue.snapshot(clock()),
        )
        self._status_lock = threading.Lock()
        self._tasks: set[asyncio.Task] = set()
        self._dry_run_logged: set[tuple[str, int, str]] = set()
        self._held_logged: set[tuple[str, int, str, str]] = set()
        self._thread: threading.Thread | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._stop: asyncio.Event | None = None
        self._wake: asyncio.Event | None = None
        self._draining = False
        self._grace = settings.SHUTDOWN_GRACE_SECONDS
        self._ready = False
        self.triggers_since: float | None = None
        self._comment_watermarks: dict[str, float] = {}

    # --- status --------------------------------------------------------------------
    def status(self) -> dict[str, Any]:
        with self._status_lock:
            return self._status.snapshot()

    def _set(self, **fields: Any) -> None:
        with self._status_lock:
            for k, v in fields.items():
                setattr(self._status, k, v)

    def _note_decision(self, pr: PullRequest, decision: Decision) -> None:
        with self._status_lock:
            self._status.recent_decisions.appendleft(
                {
                    "at": iso(utcnow()),
                    "repo": pr.repo,
                    "pr": pr.number,
                    "sha": pr.head_sha[:12],
                    "mode": decision.mode.value,
                    "reason": decision.reason,
                }
            )

    def _sync_queue_status(self) -> None:
        running = sorted(f"{r}#{n}" for r, n in self.queue.running)
        queued = sorted(f"{r}#{n}" for r, n in self.queue.queued)

        def view(job: Any) -> dict[str, Any]:
            payload: JobPayload = job.payload
            return {
                "repo": payload.pr.repo,
                "pr": payload.pr.number,
                "kind": payload.decision.kind,
                "head_sha": payload.pr.head_sha,
                "ready_at": epoch_iso(payload.ready_at),
                "started_at": epoch_iso(job.started_at),
            }

        self._set(
            running=running,
            in_flight=sorted(running + queued),
            running_jobs=[view(j) for j in self.queue.running.values() if j.payload],
            queued_jobs=sorted(
                (view(j) for j in self.queue.queued.values() if j.payload),
                key=lambda v: v["ready_at"] or "",
            ),
            queue=self.queue.snapshot(self.clock()),
        )

    def _org_of(self, repo: str) -> OrgState | None:
        return self._orgs_by_key.get(repo.split("/", 1)[0].lower())

    @property
    def cutoff(self) -> float | None:
        """The first org's go-live cutoff (single-org callers and tests)."""
        first = next(iter(self._orgs.values()), None)
        return first.cutoff if first else None

    def _sync_org_status(self) -> None:
        orgs = []
        for st in self._orgs.values():
            repos = st.repos or {}
            orgs.append(
                {
                    "org": st.org,
                    "state": st.state,
                    "installation_id": st.installation_id,
                    "repos_covered": (
                        sum(1 for r in repos.values() if not r.archived and not r.fork)
                        if st.repos is not None
                        else None
                    ),
                    "cutoff": epoch_iso(st.cutoff),
                }
            )
        self._set(
            orgs=orgs,
            org_cutoffs={o["org"]: o["cutoff"] for o in orgs if o["cutoff"]},
            org_mode_since=orgs[0]["cutoff"] if orgs else None,
        )

    def not_configured_reason(self) -> str | None:
        missing = []
        if not self.settings.github_configured:
            missing.append("GITHUB_APP_ID/GITHUB_APP_PRIVATE_KEY")
        if not self.settings.llm_configured and not self.settings.DRY_RUN:
            missing.append("LLM_API_KEY")
        if not self.direct_repos and not self.targets.orgs:
            missing.append("REVIEW_REPOS")
        return f"not configured: missing {', '.join(missing)}" if missing else None

    # --- lifecycle -----------------------------------------------------------------
    def start(self) -> None:
        reason = self.not_configured_reason()
        if reason:
            self._set(state="not_configured", reason=reason)
            logger.warning("Poller %s", reason)
            return
        self._thread = threading.Thread(target=self._thread_main, name="pr-poller", daemon=True)
        self._thread.start()

    def stop(self, timeout: float | None = None) -> None:
        """Stop polling, let in-flight reviews finish for up to the grace period, then cancel
        what is left (a cut-off review is recovered, not re-billed, after the restart)."""
        grace = self.settings.SHUTDOWN_GRACE_SECONDS if timeout is None else timeout
        self._grace = grace
        if self._loop and self._stop:
            self._loop.call_soon_threadsafe(self._stop.set)
            if self._wake:
                self._loop.call_soon_threadsafe(self._wake.set)
        if self._thread:
            self._thread.join(timeout=grace + 5)
        if self._status.state in ("running", "draining"):
            self._set(state="stopped")

    def wake(self) -> None:
        """Request an immediate reconciliation from another thread, e.g. a verified webhook."""
        if self._loop and self._wake:
            self._loop.call_soon_threadsafe(self._wake.set)

    def _thread_main(self) -> None:
        try:
            asyncio.run(self._run())
        except Exception as exc:  # pragma: no cover - last-resort guard
            logger.exception("Poller thread crashed")
            self._set(
                state="stopped",
                last_error=str(exc)[:500],
                last_error_at=iso(utcnow()),
                last_error_source="poller",
            )

    def resolve_cutoff(self, org: str, *, create: bool = True) -> float | None:
        """ORG_MODE_SINCE from env, else this org's persisted go-live cutoff.

        A new org's cutoff is the moment org mode first sees the App installed there (set
        when ``create``; ``None`` until then), so the backlog open at install time is never
        reviewed. The single-org key of earlier releases carries over to orgs it already
        covered (those with org-mode history in ``pr_state``)."""
        raw = self.settings.ORG_MODE_SINCE.strip()
        if raw:
            parsed = parse_time(raw)
            if parsed is not None:
                return parsed
            logger.error("Invalid ORG_MODE_SINCE %r; using the persisted cutoff", raw)
        key = f"{ORG_MODE_SINCE_KEY}:{org.lower()}"
        stored = self.store.get_meta(key)
        if stored is None:
            legacy = self.store.get_meta(ORG_MODE_SINCE_KEY)
            if legacy and self.store.has_org_history(org):
                value = legacy
            elif create:
                value = iso(datetime.fromtimestamp(self.clock(), tz=UTC))
            else:
                return None
            stored = self.store.meta_setdefault(key, value)
        return parse_time(stored) or self.clock()

    def _ensure_cutoff(self, st: OrgState, *, create: bool) -> None:
        if st.cutoff is None:
            st.cutoff = self.resolve_cutoff(st.org, create=create)
            if st.cutoff is not None:
                logger.info(
                    "Org mode for %s; reviewing only work after %s (pilot repos exempt: %s)",
                    st.org, epoch_iso(st.cutoff), self.pilot or "none",
                )  # fmt: skip

    async def setup(self) -> None:
        """Create clients on the current loop (poller thread, or a test's loop)."""
        if self.github is None:
            self.github = GitHubApp(
                self.settings.GITHUB_APP_ID,
                self.settings.GITHUB_APP_PRIVATE_KEY.get_secret_value(),
                api_url=self.settings.GITHUB_API_URL,
            )
        if self.reviewer is None:
            login = None
            try:
                slug = (await self.github.get_app()).get("slug")
                login = f"{slug}[bot]" if slug else None
            except Exception as exc:
                logger.warning("Could not read app slug from GET /app: %s", exc)
            self.reviewer = Reviewer(self.settings, self.store, self.github, app_login=login)
        for st in self._orgs.values():
            # Orgs not installed yet get their cutoff when discovery first finds the App.
            self._ensure_cutoff(st, create=False)
        self.triggers_since = self._resolve_triggers_since()
        self._sync_org_status()
        install_cost_tracker(self.store)
        install_incremental_anchor()
        _configure_pr_agent_logging(self.settings.LOG_LEVEL)
        self._ready = True
        self._set(state="running", reason=None)

    def _resolve_triggers_since(self) -> float:
        raw = self.store.get_meta(TRIGGERS_SINCE_KEY)
        since = parse_time(raw) if raw else None
        if since is None:
            since = self.clock()
            if not self.settings.DRY_RUN:
                self.store.set_meta(TRIGGERS_SINCE_KEY, epoch_iso(since))
        return since

    async def _run(self) -> None:
        self._loop = asyncio.get_running_loop()
        self._stop = asyncio.Event()
        self._wake = asyncio.Event()
        try:
            await self.setup()
        except Exception as exc:
            logger.exception("Poller setup failed")
            self._set(
                state="not_configured",
                reason=f"setup failed: {type(exc).__name__}: {exc}"[:500],
                last_error=str(exc)[:500],
                last_error_at=iso(utcnow()),
                last_error_source="setup",
            )
            return
        while not self._stop.is_set():
            try:
                await self.poll_once()
            except Exception as exc:  # pragma: no cover - poll_once guards its own steps
                logger.exception("Poll failed")
                self._set(
                    last_error=str(exc)[:500],
                    last_error_at=iso(utcnow()),
                    last_error_source="poller",
                )
            try:
                interval = self.settings.POLL_INTERVAL_SECONDS
                await asyncio.wait_for(self._wake.wait(), timeout=interval)
            except TimeoutError:
                pass
            finally:
                self._wake.clear()
        await self.drain(self._grace)
        if self.github:
            await self.github.aclose()
        self._set(state="stopped")

    async def drain(self, grace: float) -> None:
        """Take no new work; wait up to ``grace`` s for running reviews, then cancel them."""
        self._draining = True
        self._set(state="draining")
        dropped = len(self.queue.queued)
        self.queue.queued.clear()
        running = [t for t in self._tasks if not t.done()]
        logger.info(
            "Draining: %d review(s) running, %d queued dropped; waiting up to %.0fs",
            len(running), dropped, grace,
        )  # fmt: skip
        if running:
            _done, pending = await asyncio.wait(running, timeout=max(grace, 0))
            for task in pending:
                task.cancel()
            if pending:
                logger.warning("Cancelled %d review(s) still running at shutdown", len(pending))
                await asyncio.gather(*pending, return_exceptions=True)
        self._sync_queue_status()

    @property
    def accepting_work(self) -> bool:
        return not self._draining and not (self._stop is not None and self._stop.is_set())

    # --- polling -------------------------------------------------------------------
    async def poll_once(self) -> list[tuple[PullRequest, Decision]]:
        """One pass: list explicit repos, search orgs, queue reviews; returns what it queued."""
        assert self.github is not None and self._ready, "call setup() first"
        started = time.monotonic()
        scheduled: list[tuple[PullRequest, Decision]] = []
        errors: list[str] = []
        budget = await self.budget.check()
        for repo in self.direct_repos:
            try:
                pulls = await self.github.list_open_pulls(repo)
            except Exception as exc:
                logger.warning("Listing PRs for %s failed: %s", repo, exc)
                errors.append(f"{repo}: {type(exc).__name__}: {exc}"[:300])
                continue
            comments = await self._recent_repo_comments(repo)
            for pr in pulls:
                if comments.get(pr.number):
                    pr = replace(pr, comments=tuple(comments[pr.number]))
                decision = await self._consider_safely(pr, org=False, budget=budget)
                if decision.mode is not Mode.SKIP:
                    scheduled.append((pr, decision))
        for org in self.targets.orgs:
            st = self._orgs[org]
            try:
                scheduled.extend(await self._discover_org(st, budget))
            except Exception as exc:
                st.state = "error"
                logger.warning("Discovery for %s/* failed: %s", org, exc)
                errors.append(f"{org}/*: {type(exc).__name__}: {exc}"[:300])
        self._sync_org_status()
        self._pump()
        now_iso = iso(utcnow())
        updates: dict[str, Any] = {
            "polls": self._status.polls + 1,
            "last_poll_at": now_iso,
            "last_poll_duration_s": round(time.monotonic() - started, 3),
            "last_poll_errors": errors,
            "budget": budget.as_dict(),
            "rate_limits": _copy(getattr(self.github, "rate_limits", {}) or {}),
        }
        if self.targets.org_mode:
            updates["repo_count"] = self._repo_count()
        if errors:
            updates |= {
                "last_error": "; ".join(errors),
                "last_error_at": now_iso,
                "last_error_source": "poll",
            }
        self._set(**updates)
        return scheduled

    async def _recent_repo_comments(self, repo: str) -> dict[int, list[PrComment]]:
        """Conversation comments of an explicitly listed repo since the last poll (one call),
        by PR number: where on-demand review requests come from outside org search."""
        assert self.github is not None
        now = self.clock()
        floor = self.triggers_since if self.triggers_since is not None else now
        since = max(self._comment_watermarks.get(repo.lower(), floor), floor)
        since -= self.settings.SEARCH_OVERLAP_SECONDS
        try:
            raw = await self.github.list_repo_issue_comments(repo, _gh_time(since))
        except Exception as exc:
            logger.info("Could not read recent comments of %s: %s", repo, exc)
            return {}
        self._comment_watermarks[repo.lower()] = now
        out: dict[int, list[PrComment]] = {}
        for c in raw:
            url = str(c.get("issue_url") or c.get("html_url") or "")
            tail = url.rstrip("/").rsplit("/", 1)[-1].split("#", 1)[0]
            if not tail.isdigit() or not c.get("id"):
                continue
            user = c.get("user") or {}
            out.setdefault(int(tail), []).append(
                PrComment(
                    id=int(c["id"]),
                    body=str(c.get("body") or ""),
                    created_at=c.get("created_at"),
                    author_login=str(user.get("login") or ""),
                    author_type=str(user.get("type") or ""),
                )
            )
        return out

    def _repo_count(self) -> int | None:
        counted: set[str] = set(self._direct_set)
        known = False
        for st in self._orgs.values():
            if st.repos is None:
                continue
            known = True
            counted |= {k for k, r in st.repos.items() if not r.archived and not r.fork}
        return len(counted) if known else None

    async def _org_repos(self, st: OrgState) -> dict[str, RepoInfo] | None:
        """The org's accessible repos (cached), or ``None`` while the App is not installed
        there (the installation list is re-read every few minutes)."""
        assert self.github is not None
        now = self.clock()
        if st.installation_id is None:
            st.installation_id = await self.github.installation_id_for_owner(st.org)
            if st.installation_id is None:
                if st.state != "not_installed":
                    logger.warning("The GitHub App is not installed on %s yet", st.org)
                st.state = "not_installed"
                return None
            logger.info("GitHub App installation %s found for %s", st.installation_id, st.org)
        fresh = (
            st.repos_fetched_at is not None
            and now - st.repos_fetched_at < self.settings.REPO_CACHE_SECONDS
        )
        if st.repos is None or not fresh:
            repos = await self.github.list_installation_repos(st.installation_id)
            prefix = st.org.lower() + "/"
            st.repos = {
                r.full_name.lower(): r for r in repos if r.full_name.lower().startswith(prefix)
            }
            st.repos_fetched_at = now
        return st.repos

    async def _discover_org(
        self, st: OrgState, budget: BudgetStatus
    ) -> list[tuple[PullRequest, Decision]]:
        assert self.github is not None
        repos = await self._org_repos(st)
        if repos is None:
            with self._status_lock:
                self._status.discovery.setdefault(st.org, {})["state"] = "not_installed"
            return []
        self._ensure_cutoff(st, create=True)
        assert st.cutoff is not None
        st.state = "ok"
        now = self.clock()
        interval = self.settings.RECONCILE_INTERVAL_SECONDS
        reconcile = (
            st.last_reconcile is None or st.watermark is None or now - st.last_reconcile >= interval
        )
        query = f"org:{st.org} is:pr is:open archived:false sort:updated-desc"
        if not reconcile:
            assert st.watermark is not None
            since = st.watermark - self.settings.SEARCH_OVERLAP_SECONDS
            query += f" updated:>={_gh_time(since)}"
        assert st.installation_id is not None
        result = await self.github.search_pulls(
            st.installation_id, query, page_size=100 if reconcile else 50
        )
        if result.truncated:
            logger.warning("Search for %s hit the 1000-result cap; oldest PRs not seen", st.org)
        counts: Counter[str] = Counter()
        scheduled: list[tuple[PullRequest, Decision]] = []
        for hit in result.hits:
            pr = hit.pr
            key = pr.repo.lower()
            if key in self._direct_set:
                counts["direct_repo"] += 1  # listed directly every poll
                continue
            if not self.targets.covers(pr.repo):
                counts["not_targeted"] += 1
                continue
            info = repos.get(key)
            if info is None:
                counts["not_accessible"] += 1
                continue
            if hit.archived or info.archived:
                counts["archived"] += 1
                continue
            if hit.fork or info.fork:
                counts["fork"] += 1
                continue
            decision = await self._consider_safely(pr, org=True, budget=budget)
            counts[decision.category or decision.mode.value] += 1
            if decision.mode is not Mode.SKIP:
                scheduled.append((pr, decision))
        st.watermark = now
        if reconcile:
            st.last_reconcile = now
        summary = {
            "at": iso(datetime.fromtimestamp(now, tz=UTC)),
            "kind": "reconcile" if reconcile else "incremental",
            "hits": len(result.hits),
            "total": result.total,
            "pages": result.pages,
            "truncated": result.truncated,
            "outcomes": dict(counts),
        }
        with self._status_lock:
            org_status = self._status.discovery.setdefault(st.org, {})
            org_status["state"] = "ok"
            org_status["last"] = summary
            if reconcile:
                org_status["last_reconcile"] = summary
            org_status["repos_accessible"] = len(repos)
            org_status["installation_id"] = st.installation_id
        return scheduled

    async def _consider_safely(
        self, pr: PullRequest, *, org: bool, budget: BudgetStatus
    ) -> Decision:
        """``_consider`` that never lets one bad PR stop the rest of the poll."""
        try:
            return await self._consider(pr, org=org, budget=budget)
        except Exception as exc:
            logger.exception("Deciding %s#%s failed", pr.repo, pr.number)
            self._set(
                last_error=f"{pr.repo}#{pr.number}: {exc}"[:500],
                last_error_at=iso(utcnow()),
                last_error_source="decision",
            )
            return Decision(Mode.SKIP, f"error: {type(exc).__name__}", "error")

    async def _consider(self, pr: PullRequest, *, org: bool, budget: BudgetStatus) -> Decision:
        """Decide one open PR and queue it when it needs a review."""
        state = self.store.get_pr_state(pr.repo, pr.number)
        now = self.clock()
        if not self.settings.DRY_RUN:  # dry runs write nothing
            self.store.note_seen(pr.repo, pr.number, now, settle_update(state, pr, now))
            state = self.store.get_pr_state(pr.repo, pr.number)
        # Remember a reviewed PR going back to draft, so leaving draft gets a full review.
        if pr.draft and state and state.last_reviewed_sha and not state.draft_since_review:
            self.store.set_draft_since_review(pr.repo, pr.number, True)
            state = self.store.get_pr_state(pr.repo, pr.number)
        if (
            not pr.draft
            and state
            and state.draft_since_review
            and state.last_reviewed_sha == pr.head_sha
        ):
            # Left draft without new commits: nothing new to review.
            self.store.set_draft_since_review(pr.repo, pr.number, False)
            state = self.store.get_pr_state(pr.repo, pr.number)
        trigger = find_trigger(
            pr.comments,
            after_id=state.last_trigger_id if state else None,
            not_before=self.triggers_since,
            app_slug=self.settings.GITHUB_APP_SLUG,
        )
        if trigger is not None:
            return await self._consider_on_demand(pr, state, trigger, org=org, budget=budget)
        if state and state.failed_sha == pr.head_sha and state.failed_trigger == "on_demand":
            return await self._retry_on_demand(pr, state, org=org, budget=budget)
        if org and pr.repo.lower() not in self._pilot_set:
            org_state = self._org_of(pr.repo)
            assert org_state is not None and org_state.cutoff is not None
            cutoff = org_state.cutoff
            if not (state and (state.org_live or state.last_reviewed_sha)):
                why = post_cutoff_reason(pr, state, cutoff)
                self.store.observe(
                    pr.repo, pr.number, pr.head_sha, pr.draft, self.clock(),
                    cutoff=cutoff, state=state,
                )  # fmt: skip
                if why is None:
                    return Decision(Mode.SKIP, "backlog: before the org-mode cutoff", "backlog")
                self.store.set_org_live(pr.repo, pr.number)
                state = self.store.get_pr_state(pr.repo, pr.number)
        decision = decide(
            pr,
            state,
            now=self.clock(),
            max_attempts=self.settings.MAX_ATTEMPTS_PER_SHA,
            retry_backoff_seconds=self.settings.RETRY_BACKOFF_SECONDS,
            retry_backoff_max_seconds=self.settings.RETRY_BACKOFF_MAX_SECONDS,
            max_rounds=self.settings.MAX_REVIEW_ROUNDS,
        )
        if decision.mode is Mode.SKIP:
            self._record_skip(pr, state, decision)
            return decision
        key = (pr.repo.lower(), pr.number)
        if key in self.queue.running:
            return Decision(Mode.SKIP, "review in progress", "running")
        queued = self.queue.queued.get(key)
        if queued and queued.payload.pr.head_sha == pr.head_sha:
            return Decision(Mode.SKIP, "already queued", "queued")
        if not budget.allows_any or (decision.needs_full_budget and not budget.allows_full):
            return self._held(pr, f"budget {budget.state} ({budget.source})", "budget")
        opt_out = await self._repo_opt_out(pr.repo)
        if opt_out:
            skipped = Decision(Mode.SKIP, f"repo opted out: {opt_out}", "repo_opted_out")
            self._record_skip(pr, state, skipped, mode=decision)
            return skipped
        await self._dispatch(pr, decision, self._ready_at(state, now))
        return decision

    async def _retry_on_demand(
        self, pr: PullRequest, state: PrState, *, org: bool, budget: BudgetStatus
    ) -> Decision:
        """Resume a failed explicit request, even on a draft or after automatic rounds ran out."""
        maximum = self.settings.MAX_ATTEMPTS_PER_SHA
        if state.failed_attempts >= maximum:
            return Decision(
                Mode.SKIP, f"gave up after {state.failed_attempts} failed attempts", "gave_up"
            )
        delay = retry_delay_seconds(
            pr.repo,
            pr.number,
            pr.head_sha,
            state.failed_attempts,
            base_seconds=self.settings.RETRY_BACKOFF_SECONDS,
            max_seconds=self.settings.RETRY_BACKOFF_MAX_SECONDS,
        )
        if state.last_failure_at and self.clock() - state.last_failure_at < delay:
            return Decision(Mode.SKIP, "retry backoff", "backoff")
        trigger = PrComment(
            id=state.last_trigger_id or 0,
            body="/review force" if state.failed_force else "/review",
            created_at=None,
            author_login=state.failed_requested_by or "unknown",
            author_type="User",
        )
        return await self._consider_on_demand(
            pr, state, trigger, org=org, budget=budget, retry=True
        )

    async def _consider_on_demand(
        self,
        pr: PullRequest,
        state: PrState | None,
        trigger: PrComment,
        *,
        org: bool,
        budget: BudgetStatus,
        retry: bool = False,
    ) -> Decision:
        """A human asked for a review in a PR comment: a full review of the current head,
        now (no settle wait), whatever its draft / reviewed / backlog status. The repo
        opt-out and the cost guards still apply; until it runs the request stays pending."""
        who = trigger.author_login
        force = is_force_request(trigger.body, self.settings.GITHUB_APP_SLUG)
        if (
            not retry
            and state
            and state.failed_sha == pr.head_sha
            and state.failed_attempts >= self.settings.MAX_ATTEMPTS_PER_SHA
        ):
            return Decision(
                Mode.SKIP, f"gave up after {state.failed_attempts} failed attempts", "gave_up"
            )
        decision = Decision(Mode.FULL, f"review requested by @{who}", "on_demand", "on_demand")
        key = (pr.repo.lower(), pr.number)
        if key in self.queue.running:
            return Decision(Mode.SKIP, "review in progress (request pending)", "running")
        queued = self.queue.queued.get(key)
        if (
            queued
            and queued.payload.pr.head_sha == pr.head_sha
            and queued.payload.decision.trigger == "on_demand"
        ):
            return Decision(Mode.SKIP, "already queued", "queued")
        opt_out = await self._repo_opt_out(pr.repo)
        if opt_out:
            if not retry and not self.settings.DRY_RUN:
                self.store.set_last_trigger(pr.repo, pr.number, trigger.id)
            return Decision(Mode.SKIP, f"repo opted out: {opt_out}", "repo_opted_out")
        if not force:
            posted = self.store.posted_review_for_head(pr.repo, pr.number, pr.head_sha)
            if posted:  # nothing new to review: point at it instead of paying again
                return await self._already_reviewed(pr, state, trigger, decision, posted)
        cap = self.settings.MAX_ON_DEMAND_REVIEWS_PER_PR_PER_DAY
        if cap > 0 and not retry:
            billed = self.store.billed_reviews_since(
                pr.repo, pr.number, utcnow() - timedelta(hours=24), kind="on_demand"
            )
            if billed >= cap:
                held = self._held(
                    pr, f"daily cap: {billed} billed on-demand reviews in 24 h", "daily_cap"
                )
                self._record_skip(pr, state, held, mode=decision)
                return held
        if not budget.allows_full:
            return self._held(pr, f"budget {budget.state} ({budget.source})", "budget")
        if org and pr.repo.lower() not in self._pilot_set and not self.settings.DRY_RUN:
            if not (state and (state.org_live or state.last_reviewed_sha)):
                self.store.set_org_live(pr.repo, pr.number)  # a human engaged: live work now
        await self._dispatch(
            pr,
            decision,
            self.clock(),
            requested_by=who,
            on_demand_request_id=trigger.id,
            force=force,
        )
        if not retry and not self.settings.DRY_RUN and self.accepting_work:
            self.store.set_last_trigger(pr.repo, pr.number, trigger.id)
            await self._acknowledge(pr.repo, trigger.id)
        return decision

    async def _already_reviewed(
        self,
        pr: PullRequest,
        state: PrState | None,
        trigger: PrComment,
        requested: Decision,
        posted: dict[str, Any],
    ) -> Decision:
        """A ``/review`` of a head that already has a posted review: one short reply linking
        it (how to get another), a $0 skip row, and the request marked handled."""
        who = trigger.author_login
        decision = Decision(
            Mode.SKIP, f"already reviewed at {pr.head_sha[:8]} (request by @{who})",
            "already_reviewed", "on_demand",
        )  # fmt: skip
        if self.settings.DRY_RUN:  # dry runs write nothing
            self._held(pr, decision.reason, decision.category)
            return decision
        assert self.github is not None
        url = posted.get("comment_url")
        review = f"[already reviewed]({url})" if url else "already reviewed"
        slug = self.settings.GITHUB_APP_SLUG or "gh-pr-reviewer"
        body = (
            f"{ALREADY_REVIEWED_MARKER}\n@{who} this commit "
            f"(`{pr.head_sha[:7]}`) was {review}; nothing new to review. Push new commits for "
            f"a fresh review, or comment `/review force` (or `@{slug} review force`) to "
            "review it again anyway."
        )
        # Marked handled first: a failed or repeated reply never turns into a loop.
        self.store.set_last_trigger(pr.repo, pr.number, trigger.id)
        self._record_skip(pr, state, decision, mode=requested)
        self._note_decision(pr, decision)
        try:
            await self.github.create_issue_comment(pr.repo, pr.number, body)
        except Exception as exc:
            logger.info("Could not reply to /review on %s#%s: %s", pr.repo, pr.number, exc)
        await self._acknowledge(pr.repo, trigger.id)
        return decision

    async def _acknowledge(self, repo: str, comment_id: int) -> None:
        """React with 👀 on the request comment (best effort)."""
        assert self.github is not None
        try:
            await self.github.add_comment_reaction(repo, comment_id, "eyes")
        except Exception as exc:
            logger.info("Could not react to comment %s in %s: %s", comment_id, repo, exc)

    def _ready_at(self, state: PrState | None, now: float) -> float:
        """When this head may be reviewed: its anchor event plus the settle wait."""
        anchor = state.head_ready_at if state and state.head_ready_at is not None else now
        return anchor + max(self.settings.REVIEW_SETTLE_SECONDS, 0)

    def _record_skip(
        self,
        pr: PullRequest,
        state: PrState | None,
        decision: Decision,
        *,
        mode: Decision | None = None,
    ) -> None:
        """Persist a stats-relevant skip (draft, bot, opt-out, daily cap) once per head."""
        reason = RECORDED_SKIPS.get(decision.category)
        if not reason or self.settings.DRY_RUN:
            return
        kind = mode.kind if mode is not None else "full"  # every automatic review is full
        try:
            self.store.record_skip(
                pr.repo, pr.number, pr.head_sha, reason, pr_url=pr.html_url, title=pr.title,
                author=pr.author_login or None, kind=kind, detail=decision.reason,
            )  # fmt: skip
        except Exception:  # bookkeeping must never stop a poll
            logger.exception("Could not record skip of %s#%s", pr.repo, pr.number)

    def _held(self, pr: PullRequest, reason: str, category: str) -> Decision:
        """A review that is due but held back by a cost guard (picked up again later)."""
        decision = Decision(Mode.SKIP, reason, category)
        key = (pr.repo.lower(), pr.number, pr.head_sha, category)
        if key not in self._held_logged:  # note once per PR head and reason
            self._held_logged.add(key)
            self._note_decision(pr, decision)
            logger.info("Holding %s#%s@%s: %s", pr.repo, pr.number, pr.head_sha[:8], reason)
        return decision

    async def _repo_opt_out(self, repo: str) -> str | None:
        """Topic ``no-ai-review`` or ``enabled: false`` in the repo's reviewer config (cached)."""
        assert self.github is not None
        now = self.clock()
        cached = self._repo_policy.get(repo.lower())
        if cached and cached[0] > now:
            return cached[1]
        try:
            info = await self.github.get_repo(repo)
            text = await self.github.get_file_text(
                repo, REPO_CONFIG_PATH, info.default_branch or "HEAD"
            )
            reason = repo_opt_out_reason(info.topics, text)
            ttl = float(self.settings.REPO_CACHE_SECONDS)
        except Exception as exc:  # fail open: GitHub trouble fails the review anyway
            logger.warning("Could not read opt-out settings of %s: %s", repo, exc)
            reason, ttl = None, REPO_POLICY_ERROR_TTL
        self._repo_policy[repo.lower()] = (now + ttl, reason)
        return reason

    async def _review_size(
        self, pr: PullRequest, decision: Decision
    ) -> tuple[list[dict] | None, int]:
        """The PR's files (handed to the reviewer) and its reviewable changed lines (for an
        incremental review: of the commits since the last reviewed SHA)."""
        assert self.github is not None
        reviewable = reviewable_filter()
        try:
            files = await self.github.list_pull_files(pr.repo, pr.number)
        except Exception as exc:
            logger.info("Could not size %s#%s: %s", pr.repo, pr.number, exc)
            return None, self.settings.LARGE_REVIEW_LINES + 1
        scope = files
        if decision.mode is Mode.INCREMENTAL:
            state = self.store.get_pr_state(pr.repo, pr.number)
            if state and state.last_reviewed_sha:
                try:
                    scope = await self.github.compare_files(
                        pr.repo, state.last_reviewed_sha, pr.head_sha
                    )
                except Exception:
                    scope = files
        return files, reviewable_lines(scope, reviewable)

    async def _dispatch(
        self,
        pr: PullRequest,
        decision: Decision,
        ready_at: float,
        *,
        requested_by: str | None = None,
        on_demand_request_id: int | None = None,
        force: bool = False,
    ) -> None:
        key = (pr.repo.lower(), pr.number)
        if key in self.queue.running:
            return  # per-PR lock: never review the same PR twice concurrently
        if self.settings.DRY_RUN:
            marker = (pr.repo.lower(), pr.number, pr.head_sha)
            if marker not in self._dry_run_logged:
                self._dry_run_logged.add(marker)
                self._note_decision(pr, decision)
                logger.info(
                    "DRY_RUN: would run %s on %s (%s): %s",
                    decision.mode.value, pr.html_url, pr.head_sha[:12], decision.reason,
                )  # fmt: skip
            return
        if not self.accepting_work:
            return
        queued = self.queue.queued.get(key)
        if (
            queued
            and queued.payload.pr.head_sha == pr.head_sha
            and (decision.trigger != "on_demand" or queued.payload.decision.trigger == "on_demand")
        ):
            return  # already waiting with this head (an on-demand request replaces a wait)
        files, lines = await self._review_size(pr, decision)
        payload = JobPayload(
            pr,
            decision,
            files,
            ready_at,
            requested_by=requested_by,
            on_demand_request_id=on_demand_request_id,
            force=force,
        )
        self.queue.push(key, pr.repo, lines, self.clock(), payload, not_before=ready_at)
        self._note_decision(pr, decision)
        self._sync_queue_status()

    def _pump(self) -> None:
        """Start queued reviews while slots allow (lanes, per-repo cap, size + aging)."""
        while self.accepting_work:
            job = self.queue.next_job(self.clock())
            if job is None:
                break
            task = asyncio.create_task(self._run_job(job), name=f"review {job.key}")
            self._tasks.add(task)
            task.add_done_callback(self._tasks.discard)
        self._sync_queue_status()

    def _record_review_failure(
        self,
        pr: PullRequest,
        error: str | None,
        *,
        decision: Decision,
        requested_by: str | None,
        force: bool,
    ) -> None:
        """Persist one failed attempt, including enough on-demand intent to resume safely."""
        trigger = "on_demand" if decision.trigger == "on_demand" else "automatic"
        self.store.mark_failed(
            pr.repo,
            pr.number,
            pr.head_sha,
            self.clock(),
            trigger=trigger,
            requested_by=requested_by,
            force=force,
        )
        state = self.store.get_pr_state(pr.repo, pr.number)
        attempts = state.failed_attempts if state and state.failed_sha == pr.head_sha else 1
        maximum = self.settings.MAX_ATTEMPTS_PER_SHA
        terminal = maximum > 0 and attempts >= maximum
        label = f"Review attempt {attempts}/{maximum} failed"
        if terminal:
            label = f"Review failed after {attempts}/{maximum} attempts; retries exhausted"
        detail = f"{label}: {error or 'unknown error'}"
        self._set(
            last_error=detail[:500],
            last_error_at=iso(utcnow()),
            last_error_source="review",
        )

    def _clear_review_error(self) -> None:
        """Clear only a review failure; don't hide a simultaneous discovery/poller failure."""
        with self._status_lock:
            if self._status.last_error_source == "review":
                self._status.last_error = None
                self._status.last_error_at = None
                self._status.last_error_source = None

    async def _run_job(self, job: Any) -> None:
        payload: JobPayload = job.payload
        pr, decision = payload.pr, payload.decision
        assert self.reviewer is not None and self.github is not None
        try:
            stale, skip_reason, fresh = await self._stale_reason(
                pr, allow_draft=decision.trigger == "on_demand"
            )
            if stale:
                # Waited (settle wait or a slot) while the PR moved on; a merged / closed PR is
                # recorded as skipped, otherwise the next poll re-decides.
                logger.info("Not reviewing %s#%s@%s: %s", pr.repo, pr.number,
                            pr.head_sha[:8], stale)  # fmt: skip
                self._note_decision(pr, Decision(Mode.SKIP, f"stale when dequeued: {stale}"))
                if skip_reason:
                    self.store.record_skip(
                        pr.repo, pr.number, pr.head_sha, skip_reason, pr_url=pr.html_url,
                        title=pr.title, author=pr.author_login or None, kind=decision.kind,
                        ready_at=payload.ready_at, detail=stale,
                    )  # fmt: skip
                return
            budget = self.budget.status
            if not budget.allows_any or (decision.needs_full_budget and not budget.allows_full):
                self._note_decision(pr, Decision(Mode.SKIP, f"budget {budget.state} at start"))
                return
            state = self.store.get_pr_state(pr.repo, pr.number)
            review_round: int | None = None
            if decision.trigger != "on_demand":
                # Counted again at start: the log, not the decision, is the source of truth.
                review_round = (state.auto_rounds if state else 0) + 1
                cap = self.settings.MAX_REVIEW_ROUNDS
                if cap > 0 and review_round > cap:
                    capped = Decision(Mode.SKIP, f"max rounds: {cap} reviews posted", "max_rounds")
                    self._note_decision(pr, capped)
                    self._record_skip(pr, state, capped, mode=decision)
                    return
            installation_id = await self.github.installation_id_for_repo(pr.repo)
            with self._status_lock:
                self._status.reviews_started += 1
            result = await self.reviewer.review(
                pr,
                decision.mode,
                installation_id,
                last_reviewed_sha=state.last_reviewed_sha if state else None,
                files=payload.files,
                kind=decision.kind,
                ready_at=payload.ready_at,
                pull=fresh,
                requested_by=payload.requested_by,
                on_demand_request_id=payload.on_demand_request_id,
                review_round=review_round,
                max_rounds=self.settings.MAX_REVIEW_ROUNDS,
            )
            if result.status in SETTLED_STATUSES:
                self.store.mark_reviewed(
                    pr.repo,
                    pr.number,
                    pr.head_sha,
                    decision.mode.value,
                    diff_fingerprint=result.diff_fingerprint,
                )
                if result.status in ("ok", "posted_post_merge"):
                    self._clear_review_error()
            elif result.status == CANCELLED:
                # Stopped between model calls (merged, closed, or a newer head): neither
                # settled nor failed, and no round or daily-cap slot used. The next poll
                # re-decides: a newer head gets this round; a re-listed merged / closed PR is
                # caught by the dequeue re-check (_stale_reason) before any model call.
                self._note_decision(pr, Decision(Mode.SKIP, f"stopped: {result.error}"))
            else:
                # Every failed run, including a timeout, is one persisted attempt. The next
                # poll retries the same head after its exponential, jittered backoff.
                self._record_review_failure(
                    pr,
                    result.error,
                    decision=decision,
                    requested_by=payload.requested_by,
                    force=payload.force,
                )
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.exception("Review task for %s#%s crashed", pr.repo, pr.number)
            self._record_review_failure(
                pr,
                f"{type(exc).__name__}: {exc}",
                decision=decision,
                requested_by=payload.requested_by,
                force=payload.force,
            )
        finally:
            self.queue.finish(job.key)
            self._pump()

    async def _stale_reason(
        self, pr: PullRequest, *, allow_draft: bool = False
    ) -> tuple[str | None, str | None, dict[str, Any] | None]:
        """Why a queued review should not run any more (merged, closed, draft, newer head),
        the skip reason to record for it (``None`` for a newer head: re-decided), and the
        fresh PR (handed to the reviewer for its description)."""
        assert self.github is not None
        try:
            fresh = await self.github.get_pull(pr.repo, pr.number)
        except Exception as exc:  # fail open: review what we were asked to
            logger.info("Could not re-check %s#%s before review: %s", pr.repo, pr.number, exc)
            return None, None, None
        if fresh.get("state") != "open":
            if fresh.get("merged") or fresh.get("merged_at"):
                return "PR was merged", "merged_during_wait", fresh
            return f"PR is {fresh.get('state') or 'not open'}", "closed", fresh
        if fresh.get("draft") and not allow_draft:
            return "PR went back to draft", "draft", fresh
        head = str((fresh.get("head") or {}).get("sha") or "")
        if head and head != pr.head_sha:
            return f"head moved to {head[:8]}", None, fresh
        return None, None, fresh

    async def wait_idle(self) -> None:
        """Wait for scheduled reviews to finish (tests and graceful shutdown)."""
        while self._tasks:
            await asyncio.gather(*list(self._tasks), return_exceptions=True)


def _configure_pr_agent_logging(level: str) -> None:
    """PR-Agent logs through loguru (DEBUG + colour by default); make it plain and quieter."""
    import sys

    from loguru import logger as loguru_logger
    from pr_agent.log import inv_analytics_filter

    loguru_logger.remove()
    loguru_logger.add(
        sys.stdout,
        level=level.upper(),
        colorize=False,
        filter=inv_analytics_filter,
        format="{time:YYYY-MM-DDTHH:mm:ss} {level} pr_agent {name}:{line} {message}",
    )
