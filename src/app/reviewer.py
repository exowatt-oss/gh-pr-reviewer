"""Run one native inline review of a PR head SHA and account for what it cost.

The model gets our own prompt (``review_prompt.py``) with a line-numbered diff and returns a
JSON review; the app validates every finding against the PR's diff hunks and posts one native
GitHub pull-request review (``event: COMMENT``) with inline comments and committable
suggestions. PR-Agent is still used for what it does well: the litellm handler (model routing,
reasoning effort, the cost callback) and the hunk line-numbering of the diff.

There is no size cap. A diff too large for one model call is split into parts
(``review_format.split_diff``), each reviewed by its own call with the same instructions and
tools, and their answers are merged into the one posted review.
"""

from __future__ import annotations

import asyncio
import contextlib
import glob
import logging
import shutil
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from functools import partial
from pathlib import Path
from typing import Any, Protocol

from app.config import Settings
from app.costs import apply_footer, render_footer
from app.github_app import GitHubApp, PullRequest, ReviewRejected
from app.llm_tracking import ReviewContext, current_review, flush_llm_callbacks
from app.pr_agent_config import (
    IMPROVE_COMMAND,
    REPO_CONVENTION_FILES,
    build_overrides,
    compose_extra_instructions,
    fresh_pr_agent_settings,
    is_requirements_txt,
    load_style_guide,
    looks_pip_compiled,
    reviewable_filter,
)
from app.repo_context import CHARS_PER_TOKEN, build_repo_context, diff_fingerprint
from app.repo_snapshot import RepoSnapshot, RepoTools, build_snapshot, clear_stale_snapshots
from app.review_agent import (
    AGENT_PROMPT,
    AgentLimits,
    AgentOutcome,
    ProviderConfig,
    ProviderStep,
    run_review_agent,
)
from app.review_format import (
    MAX_FINDINGS,
    MAX_FINDINGS_SPLIT,
    SYNTH_MAX_LINES,
    DiffFile,
    Finding,
    ModelOutputError,
    ModelReview,
    PriorComment,
    RenderedDiff,
    ReviewBody,
    anchor_finding,
    extend_patch,
    is_duplicate,
    normalize_findings,
    parse_model_output,
    parts_overview,
    post_merge_comment,
    prior_from_review_body,
    prior_summary,
    review_comment_payload,
    review_marker_sha,
    split_diff,
    suggestion_applies,
    synthesize_patch,
    tally,
)
from app.review_liveness import PullLiveness, ReviewCancelled
from app.review_policy import Mode, parse_time
from app.review_prompt import (
    SYSTEM_PROMPT,
    PromptInputs,
    PromptLayout,
    build_prompt,
    style_guide_system,
)
from app.store import Store, iso

logger = logging.getLogger(__name__)

# PR-Agent "PR Reviewer Guide" comments of earlier releases (recovery after a restart only).
LEGACY_REVIEW_MARKERS = ("<!-- pr-agent:review:full -->", "<!-- pr-agent:review:incremental -->")
# The "not reviewed, too large" note of releases that had a size cap; edited once such a PR
# gets its review.
TOO_LARGE_MARKER = "<!-- gh-pr-reviewer:too-large -->"
# Context left for the model's answer and our own framing when sizing the diff. With tools the
# conversation also grows by up to AgentLimits.max_tool_chars (160k chars, ~46k tokens) of tool
# output plus the turns' tool-call messages.
OUTPUT_RESERVE_TOKENS = 24_000
TOOL_RESERVE_TOKENS = 64_000
# Parts of a review too large for one model call that run at once (still one review slot).
PART_CONCURRENCY = 2
# With the prompt-cache layout, parts 2..n start once part 1's first model call streams its
# first chunk (its prompt is then in the provider's cache, so they read the shared prefix),
# at most this long after that call was sent; at once if part 1 ends first.
PART_STAGGER_MAX_SECONDS = 10.0
# How long a review waits for a free snapshot slot before it goes ahead diff-only.
SNAPSHOT_SLOT_WAIT_S = 600.0
# The PR merged while the review ran: one short issue comment instead of the review.
POSTED_POST_MERGE = "posted_post_merge"
# Statuses that settle a head SHA (marked reviewed; no retry).
SETTLED_STATUSES = ("ok", "skipped", POSTED_POST_MERGE)
# Stopped between model calls: the PR merged or closed, or (automatic reviews) a newer head
# arrived (``review_liveness``). Nothing posted; neither settled nor failed.
CANCELLED = "cancelled"


class _Skip(Exception):
    """Internal: end a review early without an error."""

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason  # stats skip_reason (lockfile_only, unchanged_diff)


def _private(pull: dict[str, Any]) -> bool | None:
    value = ((pull.get("base") or {}).get("repo") or {}).get("private")
    return value if isinstance(value, bool) else None


def _filenames(files: list[dict]) -> list[str]:
    return [str(f.get("filename") or "") for f in files]


@dataclass
class ModelRequest:
    pr_url: str
    kind: str
    since_sha: str | None
    system: str
    user: str
    settings: Any  # per-review PR-Agent settings copy: model routing, reasoning effort, timeout
    # Repository tools on a snapshot of the PR head: the review runs as a bounded tool loop.
    tools: RepoTools | None = None
    app_settings: Settings | None = None  # provider and tool-loop bounds for the tool loop
    outcome: AgentOutcome | None = None  # set by the tool loop: turns, tool calls, stop reason
    # Awaited before every model call; raises ReviewCancelled once the PR moved on.
    checkpoint: Callable[[], Awaitable[None]] | None = None
    # The same prompt in segments (``system`` / ``user`` are its plain text): the tool loop
    # opens with ``layout.messages()`` (the prompt-cache layout when ``layout.cache``).
    layout: PromptLayout | None = None
    # Tool loop: called right before its first model request goes out and when that request
    # streams its first chunk (part 1 of a review in parts releases the other parts).
    on_sent: Callable[[], None] | None = None
    on_first_chunk: Callable[[], None] | None = None


class ModelClient(Protocol):
    async def __call__(self, request: ModelRequest) -> str: ...


async def call_model(request: ModelRequest) -> str:
    """The review's model call(s). With tools: the bounded tool loop (``review_agent``),
    straight through litellm. Without: one chat completion through PR-Agent's litellm
    handler. Both stream with `reasoning_effort`, the per-request provider timeout, no
    fallbacks and no same-model retry on timeout; the cost callback records every call.
    ``request.checkpoint`` runs before each call (the PR may have merged meanwhile)."""
    if request.tools is not None and request.app_settings is not None:
        app_settings = request.app_settings
        request.outcome = await run_review_agent(
            request.layout or PromptLayout(request.system, "", request.user),
            request.tools,
            ProviderStep(
                ProviderConfig.from_settings(app_settings),
                on_sent=request.on_sent,
                on_first_chunk=request.on_first_chunk,
            ),
            AgentLimits.from_settings(app_settings),
            checkpoint=request.checkpoint,
        )
        return request.outcome.content
    if request.checkpoint is not None:
        await request.checkpoint()

    from starlette_context import request_cycle_context

    from app.llm_handler import StreamingLiteLLMAIHandler

    with request_cycle_context({"settings": request.settings}):
        handler = StreamingLiteLLMAIHandler()
        response, _finish = await handler.chat_completion(
            model=request.settings.config.model,
            system=request.system,
            user=request.user,
            temperature=0.2,
        )
    return str(response or "")


class PrAgentInvoker(Protocol):
    async def __call__(
        self, pr_url: str, installation_id: int, pr_settings: Any, request: list[str]
    ) -> bool: ...


async def invoke_pr_agent(
    pr_url: str, installation_id: int, pr_settings: Any, request: list[str]
) -> bool:
    """Run one PR-Agent command (only `/improve`, behind ENABLE_IMPROVE). True on success."""
    from pr_agent.agent.pr_agent import PRAgent
    from starlette_context import request_cycle_context

    from app.llm_handler import StreamingLiteLLMAIHandler

    initial = {"settings": pr_settings, "installation_id": installation_id, "git_provider": {}}
    with request_cycle_context(initial):
        return bool(
            await PRAgent(ai_handler=StreamingLiteLLMAIHandler).handle_request(
                pr_url, request, propagate_tool_errors=True
            )
        )


def merge_totals(parts: list[dict[str, Any]]) -> dict[str, Any]:
    merged: dict[str, Any] = {}
    for part in parts:
        for key, value in part.items():
            if value is None:
                merged.setdefault(key, None)
            else:
                merged[key] = (merged.get(key) or 0) + value
    return merged


@dataclass
class ReviewPlan:
    """What one review posts (or would post, with publishing off)."""

    kind: str
    body: str
    comments: list[dict[str, Any]]
    findings: list[Finding]  # reported in this review: inline + folded into the body
    duplicates: int
    folded: int
    tally: dict[str, int]
    raw_output: str = ""
    prompt_chars: int = 0
    review_url: str | None = None
    fell_back: bool = False  # GitHub refused the inline comments; folded into the body
    agent: AgentOutcome | None = None  # the tool loop, when the review had repository tools
    parts: int = 1  # model calls the diff was split into
    failed_parts: int = 0


@dataclass
class _Part:
    """One model call of a review: its diff and its answer (a review is one part unless its
    diff is too large for one call)."""

    diff: RenderedDiff
    raw: str = ""
    parsed: ModelReview | None = None
    outcome: AgentOutcome | None = None
    error: str | None = None
    prompt_chars: int = 0


class _Stagger:
    """Holds parts 2..n of a review in parts until part 1's prompt is in the provider's prompt
    cache: until part 1's first model request streams its first chunk, at most ``max_wait``
    seconds after it was sent. ``release`` (part 1 ended, however) lets them go at once."""

    def __init__(self, max_wait: float) -> None:
        self.max_wait = max_wait
        self._sent = asyncio.Event()
        self._ready = asyncio.Event()
        self._sent_at = 0.0

    def sent(self) -> None:
        if not self._sent.is_set():
            self._sent_at = asyncio.get_running_loop().time()
            self._sent.set()

    def first_chunk(self) -> None:
        self._ready.set()

    def release(self) -> None:
        self._ready.set()
        self._sent.set()

    async def wait(self) -> None:
        # Part 1 sends at once (after at most its liveness check, CHECK_TIMEOUT_S); the bound
        # only makes sure no path can ever hold the other parts forever.
        with contextlib.suppress(TimeoutError):
            await asyncio.wait_for(self._sent.wait(), 2 * self.max_wait)
        if self._ready.is_set() or not self._sent.is_set():
            return
        elapsed = asyncio.get_running_loop().time() - self._sent_at
        with contextlib.suppress(TimeoutError):
            await asyncio.wait_for(self._ready.wait(), max(self.max_wait - elapsed, 0.0))


def _merge_outcomes(outcomes: list[AgentOutcome | None]) -> AgentOutcome | None:
    """The tool loops of all parts as one (the first one alone for a one-part review)."""
    loops = [o for o in outcomes if o is not None]
    if len(loops) <= 1:
        return loops[0] if loops else None
    return AgentOutcome(
        content="",
        rounds=sum(o.rounds for o in loops),
        tool_calls=sum(o.tool_calls for o in loops),
        tool_chars=sum(o.tool_chars for o in loops),
        cost_usd=sum(o.cost_usd for o in loops),
        stopped=next((o.stopped for o in loops if o.stopped), None),
        trace=[t for o in loops for t in o.trace],
    )


@dataclass
class ReviewResult:
    review_id: int
    status: str
    totals: dict[str, Any]
    duration_s: float
    error: str | None = None
    comment_url: str | None = None
    diff_fingerprint: str | None = None
    # An LLM call timed out; retained for diagnostics, while the poller counts it as one retry.
    llm_timed_out: bool = False
    # Recovered a review that a restart interrupted after it was posted.
    recovered: bool = False
    plan: ReviewPlan | None = None
    skip_reason: str | None = None


_LLM_TIMEOUT_MARKERS = ("litellm.timeout", "apitimeouterror", "request timed out")


def _timed_out(contexts: list[ReviewContext], error: str | None) -> bool:
    """Did an LLM call time out? (Not a GitHub timeout: those are worth a retry.)"""
    calls = [c.error or "" for ctx in contexts for c in ctx.calls if c.status != "success"]
    if any("timeout" in t.lower() or "timed out" in t.lower() for t in calls):
        return True
    return any(m in (error or "").lower() for m in _LLM_TIMEOUT_MARKERS)


class _FileCache:
    """Memoised file reads for one review (diff context and codebase context share them)."""

    MAX_CHARS = 400_000

    def __init__(self, github: Any) -> None:
        self.github = github
        self._texts: dict[tuple[str, str, str], str | None] = {}
        self._sem = asyncio.Semaphore(8)

    async def get_file_text(self, repo: str, path: str, ref: str) -> str | None:
        key = (repo, path, ref)
        if key not in self._texts:
            async with self._sem:
                try:
                    text = await self.github.get_file_text(repo, path, ref)
                except Exception as exc:
                    logger.info("Could not read %s@%s in %s: %s", path, ref[:8], repo, exc)
                    text = None
            self._texts[key] = text if text is None or len(text) <= self.MAX_CHARS else None
        return self._texts[key]

    async def list_dir(self, repo: str, path: str, ref: str) -> list[dict[str, Any]]:
        return await self.github.list_dir(repo, path, ref)


def _commentable_lines(files: list[dict] | None) -> dict[str, set[int]]:
    out: dict[str, set[int]] = {}
    for f in files or []:
        d = DiffFile.from_api(f)
        out[d.path] = set(d.right_lines())
    return out


@dataclass
class _Scope:
    """The part of the PR one review covers."""

    kind: str
    files: list[dict]  # PR files in scope (all, or those the new commits touched)
    diff_files: list[dict]  # patches shown to the model (PR diff, or new-commit diff)
    since_sha: str | None = None
    new_commits: int | None = None
    allowed: dict[str, set[int]] | None = None
    prior: list[PriorComment] = field(default_factory=list)
    review_round: int | None = None  # automatic round; None for on-demand reviews
    max_rounds: int = 0


class Reviewer:
    POST_RETRY_DELAY_S = 3.0
    PART_STAGGER_MAX_S = PART_STAGGER_MAX_SECONDS

    def __init__(
        self,
        settings: Settings,
        store: Store,
        github: GitHubApp,
        *,
        model: ModelClient = call_model,
        invoker: PrAgentInvoker = invoke_pr_agent,
        app_login: str | None = None,
        liveness_clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.settings = settings
        self.store = store
        self.github = github
        self.model = model
        self.invoker = invoker
        self.liveness_clock = liveness_clock  # paces the PR re-checks between model calls
        self.app_login = app_login or f"{settings.GITHUB_APP_SLUG}[bot]"
        self.snapshot_root = Path(store.path).parent / "snapshots"
        self._snapshot_slots = asyncio.Semaphore(settings.snapshot_slots)
        clear_stale_snapshots(self.snapshot_root, older_than_s=0)  # left by a crash or restart

    # --- repository snapshot -------------------------------------------------------------
    def tools_enabled(self) -> bool:
        return self.settings.REVIEW_TOOLS and hasattr(self.github, "download_tarball")

    async def open_snapshot(self, pr: PullRequest) -> RepoSnapshot | None:
        """A snapshot of the PR head holding one of the snapshot slots, or ``None`` (no slot
        in time, low disk, too large, download failed): the review then runs diff-only."""
        try:
            await asyncio.wait_for(self._snapshot_slots.acquire(), SNAPSHOT_SLOT_WAIT_S)
        except TimeoutError:
            logger.warning("No snapshot slot for %s#%s; reviewing diff-only", pr.repo, pr.number)
            return None
        snapshot = None
        try:
            self.snapshot_root.mkdir(parents=True, exist_ok=True)
            free_mb = shutil.disk_usage(self.snapshot_root).free / 1e6
            if free_mb < self.settings.REVIEW_SNAPSHOT_MIN_FREE_MB:
                logger.warning(
                    "Only %.0f MB free under %s; reviewing %s#%s diff-only",
                    free_mb, self.snapshot_root, pr.repo, pr.number,
                )  # fmt: skip
            else:
                snapshot = await build_snapshot(
                    self.github, pr.repo, pr.head_sha, self.snapshot_root
                )
        except Exception as exc:
            logger.warning("Snapshot of %s@%s failed (%s); reviewing diff-only",
                           pr.repo, pr.head_sha[:8], exc)  # fmt: skip
        finally:
            if snapshot is None:
                self._snapshot_slots.release()
        return snapshot

    def snapshot_status(self) -> dict[str, Any]:
        """Snapshot slots in use and free disk under the snapshot dir (for ``/healthz``)."""
        free_mb = None
        with contextlib.suppress(OSError):
            path = self.snapshot_root if self.snapshot_root.exists() else self.snapshot_root.parent
            free_mb = round(shutil.disk_usage(path).free / 1e6)
        limit = self.settings.snapshot_slots
        return {
            "enabled": self.tools_enabled(),
            "active": limit - self._snapshot_slots._value,
            "max_concurrent": limit,
            "disk_free_mb": free_mb,
            "min_free_mb": self.settings.REVIEW_SNAPSHOT_MIN_FREE_MB,
        }

    def close_snapshot(self, snapshot: RepoSnapshot) -> None:
        try:
            snapshot.cleanup()
        finally:
            self._snapshot_slots.release()

    async def _finish_snapshot(self, task: asyncio.Task[RepoSnapshot | None] | None) -> None:
        if task is None:
            return
        if not task.done():
            task.cancel()
        await asyncio.wait([task])  # never raises the task's own error
        if not task.cancelled() and task.exception() is None:
            snapshot = task.result()
            if snapshot is not None:
                self.close_snapshot(snapshot)

    # --- inputs ------------------------------------------------------------------------
    async def repo_conventions(self, repo: str, ref: str) -> dict[str, str]:
        found: dict[str, str] = {}
        for name in REPO_CONVENTION_FILES:
            try:
                text = await self.github.get_file_text(repo, name, ref)
            except Exception as exc:
                logger.warning("Could not fetch %s from %s@%s: %s", name, repo, ref, exc)
                continue
            if text and text.strip():
                found[name] = text
        return found

    async def pull_files(self, pr: PullRequest) -> list[dict] | None:
        """The PR's changed files, or ``None`` when GitHub could not list them."""
        try:
            return await self.github.list_pull_files(pr.repo, pr.number)
        except Exception as exc:
            logger.warning("Could not list files for %s#%s: %s", pr.repo, pr.number, exc)
            return None

    async def pip_compiled_requirements(self, pr: PullRequest, files: list[dict]) -> list[str]:
        """Exact paths of requirements*.txt files in the PR that are pip-compile output."""
        paths = []
        for f in files:
            path = str(f.get("filename") or "")
            if not is_requirements_txt(path):
                continue
            text = f.get("patch") or ""
            if not looks_pip_compiled(text) and f.get("status") != "removed":
                try:
                    text = await self.github.get_file_text(pr.repo, path, pr.head_sha) or ""
                except Exception:
                    text = ""
            if looks_pip_compiled(text):
                paths.append(glob.escape(path))  # exact-path ignore glob
        return paths

    def _is_ours(self, item: dict) -> bool:
        return str((item.get("user") or {}).get("login", "")).lower() == self.app_login.lower()

    def _ours(self, comments: list[dict], markers: tuple[str, ...]) -> list[dict]:
        return [
            c
            for c in comments
            if self._is_ours(c) and any(m in (c.get("body") or "") for m in markers)
        ]

    async def prior_comments(self, pr: PullRequest) -> list[PriorComment]:
        """Findings earlier reviews by this App raised on the PR (for de-duplication): its
        inline comments, marked when their thread was resolved, and the findings listed in
        its review bodies (outside the diff, or folded when GitHub refused them inline)."""
        prior: list[PriorComment] = []
        try:
            comments = await self.github.list_review_comments(pr.repo, pr.number)
        except Exception as exc:
            logger.warning("Could not list review comments of %s#%s: %s", pr.repo, pr.number, exc)
            comments = []
        ours = [c for c in comments if self._is_ours(c) and not c.get("in_reply_to_id")]
        resolved: set[int] = set()
        if ours and hasattr(self.github, "resolved_review_comment_ids"):
            try:
                resolved = await self.github.resolved_review_comment_ids(pr.repo, pr.number)
            except Exception as exc:  # resolution only loosens matching; best effort
                logger.info("Could not read review threads of %s#%s: %s",
                            pr.repo, pr.number, exc)  # fmt: skip
        prior += [
            PriorComment.from_api(c, resolved=int(c.get("id") or 0) in resolved) for c in ours
        ]
        try:
            reviews = await self.github.list_reviews(pr.repo, pr.number)
        except Exception as exc:
            logger.info("Could not list reviews of %s#%s: %s", pr.repo, pr.number, exc)
            reviews = []
        for review in reviews:
            if self._is_ours(review):
                prior += prior_from_review_body(str(review.get("body") or ""))
        return prior

    @staticmethod
    def context_files(files: list[dict], new_files: list[dict] | None) -> list[dict]:
        """PR files to build codebase context from: for an incremental review only those
        the new commits touched (the rest was already reviewed with its context)."""
        if new_files is None:
            return files
        changed = {str(f.get("filename")) for f in new_files}
        return [f for f in files if str(f.get("filename")) in changed]

    async def model_patches(
        self, pr: PullRequest, files: list[DiffFile], base_sha: str | None, cache: _FileCache
    ) -> dict[str, str]:
        """Patches to show the model: GitHub's hunks widened with a few more unchanged lines
        of the new file, and a diff rebuilt from both versions where GitHub sent none."""
        out: dict[str, str] = {}

        async def one(f: DiffFile) -> None:
            if f.status == "removed":
                return
            head = await cache.get_file_text(pr.repo, f.path, pr.head_sha)
            if f.patch:
                if head is not None:
                    out[f.path] = extend_patch(f.patch, head)
                return
            if head is None or not base_sha:
                return
            base = ""
            if f.status != "added":
                base = await cache.get_file_text(pr.repo, f.previous_path or f.path, base_sha)
                if base is None:
                    return
            patch = synthesize_patch(base, head)
            if patch:
                out[f.path] = patch

        await asyncio.gather(*(one(f) for f in files))
        return out

    async def merge_base(self, pr: PullRequest, pull: dict[str, Any]) -> str | None:
        base = str((pull.get("base") or {}).get("sha") or "")
        if not base:
            return None
        try:
            data = await self.github.compare(pr.repo, base, pr.head_sha)
        except Exception as exc:
            logger.info("No merge base for %s#%s: %s", pr.repo, pr.number, exc)
            return None
        return str((data.get("merge_base_commit") or {}).get("sha") or "") or None

    async def codebase_context(
        self,
        pr: PullRequest,
        files: list[dict],
        extra_ignores: list[str],
        source: Any = None,
    ) -> str:
        try:
            ctx = await build_repo_context(
                source or self.github,
                pr.repo,
                files,
                head_sha=pr.head_sha,
                base_ref=pr.base_ref or "HEAD",
                is_reviewable=reviewable_filter(extra_ignores),
            )
        except Exception as exc:  # context is best effort; the review still runs
            logger.warning("Codebase context for %s#%s failed: %s", pr.repo, pr.number, exc)
            return ""
        logger.info(
            "Codebase context for %s#%s: %d files, %d chars%s",
            pr.repo, pr.number, len(ctx.sections), ctx.chars,
            f" (skipped: {ctx.skipped_reason})" if ctx.skipped_reason else "",
        )  # fmt: skip
        return ctx.render(pr.head_sha, pr.base_ref or "default branch")

    async def incremental_scope(
        self, pr: PullRequest, since_sha: str
    ) -> tuple[list[dict], int | None] | None:
        """Files (with patches) and commit count of the commits after ``since_sha``; ``None``
        when the anchor is gone (force-push) and the whole PR is reviewed again."""
        try:
            data = await self.github.compare(pr.repo, since_sha, pr.head_sha)
        except Exception as exc:  # e.g. the anchor was force-pushed away: whole PR
            logger.info("Compare %s...%s failed: %s", since_sha[:8], pr.head_sha[:8], exc)
            return None
        if str(data.get("status") or "") in ("diverged", "behind"):
            return None  # history rewritten: the old head is not an ancestor any more
        total = data.get("total_commits")
        return list(data.get("files") or []), int(total) if total is not None else None

    async def incremental_files(self, pr: PullRequest, since_sha: str) -> list[dict] | None:
        scope = await self.incremental_scope(pr, since_sha)
        return scope[0] if scope else None

    # --- recovery after a restart ----------------------------------------------------------
    async def _adopt_posted_review(self, pr: PullRequest, started: float) -> str | None:
        reviews = await self.github.list_reviews(pr.repo, pr.number)
        for review in reversed(reviews):
            if not self._is_ours(review) or review.get("commit_id") != pr.head_sha:
                continue
            if review_marker_sha(review.get("body") or "") != pr.head_sha:
                continue
            if (parse_time(review.get("submitted_at")) or 0) >= started - 5:
                return review.get("html_url")
        return None

    async def _adopt_legacy_comment(
        self, pr: PullRequest, started: float, totals: dict
    ) -> str | None:
        """A PR-Agent "PR Reviewer Guide" comment of the previous release (footer added)."""
        since_iso = iso(datetime.fromtimestamp(started - 30, tz=UTC)).replace("+00:00", "Z")
        comments = await self.github.list_issue_comments(pr.repo, pr.number, since=since_iso)
        # PR-Agent edited its comment a minute or more into a run, so anything updated within
        # seconds of the start is the previous run's footer, not this run's comment.
        ours = [
            c
            for c in self._ours(comments, LEGACY_REVIEW_MARKERS)
            if (parse_time(c.get("updated_at")) or 0) >= started + 5
        ]
        if not ours:
            return None
        target = max(ours, key=lambda c: (c.get("updated_at") or "", c.get("id") or 0))
        footer = render_footer(
            totals["prompt_tokens"], totals["completion_tokens"], totals["cost_usd"],
            self.settings.LLM_REASONING_EFFORT,
        )  # fmt: skip
        new_body = apply_footer(target.get("body") or "", footer)
        if new_body != target.get("body"):
            await self.github.update_issue_comment(pr.repo, int(target["id"]), new_body)
        return target.get("html_url")

    async def recover_interrupted(self, pr: PullRequest) -> ReviewResult | None:
        """A restart cut off a review of this exact SHA after it had paid for LLM calls. If
        its review landed (the App's review for that SHA, submitted after the run started),
        adopt it instead of paying for the same review again."""
        prior = self.store.interrupted_billed_review(pr.repo, pr.number, pr.head_sha)
        if not prior:
            return None
        started = parse_time(prior["started_at"]) or 0.0
        totals = {
            "calls": prior["llm_calls"],
            "prompt_tokens": prior["prompt_tokens"],
            "completion_tokens": prior["completion_tokens"],
            "cost_usd": prior["cost_usd"],
        }
        try:
            url = await self._adopt_posted_review(pr, started)
            if not url:
                url = await self._adopt_legacy_comment(pr, started, totals)
        except Exception as exc:
            logger.warning("Recovery check on %s#%s failed: %s", pr.repo, pr.number, exc)
            return None
        if not url:
            return None  # the review never landed: a real re-review is needed
        self.store.mark_recovered(int(prior["id"]), url)
        logger.info(
            "Recovered interrupted review %s of %s#%s@%s (%s); not re-billing",
            prior["id"], pr.repo, pr.number, pr.head_sha[:8], url,
        )  # fmt: skip
        return ReviewResult(int(prior["id"]), "ok", totals, 0.0, comment_url=url, recovered=True)

    async def _retire_too_large_note(self, pr: PullRequest, review_url: str | None) -> None:
        """A PR that got the old "not reviewed, too large" note now has its review: say so in
        the note (once; best effort; only PRs with such a skip on record are looked at)."""
        if not self.store.has_skip(pr.repo, pr.number, "too_large"):
            return
        body = (
            f"{TOO_LARGE_MARKER}\n<sub>gh-pr-reviewer: reviewed at `{pr.head_sha[:7]}`"
            + (f" ([review]({review_url}))" if review_url else "")
            + " — PRs of any size are reviewed now; a diff too large for one model call is "
            "reviewed in parts.</sub>"
        )
        try:
            comments = await self.github.list_issue_comments(pr.repo, pr.number)
            for note in self._ours(comments, (TOO_LARGE_MARKER,)):
                if "not reviewed" in str(note.get("body") or ""):
                    await self.github.update_issue_comment(pr.repo, int(note["id"]), body)
        except Exception as exc:
            logger.info("Could not update the too-large note on %s#%s: %s",
                        pr.repo, pr.number, exc)  # fmt: skip

    # --- the review ----------------------------------------------------------------------
    @staticmethod
    def _changed_lines(files: list[dict], path: str) -> int:
        for f in files:
            if f.get("filename") == path:
                return int(f.get("additions") or 0) + int(f.get("deletions") or 0)
        return 0

    def _diff_budget_chars(self, framing_chars: int, tools: bool = False) -> int:
        """Characters of rendered diff one model call takes: what the context leaves after
        the instructions, the answer and (with tools) the tool output, and at most
        ``MAX_REVIEW_DIFF_TOKENS``."""
        tokens = self.settings.LLM_MAX_CONTEXT_TOKENS - OUTPUT_RESERVE_TOKENS
        if tools:
            tokens -= TOOL_RESERVE_TOKENS
        budget = int(tokens * CHARS_PER_TOKEN) - framing_chars
        if self.settings.MAX_REVIEW_DIFF_TOKENS > 0:
            budget = min(budget, int(self.settings.MAX_REVIEW_DIFF_TOKENS * CHARS_PER_TOKEN))
        return max(budget, 20_000)

    def split(
        self,
        files: list[DiffFile],
        patches: dict[str, str],
        system: str,
        inputs: PromptInputs,
        tools: bool,
        *,
        cache: bool = False,
    ) -> list[RenderedDiff]:
        """The review's diff in parts that each fit one model call (usually just one). The
        framing counts every segment of the prompt the model reads (``PromptLayout.chars``)."""
        framing = build_prompt(system, inputs, cache=cache).chars
        diffs = split_diff(files, self._diff_budget_chars(framing, tools), patches)
        if len(diffs) > 1:
            # each part's prompt also carries the part header and the PR's file list
            probe = replace(inputs, part=1, part_count=len(diffs), part_files=[])
            framing = build_prompt(system, probe, cache=cache).chars
            diffs = split_diff(files, self._diff_budget_chars(framing, tools), patches)
        return diffs

    async def run_parts(
        self,
        pr: PullRequest,
        scope: _Scope,
        system: str,
        inputs: PromptInputs,
        pr_settings: Any,
        tools: RepoTools | None,
        parts: list[_Part],
        *,
        checkpoint: Callable[[], Awaitable[None]] | None = None,
        cache: bool = False,
    ) -> None:
        """One model call per part, at most ``PART_CONCURRENCY`` at once, sharing the review's
        snapshot, instructions and cost attribution. Any failed part fails the whole review;
        nothing partial is posted or marked reviewed, so the poller can retry the same head
        within its persisted attempt cap. A part stopped by ``checkpoint`` (``ReviewCancelled``)
        stops the whole review: nothing partial is posted.

        With the prompt-cache layout (``cache``) parts 2..n wait for part 1's first model
        request to stream its first chunk (``_Stagger``), so they read the system text and
        the shared user text from the cache part 1 just wrote instead of each paying for
        writing it."""
        slots = asyncio.Semaphore(PART_CONCURRENCY)
        many = len(parts) > 1
        stagger = _Stagger(self.PART_STAGGER_MAX_S) if many and cache else None

        async def call(i: int, part: _Part) -> None:
            layout = build_prompt(
                system,
                replace(
                    inputs,
                    diff=part.diff.text,
                    part=i if many else 0,
                    part_count=len(parts) if many else 0,
                    part_files=part.diff.shown,
                ),
                cache=cache,
            )
            part.prompt_chars = layout.chars
            request = ModelRequest(
                pr.html_url, scope.kind, scope.since_sha, layout.system, layout.user,
                pr_settings, tools=tools, app_settings=self.settings, checkpoint=checkpoint,
                layout=layout,
            )  # fmt: skip
            if stagger is not None and i == 1:
                request.on_sent, request.on_first_chunk = stagger.sent, stagger.first_chunk
            elif stagger is not None:
                await stagger.wait()
            async with slots:
                part.raw = await self.model(request)
            part.outcome = outcome = request.outcome
            if outcome is not None:
                logger.info(
                    "Tool loop on %s#%s%s: %d turns, %d tool calls, %d chars of tool output%s",
                    pr.repo, pr.number, f" (part {i}/{len(parts)})" if many else "",
                    outcome.rounds, outcome.tool_calls, outcome.tool_chars,
                    f", stopped early: {outcome.stopped}" if outcome.stopped else "",
                )  # fmt: skip
            part.parsed = parse_model_output(part.raw)

        async def run(i: int, part: _Part) -> None:
            if stagger is None or i > 1:
                await call(i, part)
                return
            try:
                await call(i, part)
            finally:
                stagger.release()  # part 1 ended (answered, failed or cancelled)

        if not many:
            await run(1, parts[0])
            return
        logger.info("Reviewing %s#%s in %d parts", pr.repo, pr.number, len(parts))
        results = await asyncio.gather(
            *(run(i, part) for i, part in enumerate(parts, 1)), return_exceptions=True
        )
        for result in results:
            if isinstance(result, ReviewCancelled):
                raise result  # never a partial review of a PR that moved on
        errors = [r for r in results if isinstance(r, BaseException)]
        for i, (part, result) in enumerate(zip(parts, results, strict=True), 1):
            if isinstance(result, BaseException):
                part.error = f"{type(result).__name__}: {result}"[:300]
                logger.warning("Part %d/%d of %s#%s failed: %s",
                               i, len(parts), pr.repo, pr.number, part.error)  # fmt: skip
        if errors:
            # A posted partial review settles the head and makes the missing files invisible to
            # the automatic retry policy. Leave every part unposted on any error instead.
            raise errors[0]

    def plan(
        self,
        *,
        pr: PullRequest,
        scope: _Scope,
        pr_files: list[dict],
        parts: list[_Part],
        totals: dict[str, Any],
    ) -> ReviewPlan:
        """Turn the model's answer(s) into the review: anchored inline comments, committable
        suggestions where they apply, duplicates of earlier reviews dropped. The answers of a
        multi-part review are merged: each part's findings on its own files, one overview."""
        diff = {d.path: d for d in map(DiffFile.from_api, pr_files)}
        done = [p for p in parts if p.parsed is not None]
        shown_set = {path for p in parts for path in p.diff.shown}
        reviewed_set = {path for p in done for path in p.diff.shown}
        findings: list[Finding] = []
        summaries: dict[str, str] = {}
        for part in done:
            assert part.parsed is not None
            own = set(part.diff.shown)
            findings += [
                f
                for f in part.parsed.findings
                if f.path in own or (f.path in diff and f.path not in shown_set)
            ]
            for path, summary in part.parsed.files.items():
                summaries.setdefault(path, summary)
        cap = MAX_FINDINGS if len(parts) == 1 else MAX_FINDINGS_SPLIT
        findings = normalize_findings(findings, cap)
        comments: list[dict[str, Any]] = []
        reported: list[Finding] = []
        folded: list[tuple[Finding, str]] = []
        duplicates = 0
        for f in findings:
            anchor = anchor_finding(f, diff, scope.allowed)
            if is_duplicate(f, anchor, scope.prior):
                duplicates += 1
                continue
            reported.append(f)
            if anchor is None:
                where = f"{f.path}:{f.start_line}" + (
                    f"-{f.end_line}" if f.end_line != f.start_line else ""
                )
                folded.append((f, where))
                continue
            with_suggestion = suggestion_applies(f, anchor, diff[f.path])
            comments.append(review_comment_payload(f, anchor, with_suggestion))
        scope_names = _filenames(scope.files)
        reviewed = [n for n in scope_names if n in reviewed_set]
        if len(parts) == 1:
            assert parts[0].parsed is not None
            overview = parts[0].parsed.overview
        else:
            overview = parts_overview(
                [(len(p.diff.shown), p.parsed.overview if p.parsed else None) for p in parts]
            )
        unreviewed = [
            path
            for path in dict.fromkeys(n for p in parts if p.parsed is None for n in p.diff.shown)
            if path not in reviewed_set
        ]
        body = ReviewBody(
            head_sha=pr.head_sha,
            kind=scope.kind,
            overview=overview,
            file_summaries=summaries,
            reviewed_files=reviewed,
            total_files=len(scope_names),
            findings=reported,
            folded=folded,
            duplicates=duplicates,
            footer=(
                int(totals["prompt_tokens"] or 0),
                int(totals["completion_tokens"] or 0),
                float(totals["cost_usd"] or 0.0),
                self.settings.LLM_REASONING_EFFORT,
            ),
            new_commits=scope.new_commits,
            since_sha=scope.since_sha,
            unreviewed_files=unreviewed,
            round=scope.review_round,
            max_rounds=scope.max_rounds,
            parts=len(parts),
        )
        return ReviewPlan(
            kind=scope.kind,
            body=body.render(),
            comments=comments,
            findings=reported,
            duplicates=duplicates,
            folded=len(folded),
            tally=tally(reported),
            raw_output="\n\n".join(p.raw for p in parts if p.raw),
            prompt_chars=sum(p.prompt_chars for p in parts),
            agent=_merge_outcomes([p.outcome for p in parts]),
            parts=len(parts),
            failed_parts=len(parts) - len(done),
        )

    async def _create_review(
        self, pr: PullRequest, body: str, comments: list[dict[str, Any]]
    ) -> dict[str, Any]:
        """POST the review; one retry on a transient GitHub error, since failing here would
        pay for the whole model call again on the next attempt."""
        for attempt in (1, 2):
            try:
                return await self.github.create_review(
                    pr.repo, pr.number, commit_id=pr.head_sha, body=body, comments=comments
                )
            except ReviewRejected:
                raise
            except Exception as exc:
                if attempt == 2:
                    raise
                logger.warning("Posting the review on %s#%s failed (%s); retrying",
                               pr.repo, pr.number, exc)  # fmt: skip
                await asyncio.sleep(self.POST_RETRY_DELAY_S)
        raise AssertionError("unreachable")

    async def publish(self, pr: PullRequest, plan: ReviewPlan) -> str | None:
        """Post the plan as one pull-request review; when GitHub refuses an inline anchor
        (422), post the same review with every comment folded into the body instead."""
        try:
            review = await self._create_review(pr, plan.body, plan.comments)
        except ReviewRejected as exc:
            if not plan.comments:
                raise
            logger.warning(
                "GitHub refused the inline comments of %s#%s (%s); folding them into the body",
                pr.repo, pr.number, exc,
            )  # fmt: skip
            folded = "\n\n".join(
                f"**`{c['path']}:{c.get('start_line', c['line'])}`**\n\n{c['body']}"
                for c in plan.comments
            )
            body = plan.body.replace(
                "\n\n<sub>gh-pr-reviewer", f"\n\n{folded}\n\n<sub>gh-pr-reviewer", 1
            )
            review = await self._create_review(pr, body, [])
            plan.folded += len(plan.comments)
            plan.comments = []
            plan.fell_back = True
        plan.review_url = review.get("html_url")
        return plan.review_url

    async def pr_state_now(self, pr: PullRequest) -> str | None:
        """``open`` / ``merged`` / ``closed`` right now, or ``None`` when GitHub could not say
        (fail open: the review is posted as usual)."""
        try:
            pull = await self.github.get_pull(pr.repo, pr.number)
        except Exception as exc:
            logger.info("Could not re-check %s#%s before posting: %s", pr.repo, pr.number, exc)
            return None
        state = str(pull.get("state") or "")
        if state == "open":
            return "open"
        if pull.get("merged") or pull.get("merged_at"):
            return "merged"
        return "closed" if state else None

    def liveness(
        self, pr: PullRequest, *, on_demand: bool, pull: dict[str, Any] | None
    ) -> PullLiveness | None:
        """The PR re-check before each model call, or ``None`` when it is off. Automatic
        reviews also stop on a newer head; an on-demand one only on merge / close."""
        interval = self.settings.REVIEW_LIVENESS_CHECK_SECONDS
        if interval <= 0:
            return None
        return PullLiveness(
            self.github, pr.repo, pr.number, pr.head_sha, cancel_on_supersede=not on_demand,
            interval_s=interval, clock=self.liveness_clock, pull=pull,
        )  # fmt: skip

    async def publish_post_merge(self, pr: PullRequest, plan: ReviewPlan, totals: dict) -> str:
        """The PR merged mid-review: one issue comment listing the findings at the SHA."""
        repo_url = pr.html_url.split("/pull/")[0] if "/pull/" in pr.html_url else ""
        body = post_merge_comment(
            repo_url=repo_url or f"https://github.com/{pr.repo}",
            head_sha=pr.head_sha,
            base_ref=pr.base_ref,
            findings=plan.findings,
            footer=(
                int(totals["prompt_tokens"] or 0),
                int(totals["completion_tokens"] or 0),
                float(totals["cost_usd"] or 0.0),
                self.settings.LLM_REASONING_EFFORT,
            ),
        )
        comment = await self.github.create_issue_comment(pr.repo, pr.number, body)
        plan.body, plan.comments = body, []
        plan.review_url = comment.get("html_url")
        return plan.review_url or ""

    async def review(
        self,
        pr: PullRequest,
        mode: Mode,
        installation_id: int,
        *,
        last_reviewed_sha: str | None = None,
        files: list[dict] | None = None,
        kind: str | None = None,
        ready_at: float | None = None,
        pull: dict[str, Any] | None = None,
        requested_by: str | None = None,
        on_demand_request_id: int | None = None,
        publish: bool | None = None,
        review_round: int | None = None,
        max_rounds: int = 0,
    ) -> ReviewResult:
        publish = (not self.settings.DRY_RUN) if publish is None else publish
        if publish:
            recovered = await self.recover_interrupted(pr)
            if recovered:
                return recovered
        command = mode.value
        on_demand = kind == "on_demand"
        review_id = self.store.start_review(
            pr.repo, pr.number, pr.html_url, pr.title, pr.head_sha, command,
            kind=kind, ready_at=ready_at, author=pr.author_login or None,
            review_round=None if on_demand else review_round,
            on_demand_request_id=on_demand_request_id,
        )  # fmt: skip
        ctx = ReviewContext(pr.repo, pr.number, pr.head_sha, command, review_id=review_id)
        contexts = [ctx]
        started = time.monotonic()
        status, error, comment_url, fingerprint = "error", None, None, None
        skip_reason: str | None = None
        plan: ReviewPlan | None = None
        token = current_review.set(ctx)
        snapshot_task: asyncio.Task[RepoSnapshot | None] | None = None
        # Seeded with the poller's fresh read of the PR; nothing to stop when not publishing.
        liveness = self.liveness(pr, on_demand=on_demand, pull=pull) if publish else None
        try:
            if files is None:
                files = await self.pull_files(pr)
            files = files or []
            extra_ignores = await self.pip_compiled_requirements(pr, files)
            reviewable = reviewable_filter(extra_ignores)
            if files and not any(map(reviewable, _filenames(files))):
                # Lockfile / generated / binary-only PR. Nothing to review, nothing to pay.
                status = "skipped"
                error = "no reviewable files (all ignored)"
                raise _Skip("lockfile_only")
            fingerprint = diff_fingerprint(files, reviewable) if files else None
            previous = self.store.get_pr_state(pr.repo, pr.number)
            if (
                not on_demand
                and fingerprint
                and previous
                and previous.last_reviewed_diff == fingerprint
            ):
                # A rebase, an empty commit or an ignored-files-only push: the code under
                # review is identical, so a re-review would only re-bill the same findings.
                status = "skipped"
                error = "reviewable diff unchanged since the last review (rebase/empty commit)"
                raise _Skip("unchanged_diff")
            scope = _Scope(kind=kind or ("incremental" if mode is Mode.INCREMENTAL else "full"),
                           files=files, diff_files=files,
                           review_round=None if on_demand else review_round,
                           max_rounds=max_rounds)  # fmt: skip
            if mode is Mode.INCREMENTAL and last_reviewed_sha and not on_demand:
                inc = await self.incremental_scope(pr, last_reviewed_sha)
                if inc is None:
                    scope.kind = "full"  # history rewritten: review the whole PR again
                else:
                    new_files, scope.new_commits = inc
                    scope.since_sha = last_reviewed_sha
                    scope.diff_files = new_files
                    scope.files = self.context_files(files, new_files) or new_files
                    scope.allowed = _commentable_lines(new_files)
            elif scope.kind == "incremental":
                scope.kind = "full"
            changed_since: list[str] | None = None
            if scope.review_round and scope.review_round >= 2 and last_reviewed_sha:
                # A full re-review; the model is told which files the new commits touched.
                inc = await self.incremental_scope(pr, last_reviewed_sha)
                if inc is not None:
                    changed_since = [n for n in _filenames(inc[0]) if n]
            if self.tools_enabled():
                # downloads while the other inputs are gathered
                snapshot_task = asyncio.create_task(self.open_snapshot(pr))
            scope.prior = await self.prior_comments(pr)
            cache = _FileCache(self.github)
            conventions = await self.repo_conventions(pr.repo, pr.base_ref or "HEAD")
            instructions = partial(  # (style guide) -> the user prompt's instructions
                compose_extra_instructions,
                repo_conventions=conventions,
                base_ref=pr.base_ref or "default branch",
                codebase_context=await self.codebase_context(pr, scope.files, extra_ignores, cache),
                excluded_files=[n for n in _filenames(files) if n and not reviewable(n)],
                output_rules=False,
            )
            style_guide = load_style_guide(self.settings)
            if pull is None:
                try:
                    pull = await self.github.get_pull(pr.repo, pr.number)
                except Exception as exc:
                    logger.info("Could not read %s#%s: %s", pr.repo, pr.number, exc)
                    pull = {}
                if liveness is not None:
                    liveness.seed(pull)
            inputs = PromptInputs(
                repo=pr.repo,
                number=pr.number,
                url=pr.html_url,
                title=pr.title,
                description=str(pull.get("body") or ""),
                base_ref=pr.base_ref,
                head_ref=str((pull.get("head") or {}).get("ref") or ""),
                author=pr.author_login,
                kind=scope.kind,
                diff="",
                instructions=instructions(style_guide),
                since_sha=last_reviewed_sha if changed_since is not None else scope.since_sha,
                new_commits=scope.new_commits,
                prior_findings=[prior_summary(p) for p in scope.prior] or None,
                requested_by=requested_by,
                private=_private(pull),
                review_round=scope.review_round,
                max_rounds=max_rounds,
                changed_since=changed_since,
            )
            snapshot = await snapshot_task if snapshot_task else None
            tools = RepoTools(snapshot) if snapshot else None
            system = SYSTEM_PROMPT + AGENT_PROMPT if tools else SYSTEM_PROMPT
            # The tool loop's prompt-cache layout: the house rubric moves from the user prompt
            # to the end of the system text (identical for every review), so one breakpoint
            # caches tools + system for all reviews. The diff-only call keeps today's prompt.
            cache_layout = self.settings.LLM_PROMPT_CACHE_LAYOUT and tools is not None
            if cache_layout:
                system = style_guide_system(system, style_guide)
                inputs.instructions = instructions("")
            in_diff = [f for f in scope.diff_files if reviewable(str(f.get("filename") or ""))]
            diff_files = [DiffFile.from_api(f) for f in in_diff]
            inputs.tools = tools is not None
            inputs.changed_files = [
                (d.path, int(f.get("additions") or 0), int(f.get("deletions") or 0))
                for d, f in zip(diff_files, in_diff, strict=True)
            ]
            base_sha = scope.since_sha
            needs_base = any(
                not d.patch
                and d.status != "removed"
                and 0 < self._changed_lines(scope.diff_files, d.path) <= SYNTH_MAX_LINES
                for d in diff_files
            )
            if base_sha is None and needs_base:
                base_sha = await self.merge_base(pr, pull)
            patches = await self.model_patches(pr, diff_files, base_sha, cache)
            parts = [
                _Part(d)
                for d in self.split(
                    diff_files, patches, system, inputs, bool(tools), cache=cache_layout
                )
            ]
            overrides = build_overrides(
                self.settings,
                private_key_pem=self.github.private_key,
                extra_instructions="",
                extra_ignore_globs=extra_ignores,
            )
            pr_settings = fresh_pr_agent_settings(overrides)
            try:
                await self.run_parts(
                    pr, scope, system, inputs, pr_settings, tools, parts,
                    checkpoint=liveness.check if liveness is not None else None,
                    cache=cache_layout,
                )  # fmt: skip
            finally:
                await self._finish_snapshot(snapshot_task)
                snapshot_task = None
            await flush_llm_callbacks()
            plan = self.plan(pr=pr, scope=scope, pr_files=files, parts=parts, totals=ctx.totals())
            failed = [p for p in parts if p.error]
            if failed:  # posted anyway; the body names the files left unreviewed
                error = f"{len(failed)} of {len(parts)} parts failed: {failed[0].error}"[:500]
            # A long review can finish after the PR merged or closed: re-check right before
            # posting (the cost is already paid and is recorded either way).
            state_now = await self.pr_state_now(pr) if publish else None
            if state_now == "closed":
                status, skip_reason = "skipped", "closed_during_review"
                error = "PR was closed (not merged) while it was reviewed; nothing posted"
            elif state_now == "merged" and not plan.findings:
                status, skip_reason = "skipped", "merged_during_review"
                error = "PR was merged while it was reviewed; no findings, nothing posted"
            elif state_now == "merged":
                comment_url = await self.publish_post_merge(pr, plan, ctx.totals())
                status = POSTED_POST_MERGE
            else:
                if publish:
                    comment_url = await self.publish(pr, plan)
                    await self._retire_too_large_note(pr, comment_url)
                status = "ok"
            if status == "ok" and self.settings.ENABLE_IMPROVE and publish:
                # A separate context, so in-flight callbacks of the review keep their command.
                improve_ctx = ReviewContext(
                    pr.repo, pr.number, pr.head_sha, "improve", review_id=review_id
                )
                contexts.append(improve_ctx)
                improve_token = current_review.set(improve_ctx)
                try:
                    await self.invoker(pr.html_url, installation_id, pr_settings, IMPROVE_COMMAND)
                except Exception as exc:  # the review itself already landed
                    logger.warning("/improve on %s#%s failed: %s", pr.repo, pr.number, exc)
                finally:
                    current_review.reset(improve_token)
        except _Skip as skip:
            skip_reason = skip.reason
        except ReviewCancelled as cancel:
            # The calls already made stay billed (recorded by the cost callback).
            status, skip_reason, error = CANCELLED, cancel.reason, cancel.message
            logger.info("Stopped the review of %s#%s@%s: %s",
                        pr.repo, pr.number, pr.head_sha[:8], cancel.message)  # fmt: skip
        except ModelOutputError as exc:
            logger.warning("Unusable model output for %s#%s: %s", pr.repo, pr.number, exc)
            error = f"ModelOutputError: {exc}"[:500]
        except Exception as exc:
            logger.exception("Review of %s#%s failed", pr.repo, pr.number)
            error = f"{type(exc).__name__}: {exc}"[:500]
        finally:
            await self._finish_snapshot(snapshot_task)
            await flush_llm_callbacks()
            current_review.reset(token)

        totals = merge_totals([c.totals() for c in contexts])
        duration = time.monotonic() - started
        self.store.finish_review(
            review_id,
            status=status,
            duration_s=duration,
            error=error,
            comment_url=comment_url,
            skip_reason=skip_reason,
            findings=plan.tally if plan and status in ("ok", POSTED_POST_MERGE) else None,
            inline_comments=(
                len(plan.comments) if plan and status in ("ok", POSTED_POST_MERGE) else None
            ),
        )
        logger.info(
            "Review %s#%s@%s (%s) %s: %d calls, %d in / %d out tokens, $%.4f, %.0fs%s",
            pr.repo, pr.number, pr.head_sha[:8], kind or command, status, totals["calls"],
            totals["prompt_tokens"], totals["completion_tokens"], totals["cost_usd"], duration,
            f", {len(plan.findings)} findings ({len(plan.comments)} inline)" if plan else "",
        )  # fmt: skip
        return ReviewResult(
            review_id, status, totals, duration, error, comment_url,
            fingerprint if status in SETTLED_STATUSES else None,
            llm_timed_out=status == "error" and _timed_out(contexts, error),
            plan=plan, skip_reason=skip_reason,
        )  # fmt: skip
