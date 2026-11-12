"""Pure decision logic: which repos, which PRs, and which kind of review."""

from __future__ import annotations

import hashlib
import logging
import re
from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from typing import Any

from app.github_app import PullRequest

logger = logging.getLogger(__name__)

BOT_AUTHORS = frozenset(
    {
        "dependabot[bot]",
        "renovate[bot]",
        "github-actions[bot]",
        "dependabot",
        "renovate",
        "github-actions",
    }
)
_REPO_RE = re.compile(r"^[A-Za-z0-9](?:[A-Za-z0-9-]{0,38})/[A-Za-z0-9._-]{1,100}$")
_ORG_RE = re.compile(r"^([A-Za-z0-9](?:[A-Za-z0-9-]{0,38}))/\*$")

# Opt-out switches (documented in the README).
OPT_OUT_LABEL = "no-ai-review"
OPT_OUT_TOPIC = "no-ai-review"
SKIP_TITLE_MARKER = "[skip review]"
REPO_CONFIG_PATH = ".github/gh-pr-reviewer.yml"


class Mode(StrEnum):
    FULL = "review"
    INCREMENTAL = "incremental"
    SKIP = "skip"


@dataclass(frozen=True)
class PrState:
    """What we remember about a PR across restarts."""

    last_reviewed_sha: str | None = None
    last_review_mode: str | None = None
    draft_since_review: bool = False
    failed_sha: str | None = None
    failed_attempts: int = 0
    last_failure_at: float | None = None
    failed_trigger: str | None = None
    failed_requested_by: str | None = None
    failed_force: bool = False
    # Fingerprint of the reviewable diff at the last review (see repo_context.diff_fingerprint).
    last_reviewed_diff: str | None = None
    # Org mode: the last head/draft state the poller saw and when (epoch seconds), and
    # whether the PR has been confirmed as post-cutoff work (sticky).
    observed_sha: str | None = None
    observed_draft: bool = False
    observed_at: float | None = None
    org_live: bool = False
    # Settle wait: the head SHA / draft flag the anchor belongs to, when that head became
    # reviewable (epoch seconds), and when the poller last saw the PR at all.
    settle_sha: str | None = None
    settle_draft: bool = False
    head_ready_at: float | None = None
    last_seen_at: float | None = None
    # Highest on-demand trigger comment id already acted on.
    last_trigger_id: int | None = None
    # Automatic review rounds posted on this PR (full / incremental reviews with status ok;
    # on-demand reviews are not counted). Derived from the reviews log, so it survives restarts.
    auto_rounds: int = 0


@dataclass(frozen=True)
class Decision:
    mode: Mode
    reason: str
    # Short machine-readable bucket for skips (draft, bot, backlog, ...), counted on /healthz.
    category: str = ""
    # "on_demand" when a human asked for this review in a PR comment.
    trigger: str = ""
    # Automatic review round this decision starts (1 = first review, 2 = re-review after new
    # commits); 0 for skips and on-demand reviews.
    round: int = 0

    @property
    def needs_full_budget(self) -> bool:
        """First reviews and on-demand reviews pause at the budget's pause fraction;
        re-reviews of PRs already under review continue until the stop fraction."""
        return self.trigger == "on_demand" or self.round <= 1

    @property
    def kind(self) -> str:
        """full | incremental | on_demand (what the stats API reports)."""
        if self.trigger == "on_demand":
            return "on_demand"
        return "incremental" if self.mode is Mode.INCREMENTAL else "full"


def parse_repo_allowlist(raw: str) -> list[str]:
    """Parse ``owner/repo,owner/repo``; invalid entries are dropped with a warning.

    ``owner/*`` org entries are not repos and are left to :func:`parse_review_targets`.
    """
    repos: list[str] = []
    seen: set[str] = set()
    for part in (raw or "").split(","):
        name = part.strip().removeprefix("https://github.com/").strip("/")
        if not name or _ORG_RE.match(name):
            continue
        if not _REPO_RE.match(name):
            logger.warning("Ignoring invalid REVIEW_REPOS entry %r", name)
            continue
        if name.lower() in seen:
            continue
        seen.add(name.lower())
        repos.append(name)
    return repos


@dataclass(frozen=True)
class ReviewTargets:
    """What ``REVIEW_REPOS`` asks for: whole orgs (``owner/*``) and explicit repos."""

    orgs: tuple[str, ...] = ()
    repos: tuple[str, ...] = ()

    @property
    def org_mode(self) -> bool:
        return bool(self.orgs)

    def covers(self, repo: str) -> bool:
        owner = repo.split("/", 1)[0].lower()
        return owner in {o.lower() for o in self.orgs} or is_allowed_repo(repo, list(self.repos))

    def describe(self) -> list[str]:
        return [f"{o}/*" for o in self.orgs] + list(self.repos)


def parse_review_targets(raw: str) -> ReviewTargets:
    orgs: list[str] = []
    for part in (raw or "").split(","):
        name = part.strip().removeprefix("https://github.com/").strip("/")
        match = _ORG_RE.match(name)
        if match and match.group(1).lower() not in {o.lower() for o in orgs}:
            orgs.append(match.group(1))
    return ReviewTargets(tuple(orgs), tuple(parse_repo_allowlist(raw)))


def is_allowed_repo(repo: str, allowlist: list[str]) -> bool:
    return repo.lower() in {r.lower() for r in allowlist}


def is_bot_author(login: str, user_type: str) -> bool:
    login = (login or "").lower()
    return user_type == "Bot" or login in BOT_AUTHORS or login.endswith("[bot]")


def _trigger_re(app_slug: str, suffix: str = "") -> re.Pattern[str]:
    mention = re.escape(app_slug or "gh-pr-reviewer")
    return re.compile(
        rf"^\s*(?:/review|@{mention}\s+review){suffix}(?:\s|$)", re.IGNORECASE | re.MULTILINE
    )


def _request_text(body: str) -> str:
    """A comment without quoted lines and fenced code (where a request does not count)."""
    text = "\n".join(ln for ln in body.splitlines() if not ln.lstrip().startswith(">"))
    return re.sub(r"```.*?```", "", text, flags=re.DOTALL)


def is_force_request(body: str, app_slug: str = "gh-pr-reviewer") -> bool:
    """``/review force`` (or ``@<app> review force``): review again even when the current
    head already has a posted review (the daily cap and budget still apply)."""
    return bool(_trigger_re(app_slug, r"[ \t]+force").search(_request_text(body or "")))


def find_trigger(
    comments: Any,
    *,
    after_id: int | None,
    not_before: float | None,
    app_slug: str = "gh-pr-reviewer",
) -> Any | None:
    """The newest unhandled on-demand review request among a PR's comments: a line that
    starts with ``/review`` or ``@<app> review``, written by a human (not a bot, not quoted),
    newer than the last request acted on and not older than the feature (``not_before``)."""
    pattern = _trigger_re(app_slug)
    best = None
    for c in comments or ():
        if c.id <= (after_id or 0) or is_bot_author(c.author_login, c.author_type):
            continue
        created = parse_time(c.created_at)
        if not_before is not None and (created is None or created < not_before):
            continue
        if pattern.search(_request_text(c.body)) and (best is None or c.id > best.id):
            best = c
    return best


def pr_opt_out_reason(pr: PullRequest) -> str | None:
    """Why the author asked for no review of this PR, if they did."""
    if any(label.lower() == OPT_OUT_LABEL for label in pr.labels):
        return f"label {OPT_OUT_LABEL}"
    if SKIP_TITLE_MARKER in (pr.title or "").lower():
        return f"{SKIP_TITLE_MARKER} in title"
    return None


def repo_opt_out_reason(topics: list[str] | tuple[str, ...], config_text: str | None) -> str | None:
    """Why a repo opted out (topic, or ``enabled: false`` in its reviewer config)."""
    if OPT_OUT_TOPIC in {t.lower() for t in topics or ()}:
        return f"repo topic {OPT_OUT_TOPIC}"
    if config_text and not repo_config_enabled(config_text):
        return f"{REPO_CONFIG_PATH} enabled: false"
    return None


def repo_config_enabled(text: str) -> bool:
    """``False`` only for an explicit ``enabled: false`` (unparsable files keep reviews on)."""
    import yaml

    try:
        data: Any = yaml.safe_load(text)
    except yaml.YAMLError:
        logger.warning("Unparsable %s; reviews stay enabled", REPO_CONFIG_PATH)
        return True
    if not isinstance(data, dict) or "enabled" not in data:
        return True
    value = data["enabled"]
    if isinstance(value, str):
        return value.strip().lower() not in {"false", "no", "off", "0"}
    return value is not False and value != 0


def parse_time(value: str | None) -> float | None:
    """ISO 8601 (GitHub's ``...Z`` included) to epoch seconds; ``None`` when absent/bad."""
    if not value:
        return None
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00")).timestamp()
    except ValueError:
        return None


def post_cutoff_reason(pr: PullRequest, state: PrState | None, cutoff: float) -> str | None:
    """Why this PR counts as work after the org-mode cutoff, or ``None`` (backlog).

    A PR is post-cutoff when it was opened after the cutoff, its head commit was committed or
    force-pushed after it, it was marked ready after it, or the poller itself saw its head SHA
    (or draft flag) change after having observed the previous state at/after the cutoff —
    which catches an old commit pushed late, where commit dates alone would say "backlog".
    """
    state = state or PrState()
    if state.org_live or state.last_reviewed_sha:
        return "already live"
    if (
        state.observed_at is not None
        and state.observed_at >= cutoff
        and state.observed_sha
        and (state.observed_sha != pr.head_sha or (state.observed_draft and not pr.draft))
    ):
        return "changed after the cutoff"
    for label, value in (
        ("opened", pr.created_at),
        ("head committed", pr.head_committed_at),
        ("marked ready", pr.ready_at),
        ("force-pushed", pr.force_pushed_at),
    ):
        when = parse_time(value)
        if when is not None and when >= cutoff:
            return f"{label} after the cutoff"
    return None


def head_event_times(pr: PullRequest) -> list[float]:
    """GitHub's own times for events that make a head reviewable (opened, marked ready,
    force-pushed, head committed); only what the listing carried."""
    values = (pr.created_at, pr.ready_at, pr.force_pushed_at, pr.head_committed_at)
    return [t for t in map(parse_time, values) if t is not None]


def settle_update(
    state: PrState | None, pr: PullRequest, now: float
) -> tuple[str, bool, float] | None:
    """New ``(settle_sha, settle_draft, head_ready_at)`` when the PR's head or draft flag
    changed since the anchor was set, else ``None`` (keep the anchor).

    The anchor is when the current head became reviewable. On first sight of a PR, GitHub's
    latest event time is trusted (a PR opened an hour ago is ready now). When the poller saw
    the head change, the anchor is the latest event after its previous sighting; with no such
    event (a plain push on a REST-listed repo, or an old commit pushed late) it is now, so
    the wait never starts before the push could have happened.
    """
    state = state or PrState()
    if (
        state.settle_sha == pr.head_sha
        and state.settle_draft == pr.draft
        and state.head_ready_at is not None
    ):
        return None
    events = [e for e in head_event_times(pr) if e <= now + 60]  # tolerate clock skew
    if state.settle_sha is None:
        anchor = max(events) if events else now
    else:
        since = state.last_seen_at or 0.0
        recent = [e for e in events if e > since]
        anchor = max(recent) if recent else now
    return pr.head_sha, pr.draft, min(anchor, now)


def retry_delay_seconds(
    repo: str,
    pr_number: int,
    head_sha: str,
    attempt: int,
    *,
    base_seconds: float,
    max_seconds: float = 3600,
) -> float:
    """Stable per-head exponential backoff with jitter, so poll frequency doesn't redraw it."""
    if base_seconds <= 0 or max_seconds <= 0:
        return 0.0
    exponential = min(base_seconds * (2 ** max(attempt - 1, 0)), max_seconds)
    seed = f"{repo.lower()}#{pr_number}@{head_sha}:{attempt}".encode()
    fraction = int.from_bytes(hashlib.sha256(seed).digest()[:8], "big") / (2**64 - 1)
    return min(exponential * (0.8 + 0.4 * fraction), max_seconds)


def decide(
    pr: PullRequest,
    state: PrState | None,
    *,
    now: float,
    max_attempts: int = 4,
    retry_backoff_seconds: float = 60,
    retry_backoff_max_seconds: float = 3600,
    max_rounds: int = 2,
) -> Decision:
    """Decide what to do with an open PR on this poll.

    - drafts, bot-authored PRs and PRs opted out by label / title are skipped;
    - an unchanged head SHA is skipped;
    - a head SHA that keeps failing is retried with backoff, then left alone until it changes;
    - a PR with no automatic review posted yet gets a full review (round 1);
    - a new head after that gets a full re-review of the whole PR (round 2) that posts only
      findings not raised before;
    - after ``max_rounds`` automatic rounds (0 = no limit) the PR is never reviewed
      automatically again (``max_rounds``); an on-demand ``/review`` still works.
    """
    if pr.draft:
        return Decision(Mode.SKIP, "draft", "draft")
    if is_bot_author(pr.author_login, pr.author_type):
        return Decision(Mode.SKIP, f"bot author {pr.author_login}", "bot")
    opt_out = pr_opt_out_reason(pr)
    if opt_out:
        return Decision(Mode.SKIP, f"opted out: {opt_out}", "opted_out")
    if not pr.head_sha:
        return Decision(Mode.SKIP, "no head sha", "no_sha")
    state = state or PrState()
    if state.last_reviewed_sha == pr.head_sha:
        return Decision(Mode.SKIP, "already reviewed at this sha", "reviewed")
    if max_rounds > 0 and state.auto_rounds >= max_rounds:
        return Decision(
            Mode.SKIP,
            f"max rounds: {state.auto_rounds} automatic reviews posted (limit {max_rounds});"
            " /review asks for another",
            "max_rounds",
        )
    if state.failed_sha == pr.head_sha and state.failed_attempts > 0:
        if state.failed_attempts >= max_attempts:
            return Decision(
                Mode.SKIP, f"gave up after {state.failed_attempts} failed attempts", "gave_up"
            )
        delay = retry_delay_seconds(
            pr.repo,
            pr.number,
            pr.head_sha,
            state.failed_attempts,
            base_seconds=retry_backoff_seconds,
            max_seconds=retry_backoff_max_seconds,
        )
        if state.last_failure_at and now - state.last_failure_at < delay:
            return Decision(Mode.SKIP, "retry backoff", "backoff")
    next_round = state.auto_rounds + 1
    if state.auto_rounds == 0:
        return Decision(Mode.FULL, "first review", round=next_round)
    why = "left draft with new commits" if state.draft_since_review else "new commits"
    return Decision(Mode.FULL, f"round {next_round}: full re-review, {why}", round=next_round)
