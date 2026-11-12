"""SQLite persistence: per-PR review state, review runs, and every LLM call."""

from __future__ import annotations

import logging
import os
import sqlite3
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from app.review_policy import PrState

logger = logging.getLogger(__name__)

DB_FILENAME = "gh-pr-reviewer.sqlite3"

SCHEMA = """
CREATE TABLE IF NOT EXISTS pr_state (
    repo TEXT NOT NULL,
    pr_number INTEGER NOT NULL,
    last_reviewed_sha TEXT,
    last_review_mode TEXT,
    last_reviewed_at TEXT,
    draft_since_review INTEGER NOT NULL DEFAULT 0,
    failed_sha TEXT,
    failed_attempts INTEGER NOT NULL DEFAULT 0,
    last_failure_at REAL,
    failed_trigger TEXT,
    failed_requested_by TEXT,
    failed_force INTEGER NOT NULL DEFAULT 0,
    last_reviewed_diff TEXT,
    observed_sha TEXT,
    observed_draft INTEGER NOT NULL DEFAULT 0,
    observed_at REAL,
    org_live INTEGER NOT NULL DEFAULT 0,
    settle_sha TEXT,
    settle_draft INTEGER NOT NULL DEFAULT 0,
    head_ready_at REAL,
    last_seen_at REAL,
    last_trigger_id INTEGER,
    PRIMARY KEY (repo, pr_number)
);
CREATE TABLE IF NOT EXISTS reviews (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    started_at TEXT NOT NULL,
    finished_at TEXT,
    repo TEXT NOT NULL,
    pr_number INTEGER NOT NULL,
    pr_url TEXT,
    pr_title TEXT,
    head_sha TEXT NOT NULL,
    command TEXT NOT NULL,
    status TEXT NOT NULL,
    error TEXT,
    llm_calls INTEGER NOT NULL DEFAULT 0,
    prompt_tokens INTEGER NOT NULL DEFAULT 0,
    cached_tokens INTEGER NOT NULL DEFAULT 0,
    completion_tokens INTEGER NOT NULL DEFAULT 0,
    reasoning_tokens INTEGER NOT NULL DEFAULT 0,
    cost_usd REAL NOT NULL DEFAULT 0,
    litellm_cost_usd REAL,
    duration_s REAL,
    comment_url TEXT,
    kind TEXT,
    skip_reason TEXT,
    ready_at TEXT,
    author TEXT,
    origin TEXT NOT NULL DEFAULT 'reviewer',
    findings_critical INTEGER,
    findings_major INTEGER,
    findings_minor INTEGER,
    findings_suggestions INTEGER,
    inline_comments INTEGER,
    review_round INTEGER,
    on_demand_request_id INTEGER
);
CREATE INDEX IF NOT EXISTS reviews_started ON reviews (started_at);
CREATE TABLE IF NOT EXISTS llm_calls (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    call_id TEXT UNIQUE,
    review_id INTEGER,
    created_at TEXT NOT NULL,
    repo TEXT,
    pr_number INTEGER,
    head_sha TEXT,
    command TEXT,
    model TEXT,
    status TEXT NOT NULL,
    prompt_tokens INTEGER NOT NULL DEFAULT 0,
    cached_tokens INTEGER NOT NULL DEFAULT 0,
    completion_tokens INTEGER NOT NULL DEFAULT 0,
    reasoning_tokens INTEGER NOT NULL DEFAULT 0,
    cost_usd REAL NOT NULL DEFAULT 0,
    litellm_cost_usd REAL,
    duration_s REAL,
    error TEXT
);
CREATE INDEX IF NOT EXISTS llm_calls_created ON llm_calls (created_at);
CREATE INDEX IF NOT EXISTS llm_calls_pr ON llm_calls (repo, pr_number, created_at);
CREATE INDEX IF NOT EXISTS reviews_pr ON reviews (repo, pr_number, head_sha);
CREATE TABLE IF NOT EXISTS meta (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
"""

# Columns added after the first release (table -> name -> DDL type).
_MIGRATIONS = {
    "pr_state": {
        "failed_trigger": "TEXT",
        "failed_requested_by": "TEXT",
        "failed_force": "INTEGER NOT NULL DEFAULT 0",
        "last_reviewed_diff": "TEXT",
        "observed_sha": "TEXT",
        "observed_draft": "INTEGER NOT NULL DEFAULT 0",
        "observed_at": "REAL",
        "org_live": "INTEGER NOT NULL DEFAULT 0",
        "settle_sha": "TEXT",
        "settle_draft": "INTEGER NOT NULL DEFAULT 0",
        "head_ready_at": "REAL",
        "last_seen_at": "REAL",
        "last_trigger_id": "INTEGER",
    },
    "reviews": {
        "kind": "TEXT",
        "skip_reason": "TEXT",
        "ready_at": "TEXT",
        "author": "TEXT",
        "origin": "TEXT NOT NULL DEFAULT 'reviewer'",
        "findings_critical": "INTEGER",
        "findings_major": "INTEGER",
        "findings_minor": "INTEGER",
        "findings_suggestions": "INTEGER",
        "inline_comments": "INTEGER",
        "review_round": "INTEGER",
        "on_demand_request_id": "INTEGER",
    },
}

# Backfill of review rows written before `kind` / `skip_reason` existed (idempotent).
_BACKFILL_SQL = (
    "UPDATE reviews SET kind = CASE command WHEN 'incremental' THEN 'incremental'"
    " ELSE 'full' END WHERE kind IS NULL",
    "UPDATE reviews SET skip_reason = CASE"
    " WHEN error LIKE 'no reviewable files%' THEN 'lockfile_only'"
    " WHEN error LIKE 'reviewable diff unchanged%' THEN 'unchanged_diff'"
    " WHEN error LIKE 'reviewable diff too large%' THEN 'too_large'"
    " ELSE 'other' END WHERE status = 'skipped' AND skip_reason IS NULL",
)

# Skip reasons included in statistics responses.
SKIP_REASONS = (
    "draft",
    "bot",
    "lockfile_only",
    "merged_during_wait",
    "closed",
    "opted_out",
    "daily_cap",
    "too_large",  # historic: releases with a size cap skipped large PRs (no longer happens)
    "unchanged_diff",
    "max_rounds",
    "closed_during_review",  # closed unmerged while the (billed) review ran: nothing posted
    "merged_during_review",  # merged while it ran and nothing was found: nothing posted
    "superseded",  # status `cancelled`: a newer head arrived mid-review (billed, not posted)
    "already_reviewed",  # /review on a head that already has a posted review (no model call)
)
# Reviews stopped between model calls (``review_liveness``) are status ``cancelled`` with
# skip_reason merged_during_review / closed_during_review / superseded; the stats API counts
# and lists them as skips with that reason (billed, nothing posted).
# Releases with a size cap settled a too-large head as if reviewed (and later heads with the
# same diff as unchanged). Once, on the first start without the cap, every PR not reviewed
# since such a skip is released: back to its last posted review's head (or none), no diff
# fingerprint, live in org mode, so the next poll reviews its current head.
RELEASE_TOO_LARGE_KEY = "released_too_large"
_RELEASE_TOO_LARGE_SQL = """
UPDATE pr_state SET
    last_reviewed_sha = (
        SELECT r.head_sha FROM reviews r WHERE r.repo = pr_state.repo
        AND r.pr_number = pr_state.pr_number AND r.status IN ('ok', 'posted_post_merge')
        ORDER BY r.id DESC LIMIT 1),
    last_reviewed_diff = NULL,
    org_live = 1
WHERE EXISTS (
    SELECT 1 FROM reviews t WHERE t.repo = pr_state.repo AND t.pr_number = pr_state.pr_number
    AND t.status = 'skipped' AND t.skip_reason = 'too_large'
    AND NOT EXISTS (
        SELECT 1 FROM reviews r WHERE r.repo = t.repo AND r.pr_number = t.pr_number
        AND r.id > t.id AND r.status IN ('ok', 'posted_post_merge')))
"""
# Automatic review rounds of a PR: posted reviews that were not asked for in a comment.
_AUTO_ROUNDS_SQL = (
    "SELECT COUNT(*) FROM reviews WHERE repo = ? AND pr_number = ? AND status = 'ok'"
    " AND origin = 'reviewer' AND COALESCE(kind, 'full') IN ('full', 'incremental')"
)
FINDING_SEVERITIES = ("critical", "major", "minor", "suggestions")

_ROLLUP_SQL = (
    "SELECT COUNT(*) AS n, COALESCE(SUM(prompt_tokens),0) AS p,"
    " COALESCE(SUM(cached_tokens),0) AS c, COALESCE(SUM(completion_tokens),0) AS o,"
    " COALESCE(SUM(reasoning_tokens),0) AS r, COALESCE(SUM(cost_usd),0) AS cost,"
    " SUM(litellm_cost_usd) AS lcost FROM llm_calls WHERE review_id = ?"
)


def utcnow() -> datetime:
    return datetime.now(UTC)


def iso(dt: datetime) -> str:
    return dt.astimezone(UTC).isoformat(timespec="seconds")


def iso_us(dt: datetime) -> str:
    """Microsecond ISO time: review rows sort and page by it (many skips share a second).
    Mixed with ``iso`` values it still compares correctly as text (``+`` < ``.``)."""
    return dt.astimezone(UTC).isoformat(timespec="microseconds")


def epoch_iso(epoch: float | None) -> str | None:
    return None if epoch is None else iso(datetime.fromtimestamp(epoch, tz=UTC))


def resolve_data_dir(configured: str) -> Path:
    """Create the data directory, falling back to ``~/.local/share`` if needed."""
    candidates = [Path(configured).expanduser()]
    candidates.append(Path.home() / ".local" / "share" / "gh-pr-reviewer")
    for path in candidates:
        try:
            path.mkdir(parents=True, exist_ok=True)
            if os.access(path, os.W_OK):
                if path != candidates[0]:
                    logger.warning("DATA_DIR %s is not writable; using %s", configured, path)
                return path
        except OSError as exc:
            logger.warning("Cannot use data dir %s: %s", path, exc)
    raise RuntimeError(f"No writable data directory (tried {', '.join(map(str, candidates))})")


@dataclass(frozen=True)
class LlmCallRecord:
    call_id: str | None
    review_id: int | None
    repo: str | None
    pr_number: int | None
    head_sha: str | None
    command: str | None
    model: str | None
    status: str
    prompt_tokens: int
    cached_tokens: int
    completion_tokens: int
    reasoning_tokens: int
    cost_usd: float
    litellm_cost_usd: float | None
    duration_s: float | None
    error: str | None = None
    created_at: str | None = None


class Store:
    def __init__(self, path: Path) -> None:
        self.path = path
        self._write_lock = threading.Lock()
        with self._conn() as conn:
            conn.execute("PRAGMA journal_mode=WAL")
            for table, columns in _MIGRATIONS.items():
                cols = {r["name"] for r in conn.execute(f"PRAGMA table_info({table})")}
                for name, ddl in columns.items():
                    if cols and name not in cols:  # databases created before the column
                        conn.execute(f"ALTER TABLE {table} ADD COLUMN {name} {ddl}")
            conn.executescript(SCHEMA)
            for sql in _BACKFILL_SQL:
                conn.execute(sql)
            done = "SELECT 1 FROM meta WHERE key = ?"
            if conn.execute(done, (RELEASE_TOO_LARGE_KEY,)).fetchone() is None:
                released = conn.execute(_RELEASE_TOO_LARGE_SQL).rowcount
                conn.execute(
                    "INSERT INTO meta (key, value) VALUES (?, ?)",
                    (RELEASE_TOO_LARGE_KEY, iso(utcnow())),
                )
                logger.info("Released %d PR(s) skipped as too large to review", released)

    @classmethod
    def open(cls, data_dir: str) -> Store:
        return cls(resolve_data_dir(data_dir) / DB_FILENAME)

    def query(self, sql: str, params: tuple | list = ()) -> list[dict[str, Any]]:
        """Read-only helper for reporting queries (``stats.py``)."""
        with self._conn() as conn:
            return [dict(r) for r in conn.execute(sql, tuple(params)).fetchall()]

    @contextmanager
    def _conn(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self.path, timeout=30)
        conn.row_factory = sqlite3.Row
        try:
            with conn:
                yield conn
        finally:
            conn.close()

    # --- PR state ------------------------------------------------------------------
    def get_pr_state(self, repo: str, pr_number: int) -> PrState | None:
        with self._conn() as conn:
            row = conn.execute(
                "SELECT * FROM pr_state WHERE repo = ? AND pr_number = ?", (repo, pr_number)
            ).fetchone()
            rounds = conn.execute(_AUTO_ROUNDS_SQL, (repo, pr_number)).fetchone()[0]
        if row is None:
            return PrState(auto_rounds=int(rounds)) if rounds else None
        return PrState(
            last_reviewed_sha=row["last_reviewed_sha"],
            last_review_mode=row["last_review_mode"],
            draft_since_review=bool(row["draft_since_review"]),
            failed_sha=row["failed_sha"],
            failed_attempts=row["failed_attempts"],
            last_failure_at=row["last_failure_at"],
            failed_trigger=row["failed_trigger"],
            failed_requested_by=row["failed_requested_by"],
            failed_force=bool(row["failed_force"]),
            last_reviewed_diff=row["last_reviewed_diff"],
            observed_sha=row["observed_sha"],
            observed_draft=bool(row["observed_draft"]),
            observed_at=row["observed_at"],
            org_live=bool(row["org_live"]),
            settle_sha=row["settle_sha"],
            settle_draft=bool(row["settle_draft"]),
            head_ready_at=row["head_ready_at"],
            last_seen_at=row["last_seen_at"],
            last_trigger_id=row["last_trigger_id"],
            auto_rounds=int(rounds or 0),
        )

    def _upsert_state(self, repo: str, pr_number: int, **fields: Any) -> None:
        cols = ", ".join(fields)
        placeholders = ", ".join("?" for _ in fields)
        updates = ", ".join(f"{k} = excluded.{k}" for k in fields)
        sql = (
            f"INSERT INTO pr_state (repo, pr_number, {cols}) VALUES (?, ?, {placeholders}) "
            f"ON CONFLICT (repo, pr_number) DO UPDATE SET {updates}"
        )
        with self._write_lock, self._conn() as conn:
            conn.execute(sql, (repo, pr_number, *fields.values()))

    def mark_reviewed(
        self, repo: str, pr_number: int, sha: str, mode: str, diff_fingerprint: str | None = None
    ) -> None:
        self._upsert_state(
            repo,
            pr_number,
            last_reviewed_sha=sha,
            last_review_mode=mode,
            last_reviewed_diff=diff_fingerprint,
            last_reviewed_at=iso(utcnow()),
            draft_since_review=0,
            failed_sha=None,
            failed_attempts=0,
            last_failure_at=None,
            failed_trigger=None,
            failed_requested_by=None,
            failed_force=0,
        )

    def mark_failed(
        self,
        repo: str,
        pr_number: int,
        sha: str,
        now: float,
        *,
        trigger: str | None = None,
        requested_by: str | None = None,
        force: bool = False,
    ) -> None:
        """Record one failed run and the on-demand intent needed to retry it after a restart."""
        state = self.get_pr_state(repo, pr_number) or PrState()
        attempts = state.failed_attempts + 1 if state.failed_sha == sha else 1
        self._upsert_state(
            repo,
            pr_number,
            failed_sha=sha,
            failed_attempts=attempts,
            last_failure_at=now,
            failed_trigger=trigger,
            failed_requested_by=requested_by,
            failed_force=int(force),
        )

    def set_draft_since_review(self, repo: str, pr_number: int, value: bool) -> None:
        self._upsert_state(repo, pr_number, draft_since_review=int(value))

    def observe(
        self,
        repo: str,
        pr_number: int,
        sha: str,
        draft: bool,
        now: float,
        *,
        cutoff: float,
        state: PrState | None = None,
    ) -> None:
        """Remember the head/draft state the poller saw. ``observed_at`` keeps the first time
        this state was seen after the cutoff, so a later change proves a post-cutoff push."""
        if (
            state is not None
            and state.observed_sha == sha
            and state.observed_draft == draft
            and state.observed_at is not None
            and state.observed_at >= cutoff
        ):
            return
        self._upsert_state(
            repo, pr_number, observed_sha=sha, observed_draft=int(draft), observed_at=now
        )

    def set_org_live(self, repo: str, pr_number: int) -> None:
        self._upsert_state(repo, pr_number, org_live=1)

    def note_seen(
        self,
        repo: str,
        pr_number: int,
        now: float,
        settle: tuple[str, bool, float] | None,
    ) -> None:
        """Record a sighting (``last_seen_at``) and, when the head changed, its new settle
        anchor (see ``review_policy.settle_update``)."""
        fields: dict[str, Any] = {"last_seen_at": now}
        if settle is not None:
            sha, draft, ready_at = settle
            fields |= {"settle_sha": sha, "settle_draft": int(draft), "head_ready_at": ready_at}
        self._upsert_state(repo, pr_number, **fields)

    def set_last_trigger(self, repo: str, pr_number: int, comment_id: int) -> None:
        self._upsert_state(repo, pr_number, last_trigger_id=comment_id)

    def has_org_history(self, org: str) -> bool:
        """Has org mode already tracked PRs of this org (cutoff observations or live PRs)?"""
        with self._conn() as conn:
            row = conn.execute(
                "SELECT 1 FROM pr_state WHERE lower(repo) LIKE ? AND"
                " (org_live = 1 OR observed_at IS NOT NULL) LIMIT 1",
                (org.lower() + "/%",),
            ).fetchone()
        return row is not None

    # --- meta ----------------------------------------------------------------------
    def get_meta(self, key: str) -> str | None:
        with self._conn() as conn:
            row = conn.execute("SELECT value FROM meta WHERE key = ?", (key,)).fetchone()
        return row["value"] if row else None

    def set_meta(self, key: str, value: str) -> None:
        with self._write_lock, self._conn() as conn:
            conn.execute(
                "INSERT INTO meta (key, value) VALUES (?, ?) "
                "ON CONFLICT (key) DO UPDATE SET value = excluded.value",
                (key, value),
            )

    def meta_setdefault(self, key: str, value: str) -> str:
        """Store ``value`` unless the key exists; return the stored value."""
        with self._write_lock, self._conn() as conn:
            conn.execute("INSERT OR IGNORE INTO meta (key, value) VALUES (?, ?)", (key, value))
            return conn.execute("SELECT value FROM meta WHERE key = ?", (key,)).fetchone()[0]

    # --- reviews -------------------------------------------------------------------
    def start_review(
        self,
        repo: str,
        pr_number: int,
        pr_url: str,
        title: str,
        sha: str,
        command: str,
        *,
        kind: str | None = None,
        ready_at: float | None = None,
        author: str | None = None,
        review_round: int | None = None,
        on_demand_request_id: int | None = None,
    ) -> int:
        kind = kind or ("incremental" if command == "incremental" else "full")
        with self._write_lock, self._conn() as conn:
            cur = conn.execute(
                "INSERT INTO reviews (started_at, repo, pr_number, pr_url, pr_title, head_sha,"
                " command, status, kind, ready_at, author, review_round, on_demand_request_id)"
                " VALUES (?, ?, ?, ?, ?, ?, ?, 'running', ?, ?, ?, ?, ?)",
                (
                    iso_us(utcnow()),
                    repo,
                    pr_number,
                    pr_url,
                    title,
                    sha,
                    command,
                    kind,
                    epoch_iso(ready_at),
                    author,
                    review_round,
                    on_demand_request_id,
                ),
            )
            return int(cur.lastrowid)

    def has_skip(self, repo: str, pr_number: int, reason: str) -> bool:
        """Was a review of this PR ever skipped for ``reason``?"""
        with self._conn() as conn:
            row = conn.execute(
                "SELECT 1 FROM reviews WHERE repo = ? AND pr_number = ? AND status = 'skipped'"
                " AND skip_reason = ? LIMIT 1",
                (repo, pr_number, reason),
            ).fetchone()
        return row is not None

    def auto_rounds(self, repo: str, pr_number: int) -> int:
        """Automatic review rounds posted on this PR (see ``PrState.auto_rounds``)."""
        with self._conn() as conn:
            return int(conn.execute(_AUTO_ROUNDS_SQL, (repo, pr_number)).fetchone()[0] or 0)

    def record_skip(
        self,
        repo: str,
        pr_number: int,
        sha: str,
        reason: str,
        *,
        pr_url: str = "",
        title: str = "",
        author: str | None = None,
        kind: str = "full",
        ready_at: float | None = None,
        detail: str | None = None,
    ) -> int | None:
        """Record a review the poller decided not to run (draft, bot, merged while waiting,
        ...) once per PR head and reason, so the stats API can count it. Returns the new row
        id, or ``None`` when that skip was already recorded."""
        now = iso_us(utcnow())
        with self._write_lock, self._conn() as conn:
            exists = conn.execute(
                "SELECT 1 FROM reviews WHERE repo = ? AND pr_number = ? AND head_sha = ?"
                " AND status = 'skipped' AND skip_reason = ? LIMIT 1",
                (repo, pr_number, sha, reason),
            ).fetchone()
            if exists:
                return None
            cur = conn.execute(
                "INSERT INTO reviews (started_at, finished_at, repo, pr_number, pr_url, pr_title,"
                " head_sha, command, status, error, kind, skip_reason, ready_at, author, origin)"
                " VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'skipped', ?, ?, ?, ?, ?, 'poller')",
                (
                    now,
                    now,
                    repo,
                    pr_number,
                    pr_url,
                    title,
                    sha,
                    "incremental" if kind == "incremental" else "review",
                    detail or reason,
                    kind,
                    reason,
                    epoch_iso(ready_at),
                    author,
                ),
            )
            return int(cur.lastrowid)

    def finish_review(
        self,
        review_id: int,
        *,
        status: str,
        duration_s: float,
        error: str | None = None,
        comment_url: str | None = None,
        skip_reason: str | None = None,
        findings: dict[str, int] | None = None,
        inline_comments: int | None = None,
    ) -> None:
        """Close a review run, rolling up its recorded LLM calls."""
        findings = findings or {}
        with self._write_lock, self._conn() as conn:
            agg = conn.execute(_ROLLUP_SQL, (review_id,)).fetchone()
            conn.execute(
                "UPDATE reviews SET finished_at = ?, status = ?, error = ?, duration_s = ?,"
                " comment_url = COALESCE(?, comment_url), skip_reason = COALESCE(?, skip_reason),"
                " findings_critical = COALESCE(?, findings_critical),"
                " findings_major = COALESCE(?, findings_major),"
                " findings_minor = COALESCE(?, findings_minor),"
                " findings_suggestions = COALESCE(?, findings_suggestions),"
                " inline_comments = COALESCE(?, inline_comments), llm_calls = ?,"
                " prompt_tokens = ?, cached_tokens = ?, completion_tokens = ?,"
                " reasoning_tokens = ?, cost_usd = ?, litellm_cost_usd = ? WHERE id = ?",
                (
                    iso_us(utcnow()),
                    status,
                    error,
                    duration_s,
                    comment_url,
                    skip_reason,
                    *(findings.get(k) for k in FINDING_SEVERITIES),
                    inline_comments,
                    agg["n"],
                    agg["p"],
                    agg["c"],
                    agg["o"],
                    agg["r"],
                    agg["cost"],
                    agg["lcost"],
                    review_id,
                ),
            )

    def set_comment_url(self, review_id: int, url: str) -> None:
        with self._write_lock, self._conn() as conn:
            conn.execute("UPDATE reviews SET comment_url = ? WHERE id = ?", (url, review_id))

    def interrupt_running(self) -> int:
        """Mark runs left 'running' by a previous process as interrupted, rolling up the LLM
        calls they already paid for (so recovery and the per-PR cap can see them)."""
        with self._write_lock, self._conn() as conn:
            ids = [r["id"] for r in conn.execute("SELECT id FROM reviews WHERE status = 'running'")]
            for review_id in ids:
                agg = conn.execute(_ROLLUP_SQL, (review_id,)).fetchone()
                conn.execute(
                    "UPDATE reviews SET status = 'interrupted', finished_at = ?, llm_calls = ?,"
                    " prompt_tokens = ?, cached_tokens = ?, completion_tokens = ?,"
                    " reasoning_tokens = ?, cost_usd = ?, litellm_cost_usd = ? WHERE id = ?",
                    (
                        iso(utcnow()),
                        agg["n"],
                        agg["p"],
                        agg["c"],
                        agg["o"],
                        agg["r"],
                        agg["cost"],
                        agg["lcost"],
                        review_id,
                    ),
                )
            return len(ids)

    def interrupted_billed_review(
        self, repo: str, pr_number: int, sha: str
    ) -> dict[str, Any] | None:
        """The latest review of this exact head SHA that a restart cut off after it had
        already paid for LLM calls (its comment may have been posted)."""
        with self._conn() as conn:
            row = conn.execute(
                "SELECT * FROM reviews WHERE repo = ? AND pr_number = ? AND head_sha = ?"
                " AND status = 'interrupted' AND llm_calls > 0 ORDER BY id DESC LIMIT 1",
                (repo, pr_number, sha),
            ).fetchone()
        return dict(row) if row else None

    def mark_recovered(self, review_id: int, comment_url: str | None) -> None:
        with self._write_lock, self._conn() as conn:
            conn.execute(
                "UPDATE reviews SET status = 'ok', error = 'recovered after restart',"
                " comment_url = COALESCE(?, comment_url) WHERE id = ?",
                (comment_url, review_id),
            )

    def posted_review_for_head(self, repo: str, pr_number: int, sha: str) -> dict[str, Any] | None:
        """The latest review of this exact head SHA that was posted (a review, or the findings
        comment of a PR merged mid-review), of any kind."""
        with self._conn() as conn:
            row = conn.execute(
                "SELECT * FROM reviews WHERE repo = ? AND pr_number = ? AND head_sha = ?"
                " AND status IN ('ok', 'posted_post_merge') AND origin = 'reviewer'"
                " ORDER BY id DESC LIMIT 1",
                (repo, pr_number, sha),
            ).fetchone()
        return dict(row) if row else None

    def billed_reviews_since(
        self, repo: str, pr_number: int, since: datetime, *, kind: str | None = None
    ) -> int:
        """Billed review requests of this PR since, optionally filtered by ``kind``.

        On-demand retries share one request ID and therefore consume one daily-cap slot; older
        rows without a request ID remain counted as individual runs. ``cancelled`` runs do not
        count because the daily cap is for reviews that could post.
        """
        sql = (
            "SELECT COUNT(DISTINCT CASE WHEN r.on_demand_request_id IS NULL"
            " THEN 'review:' || c.review_id"
            " ELSE 'request:' || r.on_demand_request_id END)"
            " FROM llm_calls c"
            " LEFT JOIN reviews r ON r.id = c.review_id WHERE c.repo = ?"
            " AND c.pr_number = ? AND c.created_at >= ? AND c.status = 'success'"
            " AND c.review_id IS NOT NULL AND COALESCE(r.status, '') != 'cancelled'"
        )
        params: list[Any] = [repo, pr_number, iso(since)]
        if kind is not None:
            sql += " AND r.kind = ?"
            params.append(kind)
        with self._conn() as conn:
            row = conn.execute(sql, params).fetchone()
        return int(row[0] or 0)

    def get_review(self, review_id: int) -> dict[str, Any] | None:
        with self._conn() as conn:
            row = conn.execute("SELECT * FROM reviews WHERE id = ?", (review_id,)).fetchone()
        return dict(row) if row else None

    def recent_reviews(self, limit: int = 100, *, include_poller: bool = False) -> list[dict]:
        """Newest review runs; poller-level skips (drafts, bots, ...) only on request."""
        where = "" if include_poller else "WHERE origin = 'reviewer'"
        with self._conn() as conn:
            rows = conn.execute(
                f"SELECT * FROM reviews {where} ORDER BY id DESC LIMIT ?", (limit,)
            ).fetchall()
        return [dict(r) for r in rows]

    # --- LLM calls -----------------------------------------------------------------
    def insert_llm_call(self, rec: LlmCallRecord) -> bool:
        """Insert one call; a duplicate ``call_id`` is ignored. Returns True when inserted."""
        with self._write_lock, self._conn() as conn:
            cur = conn.execute(
                "INSERT OR IGNORE INTO llm_calls (call_id, review_id, created_at, repo,"
                " pr_number, head_sha, command, model, status, prompt_tokens, cached_tokens,"
                " completion_tokens, reasoning_tokens, cost_usd, litellm_cost_usd, duration_s,"
                " error) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    rec.call_id,
                    rec.review_id,
                    rec.created_at or iso(utcnow()),
                    rec.repo,
                    rec.pr_number,
                    rec.head_sha,
                    rec.command,
                    rec.model,
                    rec.status,
                    rec.prompt_tokens,
                    rec.cached_tokens,
                    rec.completion_tokens,
                    rec.reasoning_tokens,
                    rec.cost_usd,
                    rec.litellm_cost_usd,
                    rec.duration_s,
                    rec.error,
                ),
            )
            return cur.rowcount == 1

    def llm_calls_for_review(self, review_id: int) -> list[dict[str, Any]]:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT * FROM llm_calls WHERE review_id = ? ORDER BY id", (review_id,)
            ).fetchall()
        return [dict(r) for r in rows]

    def cost_since(self, since: datetime) -> dict[str, float]:
        with self._conn() as conn:
            row = conn.execute(
                "SELECT COUNT(*) AS calls, COALESCE(SUM(cost_usd),0) AS cost,"
                " COALESCE(SUM(prompt_tokens),0) AS prompt_tokens,"
                " COALESCE(SUM(completion_tokens),0) AS completion_tokens,"
                " SUM(litellm_cost_usd) AS litellm_cost FROM llm_calls WHERE created_at >= ?",
                (iso(since),),
            ).fetchone()
        return dict(row)

    def cost_by_repo(self, since: datetime | None = None) -> list[dict[str, Any]]:
        where, params = ("WHERE created_at >= ?", (iso(since),)) if since else ("", ())
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT COALESCE(repo, '(unattributed)') AS repo, COUNT(*) AS calls,"
                " COUNT(DISTINCT review_id) AS reviews, COALESCE(SUM(cost_usd),0) AS cost,"
                " COALESCE(SUM(prompt_tokens),0) AS prompt_tokens,"
                " COALESCE(SUM(completion_tokens),0) AS completion_tokens,"
                f" COALESCE(SUM(reasoning_tokens),0) AS reasoning_tokens FROM llm_calls {where}"
                " GROUP BY 1 ORDER BY cost DESC",
                params,
            ).fetchall()
        return [dict(r) for r in rows]

    def daily_by_repo(self, days: int = 14, now: datetime | None = None) -> list[dict[str, Any]]:
        """Per UTC day and repo: billed reviews, LLM calls, tokens and cost (newest first)."""
        now = (now or utcnow()).astimezone(UTC)
        start = now.replace(hour=0, minute=0, second=0, microsecond=0) - timedelta(days=days - 1)
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT substr(created_at, 1, 10) AS day, COALESCE(repo, '(unattributed)')"
                " AS repo, COUNT(DISTINCT review_id) AS reviews, COUNT(*) AS calls,"
                " COALESCE(SUM(prompt_tokens),0) AS prompt_tokens,"
                " COALESCE(SUM(completion_tokens),0) AS completion_tokens,"
                " ROUND(COALESCE(SUM(cost_usd),0), 6) AS cost FROM llm_calls"
                " WHERE created_at >= ? GROUP BY 1, 2 ORDER BY 1 DESC, cost DESC",
                (iso(start),),
            ).fetchall()
        return [dict(r) for r in rows]

    def review_counts_since(self, since: datetime) -> dict[str, int]:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT status, COUNT(*) AS n FROM reviews WHERE started_at >= ? GROUP BY 1",
                (iso(since),),
            ).fetchall()
        return {r["status"]: r["n"] for r in rows}

    def cost_summary(self, now: datetime | None = None) -> dict[str, Any]:
        from app.costs import project_month

        now = (now or utcnow()).astimezone(UTC)
        day_start = now.replace(hour=0, minute=0, second=0, microsecond=0)
        month_start = day_start.replace(day=1)
        today = self.cost_since(day_start)
        month = self.cost_since(month_start)
        last_30 = self.cost_since(now - timedelta(days=30))
        return {
            "as_of": iso(now),
            "today_usd": round(today["cost"], 6),
            "month_to_date_usd": round(month["cost"], 6),
            "projected_month_usd": round(project_month(month["cost"], now), 6),
            "last_30_days_usd": round(last_30["cost"], 6),
            "month_litellm_reported_usd": month["litellm_cost"],
            "month_calls": month["calls"],
            "by_repo_month": self.cost_by_repo(month_start),
            "by_repo_all_time": self.cost_by_repo(None),
        }
