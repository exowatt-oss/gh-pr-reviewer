"""Statistics API payloads (``/api/v1/stats/*``) for usage reporting."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any

from app.costs import MODEL_LABEL
from app.store import FINDING_SEVERITIES, SKIP_REASONS, Store, iso, utcnow

_GUARD = {"ok": "ok", "paused_full": "paused_full", "stopped": "paused_all"}
_STATUS = {
    "ok": "posted",
    "posted_post_merge": "posted_post_merge",  # merged mid-review: one findings comment
    "skipped": "skipped",
    # Stopped between model calls (merged / closed / superseded): billed, nothing posted;
    # reported like the other billed skips, with its reason.
    "cancelled": "skipped",
    "error": "failed",
    "interrupted": "failed",
}


def _parse(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00")).astimezone(UTC)
    except ValueError:
        return None


def normalize_before(value: str | None) -> str | None:
    """A client's ``before`` cursor as the stored text form (``...+00:00``)."""
    parsed = _parse(value)
    return parsed.isoformat() if parsed else None


def _seconds(start: str | None, end: str | None) -> float | None:
    a, b = _parse(start), _parse(end)
    if a is None or b is None:
        return None
    return round(max((b - a).total_seconds(), 0.0), 3)


def window_start(days: int, now: datetime) -> datetime:
    """Start of the window: midnight UTC of the first of the last ``days`` UTC days."""
    today = now.astimezone(UTC).replace(hour=0, minute=0, second=0, microsecond=0)
    return today - timedelta(days=days - 1)


def summary(
    store: Store,
    *,
    days: int,
    poller: dict[str, Any],
    version: str,
    effort: str,
    now: datetime | None = None,
) -> dict[str, Any]:
    now = (now or utcnow()).astimezone(UTC)
    start = window_start(days, now)
    since = iso(start)

    calls = store.query(
        "SELECT COUNT(DISTINCT CASE WHEN status = 'success' THEN review_id END) AS billed,"
        " COALESCE(SUM(cost_usd), 0) AS cost, COALESCE(SUM(prompt_tokens), 0) AS tin,"
        " COALESCE(SUM(completion_tokens), 0) AS tout, COALESCE(SUM(cached_tokens), 0) AS tc"
        " FROM llm_calls WHERE created_at >= ?",
        (since,),
    )[0]
    posted = store.query(
        "SELECT COUNT(*) AS n, COALESCE(SUM(inline_comments), 0) AS inline FROM reviews"
        " WHERE status = 'ok' AND started_at >= ?",
        (since,),
    )[0]
    post_merge = store.query(
        "SELECT COUNT(*) AS n FROM reviews WHERE status = 'posted_post_merge' AND started_at >= ?",
        (since,),
    )[0]
    by_round = store.query(
        "SELECT SUM(CASE WHEN review_round = 1 THEN 1 ELSE 0 END) AS r1,"
        " SUM(CASE WHEN review_round = 2 THEN 1 ELSE 0 END) AS r2,"
        " SUM(CASE WHEN review_round > 2 THEN 1 ELSE 0 END) AS more,"
        " SUM(CASE WHEN kind = 'on_demand' THEN 1 ELSE 0 END) AS od"
        " FROM reviews WHERE status = 'ok' AND started_at >= ?",
        (since,),
    )[0]
    skipped = {reason: 0 for reason in SKIP_REASONS}
    for row in store.query(
        "SELECT skip_reason, COUNT(*) AS n FROM reviews WHERE status IN ('skipped', 'cancelled')"
        " AND started_at >= ? GROUP BY 1",
        (since,),
    ):
        if row["skip_reason"] in skipped:
            skipped[row["skip_reason"]] = row["n"]

    daily_reviews = {
        r["day"]: r
        for r in store.query(
            "SELECT substr(started_at, 1, 10) AS day,"
            " SUM(CASE WHEN status = 'ok' THEN 1 ELSE 0 END) AS reviews,"
            " SUM(CASE WHEN status IN ('skipped', 'cancelled') THEN 1 ELSE 0 END) AS skipped"
            " FROM reviews WHERE started_at >= ? GROUP BY 1",
            (since,),
        )
    }
    daily_cost = {
        r["day"]: r["cost"]
        for r in store.query(
            "SELECT substr(created_at, 1, 10) AS day, COALESCE(SUM(cost_usd), 0) AS cost"
            " FROM llm_calls WHERE created_at >= ? GROUP BY 1",
            (since,),
        )
    }
    daily = []
    for offset in range(days):
        day = (start + timedelta(days=offset)).strftime("%Y-%m-%d")
        rev = daily_reviews.get(day) or {}
        daily.append(
            {
                "date": day,
                "reviews": int(rev.get("reviews") or 0),
                "cost_usd": round(float(daily_cost.get(day) or 0.0), 6),
                "skipped": int(rev.get("skipped") or 0),
            }
        )

    repo_cost = {
        r["repo"]: r["cost"]
        for r in store.query(
            "SELECT repo, COALESCE(SUM(cost_usd), 0) AS cost FROM llm_calls"
            " WHERE created_at >= ? AND repo IS NOT NULL GROUP BY 1",
            (since,),
        )
    }
    repo_reviews = {
        r["repo"]: r
        for r in store.query(
            "SELECT repo, COUNT(*) AS n, MAX(COALESCE(finished_at, started_at)) AS last"
            " FROM reviews WHERE status = 'ok' AND started_at >= ? GROUP BY 1",
            (since,),
        )
    }
    by_repo = [
        {
            "repo": repo,
            "reviews": int((repo_reviews.get(repo) or {}).get("n") or 0),
            "cost_usd": round(float(repo_cost.get(repo) or 0.0), 6),
            "last_review_at": (repo_reviews.get(repo) or {}).get("last"),
        }
        for repo in sorted(set(repo_cost) | set(repo_reviews))
    ]
    by_repo.sort(key=lambda r: (-r["cost_usd"], -r["reviews"], r["repo"]))

    budget = poller.get("budget") or {}
    return {
        "generated_at": iso(now),
        "version": version,
        "model": MODEL_LABEL,
        "effort": effort,
        "orgs": [
            {
                "org": o.get("org"),
                "installation_id": o.get("installation_id"),
                "repos_covered": o.get("repos_covered"),
                "cutoff": o.get("cutoff"),
            }
            for o in poller.get("orgs") or []
        ],
        "poller": {
            "state": poller.get("state"),
            "last_poll_at": poller.get("last_poll_at"),
            "last_error": poller.get("last_error"),
            "last_error_at": poller.get("last_error_at"),
            "running": [
                {k: j.get(k) for k in ("repo", "pr", "kind", "started_at")}
                for j in poller.get("running_jobs") or []
            ],
            "queued": [
                {k: j.get(k) for k in ("repo", "pr", "ready_at")}
                for j in poller.get("queued_jobs") or []
            ],
        },
        "budget": {
            "monthly_cap_usd": budget.get("max_budget_usd"),
            "spend_usd": budget.get("spend_usd"),
            "guard": _GUARD.get(str(budget.get("state") or "ok"), "ok"),
        },
        "totals": {
            "reviews_posted": int(posted["n"] or 0),
            "billed_reviews": int(calls["billed"] or 0),
            "cost_usd": round(float(calls["cost"] or 0.0), 6),
            "tokens_in": int(calls["tin"] or 0),
            "tokens_out": int(calls["tout"] or 0),
            "tokens_cached": int(calls["tc"] or 0),
            "inline_comments": int(posted["inline"] or 0),
            # Reviews that finished after the PR merged: one findings comment, not a review.
            "posted_post_merge": int(post_merge["n"] or 0),
            "skipped": skipped,
            # Posted reviews by automatic round (rows before rounds existed are in neither)
            # and on-demand reviews (`/review`), which are not rounds.
            "reviews_by_round": {
                "round_1": int(by_round["r1"] or 0),
                "round_2": int(by_round["r2"] or 0),
                "later": int(by_round["more"] or 0),
                "on_demand": int(by_round["od"] or 0),
            },
        },
        "daily": daily,
        "by_repo": by_repo,
    }


def _review_row(r: dict[str, Any]) -> dict[str, Any]:
    severities = [r.get(f"findings_{k}") for k in FINDING_SEVERITIES]
    findings = (
        None
        if all(v is None for v in severities)
        else {k: int(v or 0) for k, v in zip(FINDING_SEVERITIES, severities, strict=True)}
    )
    kind = r.get("kind") or ("incremental" if r.get("command") == "incremental" else "full")
    return {
        "id": r["id"],
        "repo": r["repo"],
        "pr": r["pr_number"],
        "pr_title": r.get("pr_title") or None,
        "pr_url": r.get("pr_url") or None,
        "author": r.get("author"),
        "kind": kind,
        "round": r.get("review_round"),
        "head_sha": r["head_sha"],
        "ready_at": r.get("ready_at"),
        "started_at": r.get("started_at"),
        "finished_at": r.get("finished_at"),
        "wait_s": _seconds(r.get("ready_at"), r.get("started_at")),
        "latency_s": _seconds(r.get("started_at"), r.get("finished_at")),
        "cost_usd": round(float(r.get("cost_usd") or 0.0), 6),
        "tokens_in": int(r.get("prompt_tokens") or 0),
        "tokens_out": int(r.get("completion_tokens") or 0),
        "tokens_cached": int(r.get("cached_tokens") or 0),
        "findings": findings,
        "status": _STATUS.get(r["status"], "failed"),
        "skip_reason": r.get("skip_reason") if r["status"] in ("skipped", "cancelled") else None,
        "review_url": r.get("comment_url") or None,
    }


def reviews(
    store: Store, *, limit: int, before: str | None = None, repo: str | None = None
) -> dict[str, Any]:
    """Newest first; ``next_before`` pages on to older rows (``started_at`` < it)."""
    where = ["status != 'running'"]
    params: list[Any] = []
    cursor = normalize_before(before)
    if cursor:
        where.append("started_at < ?")
        params.append(cursor)
    if repo:
        where.append("lower(repo) = ?")
        params.append(repo.lower())
    rows = store.query(
        f"SELECT * FROM reviews WHERE {' AND '.join(where)}"
        " ORDER BY started_at DESC, id DESC LIMIT ?",
        (*params, limit + 1),
    )
    more = len(rows) > limit
    rows = rows[:limit]
    return {
        "reviews": [_review_row(r) for r in rows],
        "next_before": rows[-1]["started_at"] if more and rows else None,
    }
