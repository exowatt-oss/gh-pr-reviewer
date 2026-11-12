"""Stats API (/api/v1/stats/*): contract shape, backfill, paging and bearer auth."""

from __future__ import annotations

import sqlite3
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from pydantic import SecretStr

from app import stats
from app.main import app, get_store, settings
from app.store import LlmCallRecord, Store, iso, utcnow

TOKEN = "stats-test-token-0123456789"
SUMMARY_KEYS = {
    "generated_at", "version", "model", "effort", "orgs", "poller", "budget", "totals",
    "daily", "by_repo",
}  # fmt: skip
REVIEW_KEYS = {
    "id", "repo", "pr", "pr_title", "pr_url", "author", "kind", "round", "head_sha", "ready_at",
    "started_at", "finished_at", "wait_s", "latency_s", "cost_usd", "tokens_in", "tokens_out",
    "tokens_cached", "findings", "status", "skip_reason", "review_url",
}  # fmt: skip


def billed(store: Store, rid: int, repo: str, cost: float, call_id: str) -> None:
    store.insert_llm_call(
        LlmCallRecord(
            call_id=call_id,
            review_id=rid,
            repo=repo,
            pr_number=1,
            head_sha="a" * 40,
            command="review",
            model="test-model",
            status="success",
            prompt_tokens=40_000,
            cached_tokens=4_000,
            completion_tokens=6_000,
            reasoning_tokens=5_000,
            cost_usd=cost,
            litellm_cost_usd=None,
            duration_s=60.0,
            created_at=iso(utcnow()),
        )
    )


def seed(store: Store, repo: str = "sample-org/example-project") -> int:
    rid = store.start_review(
        repo, 1, f"https://github.com/{repo}/pull/1", "Add x", "a" * 40, "review",
        kind="full", ready_at=utcnow().timestamp() - 400, author="sample-user",
    )  # fmt: skip
    billed(store, rid, repo, 0.2, f"{repo}-{rid}")
    store.finish_review(
        rid, status="ok", duration_s=90.0, comment_url=f"https://github.com/{repo}/pull/1#r",
        findings={"critical": 0, "major": 1, "minor": 2, "suggestions": 1}, inline_comments=4,
    )  # fmt: skip
    store.record_skip(repo, 2, "b" * 40, "merged_during_wait", kind="full")
    store.record_skip(repo, 3, "c" * 40, "draft")
    store.record_skip(repo, 3, "c" * 40, "draft")  # once per head
    return rid


def test_summary_has_the_contract_shape(tmp_path: Path):
    store = Store(tmp_path / "s.sqlite3")
    seed(store)
    poller = {
        "state": "running",
        "last_poll_at": "2026-10-07T20:00:00+00:00",
        "last_error": "provider unavailable",
        "last_error_at": "2026-10-07T19:59:00+00:00",
        "orgs": [
            {
                "org": "sample-org",
                "installation_id": 77,
                "repos_covered": 322,
                "cutoff": "2026-10-07T18:38:54+00:00",
                "state": "ok",
            }
        ],  # fmt: skip
        "running_jobs": [
            {"repo": "o/r", "pr": 5, "kind": "full", "started_at": "x", "ready_at": "y"}
        ],
        "queued_jobs": [
            {"repo": "o/r", "pr": 6, "kind": "full", "ready_at": "z", "started_at": None}
        ],
        "budget": {
            "state": "paused_full",
            "spend_usd": 1900.0,
            "max_budget_usd": 2000.0,
            "budget_reset_at": "2026-11-01T00:00:00Z",
        },  # fmt: skip
    }
    out = stats.summary(store, days=7, poller=poller, version="abc", effort="xhigh")
    assert set(out) == SUMMARY_KEYS
    assert (out["model"], out["effort"], out["version"]) == ("gpt-4.1", "xhigh", "abc")
    assert out["orgs"] == [{"org": "sample-org", "installation_id": 77, "repos_covered": 322,
                            "cutoff": "2026-10-07T18:38:54+00:00"}]  # fmt: skip
    assert out["poller"]["running"] == [{"repo": "o/r", "pr": 5, "kind": "full", "started_at": "x"}]
    assert out["poller"]["queued"] == [{"repo": "o/r", "pr": 6, "ready_at": "z"}]
    assert out["poller"]["last_error"] == "provider unavailable"
    assert out["poller"]["last_error_at"] == "2026-10-07T19:59:00+00:00"
    assert out["budget"] == {
        "monthly_cap_usd": 2000.0,
        "spend_usd": 1900.0,
        "guard": "paused_full",
    }
    totals = out["totals"]
    assert totals["reviews_posted"] == 1 and totals["billed_reviews"] == 1
    assert totals["cost_usd"] == pytest.approx(0.2)
    assert (totals["tokens_in"], totals["tokens_out"], totals["tokens_cached"]) == (
        40_000, 6_000, 4_000)  # fmt: skip
    assert totals["inline_comments"] == 4
    assert set(totals["skipped"]) == {
        "draft", "bot", "lockfile_only", "merged_during_wait", "closed", "opted_out",
        "daily_cap", "too_large", "unchanged_diff", "max_rounds", "closed_during_review",
        "merged_during_review", "superseded", "already_reviewed",
    }  # fmt: skip
    assert totals["posted_post_merge"] == 0
    assert totals["reviews_by_round"] == {"round_1": 0, "round_2": 0, "later": 0, "on_demand": 0}
    assert totals["skipped"]["merged_during_wait"] == 1 and totals["skipped"]["draft"] == 1
    assert len(out["daily"]) == 7 and out["daily"][-1]["date"] == utcnow().strftime("%Y-%m-%d")
    assert out["daily"][-1] == {"date": out["daily"][-1]["date"], "reviews": 1,
                                "cost_usd": pytest.approx(0.2), "skipped": 2}  # fmt: skip
    assert out["by_repo"][0]["repo"] == "sample-org/example-project"
    assert out["by_repo"][0]["reviews"] == 1


def test_reviews_rows_and_paging(tmp_path: Path):
    store = Store(tmp_path / "s.sqlite3")
    rid = seed(store)
    page = stats.reviews(store, limit=2)
    assert len(page["reviews"]) == 2 and page["next_before"]
    rest = stats.reviews(store, limit=2, before=page["next_before"])
    assert rest["next_before"] is None
    rows = page["reviews"] + rest["reviews"]
    assert len(rows) == 3 and all(set(r) == REVIEW_KEYS for r in rows)
    posted = next(r for r in rows if r["id"] == rid)
    assert posted["status"] == "posted" and posted["kind"] == "full"
    assert posted["findings"] == {"critical": 0, "major": 1, "minor": 2, "suggestions": 1}
    assert posted["wait_s"] == pytest.approx(400, abs=5) and posted["latency_s"] is not None
    assert posted["author"] == "sample-user" and posted["review_url"].endswith("#r")
    merged = next(r for r in rows if r["pr"] == 2)
    assert (merged["status"], merged["skip_reason"], merged["findings"]) == (
        "skipped", "merged_during_wait", None)  # fmt: skip
    assert stats.reviews(store, limit=10, repo="sample-org/example-project")["reviews"]
    assert stats.reviews(store, limit=10, repo="o/other")["reviews"] == []


def test_old_rows_are_backfilled(tmp_path: Path):
    path = tmp_path / "old.sqlite3"
    conn = sqlite3.connect(path)
    conn.executescript(
        """
        CREATE TABLE reviews (
            id INTEGER PRIMARY KEY AUTOINCREMENT, started_at TEXT NOT NULL, finished_at TEXT,
            repo TEXT NOT NULL, pr_number INTEGER NOT NULL, pr_url TEXT, pr_title TEXT,
            head_sha TEXT NOT NULL, command TEXT NOT NULL, status TEXT NOT NULL, error TEXT,
            llm_calls INTEGER NOT NULL DEFAULT 0, prompt_tokens INTEGER NOT NULL DEFAULT 0,
            cached_tokens INTEGER NOT NULL DEFAULT 0, completion_tokens INTEGER NOT NULL DEFAULT 0,
            reasoning_tokens INTEGER NOT NULL DEFAULT 0, cost_usd REAL NOT NULL DEFAULT 0,
            litellm_cost_usd REAL, duration_s REAL, comment_url TEXT
        );
        INSERT INTO reviews (started_at, repo, pr_number, head_sha, command, status, error)
        VALUES ('2026-10-07T10:00:00+00:00', 'o/r', 1, 'a', 'incremental', 'ok', NULL),
               ('2026-10-07T10:01:00+00:00', 'o/r', 2, 'b', 'review', 'skipped',
                'no reviewable files (all ignored)'),
               ('2026-10-07T10:02:00+00:00', 'o/r', 3, 'c', 'review', 'skipped',
                'reviewable diff unchanged since the last review (rebase/empty commit)');
        """
    )
    conn.commit()
    conn.close()
    store = Store(path)
    rows = {r["pr_number"]: r for r in store.recent_reviews()}
    assert rows[1]["kind"] == "incremental" and rows[1]["origin"] == "reviewer"
    assert rows[2]["skip_reason"] == "lockfile_only"
    assert rows[3]["skip_reason"] == "unchanged_diff"
    out = stats.reviews(store, limit=10)["reviews"]
    assert {r["pr"]: r["findings"] for r in out} == {1: None, 2: None, 3: None}
    assert next(r for r in out if r["pr"] == 1)["ready_at"] is None


def test_stats_api_requires_the_bearer_token(monkeypatch, jwks_server):
    monkeypatch.setattr(settings, "STATS_API_TOKEN", SecretStr(TOKEN))
    seed(get_store(), repo="sample-org/stats-api")
    client = TestClient(app)
    for path in ("/api/v1/stats/summary", "/api/v1/stats/reviews"):
        assert client.get(path).status_code == 401
        bad = client.get(path, headers={"Authorization": "Bearer nope"})
        assert bad.status_code == 401 and bad.headers["www-authenticate"] == "Bearer"
        assert client.get(path, headers={"Authorization": f"Bearer {TOKEN}"}).status_code == 200
    auth = {"Authorization": f"Bearer {TOKEN}"}
    summary = client.get("/api/v1/stats/summary?days=90", headers=auth).json()
    assert set(summary) == SUMMARY_KEYS and len(summary["daily"]) == 90
    assert client.get("/api/v1/stats/summary?days=91", headers=auth).status_code == 422
    assert client.get("/api/v1/stats/summary?days=0", headers=auth).status_code == 422
    page = client.get("/api/v1/stats/reviews?repo=sample-org/stats-api&limit=1", headers=auth)
    body = page.json()
    assert len(body["reviews"]) == 1 and body["next_before"]
    later = (datetime.now(UTC) + timedelta(days=1)).strftime("%Y-%m-%dT%H:%M:%SZ")
    assert client.get(f"/api/v1/stats/reviews?before={later}", headers=auth).status_code == 200
    assert client.get("/api/v1/stats/reviews?before=nope", headers=auth).status_code == 422


def test_stats_api_accepts_explicit_local_dev_auth_and_rejects_empty_token_config(
    monkeypatch, jwks_server
):
    monkeypatch.setattr(settings, "STATS_API_TOKEN", SecretStr(""))
    client = TestClient(app)
    # An empty configured token never matches an empty / any bearer.
    assert (
        client.get("/api/v1/stats/summary", headers={"Authorization": "Bearer "}).status_code == 401
    )
    for name in (
        "FORWARDED_AUTH_ISSUER",
        "FORWARDED_AUTH_JWKS_URL",
        "FORWARDED_AUTH_AUDIENCE",
        "FORWARDED_AUTH_ALLOWED_EMAIL_DOMAINS",
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("FORWARDED_AUTH_MODE", "none")
    monkeypatch.setenv("FORWARDED_AUTH_ALLOW_INSECURE_NONE_MODE", "true")
    assert client.get("/api/v1/stats/summary").status_code == 200
