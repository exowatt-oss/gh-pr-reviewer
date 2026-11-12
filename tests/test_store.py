"""SQLite state, review roll-ups, and spend aggregation."""

from __future__ import annotations

import sqlite3
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from app.store import LlmCallRecord, Store, iso, resolve_data_dir


def call(call_id: str, created: datetime, repo: str = "o/r", cost: float = 1.0, review_id=None):
    return LlmCallRecord(
        call_id=call_id,
        review_id=review_id,
        repo=repo,
        pr_number=1,
        head_sha="abc",
        command="review",
        model="test-model",
        status="success",
        prompt_tokens=100,
        cached_tokens=10,
        completion_tokens=50,
        reasoning_tokens=40,
        cost_usd=cost,
        litellm_cost_usd=None,
        duration_s=1.0,
        created_at=iso(created),
    )


def test_migrates_persisted_retry_intent_and_on_demand_request_ids(tmp_path: Path):
    path = tmp_path / "old.sqlite3"
    Store(path)
    with sqlite3.connect(path) as conn:
        conn.execute("ALTER TABLE pr_state DROP COLUMN failed_trigger")
        conn.execute("ALTER TABLE pr_state DROP COLUMN failed_requested_by")
        conn.execute("ALTER TABLE pr_state DROP COLUMN failed_force")
        conn.execute("ALTER TABLE reviews DROP COLUMN on_demand_request_id")

    store = Store(path)
    state = store.get_pr_state("o/r", 1)
    assert state is None
    review_id = store.start_review(
        "o/r",
        1,
        "https://github.com/o/r/pull/1",
        "title",
        "sha",
        "review",
        kind="on_demand",
        on_demand_request_id=12345,
    )
    assert store.get_review(review_id)["on_demand_request_id"] == 12345


def test_pr_state_round_trip(tmp_path: Path):
    store = Store(tmp_path / "s.sqlite3")
    assert store.get_pr_state("o/r", 1) is None
    store.mark_failed("o/r", 1, "sha1", now=100.0)
    store.mark_failed("o/r", 1, "sha1", now=200.0)
    state = store.get_pr_state("o/r", 1)
    assert (state.failed_sha, state.failed_attempts, state.last_failure_at) == ("sha1", 2, 200.0)
    store.mark_failed("o/r", 1, "sha2", now=300.0)
    assert store.get_pr_state("o/r", 1).failed_attempts == 1
    store.mark_reviewed("o/r", 1, "sha2", "review")
    state = store.get_pr_state("o/r", 1)
    assert (state.last_reviewed_sha, state.failed_attempts, state.failed_sha) == ("sha2", 0, None)
    store.set_draft_since_review("o/r", 1, True)
    assert store.get_pr_state("o/r", 1).draft_since_review is True
    # survives reopening the file (process restart)
    assert Store(tmp_path / "s.sqlite3").get_pr_state("o/r", 1).last_reviewed_sha == "sha2"


def test_review_rollup_and_interrupt(tmp_path: Path):
    store = Store(tmp_path / "s.sqlite3")
    rid = store.start_review("o/r", 1, "https://x/pull/1", "title", "abc", "review")
    now = datetime.now(UTC)
    store.insert_llm_call(call("c1", now, review_id=rid, cost=0.5))
    store.insert_llm_call(call("c2", now, review_id=rid, cost=0.25))
    assert not store.insert_llm_call(call("c2", now, review_id=rid, cost=0.25))  # duplicate
    store.finish_review(rid, status="ok", duration_s=12.5, comment_url="https://x/c")
    review = store.get_review(rid)
    assert (review["llm_calls"], review["prompt_tokens"], review["cost_usd"]) == (2, 200, 0.75)
    assert review["status"] == "ok" and review["comment_url"] == "https://x/c"

    rid2 = store.start_review("o/r", 2, "u", "t", "def", "incremental")
    assert store.interrupt_running() == 1
    assert store.get_review(rid2)["status"] == "interrupted"


def test_cost_summary_today_month_projection_and_repos(tmp_path: Path):
    store = Store(tmp_path / "s.sqlite3")
    now = datetime(2026, 10, 11, 0, 0, tzinfo=UTC)  # 10 of 31 days elapsed
    store.insert_llm_call(call("old", datetime(2026, 9, 30, 23, 0, tzinfo=UTC), cost=100))
    store.insert_llm_call(call("m1", datetime(2026, 10, 2, tzinfo=UTC), repo="a/b", cost=6))
    store.insert_llm_call(call("m2", datetime(2026, 10, 10, 12, tzinfo=UTC), repo="c/d", cost=4))
    store.insert_llm_call(call("t1", datetime(2026, 10, 11, 0, 0, tzinfo=UTC), repo="a/b", cost=0))
    summary = store.cost_summary(now)
    assert summary["today_usd"] == 0
    assert summary["month_to_date_usd"] == 10
    assert summary["projected_month_usd"] == pytest.approx(31)
    assert summary["month_calls"] == 3
    month = {r["repo"]: r["cost"] for r in summary["by_repo_month"]}
    assert month == {"a/b": 6, "c/d": 4}
    assert {r["repo"]: r["cost"] for r in summary["by_repo_all_time"]}["o/r"] == 100


def test_data_dir_falls_back_when_not_writable(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    blocked = tmp_path / "ro"
    blocked.mkdir()
    blocked.chmod(0o500)
    try:
        resolved = resolve_data_dir(str(blocked / "gh-pr-reviewer"))
    finally:
        blocked.chmod(0o700)
    assert resolved == tmp_path / "home" / ".local" / "share" / "gh-pr-reviewer"
    assert resolve_data_dir(str(tmp_path / "data")) == tmp_path / "data"


def test_old_database_gains_the_diff_fingerprint_column(tmp_path):
    import sqlite3

    path = tmp_path / "old.sqlite3"
    conn = sqlite3.connect(path)
    conn.execute(
        "CREATE TABLE pr_state (repo TEXT NOT NULL, pr_number INTEGER NOT NULL,"
        " last_reviewed_sha TEXT, last_review_mode TEXT, last_reviewed_at TEXT,"
        " draft_since_review INTEGER NOT NULL DEFAULT 0, failed_sha TEXT,"
        " failed_attempts INTEGER NOT NULL DEFAULT 0, last_failure_at REAL,"
        " PRIMARY KEY (repo, pr_number))"
    )
    conn.execute("INSERT INTO pr_state (repo, pr_number, last_reviewed_sha) VALUES ('o/r', 1, 's')")
    conn.commit()
    conn.close()
    store = Store(path)
    assert store.get_pr_state("o/r", 1).last_reviewed_diff is None
    store.mark_reviewed("o/r", 1, "s2", "incremental", diff_fingerprint="abc")
    state = store.get_pr_state("o/r", 1)
    assert (state.last_reviewed_sha, state.last_reviewed_diff) == ("s2", "abc")


def test_meta_setdefault_keeps_the_first_value(tmp_path: Path):
    store = Store(tmp_path / "s.sqlite3")
    assert store.meta_setdefault("k", "one") == "one"
    assert store.meta_setdefault("k", "two") == "one"
    store.set_meta("k", "three")
    assert store.get_meta("k") == "three" and store.get_meta("missing") is None


def test_old_databases_gain_the_new_pr_state_columns(tmp_path: Path):
    import sqlite3

    path = tmp_path / "old.sqlite3"
    conn = sqlite3.connect(path)
    conn.execute(
        "CREATE TABLE pr_state (repo TEXT NOT NULL, pr_number INTEGER NOT NULL,"
        " last_reviewed_sha TEXT, last_review_mode TEXT, last_reviewed_at TEXT,"
        " draft_since_review INTEGER NOT NULL DEFAULT 0, failed_sha TEXT,"
        " failed_attempts INTEGER NOT NULL DEFAULT 0, last_failure_at REAL,"
        " PRIMARY KEY (repo, pr_number))"
    )
    conn.execute("INSERT INTO pr_state (repo, pr_number, last_reviewed_sha) VALUES ('o/r', 1, 'x')")
    conn.commit()
    conn.close()
    store = Store(path)
    state = store.get_pr_state("o/r", 1)
    assert state.last_reviewed_sha == "x" and state.org_live is False
    store.observe("o/r", 1, "y", True, 10.0, cutoff=5.0)
    store.set_org_live("o/r", 1)
    state = store.get_pr_state("o/r", 1)
    assert (state.observed_sha, state.observed_draft, state.observed_at, state.org_live) == (
        "y",
        True,
        10.0,
        True,
    )


def test_observe_keeps_the_first_post_cutoff_sighting(tmp_path: Path):
    store = Store(tmp_path / "s.sqlite3")
    store.observe("o/r", 1, "a", False, 1.0, cutoff=5.0)  # before the cutoff
    state = store.get_pr_state("o/r", 1)
    store.observe("o/r", 1, "a", False, 6.0, cutoff=5.0, state=state)  # moves past it
    state = store.get_pr_state("o/r", 1)
    assert state.observed_at == 6.0
    store.observe("o/r", 1, "a", False, 9.0, cutoff=5.0, state=state)  # unchanged: kept
    assert store.get_pr_state("o/r", 1).observed_at == 6.0


def test_interrupt_rolls_up_cost_and_daily_cap_and_daily_totals(tmp_path: Path):
    store = Store(tmp_path / "s.sqlite3")
    now = datetime.now(UTC)
    rid = store.start_review("o/r", 1, "", "", "abc", "review")
    store.insert_llm_call(call("c1", now, review_id=rid, cost=0.25))
    store.insert_llm_call(call("c2", now, review_id=rid, cost=0.5))
    assert store.interrupt_running() == 1
    review = store.get_review(rid)
    assert review["status"] == "interrupted" and review["llm_calls"] == 2
    assert review["cost_usd"] == pytest.approx(0.75)
    assert store.interrupted_billed_review("o/r", 1, "abc")["id"] == rid
    assert store.interrupted_billed_review("o/r", 1, "other") is None
    store.mark_recovered(rid, "url")
    assert store.interrupted_billed_review("o/r", 1, "abc") is None

    rid2 = store.start_review("o/r", 1, "", "", "def", "incremental")
    store.insert_llm_call(call("c3", now, review_id=rid2))
    store.insert_llm_call(call("c4", now - timedelta(days=2), review_id=99))
    assert store.billed_reviews_since("o/r", 1, now - timedelta(hours=24)) == 2
    store.insert_llm_call(call("c5", now, repo="o/other", review_id=7, cost=2.0))
    daily = store.daily_by_repo(days=3, now=now)
    today = now.strftime("%Y-%m-%d")
    by = {(d["day"], d["repo"]): d for d in daily}
    assert by[(today, "o/r")]["reviews"] == 2 and by[(today, "o/r")]["cost"] == pytest.approx(1.75)
    assert by[(today, "o/other")]["cost"] == pytest.approx(2.0)
    assert daily[0]["repo"] == "o/other"  # newest day first, then by cost
    old_day = (now - timedelta(days=2)).strftime("%Y-%m-%d")
    assert by[(old_day, "o/r")]["calls"] == 1


def test_auto_rounds_count_posted_automatic_reviews_and_survive_reopening(tmp_path: Path):
    path = tmp_path / "rounds.sqlite3"
    store = Store(path)
    for sha, kind, status in (("a", "full", "ok"), ("b", "on_demand", "ok"),
                              ("c", "full", "error"), ("d", "full", "skipped"),
                              ("e", "incremental", "ok")):  # fmt: skip
        rid = store.start_review("o/r", 1, "", "", sha, "review", kind=kind, review_round=1)
        store.finish_review(rid, status=status, duration_s=1.0)
    store.record_skip("o/r", 1, "f", "max_rounds")
    assert store.auto_rounds("o/r", 1) == 2
    assert Store(path).get_pr_state("o/r", 1).auto_rounds == 2  # no pr_state row needed
    assert Store(path).get_pr_state("o/r", 2) is None
    assert store.get_review(rid)["review_round"] == 1
