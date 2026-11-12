from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.main import app, get_store
from app.store import LlmCallRecord, iso, utcnow
from tests.conftest import JwksServer


def test_healthz_is_public_and_reports_unconfigured_poller(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("FORWARDED_AUTH_MODE", "none")
    monkeypatch.setenv("FORWARDED_AUTH_ALLOW_INSECURE_NONE_MODE", "true")
    with TestClient(app) as client:
        response = client.get("/healthz")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["poller"] == {"state": "not_configured"}
    assert set(body) == {"status", "version", "poller"}
    assert "orgs" not in response.text and "installation_id" not in response.text


def test_api_health_and_preview_are_public():
    client = TestClient(app)
    assert client.get("/api/health").json()["status"] == "ok"
    preview = client.get("/preview")
    assert preview.status_code == 200 and "Pull request reviewer" in preview.text


@pytest.mark.parametrize(
    "path", ["/", "/api/reviews", "/api/costs", "/api/status", "/openapi.json", "/docs"]
)
def test_ui_and_apis_require_identity(path: str, jwks_server: JwksServer):
    response = TestClient(app).get(path)
    assert response.status_code == 401


def test_dashboard_and_json_apis_render_in_explicit_local_dev_mode(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setenv("FORWARDED_AUTH_MODE", "none")
    monkeypatch.setenv("FORWARDED_AUTH_ALLOW_INSECURE_NONE_MODE", "true")
    store = get_store()
    rid = store.start_review(
        "sample-org/example-project",
        42,
        "https://github.com/sample-org/example-project/pull/42",
        "Add example behavior",
        "f" * 40,
        "review",
    )
    store.insert_llm_call(
        LlmCallRecord(
            call_id="app-test-1",
            review_id=rid,
            repo="sample-org/example-project",
            pr_number=42,
            head_sha="f" * 40,
            command="review",
            model="gpt-4.1",
            status="success",
            prompt_tokens=50_000,
            cached_tokens=0,
            completion_tokens=8_000,
            reasoning_tokens=6_000,
            cost_usd=0.0067,
            litellm_cost_usd=None,
            duration_s=95.0,
            created_at=iso(utcnow()),
        )
    )
    store.finish_review(rid, status="ok", duration_s=101.0)
    client = TestClient(app)

    page = client.get("/")
    assert page.status_code == 200
    assert "Review activity" in page.text
    assert "sample-org/example-project" in page.text and "#42" in page.text
    assert "$0.0067" in page.text
    assert "dev@localhost" in page.text

    reviews = client.get("/api/reviews").json()
    assert reviews[0]["pr_number"] == 42 and reviews[0]["cost_usd"] == pytest.approx(0.0067)

    status = client.get("/api/status").json()
    assert "recent_decisions" in status["poller"] or status["poller"]["state"] == "not_started"

    costs = client.get("/api/costs").json()
    assert costs["month_to_date_usd"] >= 0.0067
    today = [d for d in costs["daily_by_repo"] if d["repo"] == "sample-org/example-project"][0]
    assert today["day"] == utcnow().strftime("%Y-%m-%d") and today["cost"] >= 0.0067
    assert client.get("/api/costs?days=91").status_code == 422
    assert costs["reasoning_effort"] == "default"
    assert "provider" not in costs
