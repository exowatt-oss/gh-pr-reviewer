from __future__ import annotations

import hashlib
import hmac

from fastapi.testclient import TestClient
from pydantic import SecretStr

import app.main as main

SECRET = "synthetic-webhook-secret-not-real"
BODY = b'{"action":"synchronize","pull_request":{"number":7}}'


class WakeCounter:
    def __init__(self) -> None:
        self.calls = 0

    def wake(self) -> None:
        self.calls += 1


def signature(secret: str, body: bytes) -> str:
    digest = hmac.new(secret.encode(), body, hashlib.sha256).hexdigest()
    return f"sha256={digest}"


def test_webhook_is_unavailable_without_a_signing_secret(monkeypatch):
    monkeypatch.setattr(main.settings, "GITHUB_WEBHOOK_SECRET", SecretStr(""))
    response = TestClient(main.app).post("/webhook", content=BODY)
    assert response.status_code == 503


def test_webhook_rejects_missing_or_invalid_signature(monkeypatch):
    monkeypatch.setattr(main.settings, "GITHUB_WEBHOOK_SECRET", SecretStr(SECRET))
    client = TestClient(main.app)
    headers = {"X-GitHub-Event": "pull_request"}
    assert client.post("/webhook", content=BODY, headers=headers).status_code == 401
    bad = headers | {"X-Hub-Signature-256": "sha256=invalid"}
    assert client.post("/webhook", content=BODY, headers=bad).status_code == 401


def test_valid_webhook_signature_wakes_allowlist_based_poller(monkeypatch):
    monkeypatch.setattr(main.settings, "GITHUB_WEBHOOK_SECRET", SecretStr(SECRET))
    poller = WakeCounter()
    monkeypatch.setattr(main, "_poller", poller)
    client = TestClient(main.app)
    headers = {
        "X-GitHub-Event": "pull_request",
        "X-Hub-Signature-256": signature(SECRET, BODY),
    }
    response = client.post("/webhook", content=BODY, headers=headers)
    assert response.status_code == 200
    assert response.json() == {"status": "accepted", "event": "pull_request"}
    assert poller.calls == 1


def test_valid_unrelated_webhook_does_not_wake_poller(monkeypatch):
    monkeypatch.setattr(main.settings, "GITHUB_WEBHOOK_SECRET", SecretStr(SECRET))
    poller = WakeCounter()
    monkeypatch.setattr(main, "_poller", poller)
    headers = {
        "X-GitHub-Event": "push",
        "X-Hub-Signature-256": signature(SECRET, BODY),
    }
    response = TestClient(main.app).post("/webhook", content=BODY, headers=headers)
    assert response.status_code == 200
    assert response.json()["event"] == "push"
    assert poller.calls == 0
