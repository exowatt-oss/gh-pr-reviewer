"""Shared fixtures: isolated env, generated keys, a local JWKS server, and a fake model API."""

from __future__ import annotations

import json
import os
import tempfile
import threading
from collections.abc import Iterator
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any

import jwt
import pytest
from cryptography.hazmat.primitives.asymmetric import rsa
from forwarded_auth import _jwks, _settings
from jwt.algorithms import RSAAlgorithm


def pytest_configure(config: pytest.Config) -> None:
    """Isolate the run from real config before any test module imports app.config."""
    os.environ["DATA_DIR"] = tempfile.mkdtemp(prefix="gh-pr-reviewer-test-")
    for var in (
        "GITHUB_APP_ID",
        "GITHUB_APP_PRIVATE_KEY",
        "GITHUB_WEBHOOK_SECRET",
        "LLM_API_BASE",
        "LLM_API_KEY",
        "LLM_MODEL",
        "LLM_REASONING_EFFORT",
        "LLM_PROMPT_CACHE_LAYOUT",
        "LLM_INPUT_COST_PER_MILLION_USD",
        "LLM_CACHED_INPUT_COST_PER_MILLION_USD",
        "LLM_OUTPUT_COST_PER_MILLION_USD",
        "LLM_MONTHLY_BUDGET_USD",
        "REVIEW_TOOL_MAX_COST_USD",
        "REVIEW_REPOS",
        "PILOT_REPOS",
        "STATS_API_TOKEN",
        "FORWARDED_AUTH_ISSUER",
        "FORWARDED_AUTH_JWKS_URL",
        "FORWARDED_AUTH_AUDIENCE",
        "FORWARDED_AUTH_ALLOWED_EMAIL_DOMAINS",
    ):
        os.environ.pop(var, None)
    os.environ["FORWARDED_AUTH_MODE"] = "none"
    os.environ["FORWARDED_AUTH_ALLOW_INSECURE_NONE_MODE"] = "true"
    os.environ["DRY_RUN"] = "false"


@dataclass
class JwksServer:
    server: HTTPServer
    thread: threading.Thread
    private_key: rsa.RSAPrivateKey

    @property
    def issuer(self) -> str:
        host, port = self.server.server_address
        return f"http://{host}:{port}"

    @property
    def jwks_url(self) -> str:
        return f"{self.issuer}/jwks"

    def mint(self, claims: dict[str, Any]) -> str:
        return jwt.encode(claims, self.private_key, algorithm="RS256", headers={"kid": "test"})


@pytest.fixture
def jwks_server(monkeypatch: pytest.MonkeyPatch) -> Iterator[JwksServer]:
    private_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    public_jwk = json.loads(RSAAlgorithm.to_jwk(private_key.public_key()))
    public_jwk.update({"kid": "test", "use": "sig", "alg": "RS256"})
    jwks_body = json.dumps({"keys": [public_jwk]}).encode()

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802 - stdlib callback name
            if self.path not in {"/jwks", "/.well-known/jwks.json"}:
                self.send_error(404)
                return
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(jwks_body)))
            self.end_headers()
            self.wfile.write(jwks_body)

        def log_message(self, *_args: Any) -> None:
            pass

    server = HTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    fixture = JwksServer(server=server, thread=thread, private_key=private_key)
    thread.start()

    monkeypatch.setenv("FORWARDED_AUTH_MODE", "oidc")
    monkeypatch.setenv("FORWARDED_AUTH_ISSUER", fixture.issuer)
    monkeypatch.setenv("FORWARDED_AUTH_JWKS_URL", fixture.jwks_url)
    monkeypatch.setenv("FORWARDED_AUTH_AUDIENCE", "test-audience")
    monkeypatch.setenv("FORWARDED_AUTH_ALLOWED_EMAIL_DOMAINS", "*")
    _settings._reset_for_tests()
    _jwks._reset_discovery_cache_for_tests()
    _jwks._reset_jwks_clients_for_tests()

    try:
        yield fixture
    finally:
        _settings._reset_for_tests()
        _jwks._reset_discovery_cache_for_tests()
        _jwks._reset_jwks_clients_for_tests()
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


@pytest.fixture(scope="session")
def rsa_pem() -> str:
    from cryptography.hazmat.primitives import serialization

    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    return key.private_bytes(
        serialization.Encoding.PEM,
        serialization.PrivateFormat.TraditionalOpenSSL,
        serialization.NoEncryption(),
    ).decode()


@dataclass
class FakeProvider:
    server: HTTPServer
    requests: list[dict[str, Any]]
    usage: dict[str, Any]
    # set to (status, message) to answer every request with that error instead
    error: list[tuple[int, str]] = field(default_factory=list)
    # streamed replies in order, each {"content": str} or {"tool_calls": [(id, name, args)]};
    # once used up, the default "No issues found." answer
    script: list[dict[str, Any]] = field(default_factory=list)

    @property
    def url(self) -> str:
        host, port = self.server.server_address
        return f"http://{host}:{port}"


@pytest.fixture
def fake_provider() -> Iterator[FakeProvider]:
    """OpenAI-compatible /v1/chat/completions stub that records every request."""
    requests: list[dict[str, Any]] = []
    usage = {
        "prompt_tokens": 12000,
        "completion_tokens": 3000,
        "total_tokens": 15000,
        "prompt_tokens_details": {"cached_tokens": 2000},
        "completion_tokens_details": {"reasoning_tokens": 2500},
    }
    error: list[tuple[int, str]] = []
    script: list[dict[str, Any]] = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:  # noqa: N802 - stdlib callback name
            body = self.rfile.read(int(self.headers["Content-Length"]))
            requests.append(
                {
                    "path": self.path,
                    "headers": {k.lower(): v for k, v in self.headers.items()},
                    "body": json.loads(body),
                }
            )
            if error:
                status, message = error[0]
                failure = json.dumps({"error": {"message": message, "code": str(status)}}).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(failure)))
                self.end_headers()
                self.wfile.write(failure)
                return
            if json.loads(body).get("stream"):
                self._stream()
                return
            payload = json.dumps(
                {
                    "id": "chatcmpl-test",
                    "object": "chat.completion",
                    "created": 1,
                    "model": "test-model",
                    "choices": [
                        {
                            "index": 0,
                            "message": {"role": "assistant", "content": "No issues found."},
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": usage,
                }
            ).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("x-litellm-response-cost", "0.2345")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def _stream(self) -> None:
            def chunk(delta: dict[str, Any], finish: str | None = None, **extra: Any) -> bytes:
                data = {
                    "id": "chatcmpl-test",
                    "object": "chat.completion.chunk",
                    "created": 1,
                    "model": "test-model",
                    "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
                    **extra,
                }
                return f"data: {json.dumps(data)}\n\n".encode()

            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("x-litellm-response-cost", "0.2345")
            self.end_headers()
            reply = script.pop(0) if script else {"content": "No issues found."}
            if reply.get("tool_calls"):
                for i, (call_id, name, arguments) in enumerate(reply["tool_calls"]):
                    call = {"index": i, "id": call_id, "type": "function",
                            "function": {"name": name, "arguments": arguments}}  # fmt: skip
                    delta = {"role": "assistant", "content": None} if i == 0 else {}
                    self.wfile.write(chunk(delta | {"tool_calls": [call]}))
                self.wfile.write(chunk({}, "tool_calls"))
            else:
                text = reply["content"]
                self.wfile.write(chunk({"role": "assistant", "content": text[: len(text) // 2]}))
                self.wfile.write(chunk({"content": text[len(text) // 2 :]}))
                self.wfile.write(chunk({}, "stop"))
            final = {"id": "chatcmpl-test", "object": "chat.completion.chunk", "created": 1}
            final |= {"model": "test-model", "choices": [], "usage": usage}
            self.wfile.write(f"data: {json.dumps(final)}\n\n".encode())
            self.wfile.write(b"data: [DONE]\n\n")
            self.wfile.flush()
            self.close_connection = True

        def log_message(self, *_args: Any) -> None:
            pass

    server = HTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield FakeProvider(
            server=server, requests=requests, usage=usage, error=error, script=script
        )
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
