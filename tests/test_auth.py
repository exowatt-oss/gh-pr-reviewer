from __future__ import annotations

import time

import pytest
from fastapi.testclient import TestClient

from app.main import app
from tests.conftest import JwksServer

client = TestClient(app)


def _identity_token(jwks_server: JwksServer, **claims: object) -> str:
    now = int(time.time())
    return jwks_server.mint(
        {
            "iss": jwks_server.issuer,
            "aud": "test-audience",
            "sub": "user-123",
            "email": "reviewer@example.test",
            "email_verified": True,
            "name": "Sample Reviewer",
            "iat": now - 5,
            "nbf": now - 5,
            "exp": now + 300,
            **claims,
        }
    )


def test_protected_api_rejects_missing_identity(jwks_server: JwksServer):
    response = client.get("/api/reviews")
    assert response.status_code == 401
    assert response.json() == {"detail": "Authentication required"}


@pytest.mark.parametrize("path", ["/openapi.json", "/docs", "/redoc"])
def test_api_documentation_rejects_missing_identity(path: str, jwks_server: JwksServer):
    response = client.get(path)
    assert response.status_code == 401
    assert response.json() == {"detail": "Authentication required"}


def test_oidc_identity_is_verified_and_exposed(jwks_server: JwksServer):
    token = _identity_token(jwks_server)
    response = client.get("/api/me", headers={"Authorization": f"Bearer {token}"})
    assert response.status_code == 200
    assert response.json() == {
        "email": "reviewer@example.test",
        "name": "Sample Reviewer",
        "actor": None,
        "scope": None,
    }

    schema_response = client.get("/openapi.json", headers={"Authorization": f"Bearer {token}"})
    assert schema_response.status_code == 200
    schema = schema_response.json()
    assert schema["info"]["title"] == app.title
    assert schema["components"]["securitySchemes"]["IdentityToken"] == {
        "type": "http",
        "description": "Bearer identity token verified by forwarded-auth",
        "scheme": "bearer",
    }
    assert schema["paths"]["/api/me"]["get"]["security"] == [{"IdentityToken": []}]


def test_invalid_identity_token_is_rejected(jwks_server: JwksServer):
    token = _identity_token(jwks_server, iss="https://issuer.invalid")
    response = client.get("/api/me", headers={"Authorization": f"Bearer {token}"})
    assert response.status_code == 401
    assert response.json() == {"detail": "Authentication required"}
