"""Shared request identity dependency backed by the public forwarded-auth package."""

from typing import Annotated

from fastapi import Request, Security
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from forwarded_auth import User
from forwarded_auth.fastapi import require_user

_identity_bearer = HTTPBearer(
    auto_error=False,
    scheme_name="IdentityToken",
    description="Bearer identity token verified by forwarded-auth",
)


def get_current_user(
    request: Request,
    _credentials: Annotated[
        HTTPAuthorizationCredentials | None,
        Security(_identity_bearer),
    ] = None,
) -> User:
    """Return a verified identity or raise HTTP 401; never trust forwarded headers alone."""
    return require_user(request)
