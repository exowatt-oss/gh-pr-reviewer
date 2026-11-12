import asyncio
import hashlib
import hmac
import logging
import sys
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from fastapi import Depends, FastAPI, HTTPException, Query, Request, Security
from fastapi.middleware.cors import CORSMiddleware
from fastapi.openapi.docs import get_redoc_html, get_swagger_ui_html
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from forwarded_auth import User
from forwarded_auth import get_settings as get_forwarded_auth_settings

from app import stats
from app.config import settings
from app.costs import MODEL_LABEL, format_cost
from app.deps import get_current_user
from app.poller import Poller
from app.review_policy import parse_review_targets
from app.store import Store

# --- Structured JSON logging ---
handler = logging.StreamHandler(sys.stdout)
handler.setFormatter(
    logging.Formatter(
        '{"time":"%(asctime)s","level":"%(levelname)s","logger":"%(name)s","message":"%(message)s"}'
    )
)
logging.basicConfig(level=settings.LOG_LEVEL.upper(), handlers=[handler])
logger = logging.getLogger(__name__)

_store: Store | None = None
_poller: Poller | None = None


def get_store() -> Store:
    global _store
    if _store is None:
        _store = Store.open(settings.DATA_DIR)
        logger.info("SQLite store at %s", _store.path)
    return _store


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _poller
    get_forwarded_auth_settings().validate()
    logger.info("Starting up (environment=%s, version=%s)", settings.ENVIRONMENT, settings.GIT_SHA)
    store = get_store()
    interrupted = store.interrupt_running()
    if interrupted:
        logger.warning("Marked %d review(s) from a previous process as interrupted", interrupted)
    _poller = Poller(settings, store)
    _poller.start()
    yield
    # Graceful deploys: stop taking work, let running reviews finish (up to
    # SHUTDOWN_GRACE_SECONDS, inside systemd's 90 s stop timeout).
    logger.info("Shutting down; draining reviews for up to %ss", settings.SHUTDOWN_GRACE_SECONDS)
    await asyncio.to_thread(_poller.stop)


app = FastAPI(
    title="gh-pr-reviewer",
    description=(
        "Self-hosted AI pull-request reviewer that discovers allowlisted repositories, "
        "follows a configurable review policy, and records model usage."
    ),
    # Keep the schema and interactive docs behind the same verified identity boundary as the API.
    openapi_url=None,
    docs_url=None,
    redoc_url=None,
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

_app_dir = Path(__file__).parent
app.mount("/static", StaticFiles(directory=_app_dir / "static"), name="static")
templates = Jinja2Templates(directory=_app_dir / "templates")


def poller_status() -> dict[str, Any]:
    if _poller is None:
        return {"state": "not_started", "dry_run": settings.DRY_RUN}
    return _poller.status()


def _short_time(value: str | None) -> str:
    if not value:
        return ""
    try:
        return datetime.fromisoformat(value).astimezone(UTC).strftime("%m-%d %H:%M")
    except ValueError:
        return value


def _ktok(value: int | None) -> str:
    return f"{(value or 0) / 1000:.1f}k"


templates.env.filters["short_time"] = _short_time
templates.env.filters["usd"] = lambda v: format_cost(float(v or 0))
templates.env.filters["ktok"] = _ktok


# --- Public routes -------------------------------------------------------------------
@app.get("/api/health")
async def health():
    """Unauthenticated liveness probe for the running service."""
    return {"status": "ok", "version": settings.GIT_SHA}


MAX_WEBHOOK_BYTES = 2_000_000
WEBHOOK_EVENTS = frozenset(
    {"installation", "installation_repositories", "issue_comment", "pull_request"}
)


@app.post("/webhook", include_in_schema=False)
async def github_webhook(request: Request):
    """Verify GitHub's HMAC signature before waking the normal allowlist-based poller."""
    secret = settings.GITHUB_WEBHOOK_SECRET.get_secret_value()
    if not secret:
        raise HTTPException(status_code=503, detail="Webhook signing secret is not configured")

    body_parts: list[bytes] = []
    size = 0
    async for part in request.stream():
        size += len(part)
        if size > MAX_WEBHOOK_BYTES:
            raise HTTPException(status_code=413, detail="Webhook payload is too large")
        body_parts.append(part)
    body = b"".join(body_parts)
    signature = request.headers.get("x-hub-signature-256", "")
    expected = "sha256=" + hmac.new(secret.encode(), body, hashlib.sha256).hexdigest()
    try:
        signature_bytes = signature.encode("ascii")
    except UnicodeEncodeError:
        raise HTTPException(status_code=401, detail="Invalid webhook signature") from None
    if not hmac.compare_digest(signature_bytes, expected.encode("ascii")):
        raise HTTPException(status_code=401, detail="Invalid webhook signature")

    event = request.headers.get("x-github-event", "")
    if not event:
        raise HTTPException(status_code=400, detail="Missing GitHub event header")
    if event in WEBHOOK_EVENTS and _poller is not None:
        _poller.wake()
    return {"status": "accepted", "event": event}


@app.get("/healthz")
async def healthz():
    """Public, low-detail status probe; repository and installation data stay authenticated."""
    return {
        "status": "ok",
        "version": settings.GIT_SHA,
        "poller": {"state": poller_status().get("state", "not_started")},
    }


_PREVIEW_HTML = """<!DOCTYPE html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Pull request reviewer</title><link rel="stylesheet" href="/static/app.css"></head>
<body><main><h1>Pull request reviewer</h1>
<p>Self-hosted AI-assisted code review for GitHub pull requests, with configurable review policy and
usage tracking.</p>
</main></body></html>"""


@app.get("/preview", response_class=HTMLResponse, include_in_schema=False)
async def preview():
    """Static public preview page with no user data."""
    return _PREVIEW_HTML


# --- Authenticated routes ------------------------------------------------------------
@app.get("/", response_class=HTMLResponse, include_in_schema=False)
async def dashboard(request: Request, user: User = Depends(get_current_user)):
    store = get_store()
    summary = await asyncio.to_thread(store.cost_summary)
    reviews = await asyncio.to_thread(store.recent_reviews, 100)
    return templates.TemplateResponse(
        request,
        "dashboard.html",
        {
            "user": user,
            "summary": summary,
            "reviews": reviews,
            "poller": poller_status(),
            "repos": parse_review_targets(settings.REVIEW_REPOS).describe(),
            "model_label": MODEL_LABEL,
            "effort": settings.LLM_REASONING_EFFORT or "default",
            "version": settings.GIT_SHA[:12],
        },
    )


@app.get("/api/reviews")
async def list_reviews(
    limit: int = Query(100, ge=1, le=1000), _user: User = Depends(get_current_user)
):
    """Recent review runs, newest first, with tokens and cost."""
    return await asyncio.to_thread(get_store().recent_reviews, limit)


@app.get("/api/costs")
async def costs(days: int = Query(14, ge=1, le=90), _user: User = Depends(get_current_user)):
    """Spend totals (UTC day/month, run-rate projection), per-repo totals and local spend guard."""
    store = get_store()
    summary = await asyncio.to_thread(store.cost_summary)
    daily = await asyncio.to_thread(store.daily_by_repo, days)
    return {
        **summary,
        "daily_by_repo": daily,
        "budget_guard": poller_status().get("budget") or None,
        "model": MODEL_LABEL,
        "model_alias": settings.LLM_MODEL,
        "reasoning_effort": settings.LLM_REASONING_EFFORT or "default",
    }


@app.get("/api/status")
async def full_status(_user: User = Depends(get_current_user)):
    """Poller status including the most recent review decisions."""
    return {"version": settings.GIT_SHA, "poller": poller_status()}


# --- Stats API -----------------------------------------------------------------------
# The app requires either the STATS_API_TOKEN bearer or a verified forwarded-auth identity.
_stats_bearer = HTTPBearer(
    auto_error=False,
    scheme_name="StatsToken",
    description="STATS_API_TOKEN or a forwarded-auth identity token",
)


def require_stats_reader(
    request: Request,
    credentials: HTTPAuthorizationCredentials | None = Security(_stats_bearer),
) -> str:
    token = settings.STATS_API_TOKEN.get_secret_value()
    if token and credentials and credentials.scheme.lower() == "bearer":
        if hmac.compare_digest(credentials.credentials.encode(), token.encode()):
            return "stats-token"
    try:
        return get_current_user(request, credentials).email
    except HTTPException:
        raise HTTPException(
            status_code=401,
            detail="stats API: bearer token or verified identity required",
            headers={"WWW-Authenticate": "Bearer"},
        ) from None


@app.get("/api/v1/stats/summary")
async def stats_summary(
    days: int = Query(30, ge=1, le=90), _reader: str = Depends(require_stats_reader)
):
    """Totals, per-day and per-repo usage stats for the last ``days`` UTC days."""
    return await asyncio.to_thread(
        stats.summary,
        get_store(),
        days=days,
        poller=poller_status(),
        version=settings.GIT_SHA,
        effort=settings.LLM_REASONING_EFFORT,
    )


@app.get("/api/v1/stats/reviews")
async def stats_reviews(
    limit: int = Query(100, ge=1, le=500),
    before: str | None = Query(None, description="ISO 8601; rows started before it"),
    repo: str | None = Query(None, description="owner/name"),
    _reader: str = Depends(require_stats_reader),
):
    """Review runs and recorded skips, newest first; page with ``next_before``."""
    if before and stats.normalize_before(before) is None:
        raise HTTPException(status_code=422, detail="before must be an ISO 8601 time")
    return await asyncio.to_thread(
        stats.reviews, get_store(), limit=limit, before=before, repo=repo
    )


@app.get("/api/me")
async def current_identity(user: User = Depends(get_current_user)):
    """Show the verified human identity or bot author (never its owner)."""
    return {
        "email": user.email,
        "name": user.name,
        "actor": user.actor,
        "scope": user.scope,
    }


@app.get("/openapi.json", include_in_schema=False)
async def openapi_schema(_user: User = Depends(get_current_user)):
    """Serve the API schema only after verifying the configured identity token."""
    return JSONResponse(app.openapi())


@app.get("/docs", include_in_schema=False)
async def swagger_ui(_user: User = Depends(get_current_user)):
    return get_swagger_ui_html(openapi_url="/openapi.json", title=f"{app.title} - Swagger UI")


@app.get("/redoc", include_in_schema=False)
async def redoc(_user: User = Depends(get_current_user)):
    return get_redoc_html(openapi_url="/openapi.json", title=f"{app.title} - ReDoc")
