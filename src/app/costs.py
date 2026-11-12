"""Configurable model-usage estimates and the pull-request comment footer."""

from __future__ import annotations

import calendar
import re
from dataclasses import dataclass
from datetime import UTC, datetime

from app.config import settings

# Operators may configure provider-specific rates; unknown rates default to zero.
INPUT_PER_MILLION_USD = settings.LLM_INPUT_COST_PER_MILLION_USD
CACHED_INPUT_PER_MILLION_USD = settings.LLM_CACHED_INPUT_COST_PER_MILLION_USD
OUTPUT_PER_MILLION_USD = settings.LLM_OUTPUT_COST_PER_MILLION_USD

MODEL_LABEL = settings.LLM_MODEL
FOOTER_MARKER = "<!-- gh-pr-reviewer:footer -->"
_FOOTER_RE = re.compile(r"\n*" + re.escape(FOOTER_MARKER) + r"\n<sub>.*?</sub>\n*", re.DOTALL)
# PR-Agent keeps its finding state in a trailing hidden marker; the footer goes above it.
_STATE_MARKER_RE = re.compile(r"<!-- pr-agent-review-state:v\d+\n.*?\n-->", re.DOTALL)


@dataclass(frozen=True)
class TokenUsage:
    prompt_tokens: int = 0
    cached_tokens: int = 0
    completion_tokens: int = 0
    reasoning_tokens: int = 0


def compute_cost(
    prompt_tokens: int,
    cached_tokens: int,
    completion_tokens: int,
    *,
    fresh_rate: float = INPUT_PER_MILLION_USD,
    cached_rate: float = CACHED_INPUT_PER_MILLION_USD,
    output_rate: float = OUTPUT_PER_MILLION_USD,
) -> float:
    """USD cost of one call. ``prompt_tokens`` includes ``cached_tokens``."""
    prompt_tokens = max(int(prompt_tokens or 0), 0)
    cached = min(max(int(cached_tokens or 0), 0), prompt_tokens)
    fresh = prompt_tokens - cached
    completion = max(int(completion_tokens or 0), 0)
    return (fresh * fresh_rate + cached * cached_rate + completion * output_rate) / 1_000_000


def project_month(month_to_date: float, now: datetime) -> float:
    """Linear run-rate projection of month-to-date spend to the full calendar month (UTC)."""
    now = now.astimezone(UTC)
    start = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    elapsed = (now - start).total_seconds()
    days = calendar.monthrange(now.year, now.month)[1]
    total = days * 86400
    if elapsed <= 0:
        return month_to_date
    return month_to_date * total / elapsed


def format_cost(cost: float) -> str:
    return f"${cost:.4f}" if 0 < cost < 0.01 else f"${cost:.2f}"


def render_footer(prompt_tokens: int, completion_tokens: int, cost: float, effort: str) -> str:
    effort_label = f" {effort}" if effort else ""
    return (
        f"gh-pr-reviewer · {MODEL_LABEL}{effort_label} · "
        f"{prompt_tokens / 1000:.1f}k in / {completion_tokens / 1000:.1f}k out · "
        f"{format_cost(cost)}"
    )


def apply_footer(body: str, footer_line: str) -> str:
    """Replace any previous footer and add ``footer_line`` above PR-Agent's state marker."""
    body = _FOOTER_RE.sub("\n\n", body or "").rstrip()
    block = f"{FOOTER_MARKER}\n<sub>{footer_line}</sub>"
    marker = _STATE_MARKER_RE.search(body)
    if marker:
        head = body[: marker.start()].rstrip()
        return f"{head}\n\n{block}\n\n{body[marker.start() :]}"
    return f"{body}\n\n{block}"
