"""Record every LLM call PR-Agent makes through litellm, attributed to the running review.

PR-Agent calls ``litellm.acompletion``; litellm invokes registered ``CustomLogger`` hooks on
success and failure. The hook reads the review being worked on from a ``ContextVar`` that the
reviewer sets before invoking PR-Agent (litellm's logging worker runs callbacks inside the
context captured when the call finished, so attribution survives the async hop).
"""

from __future__ import annotations

import asyncio
import contextvars
import logging
import threading
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import litellm
from litellm.integrations.custom_logger import CustomLogger

from app.costs import compute_cost
from app.store import LlmCallRecord, Store, iso

logger = logging.getLogger(__name__)


@dataclass
class ReviewContext:
    """Attribution for LLM calls made while one review command runs."""

    repo: str
    pr_number: int
    head_sha: str
    command: str
    review_id: int | None = None
    calls: list[LlmCallRecord] = field(default_factory=list)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def add(self, rec: LlmCallRecord) -> None:
        with self._lock:
            self.calls.append(rec)

    def totals(self) -> dict[str, Any]:
        with self._lock:
            calls = [c for c in self.calls if c.status == "success"]
            litellm_costs = [c.litellm_cost_usd for c in calls if c.litellm_cost_usd is not None]
            return {
                "calls": len(calls),
                "failed_calls": len(self.calls) - len(calls),
                "prompt_tokens": sum(c.prompt_tokens for c in calls),
                "cached_tokens": sum(c.cached_tokens for c in calls),
                "completion_tokens": sum(c.completion_tokens for c in calls),
                "reasoning_tokens": sum(c.reasoning_tokens for c in calls),
                "cost_usd": sum(c.cost_usd for c in calls),
                "litellm_cost_usd": sum(litellm_costs) if litellm_costs else None,
            }


current_review: contextvars.ContextVar[ReviewContext | None] = contextvars.ContextVar(
    "current_review", default=None
)


def _get(obj: Any, name: str, default: Any = None) -> Any:
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


def _int(value: Any) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def _float_or_none(value: Any) -> float | None:
    try:
        return None if value is None else float(value)
    except (TypeError, ValueError):
        return None


def _duration(start: Any, end: Any) -> float | None:
    if isinstance(start, datetime) and isinstance(end, datetime):
        return max((end - start).total_seconds(), 0.0)
    if isinstance(start, int | float) and isinstance(end, int | float):
        return max(float(end) - float(start), 0.0)
    return None


def usage_from_response(response_obj: Any) -> tuple[int, int, int, int]:
    """(prompt, cached, completion, reasoning) tokens from an OpenAI-style response."""
    usage = _get(response_obj, "usage")
    prompt = _int(_get(usage, "prompt_tokens"))
    completion = _int(_get(usage, "completion_tokens"))
    cached = _int(_get(_get(usage, "prompt_tokens_details"), "cached_tokens"))
    if not cached:  # Anthropic-style field, in case the provider passes it through
        cached = _int(_get(usage, "cache_read_input_tokens"))
    reasoning = _int(_get(_get(usage, "completion_tokens_details"), "reasoning_tokens"))
    return prompt, cached, completion, reasoning


class CostTracker(CustomLogger):
    """litellm callback that persists one row per LLM call."""

    def __init__(self, store: Store) -> None:
        super().__init__()
        self.store = store

    def _record(
        self,
        kwargs: dict[str, Any],
        response_obj: Any,
        start_time: Any,
        end_time: Any,
        *,
        status: str,
    ) -> None:
        try:
            ctx = current_review.get()
            prompt, cached, completion, reasoning = (
                usage_from_response(response_obj) if status == "success" else (0, 0, 0, 0)
            )
            error = None
            if status != "success":
                exc = kwargs.get("exception") or response_obj
                error = f"{type(exc).__name__}: {exc}"[:500] if exc is not None else "unknown"
            model = kwargs.get("model") or _get(response_obj, "model")
            provider_cost = (
                _float_or_none(kwargs.get("response_cost")) if status == "success" else None
            )
            estimated_cost = (
                provider_cost
                if provider_cost is not None
                else compute_cost(prompt, cached, completion)
            )
            rec = LlmCallRecord(
                call_id=kwargs.get("litellm_call_id"),
                review_id=ctx.review_id if ctx else None,
                repo=ctx.repo if ctx else None,
                pr_number=ctx.pr_number if ctx else None,
                head_sha=ctx.head_sha if ctx else None,
                command=ctx.command if ctx else None,
                model=str(model) if model else None,
                status=status,
                prompt_tokens=prompt,
                cached_tokens=cached,
                completion_tokens=completion,
                reasoning_tokens=reasoning,
                cost_usd=estimated_cost,
                # Provider-reported cost, when LiteLLM supplies it; otherwise use configured rates.
                litellm_cost_usd=provider_cost,
                duration_s=_duration(start_time, end_time),
                error=error,
                created_at=iso(end_time) if isinstance(end_time, datetime) else None,
            )
            if self.store.insert_llm_call(rec) and ctx is not None:
                ctx.add(rec)
            if ctx is None:
                logger.warning("LLM call %s recorded without review attribution", rec.call_id)
        except Exception:  # a broken callback must never break a review
            logger.exception("Failed to record LLM call")

    # litellm calls the sync hooks for sync completions and the async ones for acompletion;
    # call_id de-duplication makes double delivery harmless either way.
    def log_success_event(self, kwargs, response_obj, start_time, end_time):  # noqa: ANN001
        self._record(kwargs, response_obj, start_time, end_time, status="success")

    def log_failure_event(self, kwargs, response_obj, start_time, end_time):  # noqa: ANN001
        self._record(kwargs, response_obj, start_time, end_time, status="error")

    async def async_log_success_event(self, kwargs, response_obj, start_time, end_time):  # noqa: ANN001
        self._record(kwargs, response_obj, start_time, end_time, status="success")

    async def async_log_failure_event(self, kwargs, response_obj, start_time, end_time):  # noqa: ANN001
        self._record(kwargs, response_obj, start_time, end_time, status="error")


def install_cost_tracker(store: Store) -> CostTracker:
    """Register a single CostTracker in litellm's global callbacks (idempotent)."""
    tracker = CostTracker(store)
    existing = [cb for cb in litellm.callbacks if not isinstance(cb, CostTracker)]
    litellm.callbacks = [*existing, tracker]
    # litellm copies `callbacks` into its per-event lists on the first call; drop a previous
    # tracker from those too, or it would keep receiving (and storing) every call
    for name in (
        "success_callback", "failure_callback", "_async_success_callback",
        "_async_failure_callback",
    ):  # fmt: skip
        hooks = getattr(litellm, name, None)
        if isinstance(hooks, list):
            hooks[:] = [cb for cb in hooks if not isinstance(cb, CostTracker)]
    return tracker


# completion helper, and the tasks a finished (or failed) stream schedules for its callbacks
_LITELLM_LOGGING_TASKS = {
    "_client_async_logging_helper",
    "Logging.dispatch_success_handlers",
    "Logging.dispatch_failure_handlers",
}


def _pending_logging_tasks() -> list[asyncio.Task]:
    current = asyncio.current_task()
    return [
        t
        for t in asyncio.all_tasks()
        if t is not current
        and not t.done()
        and getattr(t.get_coro(), "__qualname__", "") in _LITELLM_LOGGING_TASKS
    ]


async def flush_llm_callbacks(timeout: float = 30.0) -> None:
    """Wait until litellm has delivered every success callback for finished calls.

    litellm schedules a helper task when ``acompletion`` returns; that task enqueues the
    success handler on the global logging worker, which runs it later. Wait for both hops.
    """
    from litellm.litellm_core_utils.logging_worker import GLOBAL_LOGGING_WORKER

    async def drain() -> None:
        for _ in range(20):
            await asyncio.sleep(0)
            helpers = _pending_logging_tasks()
            if helpers:
                await asyncio.gather(*helpers, return_exceptions=True)
            await GLOBAL_LOGGING_WORKER.flush()
            if not helpers and not _pending_logging_tasks():
                return

    try:
        await asyncio.wait_for(drain(), timeout=timeout)
    except TimeoutError:
        logger.warning("Timed out waiting for litellm callbacks to flush")
    except Exception:
        logger.exception("Failed to flush litellm callbacks")
