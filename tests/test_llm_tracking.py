"""litellm callback attribution and the OpenAI-compatible provider request."""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

import pytest
from starlette_context import request_cycle_context

from app.llm_handler import StreamingLiteLLMAIHandler
from app.llm_tracking import (
    CostTracker,
    ReviewContext,
    current_review,
    flush_llm_callbacks,
    install_cost_tracker,
    usage_from_response,
)
from app.pr_agent_config import build_overrides, fresh_pr_agent_settings
from app.store import Store
from tests.test_pr_agent_config import make_settings


@pytest.fixture
def store(tmp_path) -> Store:
    return Store(tmp_path / "t.sqlite3")


def _response(prompt=1000, cached=200, completion=500, reasoning=300):
    return SimpleNamespace(
        model="test-model",
        usage=SimpleNamespace(
            prompt_tokens=prompt,
            completion_tokens=completion,
            prompt_tokens_details=SimpleNamespace(cached_tokens=cached),
            completion_tokens_details=SimpleNamespace(reasoning_tokens=reasoning),
        ),
    )


def test_usage_extraction_handles_dicts_and_missing_details():
    assert usage_from_response(_response()) == (1000, 200, 500, 300)
    assert usage_from_response({"usage": {"prompt_tokens": 5, "completion_tokens": 2}}) == (
        5,
        0,
        2,
        0,
    )
    assert usage_from_response(None) == (0, 0, 0, 0)


def test_callback_attributes_call_to_running_review(store: Store):
    tracker = CostTracker(store)
    review_id = store.start_review("o/r", 3, "u", "t", "abc", "incremental")
    ctx = ReviewContext("o/r", 3, "abc", "incremental", review_id=review_id)
    start = datetime(2026, 10, 7, tzinfo=UTC)
    kwargs = {"litellm_call_id": "call-1", "model": "test-model", "response_cost": 0.01}

    async def run():
        token = current_review.set(ctx)
        try:
            await tracker.async_log_success_event(
                kwargs, _response(), start, start + timedelta(seconds=42)
            )
            # duplicate delivery (sync + async hooks) is de-duplicated by call id
            tracker.log_success_event(kwargs, _response(), start, start + timedelta(seconds=42))
        finally:
            current_review.reset(token)

    asyncio.run(run())
    rows = store.llm_calls_for_review(review_id)
    assert len(rows) == 1
    row = rows[0]
    assert (row["repo"], row["pr_number"], row["head_sha"], row["command"]) == (
        "o/r",
        3,
        "abc",
        "incremental",
    )
    assert (
        row["prompt_tokens"],
        row["cached_tokens"],
        row["completion_tokens"],
        row["reasoning_tokens"],
    ) == (
        1000,
        200,
        500,
        300,
    )
    assert row["cost_usd"] == pytest.approx(0.01)
    assert row["litellm_cost_usd"] == pytest.approx(0.01)
    assert row["duration_s"] == pytest.approx(42)
    assert ctx.totals()["calls"] == 1


def test_failure_and_unattributed_calls_are_recorded(store: Store):
    tracker = CostTracker(store)
    now = datetime.now(UTC)
    asyncio.run(
        tracker.async_log_failure_event(
            {"litellm_call_id": "f1", "model": "test-model", "exception": RuntimeError("boom")},
            None,
            now,
            now,
        )
    )
    tracker.log_success_event(
        {"litellm_call_id": "s1", "model": "test-model"}, _response(), now, now
    )
    summary = store.cost_by_repo()
    assert summary[0]["repo"] == "(unattributed)"
    with store._conn() as conn:
        statuses = dict(conn.execute("SELECT call_id, status FROM llm_calls").fetchall())
        err = conn.execute("SELECT error FROM llm_calls WHERE call_id='f1'").fetchone()[0]
    assert statuses == {"f1": "error", "s1": "success"}
    assert "boom" in err


def test_pr_agent_sends_a_streamed_provider_request_and_records_usage(
    store: Store, fake_provider, rsa_pem
):
    """Drive PR-Agent's streaming handler against a local synthetic API."""

    settings = make_settings(
        rsa_pem,
        LLM_API_BASE=fake_provider.url,
        LLM_API_KEY="test-provider-key",
        LLM_MODEL="test-model",
        LLM_REASONING_EFFORT="high",
    )
    overrides = build_overrides(settings, private_key_pem=rsa_pem, extra_instructions="")
    pr_settings = fresh_pr_agent_settings(overrides)
    install_cost_tracker(store)
    review_id = store.start_review("sample-org/example-project", 12, "u", "t", "deadbeef", "review")
    ctx = ReviewContext("sample-org/example-project", 12, "deadbeef", "review", review_id=review_id)

    async def run():
        token = current_review.set(ctx)
        try:
            with request_cycle_context({"settings": pr_settings}):
                handler = StreamingLiteLLMAIHandler()
                response, finish = await handler.chat_completion(
                    model=pr_settings.config.model, system="system", user="diff", temperature=0.2
                )
            await flush_llm_callbacks()
            return response, finish
        finally:
            current_review.reset(token)

    response, finish = asyncio.run(run())
    assert (response, finish) == ("No issues found.", "stop")

    assert len(fake_provider.requests) == 1
    sent = fake_provider.requests[0]
    assert sent["path"] == "/v1/chat/completions"
    assert sent["headers"]["authorization"] == "Bearer test-provider-key"
    body = sent["body"]
    assert body["model"] == "test-model"
    assert body["reasoning_effort"] == "high"
    assert "temperature" not in body
    assert not {"timeout", "stream_timeout", "disable_fallbacks"} & body.keys()
    assert body["stream"] is True
    assert body["stream_options"] == {"include_usage": True}

    rows = store.llm_calls_for_review(review_id)
    assert len(rows) == 1
    row = rows[0]
    usage = fake_provider.usage
    assert row["prompt_tokens"] == usage["prompt_tokens"]
    assert row["cached_tokens"] == usage["prompt_tokens_details"]["cached_tokens"]
    assert row["reasoning_tokens"] == usage["completion_tokens_details"]["reasoning_tokens"]
    assert row["cost_usd"] == pytest.approx(0.2345)
    # LiteLLM can capture a provider-reported response cost when one is available.
    assert row["litellm_cost_usd"] == pytest.approx(0.2345)
    assert ctx.totals()["cost_usd"] == pytest.approx(row["cost_usd"])


def test_provider_errors_follow_bounded_retry_policy(fake_provider, rsa_pem):
    import openai

    from app.reviewer import ModelRequest, call_model

    fake_provider.error.append((500, "provider request failed"))
    settings = make_settings(
        rsa_pem, LLM_API_BASE=fake_provider.url, LLM_API_KEY="test-provider-key"
    )
    overrides = build_overrides(settings, private_key_pem=rsa_pem, extra_instructions="")
    request = ModelRequest(
        "u", "review", None, "system", "diff", fresh_pr_agent_settings(overrides)
    )

    with pytest.raises(openai.APIError) as excinfo:
        asyncio.run(call_model(request))

    assert not isinstance(excinfo.value, openai.APITimeoutError)
    # PR-Agent's two attempts, no hidden client-side retries inside each
    assert len(fake_provider.requests) == 2


class Logging:  # stands in for litellm's Logging: a finished stream's callback task
    def __init__(self, done: list[str]) -> None:
        self.done = done

    async def dispatch_success_handlers(self) -> None:
        await asyncio.sleep(0.05)
        self.done.append("success")


def test_flush_waits_for_a_finished_streams_callback_task():
    done: list[str] = []

    async def run():
        task = asyncio.create_task(Logging(done).dispatch_success_handlers())
        await flush_llm_callbacks(timeout=5)
        return task

    asyncio.run(run())
    assert done == ["success"]
