"""Streaming LiteLLM adapter for PR-Agent."""

from __future__ import annotations

from typing import Any

from pr_agent.algo.ai_handlers.litellm_ai_handler import LiteLLMAIHandler


class StreamingLiteLLMAIHandler(LiteLLMAIHandler):
    """Use streaming responses so long model generations remain responsive."""

    def _requires_streaming(self, model: str) -> bool:
        return True

    async def _get_completion(self, **kwargs: Any) -> Any:
        kwargs.setdefault("timeout", 900)
        kwargs.setdefault("num_retries", 0)
        return await super()._get_completion(**kwargs)
