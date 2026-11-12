"""The review as a bounded tool loop: the model reads the diff, then checks the rest of the
repository before it answers.

Most problems GitHub Copilot catches and a diff-only review misses live outside the diff: a test
that still dereferences the workflow trigger the PR removed, a lockfile not regenerated for a
manifest change, a doc pointing at a path that does not exist, a validator elsewhere that rejects
what the new code produces, a sibling path applying a toggle the new one skips. So the model gets
read-only tools on a snapshot of the PR head (``repo_snapshot.RepoTools``) and decides what to
verify; its last turn is the usual JSON review (``review_prompt``).

Every loop is bounded: model turns, tool calls, tool output, dollars and wall-clock. When a bound
is hit the model is told to stop exploring and answer with what it has (one more call, tools
off), so a review always ends with an answer and a known worst-case cost. Calls go straight to
litellm (PR-Agent's handler has no tool calling) with the same provider controls as the rest of
the app: streaming, ``reasoning_effort``, the per-request provider timeout and no fallbacks. The
cost tracker records every turn like any other call.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any, Protocol

from app.config import Settings
from app.costs import compute_cost
from app.repo_snapshot import TOOL_SPECS, RepoTools
from app.review_prompt import PromptLayout

logger = logging.getLogger(__name__)

# One tool result, at most (the tools bound themselves too).
TOOL_RESULT_MAX_CHARS = 32_000


@dataclass(frozen=True)
class AgentLimits:
    max_rounds: int = 12  # model turns that may call tools
    max_tool_calls: int = 40
    max_tool_chars: int = 160_000  # total tool output fed back to the model
    max_cost_usd: float = 0.0  # optional local cost limit; zero disables it
    max_seconds: float = 240.0  # stop exploring after this long (the answer call is extra)

    @classmethod
    def from_settings(cls, settings: Settings) -> AgentLimits:
        return cls(
            max_rounds=max(1, settings.REVIEW_TOOL_MAX_ROUNDS),
            max_tool_calls=max(1, settings.REVIEW_TOOL_MAX_CALLS),
            max_cost_usd=settings.REVIEW_TOOL_MAX_COST_USD,
            max_seconds=float(settings.REVIEW_TOOL_MAX_SECONDS),
        )


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: str


@dataclass
class Step:
    """One model turn."""

    content: str
    tool_calls: list[ToolCall]
    usage: tuple[int, int, int, int] = (0, 0, 0, 0)  # prompt, cached, completion, reasoning
    finish_reason: str | None = None


class StepFn(Protocol):
    async def __call__(
        self, messages: list[dict[str, Any]], tools: list[dict[str, Any]], tool_choice: str
    ) -> Step: ...


@dataclass
class AgentOutcome:
    content: str
    rounds: int = 0  # model turns, the answer included
    tool_calls: int = 0
    tool_chars: int = 0
    cost_usd: float = 0.0
    stopped: str | None = None  # the bound that ended exploring early, if any
    trace: list[str] = field(default_factory=list)  # `name(args) -> N chars` per tool call


AGENT_PROMPT = """
## Tools: the repository at the PR head
You have read-only tools on the whole repository as of this PR's head commit: `read_file`,
`search` (literal or regex, optionally limited by a path glob) and `list_files`. The diff
shows what changed; the tools show what the change touches. Before answering, check the
effects outside the diff that could break, the way a careful reviewer opens the repo:
- Something removed, renamed or changed in meaning (function, field, route, CLI flag, env var,
  config key, workflow trigger/job/output, file path, status value): search for what still
  uses the old form — tests, callers, other workflows, scripts, docs.
- A changed signature, return shape or invariant: read the callers and the consumers of
  its output (validators, schemas, serializers, templates, goldens/fixtures).
- A new code path next to existing ones: read the siblings and check it applies the same
  guards, toggles, filters, permission checks and error handling.
- Dependency manifests (package.json, pyproject.toml, go.mod, Cargo.toml, ...): check that
  every lockfile in the repo (there may be several, e.g. package-lock.json and bun.lock)
  changed with them and resolves the version the PR intends; a lockfile still pinning the old
  version is a finding (frozen installs fail, or the old version ships). Check `replace` /
  `overrides` / `resolutions` entries that would keep the old version.
- CI/workflow changes: required checks still run what they should and still gate the merge
  where the repo's docs say they do; referenced actions/inputs/jobs exist.
- Docs, skills, READMEs, prompts: every path, command, task name, port and env var they
  mention exists in the repository as written.
- Migrations: the revision chain (one head, correct down revision), and what the models and
  application code expect.
Use the tools to confirm or rule out each concrete suspicion; do not report what you could
check but did not. Verify each finding's premise too: that the dependency, plugin or flag it
relies on is really installed and enabled, that the value really flows where you claim, that no
caller, wrapper or config already handles it. When the behaviour you rely on belongs to an
external tool, action, image or service whose source is not in the repository, report only
what the repository itself shows (its config, lockfile, docs or code), not assumptions. Before
claiming tests or jobs interfere with each other, search the repository for what would make
them run concurrently (`pytest-xdist` in pyproject/lockfiles, `-n`/`--numprocesses` in
pytest args or `addopts`); a CI flag like `--test-workers=N` passed to an external module is
not evidence, and without that evidence there is no finding.
Be efficient: issue independent calls together in one turn, search before reading, read only
the relevant line ranges, never re-read what you already have.
Stop exploring once your suspicions are settled. A finding must still anchor on lines of the
diff (the changed line that causes the problem); its body names the other file and line it
breaks.
"""

FINAL_NUDGE = (
    "Stop exploring now ({reason}). Answer with the JSON review object only, based on what "
    "you have verified so far."
)
JSON_NUDGE = "Answer with the JSON review object only (no prose, no code fence)."
NOT_RUN = "not run: exploring is over"
# Calls after exploring ends: the answer, and at most two more for a model that keeps calling
# tools or answers in prose.
MAX_CALLS_AFTER_STOP = 3


def _parse_args(raw: str) -> dict[str, Any]:
    try:
        value = json.loads(raw or "{}")
    except ValueError:
        return {}
    return value if isinstance(value, dict) else {}


def _describe(call: ToolCall, args: dict[str, Any]) -> str:
    shown = ", ".join(f"{k}={v!r}" for k, v in args.items() if v not in (None, "", False))
    return f"{call.name}({shown[:200]})"


def _assistant_message(step: Step) -> dict[str, Any]:
    return {
        "role": "assistant",
        "content": step.content or None,
        "tool_calls": [
            {"id": c.id, "type": "function", "function": {"name": c.name, "arguments": c.arguments}}
            for c in step.tool_calls
        ],
    }


async def run_review_agent(
    prompt: PromptLayout,
    tools: RepoTools,
    step: StepFn,
    limits: AgentLimits = AgentLimits(),
    *,
    clock: Any = time.monotonic,
    checkpoint: Callable[[], Awaitable[None]] | None = None,
) -> AgentOutcome:
    """Run the tool loop to the model's JSON answer. Never makes more than
    ``max_rounds + MAX_CALLS_AFTER_STOP`` model calls. ``checkpoint`` is awaited before every
    model call; whatever it raises ends the loop (``review_liveness``: the PR moved on).
    The conversation opens with ``prompt.messages()`` and only grows by appending (plain
    strings), so each turn's prompt is the previous one plus the new messages."""
    messages: list[dict[str, Any]] = prompt.messages()
    out = AgentOutcome(content="")
    started = clock()
    after_stop = 0  # calls made since exploring ended
    json_retry = False
    while True:
        exploring = out.stopped is None
        if not exploring:
            after_stop += 1
        # The first call after the stop keeps tool_choice "auto": changing it invalidates the
        # provider's prompt cache and re-bills the whole conversation at the fresh-input rate.
        # Only a model that keeps calling tools after being told to stop gets "none".
        choice = "auto" if exploring or after_stop == 1 else "none"
        if checkpoint is not None:
            await checkpoint()
        result = await step(messages, TOOL_SPECS, choice)
        out.rounds += 1
        prompt, cached, completion, _reasoning = result.usage
        out.cost_usd += compute_cost(prompt, cached, completion)
        content = (result.content or "").strip()
        if result.tool_calls and not exploring:
            if after_stop >= MAX_CALLS_AFTER_STOP:
                out.content = content
                return out
            messages.append(_assistant_message(result))
            for call in result.tool_calls:
                messages.append({"role": "tool", "tool_call_id": call.id, "content": NOT_RUN})
            messages.append({"role": "user", "content": FINAL_NUDGE.format(reason=out.stopped)})
            continue
        if not result.tool_calls:
            if "{" not in content and not json_retry and after_stop < MAX_CALLS_AFTER_STOP:
                # prose or nothing instead of the review: ask once more
                json_retry = True
                out.stopped = out.stopped or "answered without JSON"
                if content:
                    messages.append({"role": "assistant", "content": content})
                messages.append({"role": "user", "content": JSON_NUDGE})
                continue
            out.content = content
            return out
        messages.append(_assistant_message(result))
        for call in result.tool_calls:
            if out.tool_calls >= limits.max_tool_calls:
                text = "not run: tool-call budget exhausted"
            else:
                args = _parse_args(call.arguments)
                text = await asyncio.to_thread(tools.call, call.name, args)
                out.tool_calls += 1
                room = max(limits.max_tool_chars - out.tool_chars, 0)
                limit = min(TOOL_RESULT_MAX_CHARS, room)
                if len(text) > limit:
                    text = text[:limit] + "\n[... output truncated: tool output budget]"
                out.tool_chars += len(text)
                out.trace.append(f"{_describe(call, args)} -> {len(text)} chars")
            messages.append({"role": "tool", "tool_call_id": call.id, "content": text})
        elapsed = clock() - started
        if out.rounds >= limits.max_rounds:
            out.stopped = f"{limits.max_rounds} tool rounds used"
        elif out.tool_calls >= limits.max_tool_calls:
            out.stopped = f"{limits.max_tool_calls} tool calls used"
        elif out.tool_chars >= limits.max_tool_chars:
            out.stopped = "tool output budget used"
        elif limits.max_cost_usd > 0 and out.cost_usd >= limits.max_cost_usd:
            out.stopped = f"cost cap ${limits.max_cost_usd:.2f} reached"
        elif elapsed >= limits.max_seconds:
            out.stopped = f"time budget {limits.max_seconds:.0f}s used"
        if out.stopped:
            messages.append({"role": "user", "content": FINAL_NUDGE.format(reason=out.stopped)})


# --- OpenAI-compatible model provider -------------------------------------------------


@dataclass(frozen=True)
class ProviderConfig:
    model: str
    api_base: str | None
    api_key: str
    reasoning_effort: str
    client_timeout_s: int

    @classmethod
    def from_settings(cls, settings: Settings) -> ProviderConfig:
        api_base = settings.LLM_API_BASE.strip().rstrip("/")
        if api_base and not api_base.endswith("/v1"):
            api_base = f"{api_base}/v1"
        return cls(
            model=f"openai/{settings.LLM_MODEL}",
            api_base=api_base or None,
            api_key=settings.LLM_API_KEY.get_secret_value(),
            reasoning_effort=settings.LLM_REASONING_EFFORT,
            client_timeout_s=settings.llm_client_timeout_seconds,
        )


def _usage(response: Any) -> tuple[int, int, int, int]:
    from app.llm_tracking import usage_from_response

    return usage_from_response(response)


class ProviderStep:
    """One streamed model completion with tools.

    ``on_sent`` / ``on_first_chunk`` fire once, around the first call only. A review in parts
    uses them to start its other parts once the provider has accepted the shared prompt."""

    def __init__(
        self,
        config: ProviderConfig,
        *,
        on_sent: Callable[[], None] | None = None,
        on_first_chunk: Callable[[], None] | None = None,
    ) -> None:
        self.config = config
        self._on_sent = on_sent
        self._on_first_chunk = on_first_chunk

    async def __call__(
        self, messages: list[dict[str, Any]], tools: list[dict[str, Any]], tool_choice: str
    ) -> Step:
        import litellm

        cfg = self.config
        on_sent, self._on_sent = self._on_sent, None
        on_chunk, self._on_first_chunk = self._on_first_chunk, None
        chunks: list[Any] = []
        try:
            if on_sent is not None:
                on_sent()
            request: dict[str, Any] = {
                "model": cfg.model,
                "api_base": cfg.api_base,
                "api_key": cfg.api_key,
                "messages": messages,
                "tools": tools,
                "tool_choice": tool_choice,
                "stream": True,
                "stream_options": {"include_usage": True},
                "timeout": cfg.client_timeout_s,
                "num_retries": 0,
                "max_retries": 0,
            }
            if cfg.reasoning_effort:
                request["reasoning_effort"] = cfg.reasoning_effort
                request["allowed_openai_params"] = ["reasoning_effort"]
            stream = await litellm.acompletion(**request)
            async for chunk in stream:
                if on_chunk is not None:
                    on_chunk, signal = None, on_chunk
                    signal()
                chunks.append(chunk)
        finally:
            if on_chunk is not None:
                on_chunk()
        response = litellm.stream_chunk_builder(chunks, messages=messages)
        if response is None:
            return Step(content="", tool_calls=[])
        choice = response.choices[0]
        message = choice.message
        calls = [
            ToolCall(
                id=str(tc.id),
                name=str(tc.function.name or ""),
                arguments=str(tc.function.arguments or "{}"),
            )
            for tc in (getattr(message, "tool_calls", None) or [])
        ]
        return Step(
            content=str(message.content or ""),
            tool_calls=calls,
            usage=_usage(response),
            finish_reason=choice.finish_reason,
        )
