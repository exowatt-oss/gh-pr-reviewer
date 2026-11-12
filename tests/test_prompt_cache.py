"""The tool loop's prompt-cache layout (``LLM_PROMPT_CACHE_LAYOUT``) and its rollback switch,
and the staggered starts of a review in parts.

Wire facts these tests pin (live provider probes, 2026-10-28): explicit breakpoints work only on
system-role ``input_text`` parts; user-role ``input_text`` is an HTTP 500 at the provider (and a
client-side error in litellm); a breakpoint on empty text is an HTTP 400; at most 4 cache
writes per request (3 explicit + the automatic one); ``prompt_cache_options`` turns the
automatic end-of-request breakpoint off, which later tool-loop turns rely on.
"""

from __future__ import annotations

import asyncio
import json
import re
import time
from collections.abc import Awaitable, Callable
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

import app.reviewer as reviewer_module
from app.pr_agent_config import load_style_guide
from app.repo_snapshot import RepoTools
from app.review_agent import AGENT_PROMPT, ProviderConfig, ProviderStep, run_review_agent
from app.review_policy import Mode
from app.review_prompt import (
    PART_MARKER,
    SYSTEM_PROMPT,
    PromptInputs,
    PromptLayout,
    build_prompt,
    build_user_prompt,
    style_guide_system,
)
from app.reviewer import Reviewer, call_model
from app.store import Store
from tests.fakes import make_pr
from tests.test_large_prs import BIG_PR, SMALL_BUDGET, added, one_finding_per_part
from tests.test_poller import REPO, Harness, make_settings
from tests.test_review_agent import ANSWER, FILES, SHA, TarballGitHub, snapshot_of
from tests.test_review_liveness import MERGED, set_pull, ticking

BP = {"mode": "explicit"}
# build_user_prompt(representative()) as the release before the cache layout produced it.
GOLDEN_USER = (
    (Path(__file__).parent / "golden" / "prompt_layout_off_user.txt")
    .read_text(encoding="utf-8")
    .removesuffix("\n")
)


def representative() -> PromptInputs:
    """Every section of the user prompt: visibility, description, a round-2 scope, earlier
    findings, instructions, and part 2 of 3 with the repository tools."""
    return PromptInputs(
        repo="sample-org/widgets",
        number=42,
        url="https://github.com/sample-org/widgets/pull/42",
        title="Add the widget export endpoint",
        description="Adds `GET /api/widgets/export`.\n\nCloses #7.",
        base_ref="main",
        head_ref="feat/export",
        author="octocat",
        kind="full",
        diff="## File: 'api/export.py'\n\n__new hunk__\n1 +def export():\n2 +    return []\n",
        instructions=(
            "GUIDE\n\n## Repository-specific conventions (from `main`; these override the "
            "house rubric where they conflict)\n\n### CLAUDE.md\n\nAlways use uv run.\n\n"
            "## Codebase context\n\nctx"
        ),
        since_sha="b" * 40,
        prior_findings=["`api/export.py:2` [major] Export ignores auth"],
        private=True,
        review_round=2,
        max_rounds=2,
        changed_since=["api/export.py"],
        part=2,
        part_count=3,
        part_files=["api/export.py"],
        changed_files=[
            ("api/export.py", 2, 0),
            ("api/models.py", 10, 1),
            ("tests/test_export.py", 30, 0),
        ],
        tools=True,
    )


def breakpoints(messages: list[dict[str, Any]]) -> list[int]:
    """Index of the message of every content part that carries a breakpoint."""
    return [
        i
        for i, m in enumerate(messages)
        if isinstance(m.get("content"), list)
        for part in m["content"]
        if "prompt_cache_breakpoint" in part
    ]


def assert_wire_safe(messages: list[dict[str, Any]]) -> None:
    """What the provider accepts: list content (``input_text``) only in system messages, a
    breakpoint only on non-blank text, at most 3 explicit breakpoints."""
    for m in messages:
        content = m.get("content")
        if m["role"] != "system":
            assert content is None or isinstance(content, str), m  # never user input_text
            continue
        for part in [] if isinstance(content, str) else content:
            assert part["type"] == "input_text"
            if "prompt_cache_breakpoint" in part:
                assert part["text"].strip() and part["prompt_cache_breakpoint"] == BP
    assert len(breakpoints(messages)) <= 3


# --- the layout ----------------------------------------------------------------------------
def test_flag_off_sends_todays_two_plain_messages_byte_for_byte():
    system = SYSTEM_PROMPT + AGENT_PROMPT
    layout = build_prompt(system, representative(), cache=False)
    assert build_user_prompt(representative()) == GOLDEN_USER
    assert layout.messages() == [
        {"role": "system", "content": system},
        {"role": "user", "content": GOLDEN_USER},
    ]
    assert layout.user == GOLDEN_USER and layout.chars == len(system) + len(GOLDEN_USER)


def test_cache_layout_golden():
    guide = "# Review rubric\n\nRULES"
    system = style_guide_system(SYSTEM_PROMPT + AGENT_PROMPT, f"\n{guide}\n")
    assert system == SYSTEM_PROMPT + AGENT_PROMPT + "\n\n" + guide
    assert style_guide_system("S", "  ") == "S"  # no rubric: the system text alone
    layout = build_prompt(system, representative(), cache=True)
    shared, part = GOLDEN_USER.split("\n\n## This review is in parts", 1)
    assert PART_MARKER == (
        "(End of the pull request context above. The diff to review follows in the next message.)"
    )
    assert layout.messages() == [
        {
            "role": "system",
            "content": [{"type": "input_text", "text": system, "prompt_cache_breakpoint": BP}],
        },
        {"role": "user", "content": shared},
        {
            "role": "system",
            "content": [{"type": "input_text", "text": PART_MARKER, "prompt_cache_breakpoint": BP}],
        },
        {"role": "user", "content": "## This review is in parts" + part},
    ]
    # the user text is unchanged, only split where the part begins
    assert layout.user == GOLDEN_USER
    assert_wire_safe(layout.messages())
    # one call (no parts): the part segment is the diff and the closing line
    single = build_prompt(system, replace(representative(), part=0, part_count=0), cache=True)
    assert single.shared == shared and single.part.startswith("## Diff\n\n## File: ")
    assert single.part.endswith("\n\nAnswer with the JSON object only.")


def test_no_breakpoint_on_blank_text():
    # a blank system text cannot carry a breakpoint (HTTP 400): today's plain messages
    assert PromptLayout(" \n", "shared", "part", cache=True).messages() == [
        {"role": "system", "content": " \n"},
        {"role": "user", "content": "shared\n\npart"},
    ]
    # nothing shared: no marker either, the system breakpoint alone
    messages = PromptLayout("system", "", "part", cache=True).messages()
    assert messages == [
        {
            "role": "system",
            "content": [{"type": "input_text", "text": "system", "prompt_cache_breakpoint": BP}],
        },
        {"role": "user", "content": "part"},
    ]


def test_split_budgets_the_diff_with_every_segment_of_the_prompt(tmp_path, monkeypatch):
    h = Harness(Store(tmp_path / "split.sqlite3"), TarballGitHub(), MAX_REVIEW_DIFF_TOKENS=0)
    budgets: list[int] = []
    monkeypatch.setattr(
        reviewer_module, "split_diff", lambda files, budget, patches: budgets.append(budget) or []
    )
    system = style_guide_system(SYSTEM_PROMPT + AGENT_PROMPT, "RUBRIC " * 500)
    inputs = replace(representative(), part=0, part_count=0, part_files=None)
    for cache in (True, False):
        h.reviewer.split([], {}, system, inputs, True, cache=cache)
    layered = build_prompt(system, inputs, cache=True)
    assert layered.chars == (
        len(system) + len(layered.shared) + len(PART_MARKER) + len(layered.part)
    )
    assert budgets[0] == h.reviewer._diff_budget_chars(layered.chars, True)
    # the marker message counts; the "\n\n" that joined the user text does not exist there
    assert budgets[1] - budgets[0] == len(PART_MARKER) - 2


# --- through the real tool loop and provider call --------------------------------------------
@pytest.mark.parametrize("cache", [True, False])
def test_the_tool_loop_on_the_wire_with_and_without_the_cache_layout(
    tmp_path, fake_provider, cache
):
    gh = TarballGitHub({REPO: [make_pr(sha=SHA)]})
    gh.pr_files[(REPO, 1)] = [dict(f) for f in FILES]
    gh.files[(REPO, "CLAUDE.md", "main")] = "example-project convention: always use uv run."
    settings = make_settings(LLM_API_BASE=fake_provider.url, LLM_PROMPT_CACHE_LAYOUT=cache)
    fake_provider.script.extend(
        [{"tool_calls": [("call_1", "list_files", "{}")]}, {"content": ANSWER}]
    )
    reviewer = Reviewer(settings, Store(tmp_path / "wire.sqlite3"), gh, model=call_model)
    result = asyncio.run(reviewer.review(make_pr(sha=SHA), Mode.FULL, 1, publish=False))
    assert result.status == "ok"
    first, second = (r["body"] for r in fake_provider.requests)
    guide = load_style_guide(settings).strip()
    for body in (first, second):
        assert "prompt_cache_options" not in body  # keeps the automatic breakpoint
        assert body["tool_choice"] == "auto" and body["reasoning_effort"] == "xhigh"
        assert_wire_safe(body["messages"])
    # the conversation only grows: turn 2 reads all of turn 1 from the cache
    opening = first["messages"]
    assert second["messages"][: len(opening)] == opening
    assert [m["role"] for m in second["messages"][len(opening) :]] == ["assistant", "tool"]
    if cache:
        system, shared, marker, part = opening
        assert [m["role"] for m in opening] == ["system", "user", "system", "user"]
        assert breakpoints(first["messages"]) == breakpoints(second["messages"]) == [0, 2]
        assert system["content"] == [
            {
                "type": "input_text",
                "text": SYSTEM_PROMPT + AGENT_PROMPT + "\n\n" + guide,
                "prompt_cache_breakpoint": BP,
            }
        ]
        assert marker["content"][0]["text"] == PART_MARKER
        # the repo's conventions stay in the user role, the rubric is not repeated there
        assert "example-project convention" in shared["content"] and guide not in shared["content"]
        assert part["content"].startswith("## Diff\n\n")
    else:
        system, user = opening
        assert system == {"role": "system", "content": SYSTEM_PROMPT + AGENT_PROMPT}
        text = user["content"]
        assert text.index(guide) < text.index("## Repository-specific conventions (from `main`")
        assert "prompt_cache_breakpoint" not in json.dumps(first)


def test_provider_step_signals_sent_and_first_chunk_once(tmp_path, fake_provider):
    events: list[tuple[str, int]] = []
    fake_provider.script.extend(
        [{"tool_calls": [("call_1", "list_files", "{}")]}, {"content": ANSWER}]
    )
    settings = make_settings(LLM_API_BASE=fake_provider.url)
    step = ProviderStep(
        ProviderConfig.from_settings(settings),
        on_sent=lambda: events.append(("sent", len(fake_provider.requests))),
        on_first_chunk=lambda: events.append(("chunk", len(fake_provider.requests))),
    )
    layout = PromptLayout("system", "shared", "diff", cache=True)
    out = asyncio.run(run_review_agent(layout, RepoTools(snapshot_of(tmp_path)), step))
    assert out.content == ANSWER and len(fake_provider.requests) == 2
    # before the first request went out, then once it streamed; never for later turns
    assert events == [("sent", 0), ("chunk", 1)]


def test_a_failed_first_call_still_signals_so_the_other_parts_never_wait_on_it(
    tmp_path, fake_provider
):
    import openai

    events: list[str] = []
    fake_provider.error.append((400, "prompt_cache_breakpoint must be attached to ..."))
    step = ProviderStep(
        ProviderConfig.from_settings(make_settings(LLM_API_BASE=fake_provider.url)),
        on_sent=lambda: events.append("sent"),
        on_first_chunk=lambda: events.append("chunk"),
    )
    with pytest.raises(openai.BadRequestError):
        asyncio.run(
            run_review_agent(
                PromptLayout("s", "u", "d", cache=True), RepoTools(snapshot_of(tmp_path)), step
            )
        )
    assert events == ["sent", "chunk"]


# --- in reviews ------------------------------------------------------------------------------
def test_the_system_text_is_identical_for_every_review(tmp_path):
    other = "another-org/example-tools"
    gh = TarballGitHub(
        {
            REPO: [make_pr(sha=SHA)],
            other: [make_pr(repo=other, number=7, sha="c" * 40, title="Other", base_ref="dev")],
        }
    )
    gh.pr_files[(REPO, 1)] = [dict(f) for f in FILES]
    gh.pr_files[(other, 7)] = [added("svc/new.py", 30)]
    gh.files[(REPO, "CLAUDE.md", "main")] = "example-project: always use uv run."
    gh.files[(other, "AGENTS.md", "dev")] = "example-project: Dagger only."
    h = Harness(Store(tmp_path / "two.sqlite3"), gh, REVIEW_REPOS=f"{REPO},{other}")
    layouts: dict[str, PromptLayout] = {}

    def capture(request) -> str:
        layouts[request.pr_url] = request.layout
        return ANSWER

    h.agent.response = capture
    h.poll()
    a = layouts[f"https://github.com/{REPO}/pull/1"]
    b = layouts[f"https://github.com/{other}/pull/7"]
    assert a.cache and b.cache and a.shared != b.shared
    assert a.system == b.system and a.messages()[0] == b.messages()[0]
    assert (
        "example-project: always use uv run." in a.shared
        and "example-project: Dagger only." in b.shared
    )
    assert "example-project: always" not in a.system and "Dagger only" not in a.system  # user role


def parts_harness(tmp_path: Path, **settings: Any) -> tuple[TarballGitHub, Harness]:
    """A PR too large for one call (tests' budget), reviewed with the repository tools."""
    gh = TarballGitHub({REPO: [make_pr(sha=SHA)]})
    gh.pr_files[(REPO, 1)] = [dict(f) for f in BIG_PR]
    h = Harness(Store(tmp_path / "parts.sqlite3"), gh, **(SMALL_BUDGET | settings))
    return gh, h


def test_every_part_shares_the_system_and_shared_text_and_only_the_part_differs(tmp_path):
    _, h = parts_harness(tmp_path)
    layouts: list[PromptLayout] = []

    def capture(request) -> str:
        layouts.append(request.layout)
        return json.dumps(one_finding_per_part(request))

    h.agent.response = capture
    h.poll()
    n = len(layouts)
    assert n >= 2 and all(layout.cache for layout in layouts)
    assert len({layout.system for layout in layouts}) == 1
    assert len({layout.shared for layout in layouts}) == 1
    assert len({layout.part for layout in layouts}) == n
    first = layouts[0].messages()
    for layout in layouts:
        assert layout.messages()[:3] == first[:3]  # byte-identical cached prefix
        assert "in parts" not in layout.shared and "## Diff" not in layout.shared
        assert re.match(rf"## This review is in parts: part \d of {n}\n", layout.part)
    assert h.store.recent_reviews(1)[0]["status"] == "ok"


def test_without_tools_the_review_keeps_todays_prompt(tmp_path):
    gh, h = parts_harness(tmp_path)
    gh.pr_files[(REPO, 1)] = [dict(f) for f in FILES]
    gh.tarball_error = RuntimeError("codeload 502")
    h.poll()
    [call] = h.agent.invocations
    assert call["system"] == SYSTEM_PROMPT  # diff-only: no breakpoints, rubric in the user text
    assert "Review rubric" in call["user"]


def part_of(request: Any) -> int:
    found = re.search(r"part (\d+) of \d+", request.user)
    return int(found.group(1)) if found else 1


class Staged:
    """One part's tool loop as ``ProviderStep`` drives it: the checkpoint, the request sent,
    then streaming. Part 1 runs ``lead`` between being sent and streaming (``None`` = streams
    at once). Ordering is asserted with events (``until_sent``), never with sleep margins."""

    def __init__(self, lead: Callable[[Staged, Any], Awaitable[None]] | None = None) -> None:
        self.lead = lead
        self.log: list[tuple[str, int]] = []
        self.at: dict[tuple[str, int], float] = {}
        self.requests: list[Any] = []
        self._sent: dict[int, asyncio.Event] = {}

    def mark(self, what: str, part: int) -> None:
        self.log.append((what, part))
        self.at[(what, part)] = time.monotonic()
        if what == "sent":
            self._sent.setdefault(part, asyncio.Event()).set()

    async def until_sent(self, part: int) -> None:
        """Until ``part`` was sent (a TimeoutError fails part 1, so the test, if it never is)."""
        await asyncio.wait_for(self._sent.setdefault(part, asyncio.Event()).wait(), 5)

    async def __call__(self, request: Any) -> str:
        part = part_of(request)
        self.requests.append(request)
        if request.checkpoint is not None:
            await request.checkpoint()
        self.mark("sent", part)
        if request.on_sent is not None:
            request.on_sent()
        if part == 1 and self.lead is not None:
            await self.lead(self, request)
        if request.on_first_chunk is not None:
            request.on_first_chunk()  # one-shot in ProviderStep; a repeat is harmless here
        await asyncio.sleep(0)
        self.mark("end", part)
        return json.dumps(one_finding_per_part(request))


def staged_review(tmp_path: Path, lead: Any, *, max_wait: float = 30.0, **settings: Any):
    gh, h = parts_harness(tmp_path, **settings)
    model = Staged(lead)
    h.reviewer.model = model
    h.reviewer.PART_STAGGER_MAX_S = max_wait
    started = time.monotonic()
    h.poll()
    return gh, h, model, time.monotonic() - started


def test_parts_start_once_part_one_streams_without_waiting_for_its_answer(tmp_path):
    async def lead(model: Staged, request: Any) -> None:
        await asyncio.sleep(0.05)  # prefill: time for a part that does not wait to go first
        model.mark("chunk", 1)
        request.on_first_chunk()  # the prompt is cached by the first chunk
        await model.until_sent(2)  # still answering when part 2 starts

    _, h, model, _ = staged_review(tmp_path, lead)
    log = model.log
    later = [p for what, p in log if what == "sent" and p > 1]
    assert later and log[0] == ("sent", 1)
    assert all(log.index(("sent", p)) > log.index(("chunk", 1)) for p in later)
    assert log.index(("sent", 2)) < log.index(("end", 1))  # did not wait for the answer
    assert h.store.recent_reviews(1)[0]["status"] == "ok"


def test_parts_start_at_the_max_wait_when_part_one_does_not_stream(tmp_path):
    async def lead(model: Staged, request: Any) -> None:
        await model.until_sent(2)  # a long reasoning call: no chunk until part 2 started

    _, h, model, _ = staged_review(tmp_path, lead, max_wait=0.05)
    waited = model.at[("sent", 2)] - model.at[("sent", 1)]
    assert waited >= 0.04  # held for the max wait, then started without part 1's chunk
    assert model.log.index(("sent", 2)) < model.log.index(("end", 1))
    assert h.store.recent_reviews(1)[0]["status"] == "ok"


def test_the_stagger_never_holds_a_part_forever():
    async def waited(setup: Callable[[reviewer_module._Stagger], None]) -> float:
        stagger = reviewer_module._Stagger(0.05)
        setup(stagger)
        started = time.monotonic()
        await asyncio.wait_for(stagger.wait(), 5)
        return time.monotonic() - started

    def streaming(s):
        s.sent()
        s.first_chunk()

    async def run() -> tuple[float, ...]:
        return (
            await waited(lambda s: s.release()),  # part 1 already ended: at once
            await waited(streaming),  # part 1 already streaming: at once
            await waited(lambda s: s.sent()),  # sent, no chunk: the max wait
            await waited(lambda s: None),  # part 1 never even sent (a bug): still bounded
        )

    released, streamed, timed_out, never = asyncio.run(run())
    assert released < 0.04 and streamed < 0.04
    assert 0.04 <= timed_out < 1 and 0.09 <= never < 1


@pytest.mark.parametrize("sent", [True, False])
def test_a_failing_part_one_releases_others_but_posts_no_partial_review(tmp_path, sent):
    async def lead(model: Staged, request: Any) -> None:
        raise RuntimeError("provider 500")

    class FailsEarly(Staged):
        async def __call__(self, request: Any) -> str:
            if part_of(request) == 1:
                raise RuntimeError("connection refused")  # before anything was sent
            return await super().__call__(request)

    if sent:
        gh, h, model, elapsed = staged_review(tmp_path, lead)
    else:
        gh, h = parts_harness(tmp_path)
        model = FailsEarly()
        h.reviewer.model = model
        h.reviewer.PART_STAGGER_MAX_S = 30.0
        started = time.monotonic()
        h.poll()
        elapsed = time.monotonic() - started
    assert elapsed < 10  # never the 30 s stagger
    assert {p for what, p in model.log if what == "end"} >= {2}
    assert gh.reviews == {}
    [row] = h.store.recent_reviews()
    expected = "provider 500" if sent else "connection refused"
    assert row["status"] == "error" and expected in row["error"]
    state = h.store.get_pr_state(REPO, 1)
    assert state.failed_sha == SHA and state.failed_attempts == 1
    assert state.last_reviewed_sha is None


def test_a_cancel_during_part_one_still_cancels_the_whole_review(tmp_path):
    async def lead(model: Staged, request: Any) -> None:
        set_pull(gh, MERGED)()  # merged while part 1's first call runs
        await request.checkpoint()  # the tool loop's next turn: ReviewCancelled

    gh, h = parts_harness(tmp_path)
    model = Staged(lead)
    h.reviewer.model = model
    h.reviewer.PART_STAGGER_MAX_S = 30.0
    h.reviewer.liveness_clock = ticking()
    started = time.monotonic()
    h.poll()
    assert time.monotonic() - started < 10
    assert model.log == [("sent", 1)]  # the other parts stopped at their checkpoint
    assert gh.reviews == {} and gh.created_comments == []
    [row] = h.store.recent_reviews()
    assert (row["status"], row["skip_reason"]) == ("cancelled", "merged_during_review")


def test_flag_off_runs_the_parts_without_a_stagger(tmp_path):
    async def lead(model: Staged, request: Any) -> None:
        await model.until_sent(2)  # before part 1 streams: nothing held part 2 back

    _, h, model, _ = staged_review(tmp_path, lead, LLM_PROMPT_CACHE_LAYOUT=False)
    assert all(r.on_sent is None and not r.layout.cache for r in model.requests)
    assert all(r.system == SYSTEM_PROMPT + AGENT_PROMPT for r in model.requests)
    assert model.log.index(("sent", 2)) < model.log.index(("end", 1))
    assert h.store.recent_reviews(1)[0]["status"] == "ok"
