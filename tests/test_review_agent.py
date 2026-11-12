"""Repository tools for the reviewer: the head snapshot, the tools on it, the bounded tool loop
and its provider calls, and how a review uses (and always cleans up) the snapshot."""

from __future__ import annotations

import asyncio
import io
import json
import tarfile
from pathlib import Path
from typing import Any

import pytest

import app.repo_snapshot as repo_snapshot
from app.costs import compute_cost
from app.llm_tracking import (
    ReviewContext,
    current_review,
    flush_llm_callbacks,
    install_cost_tracker,
)
from app.pr_agent_config import load_style_guide
from app.repo_snapshot import (
    RepoSnapshot,
    RepoTools,
    SnapshotError,
    build_snapshot,
    clear_stale_snapshots,
    extract_tarball,
)
from app.review_agent import (
    AGENT_PROMPT,
    FINAL_NUDGE,
    AgentLimits,
    ProviderConfig,
    ProviderStep,
    Step,
    ToolCall,
    run_review_agent,
)
from app.review_prompt import SYSTEM_PROMPT, PromptLayout
from app.store import Store
from tests.fakes import FakeGitHub, make_pr
from tests.test_poller import REPO, Harness, make_settings

SHA = "a" * 40
PREFIX = "sample-org-example-project-aaaaaaa/"
REPO_FILES = {
    "README.md": "# example-project\nRun `task test`.\n",
    "src/app/main.py": (
        "def handler():\n    return check_auth()\n\n\ndef check_auth():\n    return True\n"
    ),
    "src/app/util.py": "VALUE = 1\n",
    "tests/test_main.py": (
        "from app.main import handler\n\n\ndef test_handler():\n    assert handler()\n"
    ),
    ".github/workflows/ci.yml": "on:\n  schedule:\n    - cron: '0 1 * * *'\n",
}


def tarball(files: dict[str, bytes | str], *, extra: list[tarfile.TarInfo] | None = None) -> bytes:
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        root = tarfile.TarInfo(PREFIX.rstrip("/"))
        root.type = tarfile.DIRTYPE
        tar.addfile(root)
        for name, data in files.items():
            raw = data.encode() if isinstance(data, str) else data
            info = tarfile.TarInfo(PREFIX + name)
            info.size = len(raw)
            tar.addfile(info, io.BytesIO(raw))
        for info in extra or []:
            tar.addfile(info, io.BytesIO(b"x" * info.size) if info.isfile() else None)
    return buf.getvalue()


class TarballGitHub(FakeGitHub):
    """FakeGitHub that also serves the head tarball."""

    def __init__(self, *args: Any, archive: bytes | None = None, **kw: Any) -> None:
        super().__init__(*args, **kw)
        self.archive = archive if archive is not None else tarball(REPO_FILES)
        self.tarball_error: BaseException | None = None
        self.tarball_requests: list[tuple[str, str]] = []

    async def download_tarball(self, repo: str, ref: str, dest: Path, max_bytes: int) -> int:
        self.tarball_requests.append((repo, ref))
        if self.tarball_error is not None:
            raise self.tarball_error
        dest.write_bytes(self.archive)
        return len(self.archive)


def snapshot_of(tmp_path: Path, files: dict[str, str] | None = None) -> RepoSnapshot:
    archive = tmp_path / "repo.tar.gz"
    archive.write_bytes(tarball(files or REPO_FILES))
    dest = tmp_path / "tree"
    dest.mkdir()
    return extract_tarball(archive, dest, REPO, SHA)


# --- the snapshot --------------------------------------------------------------------------
def test_extract_keeps_text_files_and_skips_binaries_links_escapes_and_large(
    tmp_path: Path, monkeypatch
):
    monkeypatch.setattr(repo_snapshot, "MAX_FILE_BYTES", 64)
    link = tarfile.TarInfo(PREFIX + "passwd")
    link.type, link.linkname = tarfile.SYMTYPE, "/etc/passwd"
    escape = tarfile.TarInfo(PREFIX + "../../evil.py")
    escape.size = 4
    archive = tmp_path / "repo.tar.gz"
    archive.write_bytes(
        tarball(
            {
                "src/a.py": "print(1)\n",
                "logo.png": b"\x89PNG\r\n",
                "data.dat": b"ab\0cd",
                "node_modules/x/index.js": "module.exports = 1\n",
                "big.txt": "y" * 65,
            },
            extra=[link, escape],
        )
    )
    dest = tmp_path / "tree"
    dest.mkdir()
    snap = extract_tarball(archive, dest, REPO, SHA)
    assert snap.files == ["src/a.py"]
    assert snap.skipped_large == ["big.txt"]
    assert snap.size_bytes == len("print(1)\n")
    assert not (tmp_path / "evil.py").exists() and not (dest / "passwd").exists()
    assert sorted(p.name for p in tmp_path.iterdir()) == ["repo.tar.gz", "tree"]


@pytest.mark.parametrize(("limit", "value"), [("MAX_FILES", 2), ("MAX_EXTRACTED_BYTES", 30)])
def test_extract_refuses_a_repository_over_the_limits(tmp_path: Path, monkeypatch, limit, value):
    monkeypatch.setattr(repo_snapshot, limit, value)
    with pytest.raises(SnapshotError, match="too large"):
        snapshot_of(tmp_path)


def test_extract_rejects_a_corrupt_archive(tmp_path: Path):
    archive = tmp_path / "repo.tar.gz"
    archive.write_bytes(b"not a tarball")
    with pytest.raises(SnapshotError, match="bad tarball"):
        extract_tarball(archive, tmp_path, REPO, SHA)


def test_build_snapshot_downloads_extracts_and_cleanup_removes_everything(tmp_path: Path):
    gh = TarballGitHub()
    parent = tmp_path / "snapshots"
    snap = asyncio.run(build_snapshot(gh, REPO, SHA, parent))
    assert gh.tarball_requests == [(REPO, SHA)]
    assert snap.files == sorted(REPO_FILES)
    assert (snap.root / "src/app/main.py").read_text() == REPO_FILES["src/app/main.py"]
    assert not list(parent.glob("*/repo.tar.gz"))  # the archive goes once extracted
    snap.cleanup()
    assert list(parent.iterdir()) == []


def test_build_snapshot_leaves_nothing_behind_when_the_download_fails(tmp_path: Path):
    gh = TarballGitHub()
    gh.tarball_error = RuntimeError("502 from codeload")
    parent = tmp_path / "snapshots"
    with pytest.raises(RuntimeError):
        asyncio.run(build_snapshot(gh, REPO, SHA, parent))
    assert list(parent.iterdir()) == []


def test_clear_stale_snapshots_removes_old_dirs_only(tmp_path: Path):
    (tmp_path / "old").mkdir()
    (tmp_path / "keep.txt").write_text("x")
    clear_stale_snapshots(tmp_path, older_than_s=0)
    assert [p.name for p in tmp_path.iterdir()] == ["keep.txt"]
    clear_stale_snapshots(tmp_path / "missing")  # no error


# --- the tools -----------------------------------------------------------------------------
def test_read_file_numbers_lines_honours_ranges_and_says_how_to_read_on(
    tmp_path: Path, monkeypatch
):
    tools = RepoTools(snapshot_of(tmp_path))
    out = tools.read_file("src/app/main.py", 2, 3)
    assert out.splitlines()[0] == "src/app/main.py (lines 2-3 of 6)"
    assert "    2     return check_auth()" in out
    assert "start_line=4" in out
    monkeypatch.setattr(repo_snapshot, "READ_MAX_LINES", 2)
    assert "(lines 1-2 of 6)" in tools.read_file("./src/app/main.py")


def test_read_file_points_at_similar_paths_and_refuses_escapes(tmp_path: Path):
    tools = RepoTools(snapshot_of(tmp_path))
    missing = tools.read_file("src/app/mian.py")
    assert "does not exist" in missing and "src/app/main.py" in missing
    assert tools.read_file("../../etc/passwd").startswith("invalid path")
    assert tools.read_file("src/app").startswith("src/app/ (2 entries)")


def test_search_literal_regex_glob_case_and_bad_regex(tmp_path: Path):
    tools = RepoTools(snapshot_of(tmp_path))
    out = tools.search("check_auth")
    assert out.startswith("2 matching lines in 1 files")
    assert "src/app/main.py:2: return check_auth()" in out
    assert "tests/test_main.py:1:" in tools.search(r"from app\.\w+ import", regex=True)
    assert tools.search("HANDLER", ignore_case=True).startswith("4 matching lines in 2 files")
    assert "tests/" not in tools.search("handler", path_glob="src/**/*.py")
    assert tools.search("schedule", path_glob="*.yml").startswith("1 matching lines")
    assert tools.search("[", regex=True).startswith("invalid regex")
    assert tools.search("nowhere-to-be-found").startswith("no matches")


def test_search_caps_its_output(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(repo_snapshot, "SEARCH_MAX_MATCHES", 2)
    files = {f"f{i}.py": "x = 1\n" for i in range(5)}
    out = RepoTools(snapshot_of(tmp_path, files)).search("x = 1")
    assert out.splitlines()[0] == "5 matching lines in 5 files (showing 2; narrow with path_glob)"
    assert len(out.splitlines()) == 3


def test_list_files_and_unknown_tools_and_bad_arguments(tmp_path: Path):
    tools = RepoTools(snapshot_of(tmp_path))
    root = tools.list_files("")
    assert root.splitlines()[0] == "./ (4 entries)"
    assert ".github/" in root and "README.md" in root
    assert tools.list_files("", "*.py").startswith("3 files match '*.py'")
    assert "does not exist" in tools.list_files("nope")
    assert tools.call("rm_rf", {}) == "unknown tool 'rm_rf'"
    assert "(lines 1-1 of 1)" in tools.call("read_file", {"path": "src/app/util.py",
                                                          "start_line": "x"})  # fmt: skip
    assert tools.call("search", {}) == "empty query"


# --- the tool loop -------------------------------------------------------------------------
ANSWER = json.dumps({"overview": "ok", "files": [], "findings": []})


class ScriptedModel:
    """A model that calls tools while allowed (per ``plan``) and otherwise answers."""

    def __init__(self, plan: list[list[tuple[str, dict]]], *, usage=(1_000, 0, 100, 0),
                 answer: str = ANSWER, insist: bool = False) -> None:  # fmt: skip
        self.plan = list(plan)
        self.usage = usage
        self.answer = answer
        self.insist = insist  # keeps calling tools even when told not to
        self.calls: list[tuple[list[dict[str, Any]], str]] = []

    async def __call__(self, messages, tools, tool_choice) -> Step:
        self.calls.append(([dict(m) for m in messages], tool_choice))
        assert {t["function"]["name"] for t in tools} == {"read_file", "search", "list_files"}
        if (tool_choice == "auto" or self.insist) and self.plan:
            batch = self.plan.pop(0) if not self.insist else self.plan[0]
            calls = [
                ToolCall(f"c{len(self.calls)}-{i}", name, json.dumps(args))
                for i, (name, args) in enumerate(batch)
            ]
            return Step(content="", tool_calls=calls, usage=self.usage)
        return Step(content=self.answer, tool_calls=[], usage=self.usage)


def run(model: ScriptedModel, tools: RepoTools, limits: AgentLimits = AgentLimits(), **kw):
    return asyncio.run(
        run_review_agent(PromptLayout("system", "", "diff"), tools, model, limits, **kw)
    )


def test_tool_loop_feeds_tool_results_back_and_returns_the_json_answer(tmp_path: Path):
    tools = RepoTools(snapshot_of(tmp_path))
    model = ScriptedModel([[("search", {"query": "check_auth"}),
                            ("read_file", {"path": "tests/test_main.py"})]])  # fmt: skip
    out = run(model, tools)
    assert out.content == ANSWER
    assert (out.rounds, out.tool_calls, out.stopped) == (2, 2, None)
    assert out.cost_usd == pytest.approx(2 * compute_cost(1_000, 0, 100))
    messages, choice = model.calls[1]
    assert choice == "auto"
    assistant, first, second = messages[2:]
    assert [c["function"]["name"] for c in assistant["tool_calls"]] == ["search", "read_file"]
    assert first["role"] == "tool" and first["tool_call_id"] == "c1-0"
    assert "src/app/main.py:2: return check_auth()" in first["content"]
    assert "def test_handler():" in second["content"]
    assert out.trace[0] == "search(query='check_auth') -> " + f"{len(first['content'])} chars"


class Obedient(ScriptedModel):
    """Stops calling tools once told to (the last message is the stop nudge)."""

    async def __call__(self, messages, tools, tool_choice) -> Step:
        if messages[-1]["role"] == "user" and messages[-1]["content"].startswith("Stop"):
            self.calls.append(([dict(m) for m in messages], tool_choice))
            return Step(content=self.answer, tool_calls=[], usage=self.usage)
        return await super().__call__(messages, tools, tool_choice)


def test_round_bound_asks_for_the_answer_keeping_the_prompt_cache(tmp_path: Path):
    tools = RepoTools(snapshot_of(tmp_path))
    model = Obedient([[("list_files", {})]] * 10)
    out = run(model, tools, AgentLimits(max_rounds=3))
    assert out.content == ANSWER
    # tool_choice stays "auto": changing it would re-bill the whole cached conversation
    assert [choice for _, choice in model.calls] == ["auto", "auto", "auto", "auto"]
    assert out.stopped == "3 tool rounds used"
    last_messages = model.calls[-1][0]
    assert last_messages[-1] == {
        "role": "user",
        "content": FINAL_NUDGE.format(reason="3 tool rounds used"),
    }


@pytest.mark.parametrize(
    ("limits", "usage", "stopped"),
    [
        (AgentLimits(max_cost_usd=0.25), (20_000, 0, 1_000, 0), "cost cap $0.25 reached"),
        (AgentLimits(max_tool_calls=2), (1_000, 0, 100, 0), "2 tool calls used"),
        (AgentLimits(max_tool_chars=50), (1_000, 0, 100, 0), "tool output budget used"),
    ],
)
def test_cost_call_and_output_bounds_stop_exploring(
    tmp_path: Path, limits, usage, stopped, monkeypatch: pytest.MonkeyPatch
):
    if limits.max_cost_usd:
        monkeypatch.setattr("app.review_agent.compute_cost", lambda *args: 0.3)
    tools = RepoTools(snapshot_of(tmp_path))
    batch = [("read_file", {"path": "README.md"})] * 3
    model = Obedient([batch] * 5, usage=usage)
    out = run(model, tools, limits)
    assert out.stopped == stopped
    assert [choice for _, choice in model.calls] == ["auto", "auto"]
    results = [m for m in model.calls[1][0] if m["role"] == "tool"]
    if limits.max_tool_calls == 2:
        assert results[2]["content"] == "not run: tool-call budget exhausted"
        assert out.tool_calls == 2
    if limits.max_tool_chars == 50:
        assert out.tool_chars <= 50 + 3 * len("\n[... output truncated: tool output budget]")
        assert results[0]["content"].endswith("[... output truncated: tool output budget]")


def test_time_bound_stops_exploring(tmp_path: Path):
    tools = RepoTools(snapshot_of(tmp_path))
    ticks = iter([0.0, 10.0, 400.0])
    model = Obedient([[("list_files", {})]] * 5)
    out = run(model, tools, AgentLimits(max_seconds=360), clock=lambda: next(ticks))
    assert out.stopped == "time budget 360s used"
    assert len(model.calls) == 3


def test_prose_answer_gets_one_json_retry(tmp_path: Path):
    class Prose(ScriptedModel):
        async def __call__(self, messages, tools, tool_choice) -> Step:
            step = await super().__call__(messages, tools, tool_choice)
            if len(self.calls) == 1:
                return Step(content="Looks fine to me.", tool_calls=[], usage=self.usage)
            return step

    model = Prose([])
    out = run(model, RepoTools(snapshot_of(tmp_path)))
    assert out.content == ANSWER and out.rounds == 2
    assert out.stopped == "answered without JSON"
    assert model.calls[1][0][-2:] == [
        {"role": "assistant", "content": "Looks fine to me."},
        {"role": "user", "content": "Answer with the JSON review object only (no prose, no "
                                    "code fence)."},
    ]  # fmt: skip


def test_a_model_that_keeps_calling_tools_after_the_stop_gets_tools_off_then_ends(
    tmp_path: Path,
):
    model = ScriptedModel([[("list_files", {})]], insist=True)
    out = run(model, RepoTools(snapshot_of(tmp_path)), AgentLimits(max_rounds=2))
    # the bound, then MAX_CALLS_AFTER_STOP calls: "auto" once, then "none"
    assert [choice for _, choice in model.calls] == ["auto", "auto", "auto", "none", "none"]
    refused = [m for m in model.calls[-1][0] if m["role"] == "tool"][-1]
    assert refused["content"] == "not run: exploring is over"
    assert out.content == ""  # parse_model_output turns this into a ModelOutputError


# --- the provider calls ---------------------------------------------------------------------
def test_tool_loop_through_the_provider_streams_with_effort_and_records_every_call(
    tmp_path: Path, fake_provider
):
    store = Store(tmp_path / "agent.sqlite3")
    settings = make_settings(LLM_API_BASE=fake_provider.url, LLM_API_KEY="test-provider-key")
    fake_provider.script.extend(
        [
            {
                "tool_calls": [
                    ("call_1", "search", '{"query": "check_auth"}'),
                    ("call_2", "list_files", "{}"),
                ]
            },  # fmt: skip
            {"content": ANSWER},
        ]
    )
    install_cost_tracker(store)
    review_id = store.start_review(REPO, 1, "u", "t", SHA, "review")
    ctx = ReviewContext(REPO, 1, SHA, "review", review_id=review_id)
    tools = RepoTools(snapshot_of(tmp_path))

    async def go():
        token = current_review.set(ctx)
        try:
            step = ProviderStep(ProviderConfig.from_settings(settings))
            out = await run_review_agent(
                PromptLayout("system", "", "diff"), tools, step, AgentLimits()
            )
            await flush_llm_callbacks()
            return out
        finally:
            current_review.reset(token)

    out = asyncio.run(go())
    assert out.content == ANSWER and out.tool_calls == 2
    first, second = (r["body"] for r in fake_provider.requests)
    assert fake_provider.requests[0]["headers"]["authorization"] == "Bearer test-provider-key"
    for body in (first, second):
        assert body["model"] == "test-model"
        assert body["reasoning_effort"] == "xhigh"
        assert body["stream"] is True and body["stream_options"] == {"include_usage": True}
        assert not {"timeout", "stream_timeout", "disable_fallbacks"} & body.keys()
        assert body["tool_choice"] == "auto"
        assert [t["function"]["name"] for t in body["tools"]] == [
            "read_file", "search", "list_files"
        ]  # fmt: skip
    assistant = second["messages"][2]
    assert [c["id"] for c in assistant["tool_calls"]] == ["call_1", "call_2"]
    results = {m["tool_call_id"]: m["content"] for m in second["messages"] if m["role"] == "tool"}
    assert "src/app/main.py:2: return check_auth()" in results["call_1"]
    assert results["call_2"].startswith("./ (4 entries)")
    rows = store.llm_calls_for_review(review_id)
    assert len(rows) == 2 and ctx.totals()["calls"] == 2
    assert ctx.totals()["cost_usd"] == pytest.approx(2 * 0.2345)


def test_provider_timeout_bubbles_without_retry(tmp_path: Path, fake_provider, monkeypatch):
    import litellm

    from app.reviewer import _timed_out

    settings = make_settings(LLM_API_BASE=fake_provider.url, LLM_API_KEY="test-provider-key")
    step = ProviderStep(ProviderConfig.from_settings(settings))
    attempts = 0

    async def timeout_once(**_kwargs):
        nonlocal attempts
        attempts += 1
        raise litellm.Timeout(
            message="synthetic provider timeout", model="openai/test-model", llm_provider="openai"
        )

    monkeypatch.setattr(litellm, "acompletion", timeout_once)
    with pytest.raises(litellm.Timeout) as excinfo:
        asyncio.run(
            run_review_agent(PromptLayout("s", "", "d"), RepoTools(snapshot_of(tmp_path)), step)
        )
    assert attempts == 1 and fake_provider.requests == []
    assert _timed_out([], str(excinfo.value))


# --- in a review ---------------------------------------------------------------------------
PATCH = "@@ -1,2 +1,2 @@\n def handler():\n-    return True\n+    return check_auth()\n"
FILES = [{"filename": "src/app/main.py", "status": "modified", "patch": PATCH, "additions": 1,
          "deletions": 1}]  # fmt: skip


def tool_harness(tmp_path: Path, **settings: Any) -> tuple[TarballGitHub, Harness, list]:
    store = Store(tmp_path / "review.sqlite3")
    gh = TarballGitHub({REPO: [make_pr(sha=SHA)]})
    gh.pr_files[(REPO, 1)] = [dict(f) for f in FILES]
    h = Harness(store, gh, **settings)
    seen: list[dict[str, Any]] = []

    def answer(request) -> str:
        tools = request.tools
        seen.append(
            {
                "system": request.system,
                "user": request.user,
                "layout": request.layout,
                "tools": tools,
                "read": tools.call("read_file", {"path": "tests/test_main.py"}) if tools else None,
                "snapshot_dirs": list((tmp_path / "snapshots").glob("*")),
            }
        )
        return ANSWER

    h.agent.response = answer
    return gh, h, seen


def test_a_review_gets_tools_on_the_head_snapshot_and_removes_it_afterwards(tmp_path: Path):
    gh, h, seen = tool_harness(tmp_path)
    h.poll()
    assert gh.tarball_requests == [(REPO, SHA)]
    [call] = seen
    # The optional prompt-cache layout keeps the reviewer rubric in the shared system prefix.
    guide = load_style_guide(h.settings).strip()
    assert call["system"] == SYSTEM_PROMPT + AGENT_PROMPT + "\n\n" + guide
    assert call["layout"].cache and "Review rubric" not in call["user"]
    assert "def test_handler():" in call["read"]
    assert len(call["snapshot_dirs"]) == 1
    assert list((tmp_path / "snapshots").iterdir()) == []
    assert h.reviewer._snapshot_slots._value == h.settings.snapshot_slots
    assert h.store.recent_reviews(1)[0]["status"] == "ok"


@pytest.mark.parametrize(
    ("settings", "error"),
    [
        ({}, RuntimeError("codeload 502")),  # the download fails
        ({"REVIEW_SNAPSHOT_MIN_FREE_MB": 10**12}, None),  # not enough disk
    ],
)
def test_without_a_snapshot_the_review_runs_diff_only(tmp_path: Path, settings, error):
    gh, h, seen = tool_harness(tmp_path, **settings)
    gh.tarball_error = error
    h.poll()
    [call] = seen
    assert call["tools"] is None and AGENT_PROMPT not in call["system"]
    assert h.store.recent_reviews(1)[0]["status"] == "ok"
    assert list((tmp_path / "snapshots").iterdir()) == []
    assert h.reviewer._snapshot_slots._value == h.settings.snapshot_slots


def test_a_review_in_parts_shares_one_snapshot_and_each_part_has_the_tools(tmp_path: Path):
    from tests.test_large_prs import added

    gh, h, seen = tool_harness(tmp_path, MAX_REVIEW_DIFF_TOKENS=8_000)
    gh.pr_files[(REPO, 1)] = [added(f"src/app/m{n}.py") for n in range(6)]
    h.poll()
    assert len(seen) >= 2 and gh.tarball_requests == [(REPO, SHA)]  # one download
    assert all(c["tools"] is seen[0]["tools"] for c in seen)
    assert all("def test_handler():" in c["read"] for c in seen)
    assert all(AGENT_PROMPT in c["system"] and c["layout"].cache for c in seen)
    users = [inv["user"] for inv in h.agent.invocations]
    assert all("including the PR's other changed files as of its head" in u for u in users)
    assert list((tmp_path / "snapshots").iterdir()) == []
    assert h.reviewer._snapshot_slots._value == h.settings.snapshot_slots
    assert h.store.recent_reviews(1)[0]["status"] == "ok"


def test_tools_off_means_no_snapshot(tmp_path: Path):
    gh, h, seen = tool_harness(tmp_path, REVIEW_TOOLS=False)
    h.poll()
    assert gh.tarball_requests == [] and seen[0]["tools"] is None


def test_a_failed_model_call_still_removes_the_snapshot(tmp_path: Path):
    gh, h, _seen = tool_harness(tmp_path)
    h.agent.ok = False
    h.poll()
    assert gh.tarball_requests == [(REPO, SHA)]
    assert h.store.recent_reviews(1)[0]["status"] == "error"
    assert list((tmp_path / "snapshots").iterdir()) == []
    assert h.reviewer._snapshot_slots._value == h.settings.snapshot_slots


def test_snapshot_slots_cap_concurrent_snapshots(tmp_path: Path):
    gh, h, _seen = tool_harness(tmp_path, REVIEW_SNAPSHOT_MAX_CONCURRENT=1)
    reviewer = h.reviewer
    pr = make_pr(sha=SHA)

    async def go():
        first = await reviewer.open_snapshot(pr)
        waiter = asyncio.create_task(reviewer.open_snapshot(pr))
        await asyncio.sleep(0.05)
        assert not waiter.done()  # waits for the slot
        reviewer.close_snapshot(first)
        second = await asyncio.wait_for(waiter, 5)
        assert second is not None
        reviewer.close_snapshot(second)

    asyncio.run(go())
    assert reviewer._snapshot_slots._value == 1


def test_snapshot_status_reports_slots_and_free_disk(tmp_path: Path):
    _gh, h, _seen = tool_harness(tmp_path)
    status = h.reviewer.snapshot_status()
    assert status["enabled"] is True and status["active"] == 0
    # REVIEW_SNAPSHOT_MAX_CONCURRENT=0: one snapshot slot per review slot
    assert status["max_concurrent"] == h.settings.MAX_CONCURRENT_REVIEWS
    assert status["disk_free_mb"] > 0
