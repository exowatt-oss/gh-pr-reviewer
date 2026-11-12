"""Bounded codebase context: full modified files, siblings of new modules, task runner."""

from __future__ import annotations

import asyncio

from app.pr_agent_config import compose_extra_instructions, reviewable_filter
from app.repo_context import MAX_DIFF_CHARS_FOR_CONTEXT, build_repo_context
from tests.fakes import FakeGitHub

REPO = "sample-org/example-project"
HEAD = "h" * 40


def build(gh: FakeGitHub, files: list[dict], **kw):
    return asyncio.run(
        build_repo_context(
            gh, REPO, files, head_sha=HEAD, base_ref="main",
            is_reviewable=reviewable_filter(), **kw,
        )
    )  # fmt: skip


def f(name: str, status: str = "modified", changes: int = 5, patch: str = "@@ +x") -> dict:
    return {"filename": name, "status": status, "changes": changes, "patch": patch}


def test_full_modified_files_but_not_added_ignored_or_docs():
    gh = FakeGitHub()
    gh.files[(REPO, "backend/src/app/services/people.py", HEAD)] = 'Person.status == "ACTIVE"'
    gh.files[(REPO, "backend/tests/test_people.py", HEAD)] = "def test(): ..."
    gh.files[(REPO, "backend/src/app/new.py", HEAD)] = "NEW"
    ctx = build(
        gh,
        [
            f("backend/tests/test_people.py", changes=50),
            f("backend/src/app/services/people.py"),
            f("backend/src/app/new.py", status="added"),
            f("frontend/package-lock.json", changes=900),
        ],
    )
    paths = [path for _, path, _ in ctx.sections]
    assert paths == ["backend/src/app/services/people.py"]  # tests stay diff-only
    text = ctx.render(HEAD, "main")
    assert 'Person.status == "ACTIVE"' in text and "full file at PR head `hhhhhhhhhhhh`" in text
    assert "package-lock" not in text and "NEW" not in text


def test_siblings_of_an_added_module_show_house_conventions():
    gh = FakeGitHub()
    gh.dirs[(REPO, "backend/src/app/routers", "main")] = [
        {"type": "file", "path": "backend/src/app/routers/__init__.py", "size": 10},
        {"type": "file", "path": "backend/src/app/routers/search.py", "size": 3000},
        {"type": "file", "path": "backend/src/app/routers/health.py", "size": 900},
        {"type": "file", "path": "backend/src/app/routers/people.py", "size": 11000},
        {"type": "file", "path": "backend/src/app/routers/huge.py", "size": 90_000},
        {"type": "file", "path": "backend/src/app/routers/README.md", "size": 100},
        {"type": "dir", "path": "backend/src/app/routers/sub", "size": 0},
    ]
    for name in ("search", "health", "people"):
        gh.files[(REPO, f"backend/src/app/routers/{name}.py", "main")] = f"# {name} router"
    ctx = build(gh, [f("backend/src/app/routers/export.py", status="added")])
    assert [(k, p) for k, p, _ in ctx.sections] == [
        ("sibling", "backend/src/app/routers/people.py"),
        ("sibling", "backend/src/app/routers/search.py"),
    ]


def test_new_test_modules_get_no_siblings():
    gh = FakeGitHub()
    gh.dirs[(REPO, "backend/tests", "main")] = [
        {"type": "file", "path": "backend/tests/test_api.py", "size": 3000},
    ]
    gh.files[(REPO, "backend/tests/test_api.py", "main")] = "def test_x(): ..."
    assert build(gh, [f("backend/tests/test_new.py", status="added")]).sections == []


def test_task_runner_only_when_docs_change():
    gh = FakeGitHub()
    gh.files[(REPO, "Taskfile.yml", "main")] = "tasks:\n  migrate:test: {}\n"
    assert build(gh, [f("backend/src/app/main.py")]).sections == []
    ctx = build(gh, [f("docs/local-development.md")])
    assert [(k, p) for k, p, _ in ctx.sections] == [("task", "Taskfile.yml")]
    assert "migrate:test" in ctx.render(HEAD, "main")


def test_budget_and_large_diffs():
    gh = FakeGitHub()
    gh.files[(REPO, "a.py", HEAD)] = "a" * 25_000
    gh.files[(REPO, "b.py", HEAD)] = "b" * 25_000
    gh.files[(REPO, "c.py", HEAD)] = "c" * 25_000
    gh.files[(REPO, "big.py", HEAD)] = "x" * 31_000  # over the per-file cap
    files = [f("a.py", changes=4), f("b.py", changes=3), f("c.py", changes=2), f("big.py", 9)]
    ctx = build(gh, files)
    assert [p for _, p, _ in ctx.sections] == ["a.py", "b.py"]  # c.py would exceed 60k
    assert ctx.chars <= 60_000
    huge = [f("a.py", patch="+" * (MAX_DIFF_CHARS_FOR_CONTEXT + 1))]
    skipped = build(gh, huge)
    assert skipped.sections == [] and "large diff" in (skipped.skipped_reason or "")
    assert skipped.render(HEAD, "main") == ""


def test_fetch_failures_are_skipped():
    class Broken(FakeGitHub):
        async def get_file_text(self, repo, path, ref):
            raise RuntimeError("boom")

        async def list_dir(self, repo, path, ref):
            raise RuntimeError("boom")

    ctx = build(Broken(), [f("a.py"), f("pkg/new.py", status="added"), f("README.md")])
    assert ctx.sections == []


def test_rendered_context_lands_in_instructions():
    gh = FakeGitHub()
    gh.files[(REPO, "a.py", HEAD)] = "uses ``` fences"
    text = build(gh, [f("a.py")]).render(HEAD, "main")
    assert "~~~~" in text  # a file containing ``` gets a different fence
    composed = compose_extra_instructions("GUIDE", {}, "main", text)
    assert composed.endswith("~~~~")


def test_diff_fingerprint_ignores_position_and_ignored_files():
    from app.repo_context import diff_fingerprint

    keep = reviewable_filter()
    base = [
        {"filename": "a.py", "status": "modified", "patch": "@@ -1,2 +1,3 @@ def f\n x\n+y"},
        {"filename": "uv.lock", "status": "modified", "patch": "+lock v1"},
    ]
    rebased = [
        {"filename": "uv.lock", "status": "modified", "patch": "+lock v2"},
        {"filename": "a.py", "status": "modified", "patch": "@@ -40,2 +41,3 @@ def f\n x\n+y"},
    ]
    changed = [{"filename": "a.py", "status": "modified", "patch": "@@ -1,2 +1,3 @@\n x\n+z"}]
    assert diff_fingerprint(base, keep) == diff_fingerprint(rebased, keep)
    assert diff_fingerprint(base, keep) != diff_fingerprint(changed, keep)
    assert diff_fingerprint([{"filename": "big.py", "status": "modified"}], keep) is None
    rename = [{"filename": "b.py", "previous_filename": "a.py", "status": "renamed", "changes": 0}]
    assert diff_fingerprint(rename, keep) is not None
    removed = [{"filename": "a.py", "status": "removed"}]
    assert diff_fingerprint(removed, keep) is not None
