"""Bounded codebase context for a review: what the diff hunks alone do not show.

PR-Agent sends the model diff hunks with a few lines of surrounding code. Regressions that
are only visible against the rest of the codebase slip through: a new route without the guard
every sibling route has, a query that skips a filter every other query in the same file
applies, a status literal spelled differently from the rest of the module, a docs command
naming a task that does not exist. This module collects, within a character budget:

1. the full head version of every modified (not added) reviewable non-test source file —
   added files are already complete in the diff;
2. for added non-test source files, up to two existing sibling files of the same type from the same
   directory on the base branch (the conventions a new module should follow);
3. when documentation changes, the repo's task-runner files (``Taskfile.yml``, ``Makefile``,
   ``justfile``) so commands and task names can be checked.

Nothing is added when the diff itself is large: the context window is then better spent on
the diff, and those reviews are already the expensive ones.
"""

from __future__ import annotations

import hashlib
import logging
import posixpath
import re
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any, Protocol

logger = logging.getLogger(__name__)

# ~4 characters per token: 60k characters is roughly 15k input tokens at most.
CONTEXT_BUDGET_CHARS = 60_000
MAX_FILE_CHARS = 30_000
MAX_SIBLING_CHARS = 12_000
MAX_SIBLINGS_PER_ADDED_FILE = 2
MAX_SIBLING_DIRS = 4
# Skip context entirely when the PR's own patches exceed this (≈ 40k tokens).
MAX_DIFF_CHARS_FOR_CONTEXT = 160_000

SOURCE_EXTENSIONS = frozenset(
    {
        ".py", ".ts", ".tsx", ".js", ".jsx", ".mjs", ".cjs", ".go", ".rs", ".java", ".kt",
        ".rb", ".php", ".cs", ".swift", ".scala", ".sql", ".sh", ".vue", ".svelte",
    }
)  # fmt: skip
DOC_EXTENSIONS = frozenset({".md", ".mdx", ".rst", ".txt", ".adoc"})
TASK_RUNNER_FILES = ("Taskfile.yml", "Taskfile.yaml", "Makefile", "justfile")
_TEST_PATH = re.compile(r"(^|/)(tests?|__tests__)/|(^|/)test_[^/]*$|\.(test|spec)\.[a-z]+$")
_SKIP_SIBLINGS = re.compile(r"(^|/)(__init__\.py|index\.(ts|js)|.*\.(test|spec)\.[a-z]+)$")


class ContextSource(Protocol):
    async def get_file_text(self, repo: str, path: str, ref: str) -> str | None: ...

    async def list_dir(self, repo: str, path: str, ref: str) -> list[dict[str, Any]]: ...


@dataclass
class RepoContext:
    sections: list[tuple[str, str, str]] = field(default_factory=list)  # (kind, path, text)
    skipped_reason: str | None = None

    @property
    def chars(self) -> int:
        return sum(len(text) for _, _, text in self.sections)

    def render(self, head_sha: str, base_ref: str) -> str:
        if not self.sections:
            return ""
        labels = {
            "modified": f"full file at PR head `{head_sha[:12]}`",
            "sibling": f"existing sibling module on `{base_ref}` (not changed by this PR)",
            "task": f"task runner on `{base_ref}`",
        }
        parts = [
            "## Codebase context (read-only reference — review the diff, not these files)",
            "Unlike the diff hunks, these are complete files. Use them to check the change "
            "against the rest of the codebase: guards and dependencies sibling routes apply, "
            "filters sibling queries apply, how constants and enum/status values are spelled, "
            "and that commands and task names in docs exist. Do not report pre-existing issues "
            "in these files.",
        ]
        for kind, path, text in self.sections:
            fence = "~~~~" if "```" in text else "```"
            parts.append(f"### `{path}` — {labels[kind]}\n{fence}\n{text}\n{fence}")
        return "\n\n".join(parts)


def _ext(path: str) -> str:
    return posixpath.splitext(path)[1].lower()


def is_source(path: str) -> bool:
    return _ext(path) in SOURCE_EXTENSIONS


def is_test(path: str) -> bool:
    return bool(_TEST_PATH.search(path))


def _is_code(path: str) -> bool:
    """Source files that are not tests (tests are visible in the diff and rarely set
    conventions worth their tokens)."""
    return is_source(path) and not is_test(path)


def is_doc(path: str) -> bool:
    return _ext(path) in DOC_EXTENSIONS or path.lower().startswith("docs/")


def _status(f: dict[str, Any]) -> str:
    return str(f.get("status") or "")


# Code and diffs average ~3.5 characters per token with GPT tokenizers.
CHARS_PER_TOKEN = 3.5


def reviewable_lines(files: Iterable[dict[str, Any]], is_reviewable: Any) -> int:
    """Changed lines (additions + deletions) in files the model will see."""
    return sum(
        int(f.get("additions") or 0) + int(f.get("deletions") or 0)
        for f in files
        if is_reviewable(str(f.get("filename") or ""))
    )


def diff_fingerprint(files: Iterable[dict[str, Any]], is_reviewable: Any) -> str | None:
    """Hash of the PR's reviewable changes, independent of where the branch sits.

    Hunk headers (line numbers) are dropped, so a rebase onto a moved base, an empty commit,
    or a lockfile-only push leaves it unchanged. ``None`` when a file's patch is missing
    (GitHub omits it for binary or very large diffs): then there is nothing safe to compare.
    """
    digest = hashlib.sha256()
    for f in sorted(files, key=lambda f: str(f.get("filename") or "")):
        name = str(f.get("filename") or "")
        if not is_reviewable(name):
            continue
        status = _status(f)
        patch = f.get("patch")
        if patch is None and status != "removed":
            if status == "renamed" and not int(f.get("changes") or 0):
                patch = ""  # pure rename
            else:
                return None
        body = "\n".join(
            line for line in str(patch or "").splitlines() if not line.startswith("@@")
        )
        digest.update(f"{name}\0{f.get('previous_filename') or ''}\0{status}\0{body}\0".encode())
    return digest.hexdigest()


async def build_repo_context(
    source: ContextSource,
    repo: str,
    files: Iterable[dict[str, Any]],
    *,
    head_sha: str,
    base_ref: str,
    is_reviewable: Any,
    budget: int = CONTEXT_BUDGET_CHARS,
) -> RepoContext:
    """Collect context for the reviewable files of one PR (``files`` = GitHub PR files)."""
    reviewable = [f for f in files if is_reviewable(str(f.get("filename") or ""))]
    ctx = RepoContext()
    diff_chars = sum(len(f.get("patch") or "") for f in reviewable)
    if diff_chars > MAX_DIFF_CHARS_FOR_CONTEXT:
        ctx.skipped_reason = f"large diff ({diff_chars} chars)"
        return ctx
    remaining = budget
    seen: set[str] = set()

    def add(kind: str, path: str, text: str | None, cap: int) -> None:
        nonlocal remaining
        if not text or not text.strip() or path in seen or len(text) > min(cap, remaining):
            return
        seen.add(path)
        ctx.sections.append((kind, path, text.rstrip()))
        remaining -= len(text)

    async def fetch(path: str, ref: str) -> str | None:
        try:
            return await source.get_file_text(repo, path, ref)
        except Exception as exc:
            logger.info("Context fetch %s@%s for %s failed: %s", path, ref, repo, exc)
            return None

    # 1. modified non-test files, most-changed first; oversized files are skipped
    changed = {str(f.get("filename")) for f in reviewable}
    modified = [
        f
        for f in reviewable
        if _status(f) in ("modified", "renamed", "changed") and _is_code(str(f["filename"]))
    ]
    modified.sort(key=lambda f: -int(f.get("changes") or 0))
    for f in modified:
        if remaining <= 0:
            break
        path = str(f["filename"])
        add("modified", path, await fetch(path, head_sha), MAX_FILE_CHARS)

    # 2. siblings of added source files (once per directory)
    added_dirs: list[tuple[str, str]] = []
    for f in reviewable:
        path = str(f["filename"])
        if _status(f) == "added" and _is_code(path):
            key = (posixpath.dirname(path), _ext(path))
            if key not in added_dirs:
                added_dirs.append(key)
    for directory, ext in added_dirs[:MAX_SIBLING_DIRS]:
        if remaining <= 0:
            break
        try:
            entries = await source.list_dir(repo, directory, base_ref)
        except Exception as exc:
            logger.info("Context listing %s@%s for %s failed: %s", directory, base_ref, repo, exc)
            continue
        candidates = sorted(
            (
                e
                for e in entries
                if e.get("type") == "file"
                and _ext(str(e.get("path"))) == ext
                and str(e.get("path")) not in changed
                and not _SKIP_SIBLINGS.search(str(e.get("path")))
                and 0 < int(e.get("size") or 0) <= MAX_SIBLING_CHARS
            ),
            key=lambda e: int(e.get("size") or 0),
            reverse=True,  # the largest that fits: most conventions per file
        )
        for entry in candidates[:MAX_SIBLINGS_PER_ADDED_FILE]:
            path = str(entry["path"])
            add("sibling", path, await fetch(path, base_ref), MAX_SIBLING_CHARS)

    # 3. task runner, when docs change
    if any(is_doc(str(f["filename"])) for f in reviewable):
        for name in TASK_RUNNER_FILES:
            if remaining <= 0:
                break
            add("task", name, await fetch(name, base_ref), MAX_SIBLING_CHARS)
    return ctx
