"""A read-only copy of the repository at the PR head, and the tools the reviewer uses on it.

The diff and the bounded codebase context miss problems that only show elsewhere in the repo:
a test that still asserts the removed behaviour, a lockfile not regenerated for a manifest
change, a doc pointing at a path that does not exist, a sibling code path applying a toggle the
new path skips. GitHub Copilot finds these by exploring the repository; so does this reviewer.

One review downloads the head commit's tarball (one REST call, streamed to disk under the data
dir, bounded), extracts the text files and lets the model ``read_file``, ``search`` and
``list_files`` on it through function calls (``review_agent.py``). The snapshot is deleted when
the review ends. All file work runs in a worker thread so the event loop (poller, web app)
never blocks on it.
"""

from __future__ import annotations

import asyncio
import difflib
import fnmatch
import logging
import os
import posixpath
import re
import shutil
import tarfile
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from app.pr_agent_config import BINARY_EXTENSIONS

logger = logging.getLogger(__name__)

MAX_TARBALL_BYTES = 200 * 1024 * 1024
MAX_EXTRACTED_BYTES = 400 * 1024 * 1024
MAX_FILES = 40_000
MAX_FILE_BYTES = 2 * 1024 * 1024
# Tool output bounds (characters) and search limits.
READ_MAX_LINES = 500
READ_MAX_CHARS = 30_000
SEARCH_MAX_MATCHES = 60
SEARCH_MAX_PER_FILE = 8
SEARCH_LINE_CHARS = 220
SEARCH_TIME_BUDGET_S = 8.0
LIST_MAX_ENTRIES = 300
_BINARY_EXT = frozenset(BINARY_EXTENSIONS)
_SKIP_DIRS = frozenset({".git", "node_modules", "__pycache__", ".venv", ".mypy_cache"})


class SnapshotError(Exception):
    """The snapshot could not be built (too large, download failed, bad archive)."""


@dataclass
class RepoSnapshot:
    root: Path
    repo: str
    sha: str
    files: list[str] = field(default_factory=list)  # repo-relative text files, sorted
    skipped_large: list[str] = field(default_factory=list)
    size_bytes: int = 0  # extracted text, bytes
    workdir: Path | None = None  # removed with the snapshot (holds root and the archive)

    @property
    def file_set(self) -> frozenset[str]:
        return frozenset(self.files)

    def cleanup(self) -> None:
        shutil.rmtree(self.workdir or self.root, ignore_errors=True)


def _safe_member_path(name: str) -> str | None:
    """Repo-relative path of a tarball member (GitHub prefixes ``owner-repo-sha/``)."""
    parts = name.split("/", 1)
    if len(parts) < 2 or not parts[1]:
        return None
    rel = posixpath.normpath(parts[1])
    if rel.startswith(("/", "../")) or rel == ".." or "/../" in f"/{rel}/":
        return None
    return rel


def _skip_path(rel: str) -> bool:
    if any(part in _SKIP_DIRS for part in rel.split("/")[:-1]):
        return True
    ext = posixpath.splitext(rel)[1].lower().lstrip(".")
    return ext in _BINARY_EXT


def extract_tarball(archive: Path, dest: Path, repo: str, sha: str) -> RepoSnapshot:
    """Extract the text files of a GitHub tarball into ``dest`` (bounded, no links)."""
    snap = RepoSnapshot(root=dest, repo=repo, sha=sha)
    total = 0
    try:
        with tarfile.open(archive, mode="r:gz") as tar:
            for member in tar:
                if not member.isfile():
                    continue  # directories are implied; links and devices are never followed
                rel = _safe_member_path(member.name)
                if rel is None or _skip_path(rel):
                    continue
                if member.size > MAX_FILE_BYTES:
                    snap.skipped_large.append(rel)
                    continue
                if len(snap.files) >= MAX_FILES or total + member.size > MAX_EXTRACTED_BYTES:
                    raise SnapshotError("repository too large to snapshot")
                handle = tar.extractfile(member)
                if handle is None:
                    continue
                data = handle.read()
                if b"\0" in data[:8192]:
                    continue  # binary
                target = dest / rel
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(data)
                snap.files.append(rel)
                total += len(data)
    except (tarfile.TarError, OSError, EOFError) as exc:
        raise SnapshotError(f"bad tarball: {exc}") from exc
    snap.files.sort()
    snap.size_bytes = total
    return snap


async def build_snapshot(github: Any, repo: str, sha: str, parent: Path) -> RepoSnapshot:
    """Download and extract ``repo@sha`` under ``parent``; the caller calls ``cleanup()``."""
    parent.mkdir(parents=True, exist_ok=True)
    workdir = Path(tempfile.mkdtemp(prefix=f"{sha[:12]}-", dir=parent))
    archive = workdir / "repo.tar.gz"
    tree = workdir / "tree"
    tree.mkdir()
    started = time.monotonic()
    try:
        await github.download_tarball(repo, sha, archive, MAX_TARBALL_BYTES)
        snap = await asyncio.to_thread(extract_tarball, archive, tree, repo, sha)
    except BaseException:
        shutil.rmtree(workdir, ignore_errors=True)
        raise
    archive.unlink(missing_ok=True)
    snap.workdir = workdir
    logger.info(
        "Snapshot of %s@%s: %d files, %.1f MB in %.1fs", repo, sha[:8], len(snap.files),
        snap.size_bytes / 1e6, time.monotonic() - started,
    )  # fmt: skip
    return snap


def clear_stale_snapshots(parent: Path, older_than_s: float = 3600) -> None:
    """Remove snapshot dirs a crash or restart left behind."""
    if not parent.is_dir():
        return
    now = time.time()
    for entry in parent.iterdir():
        try:
            if entry.is_dir() and now - entry.stat().st_mtime > older_than_s:
                shutil.rmtree(entry, ignore_errors=True)
        except OSError:
            continue


# --- tools ------------------------------------------------------------------------------


def _norm(path: str) -> str:
    path = (path or "").strip().strip("`'\"")
    path = re.sub(r"^\./+", "", path).lstrip("/")
    return posixpath.normpath(path) if path else ""


def _glob_match(path: str, pattern: str) -> bool:
    pattern = _norm(pattern)
    if not pattern:
        return True
    if fnmatch.fnmatchcase(path, pattern) or fnmatch.fnmatchcase(path, pattern.replace("**/", "")):
        return True
    if "/" not in pattern:  # a bare `*.ts` matches at any depth
        return fnmatch.fnmatchcase(posixpath.basename(path), pattern)
    return False


class RepoTools:
    """The model's read-only view of the snapshot. Every method returns text for the model."""

    def __init__(self, snapshot: RepoSnapshot) -> None:
        self.snap = snapshot
        self._files = snapshot.file_set
        self._dirs = {posixpath.dirname(f) for f in snapshot.files}
        for d in list(self._dirs):
            while d:
                d = posixpath.dirname(d)
                self._dirs.add(d)

    def _resolve(self, path: str) -> str | None:
        rel = _norm(path)
        if rel in ("", "."):
            return ""
        if rel.startswith("..") or os.path.isabs(rel):
            return None
        return rel

    def _missing(self, rel: str) -> str:
        if rel in self.snap.skipped_large:
            return f"`{rel}` exists but is larger than {MAX_FILE_BYTES // 1024} KB; use search."
        names = difflib.get_close_matches(rel, self.snap.files, n=5, cutoff=0.6)
        base = posixpath.basename(rel)
        names += [f for f in self.snap.files if posixpath.basename(f) == base and f not in names]
        hint = ("; similar paths: " + ", ".join(names[:6])) if names else ""
        return f"`{rel}` does not exist at {self.snap.sha[:12]}{hint}"

    def read_file(
        self, path: str, start_line: int | None = None, end_line: int | None = None
    ) -> str:
        rel = self._resolve(path)
        if rel is None:
            return f"invalid path {path!r}"
        if rel in self._dirs or rel == "":
            return self.list_files(rel)
        if rel not in self._files:
            return self._missing(rel)
        lines = (self.snap.root / rel).read_text(errors="replace").splitlines()
        total = len(lines)
        start = max(1, int(start_line or 1))
        end = min(total, int(end_line) if end_line else start + READ_MAX_LINES - 1)
        end = min(end, start + READ_MAX_LINES - 1)
        out, size = [], 0
        for n in range(start, end + 1):
            row = f"{n:>5} {lines[n - 1]}"
            size += len(row) + 1
            if size > READ_MAX_CHARS:
                end = n - 1
                break
            out.append(row)
        head = f"{rel} (lines {start}-{end} of {total})"
        more = ""
        if end < total:
            more = f"\n[... {total - end} more lines; read on with start_line={end + 1}]"
        return head + "\n" + "\n".join(out) + more

    def search(
        self, query: str, path_glob: str | None = None, regex: bool = False,
        ignore_case: bool = False,
    ) -> str:  # fmt: skip
        if not query:
            return "empty query"
        flags = re.IGNORECASE if ignore_case else 0
        try:
            pattern = re.compile(query if regex else re.escape(query), flags)
        except re.error as exc:
            return f"invalid regex: {exc}"
        deadline = time.monotonic() + SEARCH_TIME_BUDGET_S
        out: list[str] = []
        files_hit, total, truncated = 0, 0, False
        for rel in self.snap.files:
            if path_glob and not _glob_match(rel, path_glob):
                continue
            if time.monotonic() > deadline:
                truncated = True
                break
            try:
                text = (self.snap.root / rel).read_text(errors="replace")
            except OSError:
                continue
            if not pattern.search(text):
                continue
            files_hit += 1
            shown = 0
            for n, line in enumerate(text.splitlines(), 1):
                if pattern.search(line):
                    total += 1
                    if shown < SEARCH_MAX_PER_FILE and len(out) < SEARCH_MAX_MATCHES:
                        out.append(f"{rel}:{n}: {line.strip()[:SEARCH_LINE_CHARS]}")
                        shown += 1
        if not out:
            where = f" in {path_glob}" if path_glob else ""
            budget = " (search time budget hit)" if truncated else ""
            return f"no matches for {query!r}{where}{budget}"
        note = f"{total} matching lines in {files_hit} files"
        if total > len(out):
            note += f" (showing {len(out)}; narrow with path_glob)"
        if truncated:
            note += " (search time budget hit: partial)"
        return note + "\n" + "\n".join(out)

    def list_files(self, path: str = "", pattern: str | None = None) -> str:
        rel = self._resolve(path)
        if rel is None:
            return f"invalid path {path!r}"
        if pattern:
            prefix = f"{rel}/" if rel else ""
            hits = [f for f in self.snap.files if f.startswith(prefix) and _glob_match(f, pattern)]
            shown = hits[:LIST_MAX_ENTRIES]
            more = f"\n[... {len(hits) - len(shown)} more]" if len(hits) > len(shown) else ""
            return (f"{len(hits)} files match {pattern!r}" + (f" under {rel}/" if rel else "")
                    + "\n" + "\n".join(shown) + more)  # fmt: skip
        if rel and rel not in self._dirs:
            return self._missing(rel) if rel not in self._files else f"`{rel}` is a file"
        prefix = f"{rel}/" if rel else ""
        entries: set[str] = set()
        for f in self.snap.files:
            if f.startswith(prefix):
                head = f[len(prefix) :].split("/", 1)
                entries.add(head[0] + ("/" if len(head) > 1 else ""))
        for f in self.snap.skipped_large:
            if f.startswith(prefix) and "/" not in f[len(prefix) :]:
                entries.add(f[len(prefix) :] + " (large)")
        shown = sorted(entries)[:LIST_MAX_ENTRIES]
        more = f"\n[... {len(entries) - len(shown)} more]" if len(entries) > len(shown) else ""
        return f"{rel or '.'}/ ({len(entries)} entries)\n" + "\n".join(shown) + more

    def call(self, name: str, args: dict[str, Any]) -> str:
        try:
            if name == "read_file":
                return self.read_file(
                    str(args.get("path") or ""), _int(args.get("start_line")),
                    _int(args.get("end_line")),
                )  # fmt: skip
            if name == "search":
                return self.search(
                    str(args.get("query") or ""), args.get("path_glob") or None,
                    bool(args.get("regex")), bool(args.get("ignore_case")),
                )  # fmt: skip
            if name == "list_files":
                return self.list_files(str(args.get("path") or ""), args.get("pattern") or None)
        except Exception as exc:  # a tool error is the model's to handle, never the review's
            logger.info("Tool %s(%s) failed: %s", name, args, exc)
            return f"tool error: {type(exc).__name__}: {exc}"
        return f"unknown tool {name!r}"


def _int(value: Any) -> int | None:
    try:
        return int(value) if value is not None and value != "" else None
    except (TypeError, ValueError):
        return None


TOOL_SPECS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": (
                "Read a file of the repository at the PR head (line-numbered, up to "
                f"{READ_MAX_LINES} lines per call). A directory path lists it."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "repo-relative path"},
                    "start_line": {"type": "integer", "description": "first line (1-based)"},
                    "end_line": {"type": "integer", "description": "last line (inclusive)"},
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search",
            "description": (
                "Search every text file of the repository at the PR head; returns "
                "`path:line: text` for matching lines. Literal by default."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "text (or regex) to find"},
                    "path_glob": {
                        "type": "string",
                        "description": "only files matching this glob, e.g. `**/*.test.ts`",
                    },
                    "regex": {"type": "boolean", "description": "query is a Python regex"},
                    "ignore_case": {"type": "boolean"},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_files",
            "description": (
                "List a directory of the repository at the PR head, or with `pattern` every "
                "file under it matching a glob (e.g. `*.lock`, `tests/**/*.py`)."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "directory; empty = repo root"},
                    "pattern": {"type": "string", "description": "optional file glob"},
                },
            },
        },
    },
]
