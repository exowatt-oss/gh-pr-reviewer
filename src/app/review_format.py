"""Pure pieces of a native inline review: diff rendering, model output parsing, inline-comment
anchoring against the PR's diff hunks, de-duplication, and the review / comment markdown.

Nothing here talks to GitHub or the model (``reviewer.py`` does), so every rule is unit-tested.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import quote

from app.costs import render_footer

REVIEW_MARKER = "<!-- gh-pr-reviewer:review"
FINDING_MARKER = "<!-- gh-pr-reviewer:finding"
POST_MERGE_MARKER = "<!-- gh-pr-reviewer:post-merge"
POST_MERGE_MAX_FINDINGS = 8
MAX_FINDINGS = 12
MAX_FINDINGS_SPLIT = 24  # a review in several parts (a diff too large for one model call)
MAX_SUGGESTIONS = 3  # severity "suggestion" (non-defect improvements), not suggestion blocks
MAX_SUGGESTION_LINES = 40
MAX_RANGE_LINES = 60
NEAREST_LINE_DISTANCE = 3
MAX_TABLE_ROWS = 80
SEVERITIES = ("critical", "major", "minor", "suggestion")
ISSUE_SEVERITIES = ("critical", "major", "minor")
PREFIXED_SEVERITIES = ("critical", "major")  # only real bugs / security get a label

_SEVERITY_ALIASES = {
    "critical": "critical",
    "blocker": "critical",
    "security": "critical",
    "high": "major",
    "major": "major",
    "bug": "major",
    "error": "major",
    "medium": "minor",
    "minor": "minor",
    "low": "minor",
    "warning": "minor",
    "nit": "suggestion",
    "suggestion": "suggestion",
    "info": "suggestion",
    "improvement": "suggestion",
}
_HUNK_RE = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@")


# --- diff hunks --------------------------------------------------------------------------
@dataclass
class Hunk:
    """The RIGHT (new-file) side of one diff hunk: GitHub accepts inline comments on these
    lines only, and a multi-line comment must start and end in the same hunk."""

    start: int
    end: int  # inclusive; end < start for a pure-deletion hunk
    added: set[int] = field(default_factory=set)
    text: dict[int, str] = field(default_factory=dict)

    def __contains__(self, line: int) -> bool:
        return self.start <= line <= self.end


def parse_hunks(patch: str | None) -> list[Hunk]:
    hunks: list[Hunk] = []
    current: Hunk | None = None
    line_no = 0
    for raw in (patch or "").splitlines():
        match = _HUNK_RE.match(raw)
        if match:
            line_no = int(match.group(3))
            current = Hunk(start=line_no, end=line_no - 1)
            hunks.append(current)
            continue
        if current is None or raw.startswith("\\"):
            continue  # file headers before the first hunk / "\ No newline at end of file"
        if raw.startswith("-"):
            continue
        if raw.startswith("+"):
            current.added.add(line_no)
            current.text[line_no] = raw[1:]
        else:  # context (" ..." or an empty line GitHub sometimes sends for " ")
            current.text[line_no] = raw[1:] if raw.startswith(" ") else raw
        current.end = line_no
        line_no += 1
    return [h for h in hunks if h.end >= h.start]


@dataclass
class DiffFile:
    path: str
    status: str
    patch: str | None
    hunks: list[Hunk]
    previous_path: str | None = None

    @classmethod
    def from_api(cls, data: dict[str, Any]) -> DiffFile:
        patch = data.get("patch")
        return cls(
            path=str(data.get("filename") or ""),
            status=str(data.get("status") or "modified"),
            patch=patch,
            hunks=parse_hunks(patch),
            previous_path=data.get("previous_filename"),
        )

    def hunk_of(self, line: int) -> Hunk | None:
        return next((h for h in self.hunks if line in h), None)

    def right_lines(self) -> list[int]:
        return [n for h in self.hunks for n in range(h.start, h.end + 1)]


EXTRA_LINES_BEFORE = 8
EXTRA_LINES_AFTER = 3
SYNTH_MAX_LINES = 8000  # larger patchless files stay "no diff" (generated data, mostly)


def extend_patch(patch: str, head_text: str | None, before: int = EXTRA_LINES_BEFORE,
                 after: int = EXTRA_LINES_AFTER) -> str:  # fmt: skip
    """Widen each hunk's context with unchanged lines of the new file. Lines outside the
    hunks are identical on both sides, so the head version alone is enough; hunks never grow
    into each other. Only what the model reads changes: anchors use GitHub's own hunks."""
    if not patch or not head_text:
        return patch or ""
    head = head_text.split("\n")
    lines = patch.split("\n")
    starts = [i for i, ln in enumerate(lines) if _HUNK_RE.match(ln)]
    if not starts:
        return patch
    bounds = []  # (old_start, old_len, new_start, new_len, body lines)
    for k, i in enumerate(starts):
        m = _HUNK_RE.match(lines[i])
        assert m
        tail = lines[i].split("@@", 2)[2] if lines[i].count("@@") >= 2 else ""
        body = lines[i + 1 : starts[k + 1] if k + 1 < len(starts) else len(lines)]
        if body and body[-1] == "" and k + 1 == len(starts):
            body = body[:-1]
        old_len = int(m.group(2)) if m.group(2) is not None else 1
        new_len = int(m.group(4)) if m.group(4) is not None else 1
        bounds.append([int(m.group(1)), old_len, int(m.group(3)), new_len, body, tail])
    out = []
    prev_new_end = 0  # last new-file line already shown
    for k, (o_start, o_len, n_start, n_len, body, tail) in enumerate(bounds):
        if n_len == 0 or n_start == 0:
            out.append(f"@@ -{o_start},{o_len} +{n_start},{n_len} @@{tail}")
            out.extend(body)
            prev_new_end = max(prev_new_end, n_start)
            continue
        add_before = max(0, min(before, n_start - 1 - prev_new_end))
        next_start = bounds[k + 1][2] if k + 1 < len(bounds) else len(head) + 1
        n_end = n_start + n_len - 1
        add_after = max(0, min(after, next_start - 1 - n_end - before, len(head) - n_end))
        pre = [" " + head[n] for n in range(n_start - 1 - add_before, n_start - 1)]
        post = [" " + head[n] for n in range(n_end, n_end + add_after)]
        out.append(
            f"@@ -{o_start - add_before},{o_len + add_before + add_after} "
            f"+{n_start - add_before},{n_len + add_before + add_after} @@{tail}"
        )
        out.extend(pre + body + post)
        prev_new_end = n_end + add_after
    return "\n".join(out)


def synthesize_patch(base_text: str | None, head_text: str | None) -> str | None:
    """A unified diff for a file GitHub sent without a patch (too large for the API)."""
    import difflib

    old = (base_text or "").splitlines()
    new = (head_text or "").splitlines()
    if len(old) + len(new) > 6 * SYNTH_MAX_LINES:
        return None
    diff = list(difflib.unified_diff(old, new, lineterm="", n=3))[2:]  # drop ---/+++ headers
    if not diff or sum(1 for ln in diff if ln[:1] in "+-") > SYNTH_MAX_LINES:
        return None
    return "\n".join(diff)


def render_file_diff(f: DiffFile, patch: str | None = None) -> str:
    """One file in PR-Agent's line-numbered ``__new hunk__`` / ``__old hunk__`` format
    (``patch``: a widened or synthesized patch to show instead of GitHub's)."""
    from pr_agent.algo.git_patch_processing import (
        decouple_and_convert_to_hunks_with_lines_numbers,
    )
    from pr_agent.algo.types import EDIT_TYPE, FilePatchInfo

    if f.status == "removed":
        return f"\n\n## File '{f.path}' was deleted\n"
    patch = patch or f.patch
    if not patch:
        return (
            f"\n\n## File: '{f.path}'\n(GitHub sent no diff for this file — binary or too "
            "large; do not comment on it)\n"
        )
    edit = {"added": EDIT_TYPE.ADDED, "renamed": EDIT_TYPE.RENAMED}.get(
        f.status, EDIT_TYPE.MODIFIED
    )
    info = FilePatchInfo("", "", patch, f.path, edit_type=edit, old_filename=f.previous_path)
    rendered = decouple_and_convert_to_hunks_with_lines_numbers(patch, info)
    if not f.patch:
        rendered = rendered.replace(
            f"## File: '{f.path}'",
            f"## File: '{f.path}' (diff rebuilt from the file contents: too large for "
            "GitHub's API)",
            1,
        )
    if f.status == "renamed" and f.previous_path:
        rendered = rendered.replace(
            f"## File: '{f.path}'", f"## File: '{f.path}' (renamed from '{f.previous_path}')", 1
        )
    return rendered


@dataclass
class RenderedDiff:
    text: str
    shown: list[str]  # files in this diff (a file split between parts is in each of them)


def _patch_hunks(patch: str) -> list[list[str]]:
    """A unified diff as its hunks, each a header line followed by its body lines."""
    hunks: list[list[str]] = []
    for line in patch.split("\n"):
        if _HUNK_RE.match(line):
            hunks.append([line])
        elif hunks:
            hunks[-1].append(line)
    return hunks


def _cut_hunk(hunk: list[str], max_chars: int) -> list[list[str]]:
    """A hunk larger than ``max_chars`` as consecutive smaller hunks, each with a header
    carrying its own line numbers, so every line is still shown with the right number."""
    if sum(len(line) + 1 for line in hunk) <= max_chars:
        return [hunk]
    match = _HUNK_RE.match(hunk[0])
    assert match
    old_start, new_start = int(match.group(1)), int(match.group(3))
    out: list[list[str]] = []
    body: list[str] = []
    size = old_n = new_n = 0
    for line in [*hunk[1:], None]:
        if body and (line is None or size + len(line) + 1 > max_chars):
            out.append([f"@@ -{old_start},{old_n} +{new_start},{new_n} @@", *body])
            old_start, new_start = old_start + old_n, new_start + new_n
            body, size, old_n, new_n = [], 0, 0, 0
        if line is None:
            break
        body.append(line)
        size += len(line) + 1
        if not line.startswith("\\"):  # "\ No newline at end of file" is on neither side
            old_n += not line.startswith("+")
            new_n += not line.startswith("-")
    return out


def _split_patch(patch: str, max_chars: int) -> list[str]:
    """Consecutive pieces of a unified diff of about ``max_chars`` at most: whole hunks where
    they fit, a hunk too large on its own cut between lines."""
    pieces: list[list[str]] = [[]]
    size = 0
    for hunk in _patch_hunks(patch):
        for part in _cut_hunk(hunk, max_chars):
            text = "\n".join(part)
            if pieces[-1] and size + len(text) + 1 > max_chars:
                pieces.append([])
                size = 0
            pieces[-1].append(text)
            size += len(text) + 1
    return ["\n".join(p) for p in pieces if p]


TRUNCATED_LINE_NOTE = "\n[... cut here: a single diff line too long to show in one model call]"


def _file_pieces(f: DiffFile, patch: str | None, budget_chars: int, rendered: str) -> list[str]:
    """One file's rendered diff in pieces of at most ``budget_chars``. Rendering adds line
    numbers and repeats context lines, so the patch is cut by the file's own rendered-to-raw
    ratio, and finer until every piece fits."""
    patch = patch or f.patch or ""
    limit = int(budget_chars * 0.95 * len(patch) / max(len(rendered), 1))
    while True:
        texts = [render_file_diff(f, piece) for piece in _split_patch(patch, limit) or [patch]]
        if all(len(t) <= budget_chars for t in texts) or limit < 2_000:
            break
        limit = int(limit * 0.8)
    header = f"## File: '{f.path}'"
    k = len(texts)
    out = []
    for i, text in enumerate(texts, 1):
        if k > 1:
            text = text.replace(header, f"{header} (part {i} of {k} of this file's diff)", 1)
        if len(text) > budget_chars:  # one line longer than a whole call: show its start
            text = text[: budget_chars - len(TRUNCATED_LINE_NOTE)] + TRUNCATED_LINE_NOTE
        out.append(text)
    return out


def split_diff(
    files: list[DiffFile], budget_chars: int, patches: dict[str, str] | None = None
) -> list[RenderedDiff]:
    """Line-numbered diff of ``files`` (``patches``: widened or synthesized patches by path) in
    parts of at most ``budget_chars`` each, one model call per part. Usually one part. When the
    diff does not fit, the files are split, in path order (a directory's files stay together),
    into the fewest parts of about equal size; a file too large for one part is itself split
    between hunks (a giant hunk between lines). No file is ever left out."""
    patches = patches or {}
    rendered = [(f, render_file_diff(f, patches.get(f.path))) for f in files]
    total = sum(len(text) for _, text in rendered)
    if total <= budget_chars:
        text = "".join(text for _, text in rendered).strip("\n")
        return [RenderedDiff(text, [f.path for f in files])]
    pieces: list[tuple[str, str]] = []
    for f, text in sorted(rendered, key=lambda p: p[0].path):
        if len(text) <= budget_chars:
            pieces.append((f.path, text))
        else:
            texts = _file_pieces(f, patches.get(f.path), budget_chars, text)
            pieces += [(f.path, t) for t in texts]
    total = sum(len(text) for _, text in pieces)
    target = total / math.ceil(total / budget_chars)
    groups: list[list[tuple[str, str]]] = [[]]
    size = 0
    for path, text in pieces:
        if groups[-1] and (size >= target or size + len(text) > budget_chars):
            groups.append([])
            size = 0
        groups[-1].append((path, text))
        size += len(text)
    return [
        RenderedDiff(
            "".join(text for _, text in group).strip("\n"),
            list(dict.fromkeys(path for path, _ in group)),
        )
        for group in groups
    ]


# --- model output ------------------------------------------------------------------------
@dataclass
class Finding:
    path: str
    start_line: int
    end_line: int
    severity: str
    title: str
    body: str
    suggestion: str | None = None

    @property
    def key(self) -> str:
        """Stable identity of a finding for de-duplication across reviews."""
        norm = re.sub(r"[^a-z0-9]+", " ", self.title.lower()).strip()
        return hashlib.sha1(f"{self.path}\n{norm}".encode()).hexdigest()[:12]


@dataclass
class ModelReview:
    overview: str
    files: dict[str, str]
    findings: list[Finding]


class ModelOutputError(ValueError):
    """The model's answer is not the JSON object the prompt asked for."""


def _extract_json(text: str) -> Any:
    text = (text or "").strip()
    fenced = re.search(r"```(?:json)?\s*\n(.*?)\n```", text, re.DOTALL)
    candidates = [fenced.group(1)] if fenced else []
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end > start:
        candidates.append(text[start : end + 1])
    candidates.append(text)
    for candidate in candidates:
        for attempt in (candidate, re.sub(r",\s*([}\]])", r"\1", candidate)):  # trailing commas
            try:
                return json.loads(attempt)
            except (json.JSONDecodeError, ValueError):
                continue
    raise ModelOutputError(f"no JSON object in the model output: {text[:200]!r}")


def _int(value: Any) -> int | None:
    try:
        number = int(str(value).strip())
    except (TypeError, ValueError):
        return None
    return number if number > 0 else None


def _strip_fence(text: str) -> str:
    """A suggestion is replacement lines; drop a code fence the model may have wrapped."""
    match = re.fullmatch(r"\s*(`{3,}|~{3,})[^\n]*\n(.*?)\n?\1\s*", text, re.DOTALL)
    return match.group(2) if match else text


def parse_model_output(text: str) -> ModelReview:
    data = _extract_json(text)
    if isinstance(data, dict) and isinstance(data.get("review"), dict):
        data = data["review"]
    if not isinstance(data, dict):
        raise ModelOutputError("the model output is not a JSON object")
    files: dict[str, str] = {}
    for item in data.get("files") or []:
        if isinstance(item, dict) and item.get("path"):
            files[str(item["path"]).strip()] = " ".join(str(item.get("summary") or "").split())
    findings = []
    for item in data.get("findings") or []:
        if not isinstance(item, dict):
            continue
        path = str(item.get("path") or "").strip().lstrip("/")
        start = _int(item.get("start_line"))
        end = _int(item.get("end_line")) or start
        body = str(item.get("body") or "").strip()
        if not path or start is None or end is None or not body:
            continue
        severity = _SEVERITY_ALIASES.get(str(item.get("severity") or "").strip().lower(), "minor")
        suggestion = item.get("suggestion")
        if isinstance(suggestion, str) and suggestion.strip():
            suggestion = _strip_fence(suggestion.rstrip("\n"))
        else:
            suggestion = None
        findings.append(
            Finding(
                path=path,
                start_line=min(start, end),
                end_line=max(start, end),
                severity=severity,
                title=" ".join(str(item.get("title") or "").split())[:120] or body[:60],
                body=body,
                suggestion=suggestion,
            )
        )
    overview = " ".join(str(data.get("overview") or "").split())
    return ModelReview(overview=overview, files=files, findings=findings)


def normalize_findings(findings: list[Finding], cap: int = MAX_FINDINGS) -> list[Finding]:
    """Severity-ordered (stable), duplicates merged, non-defect suggestions limited, capped."""
    seen_keys: set[str] = set()
    seen_spots: set[tuple[str, int, int]] = set()
    unique: list[Finding] = []
    for f in sorted(findings, key=lambda f: SEVERITIES.index(f.severity)):
        spot = (f.path, f.start_line, f.end_line)
        if f.key in seen_keys or spot in seen_spots:
            continue
        seen_keys.add(f.key)
        seen_spots.add(spot)
        unique.append(f)
    out, suggestions = [], 0
    for f in unique:
        if f.severity == "suggestion":
            suggestions += 1
            if suggestions > MAX_SUGGESTIONS:
                continue
        out.append(f)
    return out[:cap]


# --- anchoring -----------------------------------------------------------------------------
@dataclass
class Anchor:
    path: str
    start_line: int
    end_line: int
    exact: bool  # the model's range as given: a suggestion block may replace it


def anchor_finding(
    f: Finding, diff: dict[str, DiffFile], allowed: dict[str, set[int]] | None = None
) -> Anchor | None:
    """Where GitHub will accept this comment: the model's range when it lies in one hunk
    (and, for an incremental review, in the new commits), else a commentable line inside the
    range, else (full reviews) one at most a few lines away; ``None``: fold into the body."""
    file = diff.get(f.path)
    if file is None or not file.hunks:
        return None
    ok_lines = set(file.right_lines())
    if allowed is not None:
        ok_lines &= allowed.get(f.path, set())
    if not ok_lines:
        return None
    start, end = f.start_line, f.end_line
    hunk = file.hunk_of(end)
    if (
        hunk is not None
        and start in hunk
        and end - start < MAX_RANGE_LINES
        and all(n in ok_lines for n in range(start, end + 1))
    ):
        return Anchor(f.path, start, end, exact=True)
    # A single line inside the model's range, preferring its end (GitHub shows the comment
    # under the last line), then the nearest commentable line.
    inside = [n for n in range(end, start - 1, -1) if n in ok_lines]
    if inside:
        return Anchor(f.path, inside[0], inside[0], exact=False)
    if allowed is not None:
        return None  # incremental: never move a comment onto lines the new commits touched
    nearest = min(ok_lines, key=lambda n: (min(abs(n - start), abs(n - end)), n))
    if min(abs(nearest - start), abs(nearest - end)) <= NEAREST_LINE_DISTANCE:
        return Anchor(f.path, nearest, nearest, exact=False)
    return None


def suggestion_applies(f: Finding, anchor: Anchor, file: DiffFile) -> bool:
    """A committable suggestion must replace exactly the anchored lines and change them."""
    if f.suggestion is None or not anchor.exact:
        return False
    lines = f.suggestion.split("\n")
    if len(lines) > MAX_SUGGESTION_LINES:
        return False
    hunk = file.hunk_of(anchor.end_line)
    if hunk is None:
        return False
    current = [hunk.text.get(n) for n in range(anchor.start_line, anchor.end_line + 1)]
    if any(c is None for c in current):
        return False
    if [c.rstrip() for c in current] == [ln.rstrip() for ln in lines]:
        return False  # no change: not a suggestion
    # Diff output pasted as a "suggestion" would be committed verbatim.
    if any(ln.startswith("@@ ") for ln in lines):
        return False
    marked = [ln for ln in lines if ln.strip()]
    return not (
        marked
        and all(ln[0] in "+-" for ln in marked)
        and any(ln.startswith("+") for ln in marked)
        and not any((c or "").lstrip().startswith(("+", "-")) for c in current)
    )


# --- de-duplication against earlier reviews ------------------------------------------------
_FINDING_KEY_RE = re.compile(re.escape(FINDING_MARKER) + r" key=([0-9a-f]+)")
_WORD_RE = re.compile(r"[a-z0-9_]{3,}")


_HTML_COMMENT_RE = re.compile(r"<!--.*?-->", re.DOTALL)
_FENCE_RE = re.compile(r"(`{3,})suggestion\n.*?\n\1", re.DOTALL)
_CODE_RE = re.compile(r"`([^`\n]{1,80})`")
_STOPWORDS = frozenset(
    "the and for but not are was were this that these those with from into onto when then"
    " than there their they them its it's has have had can could will would should must may"
    " also only just any all each every one two use used uses using via which what who how"
    " why where does did done here been being out off our your you while because so such"
    " instead still now new old same other more most less very like make makes made per"
    " major minor critical suggestion fix fixes".split()
)


@dataclass(frozen=True)
class PriorComment:
    """A finding an earlier review of this App raised on the PR: an inline comment, or an
    item of a review body (comments outside the diff, or inline comments GitHub refused)."""

    path: str
    line: int | None
    key: str | None
    text: str
    resolved: bool = False  # its review thread was resolved on GitHub
    comment_id: int | None = None

    @classmethod
    def from_api(cls, c: dict[str, Any], *, resolved: bool = False) -> PriorComment:
        body = str(c.get("body") or "")
        match = _FINDING_KEY_RE.search(body)
        line = c.get("line") or c.get("original_line")
        return cls(
            path=str(c.get("path") or ""),
            line=int(line) if line else None,
            key=match.group(1) if match else None,
            text=body,
            resolved=resolved,
            comment_id=int(c["id"]) if c.get("id") else None,
        )


# Body items rendered by ``ReviewBody`` ("comments outside the diff") and by the 422
# fallback of ``Reviewer.publish`` (each refused inline comment folded into the body).
_BODY_ITEM_RE = re.compile(
    r"^- `([^`\n]+?):(\d+)(?:-\d+)?` — (?:\*\*\[\w+\]\*\* )?\*\*(.+?)\.\*\* (.*)$", re.MULTILINE
)
_FOLDED_HEAD_RE = re.compile(r"^\*\*`([^`\n]+?):(\d+)`\*\*$", re.MULTILINE)


def prior_from_review_body(body: str) -> list[PriorComment]:
    """Findings an earlier review listed in its body instead of inline."""
    out: list[PriorComment] = []
    for m in _BODY_ITEM_RE.finditer(body or ""):
        path, line, title, text = m.group(1), int(m.group(2)), m.group(3), m.group(4)
        key = Finding(path, line, line, "minor", title, "").key
        out.append(PriorComment(path, line, key, f"{title}. {text}"))
    heads = list(_FOLDED_HEAD_RE.finditer(body or ""))
    for i, m in enumerate(heads):
        stop = heads[i + 1].start() if i + 1 < len(heads) else len(body)
        chunk = body[m.end() : stop].split("\n\n<sub>gh-pr-reviewer", 1)[0]
        key = _FINDING_KEY_RE.search(chunk)
        out.append(PriorComment(m.group(1), int(m.group(2)), key.group(1) if key else None, chunk))
    return out


def prior_summary(p: PriorComment) -> str:
    """One line for the prompt's "already reported" list."""
    text = " ".join(_clean(p.text).split())
    where = f"{p.path}:{p.line}" if p.line else p.path
    return f"`{where}` — {text[:200]}" + (" (thread resolved)" if p.resolved else "")


def _clean(text: str) -> str:
    """A finding's prose: no hidden markers, suggestion blocks or severity labels."""
    text = _HTML_COMMENT_RE.sub(" ", text or "")
    text = _FENCE_RE.sub(" ", text)
    return re.sub(r"\*\*\[\w+\]\*\*", " ", text)


def _words(text: str) -> set[str]:
    """Content words, cut to 5-letter stems so "expires" / "expiry" or "token" / "tokens"
    count as the same word."""
    return {w[:5] for w in _WORD_RE.findall(_clean(text).lower()) if w not in _STOPWORDS}


def _code_terms(text: str) -> set[str]:
    terms = {t.strip().lower() for t in _CODE_RE.findall(_clean(text))}
    return {t for t in terms if len(t) >= 2}  # `x` alone says little


def _jaccard(a: set[str], b: set[str]) -> float:
    return len(a & b) / len(a | b) if a and b else 0.0


def is_duplicate(f: Finding, anchor: Anchor | None, prior: list[PriorComment]) -> bool:
    """Already raised by an earlier review of this App on the same file:

    - the same finding key (same file and title), or
    - on lines within a few of it, with mostly the same wording, or
    - anywhere in the file (lines move between pushes) with closely matching wording, or
      the same code identifiers and similar wording, or
    - anywhere in the file when the earlier comment's thread was resolved (a human already
      dealt with it), with loosely similar wording or a shared identifier.
    """
    text = f"{f.title} {f.body}"
    words, terms = _words(text), _code_terms(text)
    lo = (anchor.start_line if anchor else f.start_line) - NEAREST_LINE_DISTANCE
    hi = (anchor.end_line if anchor else f.end_line) + NEAREST_LINE_DISTANCE
    for p in prior:
        if p.path != f.path:
            continue
        if p.key and p.key == f.key:
            return True
        sim = _jaccard(words, _words(p.text))
        shared = len(terms & _code_terms(p.text))
        if p.line is not None and lo <= p.line <= hi and sim >= 0.4:
            return True
        if sim >= 0.5 or (shared >= 2 and sim >= 0.3):
            return True
        if p.resolved and (sim >= 0.3 or (shared >= 1 and sim >= 0.2)):
            return True
    return False


# --- markdown ------------------------------------------------------------------------------
def render_comment(f: Finding, with_suggestion: bool) -> str:
    prefix = f"**[{f.severity}]** " if f.severity in PREFIXED_SEVERITIES else ""
    parts = [prefix + f.body.strip()]
    if with_suggestion and f.suggestion is not None:
        fence = "````" if "```" in f.suggestion else "```"
        parts.append(f"{fence}suggestion\n{f.suggestion}\n{fence}")
    parts.append(f"{FINDING_MARKER} key={f.key} severity={f.severity} -->")
    return "\n\n".join(parts)


def review_comment_payload(f: Finding, anchor: Anchor, with_suggestion: bool) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "path": anchor.path,
        "line": anchor.end_line,
        "side": "RIGHT",
        "body": render_comment(f, with_suggestion),
    }
    if anchor.start_line != anchor.end_line:
        payload |= {"start_line": anchor.start_line, "start_side": "RIGHT"}
    return payload


def tally(findings: list[Finding]) -> dict[str, int]:
    counts = {s: 0 for s in SEVERITIES}
    for f in findings:
        counts[f.severity] += 1
    return {
        "critical": counts["critical"],
        "major": counts["major"],
        "minor": counts["minor"],
        "suggestions": counts["suggestion"],
    }


def _plural(n: int, word: str) -> str:
    return f"{n} {word}{'' if n == 1 else 's'}"


def tally_line(findings: list[Finding]) -> str:
    counts = tally(findings)
    issues = counts["critical"] + counts["major"] + counts["minor"]
    return (
        f"**{_plural(len(findings), 'comment')}:** {_plural(issues, 'issue')}, "
        f"{_plural(counts['suggestions'], 'suggestion')}"
    )


def _cell(text: str) -> str:
    return " ".join((text or "").split()).replace("|", "\\|")


def parts_overview(parts: list[tuple[int, str | None]]) -> str:
    """The overview of a review run in several model calls: each part's own overview
    (``(files in the part, overview)``; ``None``: the part failed)."""
    lines = [f"Reviewed in {len(parts)} parts (the diff is too large for one model call):", ""]
    for i, (files, overview) in enumerate(parts, 1):
        text = "not reviewed: this part failed" if overview is None else overview or "—"
        lines.append(f"- **Part {i}** ({_plural(files, 'file')}): {text}")
    return "\n".join(lines)


@dataclass
class ReviewBody:
    head_sha: str
    kind: str  # full | incremental | on_demand
    overview: str
    file_summaries: dict[str, str]
    reviewed_files: list[str]
    total_files: int
    findings: list[Finding]  # everything reported in this review (inline + folded)
    folded: list[tuple[Finding, str]]  # findings that could not be anchored: (finding, where)
    duplicates: int
    footer: tuple[int, int, float, str]  # prompt tokens, completion tokens, cost, effort
    new_commits: int | None = None
    since_sha: str | None = None
    # Files not reviewed because the part of a multi-part review that held them failed.
    unreviewed_files: list[str] = field(default_factory=list)
    # Automatic review round (1 = first review, 2+ = full re-review after new commits) and the
    # configured limit (0 = none); ``None`` for on-demand reviews.
    round: int | None = None
    max_rounds: int = 0
    parts: int = 1  # model calls the diff was split into (too large for one)

    @property
    def rereview(self) -> bool:
        return bool(self.round and self.round >= 2)

    def _round_note(self) -> str | None:
        if self.round and self.max_rounds and self.round >= self.max_rounds:
            return (
                "This was the last automatic review of this PR; comment `/review` to ask for "
                "another."
            )
        return None

    def _marker(self) -> str:
        tail = f" round={self.round}" if self.round else ""
        return f"{REVIEW_MARKER} sha={self.head_sha} kind={self.kind}{tail} -->"

    def _footer(self) -> str:
        prompt, completion, cost, effort = self.footer
        tail = f" · reviewed in {self.parts} parts" if self.parts > 1 else ""
        return f"<sub>{render_footer(prompt, completion, cost, effort)}{tail}</sub>"

    def render(self) -> str:
        if self.rereview and not self.findings:
            return self.render_nothing_new()
        parts = [self._marker()]
        parts.append("## Pull request overview")
        lead = self.overview or "No summary available."
        if self.rereview:
            of = f" of {self.max_rounds}" if self.max_rounds else ""
            lead = (
                f"Re-reviewed the whole PR at `{self.head_sha[:7]}` after new commits "
                f"(automatic review {self.round}{of}); only issues not raised by an earlier "
                f"review are commented. {lead}"
            )
        if self.kind == "incremental" and self.new_commits is not None and self.since_sha:
            lead = (
                f"Reviewed {_plural(self.new_commits, 'new commit')} since "
                f"`{self.since_sha[:7]}`. {lead}"
            )
        parts.append(lead)
        n = len(self.findings)
        scope = _plural(self.total_files, "changed file")
        if self.kind == "incremental":
            scope = _plural(self.total_files, "file") + " changed by the new commits"
        summary_line = (
            f"Reviewed {len(self.reviewed_files)} of {scope} and generated {_plural(n, 'comment')}."
        )
        rows = [
            f"| `{_cell(path)}` | {_cell(self.file_summaries.get(path, ''))} |"
            for path in self.reviewed_files[:MAX_TABLE_ROWS]
        ]
        more = len(self.reviewed_files) - len(rows)
        table = "| File | Description |\n| ---- | ----------- |\n" + "\n".join(rows)
        if more > 0:
            table += f"\n| … | {more} more files |"
        parts.append(
            "### Reviewed changes\n\n"
            + summary_line
            + "\n\n<details>\n<summary>Show a summary per file</summary>\n\n"
            + table
            + "\n</details>"
        )
        if self.unreviewed_files:
            parts.append(
                f"{_plural(len(self.unreviewed_files), 'file')} could not be reviewed (the part "
                "of this review that held them failed; comment `/review` to try again): "
                + ", ".join(f"`{p}`" for p in self.unreviewed_files[:20])
                + (" …" if len(self.unreviewed_files) > 20 else "")
            )
        if n:
            parts.append(tally_line(self.findings))
        else:
            parts.append("No new issues found." if self.duplicates else "No issues found.")
        if self.duplicates:
            parts.append(
                f"{_plural(self.duplicates, 'finding')} already raised in an earlier review "
                f"{'is' if self.duplicates == 1 else 'are'} not repeated."
            )
        if self.folded:
            items = []
            for f, where in self.folded:
                prefix = f"**[{f.severity}]** " if f.severity in PREFIXED_SEVERITIES else ""
                items.append(f"- `{where}` — {prefix}**{_cell(f.title)}.** {f.body.strip()}")
            parts.append(
                f"<details>\n<summary>{_plural(len(self.folded), 'comment')} outside the "
                "diff</summary>\n\n" + "\n".join(items) + "\n</details>"
            )
        note = self._round_note()
        if note:
            parts.append(note)
        parts.append(self._footer())
        return "\n\n".join(parts)

    def render_nothing_new(self) -> str:
        """A re-review that found nothing not already raised: a short body, no overview."""
        of = f" of {self.max_rounds}" if self.max_rounds else ""
        line = (
            f"Re-reviewed the whole PR at `{self.head_sha[:7]}` after new commits (automatic "
            f"review {self.round}{of}): {len(self.reviewed_files)} of "
            f"{_plural(self.total_files, 'changed file')}, nothing new beyond the earlier review."
        )
        if self.duplicates:
            line += (
                f" {_plural(self.duplicates, 'finding')} already raised "
                f"{'is' if self.duplicates == 1 else 'are'} not repeated."
            )
        if self.unreviewed_files:
            line += (
                f" {_plural(len(self.unreviewed_files), 'file')} could not be reviewed (a part "
                "of this review failed)."
            )
        parts = [self._marker(), line]
        note = self._round_note()
        if note:
            parts.append(note)
        parts.append(self._footer())
        return "\n\n".join(parts)


def review_marker_sha(body: str) -> str | None:
    match = re.search(re.escape(REVIEW_MARKER) + r" sha=([0-9a-f]+)", body or "")
    return match.group(1) if match else None


def post_merge_comment(
    *,
    repo_url: str,
    head_sha: str,
    base_ref: str | None,
    findings: list[Finding],
    footer: tuple[int, int, float, str],
) -> str:
    """The one issue comment posted instead of a review when the PR merged while it ran:
    the most severe findings (at most ``POST_MERGE_MAX_FINDINGS``) linked at the reviewed SHA,
    so the author can follow up on the base branch."""
    base = repo_url.rstrip("/")
    short = head_sha[:7]
    ranked = sorted(findings, key=lambda f: SEVERITIES.index(f.severity))
    shown = ranked[:POST_MERGE_MAX_FINDINGS]
    lines = []
    for f in shown:
        span = f"{f.start_line}" + (f"-{f.end_line}" if f.end_line != f.start_line else "")
        anchor = f"#L{f.start_line}" + (f"-L{f.end_line}" if f.end_line != f.start_line else "")
        link = f"{base}/blob/{head_sha}/{quote(f.path)}{anchor}"
        label = f"**[{f.severity}]** " if f.severity in PREFIXED_SEVERITIES else ""
        lines.append(f"- {label}[`{f.path}:{span}`]({link}) — {_cell(f.title)}")
    more = len(ranked) - len(shown)
    if more > 0:
        lines.append(f"- …and {more} more")
    target = f"`{base_ref}`" if base_ref else "the base branch"
    prompt, completion, cost, effort = footer
    return "\n\n".join(
        [
            f"{POST_MERGE_MARKER} sha={head_sha} -->",
            f"**Merged before review finished** — {_plural(len(findings), 'finding')} at "
            f"[`{short}`]({base}/commit/{head_sha}). Not posted inline; follow up on "
            f"{target} if they still apply:",
            "\n".join(lines),
            f"<sub>{render_footer(prompt, completion, cost, effort)}</sub>",
        ]
    )
