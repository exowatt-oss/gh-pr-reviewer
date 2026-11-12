"""The reviewer's own prompt and output contract (native inline inline review).

PR-Agent's `/review` prompt produced a "PR Reviewer Guide" comment; this one asks for a JSON
object the app turns into a native GitHub pull-request review with inline comments.

One model call's prompt is a ``PromptLayout`` of three segments, from most to least shared:
the system text (identical for every review), the user text every part of one review shares
(PR header, description, scope, earlier findings, repo conventions, codebase context) and
the part's own text (part header, diff, closing line). With ``LLM_PROMPT_CACHE_LAYOUT`` the
tool loop sends them as separate messages with explicit prompt-cache breakpoints after the
system text and after the shared text, so every review reads the system prefix from the
provider's cache and the parts of a review in parts read each other's shared prefix.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from app.pr_agent_config import truncate
from app.review_format import MAX_FINDINGS, MAX_SUGGESTIONS

DESCRIPTION_MAX_CHARS = 4000
PRIOR_FINDINGS_MAX = 40
PART_FILES_MAX = 300  # changed files listed in each part of a multi-part review

SYSTEM_PROMPT = f"""\
You are gh-pr-reviewer, a senior engineer doing code review on a GitHub pull request,
the way GitHub Copilot code review does: a short overview of the change plus inline review
comments anchored on the exact changed lines that have a problem.

## The diff format
Each file starts with `## File: 'path'`. Each hunk has a `__new hunk__` section (the new
code; every line starts with its line number in the new file, `+` marks an added line) and,
when lines were removed, an `__old hunk__` section (old code, no line numbers). Review the new
code: the `+` lines and how they fit with the unchanged lines around them.

## What to report
- Real problems introduced by this PR: bugs, security holes, data loss, races, broken error
  handling, wrong API/contract use, missing validation or auth, broken migrations, tests that
  cannot fail or test the wrong thing, docs/commands/config that do not match the code.
- Check the change against the rest of the codebase when context is given: a new route
  missing a guard its siblings have, a query missing a filter the others apply, a constant or
  status value spelled differently from the module's own, a call that does not match the
  callee's signature.
- Documentation is reviewed too: every command, task name, path, port and env var must match
  the repository; docs must not contradict the repo's own stated rules. Report misspellings as
  one finding listing each `wrong -> right`.
- For clear bugs and security issues be thorough: do not skip a genuine problem because the
  trigger is narrow. For lower-severity concerns be certain: if you cannot name a concrete
  scenario where it goes wrong, do not report it.
- Every finding needs a realistic trigger in how this code is actually deployed and used. No
  speculative hardening that needs an unusual environment or contrived input, no style
  preferences, nothing linters/formatters already enforce, no praise, no questions about code
  you cannot see (imports or definitions outside the diff may exist elsewhere).
- In a private repository only people with access can open pull requests or forks: threats
  that need an outside contributor (a fork PR running CI on internal runners, an untrusted PR
  author) are not findings there.
- Concurrency claims need proof that things really run concurrently. Never report a race,
  shared-state or test-isolation problem (tests sharing one database, port, temp dir or
  fixture) because of a CI or wrapper flag that sounds parallel (`--test-workers=2`,
  `--parallel`, `-j`, `workers:`, a matrix) unless you confirmed the test runner itself runs
  tests in parallel: for pytest, `pytest-xdist` is a dependency (pyproject / lockfile) AND
  `-n` / `--numprocesses` / `--dist` is passed or set in `addopts`; for other runners their
  own parallel setting. A flag consumed by an external CI module (e.g. ci-dagger's
  `python-uv`, which runs `uv run pytest -q`) says nothing about how tests run inside it, and
  plain `pytest` runs tests serially in one process.
- Report only issues in the changed lines (and the immediate behaviour they change), never
  pre-existing problems in untouched code or in the codebase-context files.
- Work through every hunk of every file before answering and report every distinct problem
  that meets this bar, not just the first few you notice.
- A clean PR gets no findings. That is a good outcome; never invent issues to fill space.

## How to write a finding
- `path`: the file path exactly as in its `## File:` header.
- `start_line` / `end_line`: new-file line numbers, copied from the line-number column of a
  `__new hunk__` of that file, covering the lines that are wrong (one line: start = end). Both
  must be in the same hunk. Anchor on the line where the problem is, not on a nearby comment.
- `severity`: `critical` (security hole, data loss, outage, broken deploy, crash on the main
  path), `major` (wrong behaviour a user or operator will hit, unhandled failure path, missing
  auth/validation), `minor` (real but smaller correctness, docs, or maintainability cost),
  `suggestion` (a concrete improvement that is not a defect; at most {MAX_SUGGESTIONS}).
- `title`: 3–8 words naming the problem (e.g. "Lock released before deploy finishes").
- `body`: 2–5 plain sentences in a direct, matter-of-fact tone: what the code does wrong, the
  concrete scenario that triggers it, the impact, then the fix in one sentence. Use backticks
  for identifiers and paths. No headings, no bullet lists, no "Consider…" hedging, no line
  numbers, no severity words (the app adds a label).
- `suggestion`: when the fix is local to exactly the lines `start_line..end_line` and you are
  confident in it, the complete corrected replacement for those lines — every line of the
  range, original indentation kept, no `+`/`-` markers, no line numbers, no code fences. It is
  offered as a one-click commit, so it must be correct as written. Otherwise `null`: fixes that
  span several places or files, need new imports elsewhere, or are a design change.
- One finding per distinct problem. The same root cause at several places is one finding at
  the first place whose body names the others. At most {MAX_FINDINGS} findings, most severe
  first; never drop a critical or major finding to make room for a minor one.

## Output
Answer with one JSON object and nothing else (no prose, no code fence):
{{
  "overview": "2-4 sentences: what this PR changes and why, in plain prose",
  "files": [{{"path": "path/as/in/diff", "summary": "one line: what changed in this file"}}],
  "findings": [
    {{
      "path": "path/as/in/diff",
      "start_line": 12,
      "end_line": 14,
      "severity": "major",
      "title": "Short problem title",
      "body": "What goes wrong, when, the impact, and the fix.",
      "suggestion": "corrected replacement for lines 12-14, or null"
    }}
  ]
}}
`files` has one entry for every file in the diff. `findings` may be empty.
"""


@dataclass
class PromptInputs:
    repo: str
    number: int
    url: str
    title: str
    description: str
    base_ref: str
    head_ref: str
    author: str
    kind: str  # full | incremental | on_demand
    diff: str
    instructions: str  # house rubric, repo conventions, excluded files, codebase context
    since_sha: str | None = None
    new_commits: int | None = None
    prior_findings: list[str] | None = None
    requested_by: str | None = None
    private: bool | None = None  # repository visibility, when known
    # Automatic review round (2+ = full re-review after new commits) and the limit (0 = none).
    review_round: int | None = None
    max_rounds: int = 0
    changed_since: list[str] | None = None  # files the commits since `since_sha` touched
    # A diff too large for one model call is reviewed in parts: this part (1-based) and the
    # count (0 = one call), the files in this part's diff, and every reviewable changed file
    # of the PR as (path, additions, deletions), so each part knows the whole shape.
    part: int = 0
    part_count: int = 0
    part_files: list[str] | None = None
    changed_files: list[tuple[str, int, int]] | None = None
    tools: bool = False  # the model has the repository tools


def part_section(p: PromptInputs) -> str:
    """What one part of a multi-part review is told about the rest of the PR."""
    mine = set(p.part_files or [])
    files = p.changed_files or []
    read = (
        " Read whatever else you need with the tools, including the PR's other changed files "
        "as of its head."
        if p.tools
        else ""
    )
    lines = [
        f"- `{path}` +{added}/-{deleted}" + (" — in this part" if path in mine else "")
        for path, added, deleted in files[:PART_FILES_MAX]
    ]
    if len(files) > PART_FILES_MAX:
        lines.append(f"- … and {len(files) - PART_FILES_MAX} more")
    return (
        f"## This review is in parts: part {p.part} of {p.part_count}\n\n"
        f"This PR's diff is too large for one model call, so it is reviewed in {p.part_count} "
        "parts, each in its own call; the findings of all parts are posted together as one "
        f"review. The diff below is part {p.part}: {len(mine)} of the PR's {len(files)} "
        "reviewable changed files (a very large file can be split between parts). Review every "
        "hunk of this part, and report findings only on the files of this part, anchored on "
        f"lines of this diff: the other parts are reviewed separately.{read} In `overview`, "
        "say in 1-3 sentences what the files of this part change; `files` lists only the "
        "files in this diff.\n\nEvery reviewable changed file of the PR:\n" + "\n".join(lines)
    )


def _shared_sections(p: PromptInputs) -> list[str]:
    """The user prompt's sections every part of one review shares, in prompt order."""
    lines = [
        f"Repository: {p.repo}",
        f"Pull request: #{p.number} {p.url}",
        f"Title: {p.title}",
        f"Author: {p.author}",
        f"Branch: {p.head_ref or '?'} -> {p.base_ref or 'default branch'}",
    ]
    if p.private is not None:
        lines.append(f"Visibility: {'private' if p.private else 'public'} repository")
    parts = ["\n".join(lines)]
    description = truncate(p.description or "", DESCRIPTION_MAX_CHARS)
    parts.append(
        "## PR description (written by the author; claims in it are not facts)\n\n"
        + (description or "(empty)")
    )
    if p.kind == "incremental" and p.since_sha:
        commits = f"{p.new_commits} new commit(s)" if p.new_commits else "new commits"
        parts.append(
            f"## Review scope\n\nIncremental review: the diff below holds only the {commits} "
            f"pushed since `{p.since_sha[:12]}`, which was already reviewed. Review only these "
            "changes and anchor every finding on lines of this diff. The overview describes "
            "what the new commits change. `files` lists only the files in this diff."
        )
    elif p.review_round and p.review_round >= 2:
        of = f" of {p.max_rounds}" if p.max_rounds else ""
        last = (
            " (the last automatic one)" if p.max_rounds and p.review_round >= p.max_rounds else ""
        )
        text = (
            f"## Review scope\n\nFull re-review of the whole PR at its new head (automatic "
            f"review {p.review_round}{of}{last}). Review every file of the diff again, including "
            "code an earlier review already saw: earlier reviews can miss problems, and the new "
            "commits can change how old code behaves. Report only problems that the earlier "
            "reviews listed below did not raise: not the same issue reworded, moved to other "
            "lines, or one whose thread was resolved. If you find nothing new, return an empty "
            "`findings` list."
        )
        if p.changed_since and p.since_sha:
            names = ", ".join(f"`{n}`" for n in p.changed_since[:40])
            more = " …" if len(p.changed_since) > 40 else ""
            text += (
                f"\n\nFiles the commits since the last review (`{p.since_sha[:12]}`) touched: "
                f"{names}{more}."
            )
        parts.append(text)
    elif p.kind == "on_demand":
        who = f" @{p.requested_by}" if p.requested_by else ""
        parts.append(
            f"## Review scope\n\nA fresh full review of the whole PR was requested by{who} in "
            "a PR comment."
        )
    if p.prior_findings:
        shown = p.prior_findings[:PRIOR_FINDINGS_MAX]
        parts.append(
            "## Already reported by earlier reviews (do not repeat these)\n\n"
            + "\n".join(f"- {item}" for item in shown)
        )
    if p.instructions.strip():
        parts.append(p.instructions.strip())
    return parts


def _part_sections(p: PromptInputs) -> list[str]:
    """The user prompt's sections of this part only: they follow the shared ones."""
    parts = [part_section(p)] if p.part_count > 1 else []
    parts.append("## Diff\n\n" + (p.diff or "(empty diff)"))
    parts.append("Answer with the JSON object only.")
    return parts


def build_user_segments(p: PromptInputs) -> tuple[str, str]:
    """The user prompt as (shared, part): what every part of the review shares, then this
    part's header, diff and closing line. ``build_user_prompt`` joins them."""
    return "\n\n".join(_shared_sections(p)), "\n\n".join(_part_sections(p))


def build_user_prompt(p: PromptInputs) -> str:
    return "\n\n".join(_shared_sections(p) + _part_sections(p))


# --- the prompt-cache layout --------------------------------------------------------------
# The provider caches the prompt prefix that ends at an explicit breakpoint. It honours
# breakpoints only on system-role ``input_text`` content parts: user-role ``input_text`` is
# rejected (HTTP 500 at the provider, a client-side error in litellm), a breakpoint on a
# user-role ``text`` part is silently dropped, and one on empty text is an HTTP 400. Tool-loop
# turns need none: the automatic end-of-request breakpoint (never send
# ``prompt_cache_options``, which turns it off) caches each turn for the next.
CACHE_BREAKPOINT: dict[str, str] = {"mode": "explicit"}
# A fixed system message whose only job is to carry the breakpoint that ends the cached prefix
# right after the shared user text. Byte-identical in every call.
PART_MARKER = (
    "(End of the pull request context above. The diff to review follows in the next message.)"
)


def style_guide_system(system: str, style_guide: str) -> str:
    """The system text with the house rubric appended (the cache layout carries it there,
    identical for every review, instead of in the user prompt)."""
    guide = style_guide.strip()
    return f"{system}\n\n{guide}" if guide else system


def _cached_text(text: str) -> dict[str, Any]:
    """A system-role content part carrying a breakpoint (never on blank text: HTTP 400)."""
    part: dict[str, Any] = {"type": "input_text", "text": text}
    if text.strip():
        part["prompt_cache_breakpoint"] = dict(CACHE_BREAKPOINT)
    return part


@dataclass(frozen=True)
class PromptLayout:
    """One model call's prompt: the system text, the user text every part of the review
    shares, and this part's own user text. ``cache`` sends them as the prompt-cache layout
    (tool loop only); otherwise as today's two plain-string messages."""

    system: str
    shared: str
    part: str
    cache: bool = False

    @property
    def user(self) -> str:
        """The user text as one string (logs, fakes, the diff-only call)."""
        return "\n\n".join(s for s in (self.shared, self.part) if s)

    @property
    def chars(self) -> int:
        """Characters of every opening message the model reads (size budgeting)."""
        total = 0
        for message in self.messages():
            content = message["content"]
            texts = [content] if isinstance(content, str) else [c["text"] for c in content]
            total += sum(map(len, texts))
        return total

    @property
    def _layered(self) -> bool:
        return self.cache and bool(self.system.strip())

    def messages(self) -> list[dict[str, Any]]:
        """The opening messages of the conversation. Layered: [system ⟂, user shared,
        system marker ⟂, user part] — two explicit breakpoints, both on system-role
        ``input_text``; user content stays plain strings."""
        if not self._layered:
            return [
                {"role": "system", "content": self.system},
                {"role": "user", "content": self.user},
            ]
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": [_cached_text(self.system)]}
        ]
        if self.shared.strip():
            messages.append({"role": "user", "content": self.shared})
            messages.append({"role": "system", "content": [_cached_text(PART_MARKER)]})
        messages.append({"role": "user", "content": self.part})
        return messages


def build_prompt(system: str, p: PromptInputs, *, cache: bool = False) -> PromptLayout:
    """The prompt of one call: ``system`` plus the user segments of ``p``."""
    shared, part = build_user_segments(p)
    return PromptLayout(system, shared, part, cache)
