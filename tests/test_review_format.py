"""Pure pieces of the native inline review: hunks, anchors, suggestions, parsing, rendering."""

from __future__ import annotations

import json
import re
from dataclasses import replace

import pytest

from app.review_format import (
    MAX_FINDINGS,
    Anchor,
    DiffFile,
    Finding,
    ModelOutputError,
    PriorComment,
    ReviewBody,
    anchor_finding,
    extend_patch,
    is_duplicate,
    normalize_findings,
    parse_hunks,
    parse_model_output,
    parts_overview,
    prior_from_review_body,
    review_comment_payload,
    review_marker_sha,
    split_diff,
    suggestion_applies,
    synthesize_patch,
    tally_line,
)

PATCH = (
    "@@ -1,4 +1,5 @@\n"
    " import os\n"
    "-x = 1\n"
    "+x = 2\n"
    "+y = x + 1\n"
    " def f():\n"
    "     return x\n"
    "@@ -20,3 +21,4 @@ def g():\n"
    "     a = 1\n"
    "+    b = 2\n"
    "     return a\n"
    "\\ No newline at end of file"
)


def diff_file(patch: str = PATCH, path: str = "a.py") -> DiffFile:
    return DiffFile.from_api({"filename": path, "status": "modified", "patch": patch})


def finding(start: int, end: int | None = None, **kw) -> Finding:
    base = dict(
        path="a.py",
        start_line=start,
        end_line=end if end is not None else start,
        severity="major",
        title="Wrong value",
        body="`x` must stay 1: callers divide by it.",
        suggestion=None,
    )
    return Finding(**(base | kw))


# --- hunks -----------------------------------------------------------------------------
def test_parse_hunks_tracks_right_side_lines_and_additions():
    hunks = parse_hunks(PATCH)
    assert [(h.start, h.end) for h in hunks] == [(1, 5), (21, 23)]
    assert hunks[0].added == {2, 3} and hunks[1].added == {22}
    assert hunks[0].text[3] == "y = x + 1" and hunks[1].text[23] == "    return a"
    assert parse_hunks("") == [] and parse_hunks(None) == []
    # a deletion-only hunk has no RIGHT-side line to comment on
    assert parse_hunks("@@ -3,2 +2,0 @@\n-a\n-b") == []


def test_split_diff_is_one_numbered_part_when_the_diff_fits():
    small, big = diff_file(), diff_file("@@ -1 +1 @@\n-a\n+" + "b" * 5000, path="big.py")
    [rendered] = split_diff([small, big], budget_chars=10_000)
    assert "## File: 'a.py'" in rendered.text and "3 +y = x + 1" in rendered.text
    assert "__old hunk__" in rendered.text
    assert rendered.shown == ["a.py", "big.py"]
    assert rendered.text.index("'a.py'") < rendered.text.index("'big.py'")  # input order


def added_file(path: str, lines: int, width: int = 40) -> DiffFile:
    body = "\n".join("+" + f"line {n} ".ljust(width, "x") for n in range(1, lines + 1))
    return DiffFile.from_api(
        {"filename": path, "status": "added", "patch": f"@@ -0,0 +1,{lines} @@\n{body}"}
    )


def new_line_numbers(text: str, path: str) -> list[int]:
    """New-file line numbers of ``path``'s added lines shown in a rendered diff."""
    numbers, current = [], None
    for line in text.splitlines():
        if line.startswith("## File: "):
            current = line.split("'")[1]
        elif current == path and " +line " in line:
            numbers.append(int(line.split(" ", 1)[0]))
    return numbers


def test_split_diff_splits_an_oversized_diff_into_balanced_parts_by_path():
    files = [added_file(f"{d}/m{n}.py", 120) for d in ("src", "tests", "docs") for n in (1, 2)]
    one = sum(len(p.text) for p in split_diff(files, 10**9))
    parts = split_diff(files, budget_chars=one // 2)
    assert len(parts) in (2, 3)
    assert all(len(p.text) <= one // 2 for p in parts)
    shown = [path for p in parts for path in p.shown]
    assert sorted(shown) == sorted(f.path for f in files) and len(set(shown)) == len(shown)
    assert shown == sorted(shown)  # path order: a directory's files stay together
    for f in files:  # every line of every file, once
        lines = [n for p in parts for n in new_line_numbers(p.text, f.path)]
        assert lines == list(range(1, 121))


def test_split_diff_cuts_a_file_larger_than_a_part_between_lines_and_keeps_every_line():
    giant, small = added_file("big.py", 3000), added_file("a.py", 10)
    budget = 40_000
    parts = split_diff([giant, small], budget_chars=budget)
    assert len(parts) >= 4 and all(len(p.text) <= budget for p in parts)
    lines = [n for p in parts for n in new_line_numbers(p.text, "big.py")]
    assert lines == list(range(1, 3001))  # nothing dropped, numbering carried over
    text = "\n".join(p.text for p in parts)
    assert "## File: 'big.py' (part 1 of " in text
    assert [path for p in parts for path in p.shown].count("a.py") == 1
    # the cut hunks carry their own new-file start lines
    starts = [int(m) for m in re.findall(r"^@@ -0,0 \+(\d+),\d+ @@", text, re.MULTILINE)]
    assert starts[0] == 1 and starts == sorted(starts) and len(starts) >= 4


def test_split_diff_shows_the_start_of_a_line_too_long_for_any_part():
    huge = DiffFile.from_api(
        {"filename": "min.js", "status": "added", "patch": "@@ -0,0 +1 @@\n+" + "z" * 90_000}
    )
    parts = split_diff([huge], budget_chars=30_000)
    assert len(parts) == 1 and parts[0].shown == ["min.js"]
    assert len(parts[0].text) <= 30_000 and "too long to show" in parts[0].text


def test_extend_patch_widens_context_without_merging_hunks():
    head = "\n".join(f"l{n}" for n in range(1, 41))
    patch = "@@ -10,2 +10,3 @@\n l10\n+l11\n l12\n@@ -20,2 +21,2 @@\n-x\n+l21\n l22"
    widened = extend_patch(patch, head, before=4, after=2)
    hunks = parse_hunks(widened)
    assert [(h.start, h.end) for h in hunks] == [(6, 14), (17, 24)]
    assert hunks[0].text[6] == "l6" and hunks[1].text[24] == "l24"
    assert extend_patch(patch, None) == patch


def test_synthesize_patch_for_files_github_sent_without_one():
    patch = synthesize_patch("a\nb\nc\n", "a\nB\nc\nd\n")
    hunks = parse_hunks(patch)
    assert hunks and hunks[0].added == {2, 4}
    assert synthesize_patch("same", "same") is None


# --- anchors and suggestions -----------------------------------------------------------
def test_anchor_keeps_a_range_inside_one_hunk():
    diff = {"a.py": diff_file()}
    assert anchor_finding(finding(2, 3), diff) == Anchor("a.py", 2, 3, exact=True)
    assert anchor_finding(finding(22), diff) == Anchor("a.py", 22, 22, exact=True)


def test_anchor_falls_back_to_a_line_inside_the_range_or_nearby():
    diff = {"a.py": diff_file()}
    # spans both hunks: GitHub rejects that, so the comment goes on the last valid line
    assert anchor_finding(finding(4, 22), diff) == Anchor("a.py", 22, 22, exact=False)
    # just past the hunk: nearest commentable line
    assert anchor_finding(finding(7), diff) == Anchor("a.py", 5, 5, exact=False)
    # far from any hunk, or a file not in the diff: folded into the review body
    assert anchor_finding(finding(12), diff) is None
    assert anchor_finding(finding(2, path="other.py"), diff) is None


def test_incremental_anchor_is_limited_to_lines_of_the_new_commits():
    diff = {"a.py": diff_file()}
    allowed = {"a.py": {21, 22, 23}}
    assert anchor_finding(finding(22), diff, allowed) == Anchor("a.py", 22, 22, exact=True)
    assert anchor_finding(finding(2, 3), diff, allowed) is None  # already-reviewed code
    assert anchor_finding(finding(2), diff, {}) is None


def test_suggestion_must_replace_exactly_the_anchored_lines_and_change_them():
    file = diff_file()
    good = finding(2, 3, suggestion="x = 1\ny = x + 1")
    assert suggestion_applies(good, Anchor("a.py", 2, 3, True), file)
    assert not suggestion_applies(good, Anchor("a.py", 3, 3, False), file)  # moved anchor
    same = finding(2, 3, suggestion="x = 2\ny = x + 1")
    assert not suggestion_applies(same, Anchor("a.py", 2, 3, True), file)
    diffy = finding(2, 3, suggestion="-x = 2\n+x = 1")
    assert not suggestion_applies(diffy, Anchor("a.py", 2, 3, True), file)
    assert not suggestion_applies(finding(2), Anchor("a.py", 2, 2, True), file)


def test_review_comment_payload_for_single_and_multi_line_comments():
    f = finding(2, 3, severity="critical", suggestion="x = 1\ny = x + 1")
    multi = review_comment_payload(f, Anchor("a.py", 2, 3, True), with_suggestion=True)
    assert multi["path"] == "a.py" and multi["side"] == "RIGHT"
    assert (multi["start_line"], multi["line"], multi["start_side"]) == (2, 3, "RIGHT")
    assert multi["body"].startswith("**[critical]** `x` must stay 1")
    assert "```suggestion\nx = 1\ny = x + 1\n```" in multi["body"]
    assert f"key={f.key}" in multi["body"]
    single = review_comment_payload(
        finding(22, severity="minor"), Anchor("a.py", 22, 22, False), with_suggestion=False
    )
    assert "start_line" not in single and single["line"] == 22
    assert not single["body"].startswith("**[")  # only bugs / security get a label
    assert "```suggestion" not in single["body"]


# --- model output ----------------------------------------------------------------------
def test_parse_model_output_tolerates_fences_trailing_commas_and_aliases():
    raw = """Here you go:
```json
{"overview": "Adds  a thing.", "files": [{"path": "a.py", "summary": "Adds x."}],
 "findings": [
   {"path": "/a.py", "start_line": 3, "end_line": 2, "severity": "HIGH", "title": "Bug",
    "body": "Breaks.", "suggestion": "```python\\nx = 1\\n```"},
   {"path": "a.py", "start_line": "x", "body": "no line"},
   {"path": "a.py", "start_line": 4, "severity": "nit", "body": "Rename.", "suggestion": null},
 ]}
```"""
    review = parse_model_output(raw)
    assert review.overview == "Adds a thing." and review.files == {"a.py": "Adds x."}
    first, second = review.findings
    assert (first.path, first.start_line, first.end_line) == ("a.py", 2, 3)
    assert first.severity == "major" and first.suggestion == "x = 1"
    assert second.severity == "suggestion" and second.end_line == 4 and second.title
    with pytest.raises(ModelOutputError):
        parse_model_output("I could not review this PR.")


def test_normalize_findings_orders_merges_and_caps():
    items = [
        finding(1, severity="minor", title="Typo in docs"),
        finding(2, severity="critical", title="SQL injection"),
        finding(2, severity="major", title="SQL injection"),  # same key: merged
        finding(3, severity="major", title="Other bug"),
    ] + [finding(10 + n, severity="suggestion", title=f"Idea {n}") for n in range(5)]
    out = normalize_findings(items)
    assert [f.title for f in out[:3]] == ["SQL injection", "Other bug", "Typo in docs"]
    assert sum(f.severity == "suggestion" for f in out) == 3
    many = [finding(n, title=f"Bug {n}") for n in range(1, 30)]
    assert len(normalize_findings(many)) == MAX_FINDINGS


def test_duplicates_of_earlier_reviews_by_key_or_by_place_and_wording():
    f = finding(22, title="Unchecked write")
    by_key = PriorComment("a.py", 90, f.key, "anything")
    assert is_duplicate(f, None, [by_key])
    nearby = PriorComment("a.py", 23, None, "**[major]** `x` must stay 1: callers divide by it.")
    assert is_duplicate(f, Anchor("a.py", 22, 22, True), [nearby])
    unrelated = PriorComment("a.py", 23, None, "Log message has a typo in `connect`.")
    assert not is_duplicate(f, Anchor("a.py", 22, 22, True), [unrelated])
    other_file = PriorComment("b.py", 22, f.key, "same")
    assert not is_duplicate(f, None, [other_file])
    api = PriorComment.from_api(
        {"path": "a.py", "line": None, "original_line": 7,
         "body": f"text\n\n<!-- gh-pr-reviewer:finding key={f.key} severity=major -->"}
    )  # fmt: skip
    assert (api.line, api.key) == (7, f.key)


def test_duplicates_tolerate_line_moves_and_rewording_of_the_same_issue():
    f = finding(40, title="Token never expires")
    f.body = "`create_token()` sets no `exp` claim, so a leaked token works forever."
    moved = PriorComment("a.py", 12, None, "**[major]** `create_token()` sets no `exp` claim: a "
                         "leaked token stays valid forever.\n\n<!-- gh-pr-reviewer:finding "
                         f"key={f.key} severity=major -->")  # fmt: skip
    assert is_duplicate(f, None, [moved])  # 28 lines away after the push, same issue
    other = PriorComment("a.py", 12, None, "`load_config()` ignores `TIMEOUT` from the env.")
    assert not is_duplicate(f, None, [other])  # same file, different issue
    loose = PriorComment("a.py", 90, None, "`create_token()` tokens have no expiry.")
    assert not is_duplicate(f, None, [loose])  # loosely similar: kept while the thread is open
    resolved = PriorComment("a.py", 90, None, loose.text, resolved=True)
    assert is_duplicate(f, None, [resolved])  # a human resolved it: not raised again


def test_findings_listed_in_earlier_review_bodies_are_prior_findings():
    folded = body(folded=[(finding(9, title="Missing env check"), "a.py:9-11")],
                  findings=[finding(9, title="Missing env check")]).render()  # fmt: skip
    fallback = (
        "<!-- gh-pr-reviewer:review sha=abc kind=full -->\n\n## Pull request overview\n\n"
        "**`b.py:3`**\n\n**[major]** `f()` drops errors.\n\n<!-- gh-pr-reviewer:finding "
        "key=feedfacecafe severity=major -->\n\n<sub>gh-pr-reviewer · footer</sub>"
    )
    [item] = prior_from_review_body(folded)
    assert (item.path, item.line) == ("a.py", 9)
    assert item.key == finding(30, title="Missing env check").key  # same file + title
    [refused] = prior_from_review_body(fallback)
    assert (refused.path, refused.line, refused.key) == ("b.py", 3, "feedfacecafe")
    assert "footer" not in refused.text
    assert prior_from_review_body("## Pull request overview\n\nNo issues found.") == []


def test_round_two_body_with_and_without_new_findings():
    new = finding(5, title="w is never used")
    text = body(round=2, max_rounds=2, findings=[new], duplicates=2).render()
    assert "round=2 -->" in text and review_marker_sha(text) == "c" * 40
    assert "Re-reviewed the whole PR at `ccccccc` after new commits (automatic review 2 of 2)" in (
        text
    )
    assert "2 findings already raised in an earlier review are not repeated." in text
    assert "This was the last automatic review of this PR" in text
    quiet = body(round=2, max_rounds=2, duplicates=1).render()
    assert review_marker_sha(quiet) == "c" * 40 and "## Pull request overview" not in quiet
    assert (
        "(automatic review 2 of 2): 1 of 2 changed files, nothing new beyond the earlier review."
        " 1 finding already raised is not repeated."
    ) in quiet
    assert "comment `/review` to ask for another" in quiet
    assert "This was the last" not in body(round=2, max_rounds=3).render()
    assert "round=" not in body().render() and "Re-reviewed" not in body(round=1).render()


# --- review body ----------------------------------------------------------------------------
def body(**kw) -> ReviewBody:
    base = dict(
        head_sha="c" * 40,
        kind="full",
        overview="Adds a deactivation endpoint.",
        file_summaries={"a.py": "Adds the route | and service."},
        reviewed_files=["a.py"],
        total_files=2,
        findings=[],
        folded=[],
        duplicates=0,
        footer=(72_000, 12_500, 0.0067, "high"),
    )
    return ReviewBody(**(base | kw))


def test_clean_review_body_is_short_with_overview_table_and_footer():
    text = body().render()
    assert review_marker_sha(text) == "c" * 40
    assert "## Pull request overview\n\nAdds a deactivation endpoint." in text
    assert "Reviewed 1 of 2 changed files and generated 0 comments." in text
    assert "| `a.py` | Adds the route \\| and service. |" in text
    assert "<summary>Show a summary per file</summary>" in text
    assert "No issues found." in text
    assert "<sub>gh-pr-reviewer · gpt-4.1 high · 72.0k in / 12.5k out · $0.0067</sub>" in (text)
    for chrome in ("Estimated effort", "🔍", "key observations", "PR Reviewer Guide"):
        assert chrome not in text


def test_review_body_tally_folded_comments_and_incremental_lead():
    issues = [finding(2, severity="major"), finding(9, severity="suggestion", title="Idea")]
    text = body(
        kind="incremental",
        findings=issues,
        folded=[(issues[1], "a.py:9")],
        new_commits=2,
        since_sha="abcdef1234",
        duplicates=1,
    ).render()
    assert text.count("Reviewed 2 new commits since `abcdef1`.") == 1
    assert "Reviewed 1 of 2 files changed by the new commits and generated 2 comments." in text
    assert tally_line(issues) in text and "**2 comments:** 1 issue, 1 suggestion" in text
    assert "1 finding already raised in an earlier review is not repeated." in text
    assert "<summary>1 comment outside the diff</summary>" in text
    assert "- `a.py:9` — **Idea.**" in text
    clean_again = body(duplicates=2).render()
    assert "No new issues found." in clean_again


def test_review_body_of_a_review_in_parts_says_so_and_names_unreviewed_files():
    overview = parts_overview([(40, "Adds the queue."), (12, None), (30, "Tests it.")])
    assert overview.startswith("Reviewed in 3 parts (the diff is too large for one model call)")
    assert "- **Part 1** (40 files): Adds the queue." in overview
    assert "- **Part 2** (12 files): not reviewed: this part failed" in overview
    text = body(overview=overview, parts=3, unreviewed_files=["b.py"]).render()
    assert "## Pull request overview\n\nReviewed in 3 parts" in text
    assert "1 file could not be reviewed (the part of this review that held them failed;" in text
    assert "`b.py`" in text
    assert text.endswith("72.0k in / 12.5k out · $0.0067 · reviewed in 3 parts</sub>")
    assert "reviewed in" not in body().render()
    quiet = body(round=2, max_rounds=2, parts=2, unreviewed_files=["b.py"]).render()
    assert "1 file could not be reviewed (a part of this review failed)." in quiet
    assert quiet.endswith("reviewed in 2 parts</sub>")


def test_model_contract_example_in_prompt_is_valid_json():
    from app.review_prompt import SYSTEM_PROMPT

    start = SYSTEM_PROMPT.index("{\n")
    example = SYSTEM_PROMPT[start : SYSTEM_PROMPT.index("\n}\n", start) + 2]
    data = json.loads(example)
    assert set(data) == {"overview", "files", "findings"}
    assert set(data["findings"][0]) == {
        "path", "start_line", "end_line", "severity", "title", "body", "suggestion",
    }  # fmt: skip


def test_prompt_states_repository_visibility_when_known():
    from app.review_prompt import PromptInputs, build_user_prompt

    def prompt(private):
        return build_user_prompt(
            PromptInputs(
                repo="o/r", number=1, url="u", title="t", description="", base_ref="main",
                head_ref="f", author="a", kind="full", diff="d", instructions="", private=private,
            )
        )  # fmt: skip

    assert "Visibility: private repository" in prompt(True)
    assert "Visibility: public repository" in prompt(False)
    assert "Visibility" not in prompt(None)


def test_a_part_prompt_shows_the_whole_pr_and_comes_right_before_the_diff():
    from app.review_prompt import PART_FILES_MAX, PromptInputs, build_user_prompt

    files = [(f"src/m{n}.py", n, 1) for n in range(PART_FILES_MAX + 5)]
    base = PromptInputs(
        repo="o/r", number=1, url="u", title="t", description="", base_ref="main",
        head_ref="f", author="a", kind="full", diff="DIFF", instructions="## House rules",
        changed_files=files, part_files=["src/m1.py", "src/m2.py"],
    )  # fmt: skip
    assert "in parts" not in build_user_prompt(base)  # one call: no part section
    text = build_user_prompt(replace(base, part=2, part_count=3, tools=True))
    assert "## This review is in parts: part 2 of 3" in text
    assert "part 2: 2 of the PR's 305 reviewable changed files" in text
    assert "- `src/m1.py` +1/-1 — in this part" in text and "- `src/m3.py` +3/-1\n" in text
    assert "- … and 5 more" in text
    assert "other changed files as of its head" in text
    assert (
        text.index("## House rules")
        < text.index("## This review is in parts")
        < (text.index("## Diff"))
    )
    no_tools = build_user_prompt(replace(base, part=1, part_count=3))
    assert "other changed files as of its head" not in no_tools


def test_prompts_forbid_concurrency_claims_from_ci_worker_flags():
    # Generic regression guard: do not infer unsafe test concurrency from undocumented flags.
    from app.review_agent import AGENT_PROMPT
    from app.review_prompt import SYSTEM_PROMPT

    for needle in ("`--test-workers=2`", "`pytest-xdist` is a dependency", "`-n`",
                   "plain `pytest` runs tests serially"):  # fmt: skip
        assert needle in SYSTEM_PROMPT
    assert "`--test-workers=N`" in AGENT_PROMPT and "no finding" in AGENT_PROMPT


def test_post_merge_comment_lists_the_most_severe_findings_capped():
    from app.review_format import POST_MERGE_MAX_FINDINGS, Finding, post_merge_comment

    findings = [Finding("src/a b.py", i, i, "minor", f"minor {i}", "b") for i in range(1, 10)]
    findings.append(Finding("x.py", 5, 7, "critical", "Leaks | the token", "b"))
    body = post_merge_comment(
        repo_url="https://github.com/o/r", head_sha="f" * 40, base_ref=None,
        findings=findings, footer=(1000, 200, 0.5, "xhigh"),
    )  # fmt: skip
    bullets = [line for line in body.splitlines() if line.startswith("- ")]
    assert len(bullets) == POST_MERGE_MAX_FINDINGS + 1 and bullets[-1] == "- …and 2 more"
    assert bullets[0] == (
        f"- **[critical]** [`x.py:5-7`](https://github.com/o/r/blob/{'f' * 40}/x.py#L5-L7)"
        " — Leaks \\| the token"
    )
    assert f"(https://github.com/o/r/blob/{'f' * 40}/src/a%20b.py#L1)" in bullets[1]
    assert "10 findings at" in body and "follow up on the base branch" in body
