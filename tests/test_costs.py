"""Configurable cost estimates, month projection, and the pull-request comment footer."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from app.costs import (
    CACHED_INPUT_PER_MILLION_USD,
    FOOTER_MARKER,
    INPUT_PER_MILLION_USD,
    OUTPUT_PER_MILLION_USD,
    apply_footer,
    compute_cost,
    format_cost,
    project_month,
    render_footer,
)


def test_default_rates_are_unknown_not_provider_specific():
    assert (
        INPUT_PER_MILLION_USD,
        CACHED_INPUT_PER_MILLION_USD,
        OUTPUT_PER_MILLION_USD,
    ) == (0.0, 0.0, 0.0)


def test_cost_splits_fresh_and_cached_input():
    # 1M prompt of which 400k is cached, plus 100k output.
    cost = compute_cost(
        1_000_000,
        400_000,
        100_000,
        fresh_rate=1.0,
        cached_rate=0.2,
        output_rate=3.0,
    )
    assert cost == pytest.approx(0.6 * 1.0 + 0.4 * 0.2 + 0.1 * 3.0)


def test_cost_edge_cases():
    assert compute_cost(0, 0, 0) == 0
    assert compute_cost(None, None, None) == 0  # type: ignore[arg-type]
    # Cached input can never exceed the prompt, and negative fresh input is impossible.
    assert compute_cost(100, 500, 0, fresh_rate=1.0, cached_rate=0.2) == pytest.approx(
        100 * 0.2 / 1e6
    )
    assert compute_cost(1_000_000, 0, 0, fresh_rate=1.0) == pytest.approx(1.0)


def test_project_month_run_rate():
    now = datetime(2026, 10, 11, 0, 0, tzinfo=UTC)
    assert project_month(10.0, now) == pytest.approx(31.0)
    assert project_month(5.0, datetime(2026, 10, 1, 0, 0, tzinfo=UTC)) == 5.0


def test_format_cost():
    assert format_cost(0) == "$0.00"
    assert format_cost(0.004) == "$0.0040"
    assert format_cost(1.234) == "$1.23"


def test_render_footer():
    line = render_footer(12_345, 4_560, 0.1234, "high")
    assert line == "gh-pr-reviewer · gpt-4.1 high · 12.3k in / 4.6k out · $0.12"


def test_apply_footer_appends_and_replaces():
    body = "## Review summary\n\nstuff"
    once = apply_footer(body, "gh-pr-reviewer · a")
    assert once.endswith(f"{FOOTER_MARKER}\n<sub>gh-pr-reviewer · a</sub>")
    twice = apply_footer(once, "gh-pr-reviewer · b")
    assert twice.count(FOOTER_MARKER) == 1
    assert "· a</sub>" not in twice and twice.endswith("<sub>gh-pr-reviewer · b</sub>")


def test_apply_footer_keeps_pr_agent_state_marker_last_and_parseable():
    from pr_agent.algo.review_finding_state import parse_review_state

    state = '<!-- pr-agent-review-state:v1\n{"version": 1, "findings": []}\n-->'
    body = f"<!-- pr-agent:review:full -->\n## Review summary\n\nfindings\n\n{state}"
    out = apply_footer(body, "gh-pr-reviewer · x")
    assert out.endswith(state)
    assert out.index(FOOTER_MARKER) < out.index("pr-agent-review-state")
    assert out.splitlines()[0] == "<!-- pr-agent:review:full -->"
    assert parse_review_state(out).present
    assert apply_footer(out, "gh-pr-reviewer · y").count(FOOTER_MARKER) == 1
