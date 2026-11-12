"""Our overrides reach PR-Agent's own settings object and behave as intended there."""

from __future__ import annotations

from dataclasses import dataclass

import pytest
from starlette_context import request_cycle_context

from app.config import Settings
from app.pr_agent_config import (
    IGNORE_GLOBS,
    NUM_MAX_FINDINGS,
    REVIEW_COMMANDS,
    build_overrides,
    compose_extra_instructions,
    fresh_pr_agent_settings,
    is_requirements_txt,
    load_style_guide,
    looks_pip_compiled,
    reviewable_filter,
)


@dataclass
class F:
    filename: str


def test_configuration_defaults_are_unconfigured_and_fail_closed():
    settings = Settings(_env_file=None)
    assert settings.GITHUB_APP_ID == ""
    assert settings.GITHUB_APP_PRIVATE_KEY.get_secret_value() == ""
    assert settings.GITHUB_WEBHOOK_SECRET.get_secret_value() == ""
    assert settings.LLM_API_KEY.get_secret_value() == ""
    assert settings.REVIEW_REPOS == ""
    assert settings.LLM_MODEL == "gpt-4.1"
    assert settings.LLM_PROMPT_CACHE_LAYOUT is False
    assert settings.LLM_MONTHLY_BUDGET_USD == 0 and settings.REVIEW_TOOL_MAX_COST_USD == 0


def make_settings(rsa_pem: str, **kw) -> Settings:
    values = dict(
        LLM_API_BASE="https://api.example.test/v1",
        LLM_API_KEY="test-provider-key",
        LLM_MODEL="test-model",
        LLM_REASONING_EFFORT="high",
        GITHUB_APP_ID="4242",
        GITHUB_APP_PRIVATE_KEY=rsa_pem,
    )
    return Settings(_env_file=None, **(values | kw))


@pytest.fixture
def pr_settings(rsa_pem: str):
    settings = make_settings(rsa_pem)
    overrides = build_overrides(
        settings,
        private_key_pem=rsa_pem,
        extra_instructions="RULES",
        extra_ignore_globs=["services/api/requirements.txt"],
    )
    return fresh_pr_agent_settings(overrides)


def test_model_reasoning_and_token_settings(pr_settings):
    cfg = pr_settings.config
    assert cfg.model == "openai/test-model"
    assert list(cfg.fallback_models) == []
    assert cfg.reasoning_effort == "high"
    assert "test-model" in cfg.additional_reasoning_effort_models
    assert "test-model" in cfg.no_temperature_models
    assert cfg.custom_model_max_tokens == 272000
    assert cfg.max_model_tokens == 272000
    assert pr_settings.openai.api_base == "https://api.example.test/v1"
    assert pr_settings.openai.key == "test-provider-key"
    assert pr_settings.litellm.custom_llm_provider == "openai"


def test_review_only_no_describe_no_labels(pr_settings):
    assert all(cmd[0] == "auto_review" for cmd in REVIEW_COMMANDS.values())
    assert REVIEW_COMMANDS["incremental"] == ["auto_review", "-i"]
    rv = pr_settings.pr_reviewer
    assert rv.enable_review_labels_security is False
    assert rv.enable_review_labels_effort is False
    assert rv.extra_instructions == "RULES"
    assert pr_settings.config.use_repo_settings_file is False
    assert pr_settings.config.use_global_settings_file is False
    assert list(pr_settings.config.repo_context_files) == []
    assert pr_settings.github.deployment_type == "app"
    assert pr_settings.github.app_id == 4242


def test_global_settings_are_not_mutated(pr_settings):
    from pr_agent.config_loader import global_settings

    assert global_settings.config.model != "openai/test-model"
    assert global_settings.get("openai.key", None) != "test-provider-key"


def test_settings_visible_through_pr_agent_get_settings(pr_settings):
    from pr_agent.algo.token_budget import get_max_tokens
    from pr_agent.config_loader import get_settings

    with request_cycle_context({"settings": pr_settings}):
        assert get_settings().config.reasoning_effort == "high"
        assert get_max_tokens("openai/test-model") == 272000
        globs = list(get_settings().ignore.glob)
        assert set(IGNORE_GLOBS) <= set(globs)
        assert "services/api/requirements.txt" in globs


IGNORED = [
    "package-lock.json",
    "web/package-lock.json",
    "npm-shrinkwrap.json",
    "yarn.lock",
    "pnpm-lock.yaml",
    "bun.lock",
    "bun.lockb",
    "poetry.lock",
    "uv.lock",
    "backend/uv.lock",
    "Pipfile.lock",
    "Cargo.lock",
    "go.sum",
    "Gemfile.lock",
    "composer.lock",
    "flake.lock",
    "deps/something.lock",
    "dist/app.js",
    "frontend/dist/index.html",
    "build/lib/x.py",
    "out/a.txt",
    "vendor/github.com/x/y.go",
    "node_modules/react/index.js",
    "src/__generated__/schema.ts",
    "api/generated/client.py",
    ".next/cache/x",
    "coverage/lcov.info",
    "static/app.min.js",
    "static/app.min.css",
    "static/app.js.map",
    "proto/foo_pb2.py",
    "proto/foo_pb2_grpc.py",
    "proto/foo.pb.go",
    "tests/__snapshots__/a.snap",
    "ui/Button.test.tsx.snap",
    "tests/fixtures/payload.json",
    "pkg/testdata/case1.txt",
    "data/sites.csv",
    "data/a.tsv",
    "data/a.parquet",
    "data/a.jsonl",
    "data/a.ndjson",
    "data/session.har",
    "notebooks/explore.ipynb",
    "docs/diagram.png",
    "docs/Photo.JPG",
    "assets/logo.svg",
    "docs/spec.pdf",
    "fonts/Inter.woff2",
    "release.zip",
    "docs/plan.docx",
    "sheets/model.xlsx",
    "cad/bracket.STEP",
    "cad/part.dwg",
    "cad/part.stl",
    "services/api/requirements.txt",  # pip-compile output detected for this PR
]
KEPT = [
    "src/app/main.py",
    "README.md",
    "app-manifest.yml",
    "pyproject.toml",
    "package.json",
    "frontend/src/build.ts",  # a file named build, not a build/ dir
    "src/locker.py",
    "requirements.txt",  # hand-written requirements stay reviewed
    "src/app/templates/dashboard.html",
    ".github/workflows/tests.yml",
]


def test_ignore_patterns_filter_files_in_pr_agent(pr_settings):
    from pr_agent.algo.file_filter import filter_ignored

    with request_cycle_context({"settings": pr_settings}):
        kept = {f.filename for f in filter_ignored([F(p) for p in IGNORED + KEPT])}
    assert kept == set(KEPT), sorted(set(IGNORED) & kept)


def test_pip_compile_detection():
    assert is_requirements_txt("requirements.txt")
    assert is_requirements_txt("svc/requirements-dev.txt")
    assert not is_requirements_txt("docs/requirements.md")
    assert looks_pip_compiled("#\n# This file is autogenerated by pip-compile with Python 3.12\n")
    assert looks_pip_compiled("# This file was autogenerated by uv via the following command:\n")
    assert not looks_pip_compiled("fastapi>=0.1\nhttpx\n")


def test_extra_instructions_compose_style_guide_and_repo_conventions(rsa_pem: str):
    guide = load_style_guide(make_settings(rsa_pem))
    assert "Review rubric" in guide
    assert len(guide) <= 6000
    text = compose_extra_instructions(
        guide, {"CLAUDE.md": "use uv\n" * 2000, "AGENTS.md": "agents"}, "main"
    )
    assert text.index("Output rules") < text.index("Review rubric")
    assert "Repository-specific conventions (from `main`" in text
    assert "### CLAUDE.md" in text and "[... truncated ...]" in text
    assert "path:line" in text
    assert len(text) < 14_000
    assert "No issues found." in text


def test_reviewable_filter_matches_pr_agent_ignore_rules():
    keep = reviewable_filter(["services/api/requirements.txt"])
    assert {p for p in IGNORED + KEPT if keep(p)} == set(KEPT)
    assert not keep("")


def test_findings_cap_leaves_room_for_every_seeded_class(pr_settings):
    # At 6, multi-defect PRs saturated the cap: PR-Agent then records the run as "partial"
    # (no finding can be auto-resolved later) and lower-ranked real defects were dropped.
    assert NUM_MAX_FINDINGS == 10
    assert pr_settings.pr_reviewer.num_max_findings == 10


def test_codebase_context_is_appended_last():
    text = compose_extra_instructions(
        "GUIDE", {"CLAUDE.md": "conv"}, "main", "## Codebase context X"
    )
    assert text.index("GUIDE") < text.index("### CLAUDE.md") < text.index("## Codebase context X")
    assert "every critical and major issue comes before any minor one" in text
    assert "Documentation changes are reviewed too" in text
    assert compose_extra_instructions("GUIDE", {}, "main", "  ").endswith("GUIDE")


def test_excluded_changed_files_are_named_but_capped():
    from app.pr_agent_config import MAX_LISTED_EXCLUDED_FILES, excluded_files_note

    text = compose_extra_instructions("GUIDE", {}, "main", "", ["frontend/package-lock.json"])
    assert "## Changed files not shown in the diff" in text
    assert "- `frontend/package-lock.json`" in text
    assert "never report them as missing" in text
    many = excluded_files_note([f"gen/{i:03}.py" for i in range(MAX_LISTED_EXCLUDED_FILES + 5)])
    assert many.count("- `gen/") == MAX_LISTED_EXCLUDED_FILES and "and 5 more" in many
    assert excluded_files_note([]) == ""
    assert "realistic trigger" in text


def test_provider_timeout_and_retry_settings_reach_pr_agent(rsa_pem: str):
    overrides = build_overrides(
        make_settings(rsa_pem), private_key_pem=rsa_pem, extra_instructions=""
    )
    assert overrides["config.ai_timeout"] == 900
    assert overrides["config.num_retries"] == 0

    longer = make_settings(rsa_pem, LLM_TIMEOUT_SECONDS=1200)
    overrides = build_overrides(longer, private_key_pem=rsa_pem, extra_instructions="")
    assert overrides["config.ai_timeout"] == 1200
