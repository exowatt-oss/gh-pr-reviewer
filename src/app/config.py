from pathlib import Path

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings

# repo root = src/app/config.py -> parents[2]
REPO_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_REVIEW_REPOS = ""


class Settings(BaseSettings):
    """Application settings loaded from environment variables or a local `.env` file.

    Secrets and the review allowlist default to empty; configure them explicitly before use.
    """

    ENVIRONMENT: str = "development"
    LOG_LEVEL: str = "INFO"
    CORS_ORIGINS: list[str] = ["http://localhost:8000"]
    # Commit actually checked out, supplied by the deployment environment when available.
    GIT_SHA: str = "unknown"

    # --- GitHub App credentials (configure via environment) ---
    GITHUB_APP_ID: str = ""
    # Raw PEM, PEM with literal "\n", or base64-encoded PEM. See github_app.normalize_pem.
    GITHUB_APP_PRIVATE_KEY: SecretStr = Field(default=SecretStr(""), repr=False, exclude=True)
    GITHUB_APP_SLUG: str = "gh-pr-reviewer"
    GITHUB_API_URL: str = "https://api.github.com"
    GITHUB_WEBHOOK_SECRET: SecretStr = Field(default=SecretStr(""), repr=False, exclude=True)

    # --- What to review ---
    # Comma-separated targets: ``owner/repo`` entries (listed directly every poll) and/or
    # ``owner/*`` (org mode: every repo the installation can access, discovered by search).
    REVIEW_REPOS: str = DEFAULT_REVIEW_REPOS
    # Repos reviewed before org mode existed: exempt from the ORG_MODE_SINCE cutoff and still
    # listed directly every poll. Must also be covered by REVIEW_REPOS.
    PILOT_REPOS: str = ""
    # Org mode go-live cutoff (ISO 8601) for every org. Empty = each org's first org-mode
    # start, persisted in SQLite per org. Only PRs opened, pushed or marked ready after it are
    # reviewed.
    ORG_MODE_SINCE: str = ""
    # Wait this long after a PR's head commit / ready-for-review / opened event before
    # reviewing it (a third of PRs merge within minutes). A newer push restarts the wait.
    REVIEW_SETTLE_SECONDS: int = 300
    # Full open-PR sweep of each org (catches anything the incremental search missed).
    RECONCILE_INTERVAL_SECONDS: int = 900
    # The incremental search re-reads this much before its watermark (search index lag).
    SEARCH_OVERLAP_SECONDS: int = 180
    # How long the installation's repo list and per-repo opt-out settings are cached.
    REPO_CACHE_SECONDS: int = 600
    POLL_INTERVAL_SECONDS: int = 60
    MAX_CONCURRENT_REVIEWS: int = 2
    # Of those slots, at most this many run large reviews (0 = half, rounded down, min 1),
    # so small PRs are never stuck behind giant ones.
    MAX_CONCURRENT_LARGE_REVIEWS: int = 0
    # A review is "large" above this many reviewable changed lines (additions + deletions).
    LARGE_REVIEW_LINES: int = 400
    # At most this many reviews of one repo run at once (0 = half the slots, at least 2).
    MAX_CONCURRENT_PER_REPO: int = 0
    # Shortest-job-first with aging: every this many seconds waited halves a job's size rank.
    QUEUE_AGING_SECONDS: int = 300
    # A failing head SHA is retried at most this many times, spaced by exponential backoff.
    MAX_ATTEMPTS_PER_SHA: int = 4
    RETRY_BACKOFF_SECONDS: int = 60
    RETRY_BACKOFF_MAX_SECONDS: int = 3600
    DRY_RUN: bool = False
    ENABLE_IMPROVE: bool = False

    # --- Cost guards ---
    # Automatic review rounds per PR: round 1 is the first full review, round 2 a full
    # re-review on the first push after it that posts only findings not raised before. After
    # that the PR is never reviewed automatically again (skip reason max_rounds). 0 = no limit.
    MAX_REVIEW_ROUNDS: int = 2
    # On-demand `/review` requests are not limited by the rounds; at most this many billed
    # on-demand reviews per PR in any rolling 24 h (0 = no limit).
    MAX_ON_DEMAND_REVIEWS_PER_PR_PER_DAY: int = 2
    # There is no size cap: every PR is reviewed. This is the most diff one model call gets
    # (estimated tokens of the rendered diff; LLM_MAX_CONTEXT_TOKENS minus the instructions,
    # the answer and the tool output also bounds it). A larger diff is reviewed in parts of at
    # most this size, one call each, merged into one posted review. 0 = the context bound only.
    MAX_REVIEW_DIFF_TOKENS: int = 160_000
    # Pause new full reviews at this fraction of the locally configured monthly limit; pause
    # everything at BUDGET_STOP_FRACTION. Spend comes from the app's recorded usage.
    BUDGET_PAUSE_FRACTION: float = 0.9
    BUDGET_STOP_FRACTION: float = 0.99
    LLM_MONTHLY_BUDGET_USD: float = 0.0
    # A running review re-reads its PR before each model call (at most once per this many
    # seconds) and stops when the PR merged or closed, or, for automatic reviews, a newer head
    # arrived: nothing posted, the calls already made stay billed (status `cancelled`). 0 = off
    # (only the re-check right before posting).
    REVIEW_LIVENESS_CHECK_SECONDS: int = 15

    # --- Shutdown ---
    # On SIGTERM stop taking work and let in-flight reviews finish for up to this long
    # (systemd's default stop timeout is 90 s).
    SHUTDOWN_GRACE_SECONDS: float = 80.0

    # --- OpenAI-compatible model provider ---
    # API base is optional for providers using the standard OpenAI endpoint.
    LLM_API_BASE: str = ""
    LLM_API_KEY: SecretStr = Field(default=SecretStr(""), repr=False, exclude=True)
    LLM_MODEL: str = "gpt-4.1"
    LLM_REASONING_EFFORT: str = ""
    LLM_MAX_CONTEXT_TOKENS: int = 272_000
    # Per-request client deadline in seconds.
    LLM_TIMEOUT_SECONDS: int = 900
    # Optional USD rates per million tokens; zero means no estimate is available.
    LLM_INPUT_COST_PER_MILLION_USD: float = 0.0
    LLM_CACHED_INPUT_COST_PER_MILLION_USD: float = 0.0
    LLM_OUTPUT_COST_PER_MILLION_USD: float = 0.0
    # Optional provider-specific tool-loop prompt-cache layout: shared guidance and user text
    # use explicit cache breakpoints when the configured endpoint supports that format.
    # false = the previous wire format exactly (rollback switch).
    LLM_PROMPT_CACHE_LAYOUT: bool = False

    # --- Repository tools (review_agent.py) ---
    # Give the model read-only read_file / search / list_files on a snapshot of the PR head
    # (its tarball, deleted after the review). Off, or a failed snapshot = diff-only review.
    REVIEW_TOOLS: bool = True
    # Bounds of one review's tool loop: model turns, tool calls, optional local cost limit,
    # and seconds spent exploring. Past any enabled limit the model answers with what it has.
    REVIEW_TOOL_MAX_ROUNDS: int = 12
    REVIEW_TOOL_MAX_CALLS: int = 40
    REVIEW_TOOL_MAX_COST_USD: float = 0.0
    REVIEW_TOOL_MAX_SECONDS: int = 240
    # Snapshots on disk at once (each at most a 200 MB tarball, then 400 MB of extracted text;
    # typically a few MB); 0 = one per review slot (MAX_CONCURRENT_REVIEWS). A review waits up
    # to 10 minutes for a slot, then goes diff-only.
    REVIEW_SNAPSHOT_MAX_CONCURRENT: int = 0
    # No snapshot (diff-only review) when the data dir's disk has less free space than this.
    REVIEW_SNAPSHOT_MIN_FREE_MB: int = 2000

    # --- Persistence ---
    # SQLite lives here, outside the git checkout. Falls back to
    # ~/.local/share/gh-pr-reviewer when this path cannot be created.
    DATA_DIR: str = "/var/lib/gh-pr-reviewer"

    STYLE_GUIDE_PATH: str = str(REPO_ROOT / "review" / "STYLE_GUIDE.md")

    # --- Statistics API (/api/v1/stats/*) ---
    # Optional bearer token for machine readers of the statistics API.
    STATS_API_TOKEN: SecretStr = Field(default=SecretStr(""), repr=False, exclude=True)

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8", "extra": "ignore"}

    @property
    def snapshot_slots(self) -> int:
        limit = self.REVIEW_SNAPSHOT_MAX_CONCURRENT or self.MAX_CONCURRENT_REVIEWS
        return max(1, limit)

    @property
    def github_configured(self) -> bool:
        return bool(self.GITHUB_APP_ID.strip() and self.GITHUB_APP_PRIVATE_KEY.get_secret_value())

    @property
    def max_concurrent_reviews(self) -> int:
        return max(1, self.MAX_CONCURRENT_REVIEWS)

    @property
    def max_concurrent_large_reviews(self) -> int:
        n = self.MAX_CONCURRENT_LARGE_REVIEWS or self.max_concurrent_reviews // 2
        return max(1, min(n, self.max_concurrent_reviews))

    @property
    def max_concurrent_per_repo(self) -> int:
        n = self.MAX_CONCURRENT_PER_REPO or max(2, self.max_concurrent_reviews // 2)
        return max(1, min(n, self.max_concurrent_reviews))

    @property
    def llm_client_timeout_seconds(self) -> int:
        """Client timeout used by both review paths."""
        return self.LLM_TIMEOUT_SECONDS

    @property
    def llm_configured(self) -> bool:
        return bool(self.LLM_API_KEY.get_secret_value())


settings = Settings()
