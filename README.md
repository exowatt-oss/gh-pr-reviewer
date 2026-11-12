# gh-pr-reviewer

A self-hosted GitHub App that reviews pull requests with an OpenAI-compatible language model. It combines a configurable repository allowlist, review-round limits, spend tracking, and a review engine that posts native GitHub review comments.

![Status: experimental](https://img.shields.io/badge/status-experimental-orange) ![License: Apache--2.0](https://img.shields.io/badge/license-Apache--2.0-blue)

Created at [Exowatt](https://exowatt.com).

## Status

Experimental. The project is provided for evaluation and self-hosting; it has no support or availability commitment.

## Requirements

- Python 3.12
- A GitHub App installed on the repositories it may review
- An API key for an OpenAI-compatible chat-completions provider
- A publicly reachable HTTPS endpoint for GitHub webhooks (optional; polling-only setups need no endpoint)

## Install and quickstart

```sh
uv sync --locked
cp .env.example .env
```

Create a GitHub App with **Contents: read**, **Issues: read and write**, **Pull requests: read and write**, and **Metadata: read-only** permissions, then install it only on repositories you intend to review. For webhook-triggered reconciliation, subscribe to `pull_request`, `issue_comment`, `installation`, and `installation_repositories`, set the webhook URL to `https://<your-host>/webhook`, and generate a webhook secret. Polling-only setups can leave webhook delivery disabled and the secret blank.

Set these values in `.env` (never commit the file):

```dotenv
GITHUB_APP_ID=123456
GITHUB_APP_PRIVATE_KEY=<base64-encoded-private-key-pem>
GITHUB_WEBHOOK_SECRET=<random-webhook-secret-if-using-webhooks>
REVIEW_REPOS=owner/repository
LLM_API_KEY=<your-provider-key>
LLM_MODEL=gpt-4.1
LLM_API_BASE=https://api.openai.com/v1
```

`LLM_API_BASE` is optional for providers that use the default OpenAI API endpoint. `REVIEW_REPOS` is an explicit allowlist of `owner/repository` entries; it defaults to empty, so the App reviews nothing until you opt in. `owner/*` enables installation-wide organization discovery and should be used only when that breadth is intended.

For a local-only smoke run, configure forwarded-auth's explicit dev mode in `.env`:

```dotenv
FORWARDED_AUTH_MODE=none
FORWARDED_AUTH_ALLOW_INSECURE_NONE_MODE=true
DRY_RUN=true
```

These two auth settings intentionally disable request identity verification and must not be used on a network-accessible deployment. Before deploying, configure `forwarded-auth` with your OIDC issuer, audience, and allowed email domains (see its [configuration guide](https://github.com/exowatt-oss/forwarded-auth#configuration)). App dashboard and API routes otherwise fail closed and require a verified identity.

Start the service:

```sh
uv run uvicorn app.main:app --host 127.0.0.1 --port 8000
```

GitHub webhook signatures are always verified with `GITHUB_WEBHOOK_SECRET`; requests are rejected when the secret is missing or the signature is invalid. A valid event wakes the poller for an immediate reconciliation. Periodic polling remains enabled as a recovery path. The model can make mistakes: review its suggestions before applying them.

## Review policy and configuration

Automatic review policy defaults to at most two full review rounds per pull request. A reviewer may request another full review by commenting `/review`; `MAX_ON_DEMAND_REVIEWS_PER_PR_PER_DAY` limits billed on-demand runs. Owners can opt a repository out with the `no-ai-review` topic, the `no-ai-review` pull-request label, `[skip review]` in the title, or `.github/gh-pr-reviewer.yml` containing `enabled: false`.

The application records token usage and estimated cost in SQLite. It uses a provider-reported cost
when LiteLLM supplies one; otherwise, optionally set `LLM_INPUT_COST_PER_MILLION_USD`,
`LLM_CACHED_INPUT_COST_PER_MILLION_USD`, and `LLM_OUTPUT_COST_PER_MILLION_USD` to your
provider/model rates. These rates default to zero; no provider price or quota is assumed. An
optional `LLM_MONTHLY_BUDGET_USD` applies a local spend guard using recorded usage.
`REVIEW_TOOL_MAX_COST_USD` optionally limits model-assisted repository exploration for one pull
request; zero disables the cap. Provider account limits remain the operator's responsibility.

Configuration lives in environment variables; `.env.example` lists the main settings. Review every permission, webhook event, repository allowlist, provider rate, and limit for your deployment before enabling real reviews.

The dashboard and application APIs require a verified forwarded-auth identity. The machine-readable
`/api/v1/stats/*` endpoints also accept an optional `STATS_API_TOKEN` bearer; leave it empty unless
you need a trusted machine reader.

## Development and tests

```sh
uv run pytest -n 4
uv run ruff check .
uv run ruff format --check .
```

Tests use generated keys, local HTTP stubs, and synthetic GitHub fixtures. They do not connect to GitHub or an LLM provider.

## License

Original code is licensed under [Apache-2.0](LICENSE). See [NOTICE](NOTICE) for attribution.

## Contributing

See [the organization contributing guide](https://github.com/exowatt-oss/.github/blob/main/CONTRIBUTING.md). Contributions require DCO sign-off (`git commit -s`); no CLA is required.

## Security

Use [GitHub private vulnerability reporting](https://github.com/exowatt-oss/gh-pr-reviewer/security/advisories/new) for security reports, or contact software@exowatt.com. Do not open a public issue for a suspected vulnerability.
