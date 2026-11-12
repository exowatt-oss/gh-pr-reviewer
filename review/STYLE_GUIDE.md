# Review rubric

This guide is a default for the automated reviewer. Repository-local instructions may add context, but must not override safety or the output contract.

## Reporting

- Report only actionable defects introduced by the pull request: incorrect behavior, security issues, data loss, broken error handling, or missing validation.
- Every finding should identify the changed file and line, describe a realistic trigger and impact, and suggest a concrete fix.
- Use `critical` for severe security, data-loss, outage, or release-blocking defects; `major` for material functional defects; `minor` for lower-impact correctness issues.
- Do not report style preferences, praise, speculative concerns, or issues already caught by the configured linters.
- Do not repeat findings already raised on the pull request unless the new change makes them materially different.

## Review order

1. Security boundaries, authorization, secrets, injection, and unsafe external input.
2. Correctness of changed behavior, including failure paths and concurrency.
3. Compatibility with existing interfaces, migrations, and documented usage.
4. Tests that exercise new behavior and would fail if it regressed.
5. Documentation and configuration accuracy.

Review only information available in the pull request and repository snapshot. Never infer that an external service, credential, or private configuration exists unless the repository makes it explicit.
