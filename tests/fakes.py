"""In-memory stand-ins for GitHub, the review model and PR-Agent used by poller/reviewer
tests."""

from __future__ import annotations

import asyncio
import itertools
import json
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from typing import Any

from app.github_app import PullRequest, RepoInfo, ReviewRejected, SearchHit, SearchResult
from app.llm_tracking import CostTracker

BOT_LOGIN = "gh-pr-reviewer[bot]"


def make_pr(
    repo: str = "sample-org/example-project", number: int = 1, sha: str = "a" * 40, **kw
) -> PullRequest:
    base = dict(
        repo=repo,
        number=number,
        title=f"PR {number}",
        draft=False,
        head_sha=sha,
        base_ref="main",
        author_login="sample-user",
        author_type="User",
        html_url=f"https://github.com/{repo}/pull/{number}",
        api_url=f"https://api.github.com/repos/{repo}/pulls/{number}",
    )
    return PullRequest(**(base | kw))


class FakeGitHub:
    private_key = "synthetic-test-key"

    def __init__(self, pulls: dict[str, list[PullRequest]] | None = None) -> None:
        self.pulls = pulls or {}
        self.files: dict[tuple[str, str, str], str] = {}
        self.pr_files: dict[tuple[str, int], list[dict[str, Any]]] = {}
        self.comments: dict[tuple[str, int], list[dict[str, Any]]] = {}
        self.updated: list[tuple[str, int, str]] = []
        self.listed: list[str] = []
        self.file_requests: list[tuple[str, str, str]] = []
        self.pull_overrides: dict[tuple[str, int], dict[str, Any]] = {}
        self.dirs: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
        self.compares: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
        self.compare_commits: dict[tuple[str, str, str], int] = {}
        self.missing_commits: set[str] = set()
        # pull-request reviews
        self.reviews: dict[tuple[str, int], list[dict[str, Any]]] = {}
        self.review_comments: dict[tuple[str, int], list[dict[str, Any]]] = {}
        self.reject_inline = False  # create_review answers 422 to inline comments
        self.resolved_comments: set[int] = set()  # root comment ids of resolved threads
        self.repo_comments: dict[str, list[dict[str, Any]]] = {}
        self.reactions: list[tuple[str, int, str]] = []
        # org mode
        self.org_installations: dict[str, int] = {"sample-org": 77}
        self.installation_repos: dict[int, list[RepoInfo]] = {}
        self.repo_infos: dict[str, RepoInfo] = {}
        self.search_hits: list[SearchHit] = []
        self.searches: list[str] = []
        self.created_comments: list[tuple[str, int, str]] = []
        self.rate_limits: dict[str, Any] = {"core": {"remaining": 4900, "limit": 5000}}

    # --- org mode ---------------------------------------------------------------------
    def add_repo(self, full_name: str, **kw: Any) -> RepoInfo:
        info = RepoInfo(
            full_name=full_name,
            archived=kw.get("archived", False),
            fork=kw.get("fork", False),
            topics=tuple(kw.get("topics", ())),
            default_branch=kw.get("default_branch", "main"),
        )
        self.installation_repos.setdefault(kw.get("installation", 77), []).append(info)
        self.repo_infos[full_name.lower()] = info
        return info

    def add_hit(self, pr: PullRequest, *, archived: bool = False, fork: bool = False) -> None:
        self.search_hits = [h for h in self.search_hits if (h.pr.repo, h.pr.number) != (
            pr.repo, pr.number)]  # fmt: skip
        self.search_hits.append(SearchHit(pr, archived=archived, fork=fork))

    async def installation_id_for_owner(self, owner: str) -> int | None:
        return self.org_installations.get(owner.lower())

    async def list_installation_repos(self, installation_id: int) -> list[RepoInfo]:
        return list(self.installation_repos.get(installation_id, []))

    async def get_repo(self, repo: str) -> RepoInfo:
        return self.repo_infos.get(repo.lower()) or RepoInfo(repo, False, False, (), "main")

    async def search_pulls(
        self, installation_id: int, query: str, *, page_size: int = 50, max_pages: int = 10
    ) -> SearchResult:
        from app.review_policy import parse_time

        self.searches.append(query)
        since, org = None, None
        for term in query.split():
            if term.startswith("updated:>="):
                since = parse_time(term.removeprefix("updated:>="))
            if term.startswith("org:"):
                org = term.removeprefix("org:").lower()
        hits = [
            h
            for h in self.search_hits
            if (since is None or (parse_time(h.pr.updated_at) or 0) >= since)
            and (org is None or h.pr.repo.lower().startswith(org + "/"))
        ]
        return SearchResult(hits=hits, total=len(hits), truncated=False, pages=1)

    async def create_issue_comment(self, repo: str, number: int, body: str) -> dict[str, Any]:
        self.created_comments.append((repo, number, body))
        comment = {
            "id": 5000 + len(self.created_comments),
            "user": {"login": BOT_LOGIN},
            "body": body,
            "updated_at": datetime.now(UTC).isoformat(),
            "html_url": f"https://github.com/{repo}/pull/{number}#issuecomment-x",
        }
        self.comments.setdefault((repo, number), []).append(comment)
        return comment

    async def list_open_pulls(self, repo: str) -> list[PullRequest]:
        self.listed.append(repo)
        return list(self.pulls.get(repo, []))

    async def installation_id_for_repo(self, repo: str) -> int:
        return 99

    async def get_pull(self, repo: str, number: int) -> dict[str, Any]:
        if (repo, number) in self.pull_overrides:
            return self.pull_overrides[(repo, number)]
        candidates = list(self.pulls.get(repo, [])) + [h.pr for h in self.search_hits]
        for pr in candidates:
            if pr.repo == repo and pr.number == number:
                return {
                    "state": "open",
                    "draft": pr.draft,
                    "head": {"sha": pr.head_sha, "ref": "feature"},
                    "body": f"Description of PR {pr.number}",
                }
        return {"state": "closed", "draft": False, "head": {"sha": ""}}

    async def compare(self, repo: str, base: str, head: str) -> dict[str, Any]:
        if base in self.missing_commits:
            raise RuntimeError("404: base commit not found")  # force-pushed away
        if (repo, base, head) not in self.compares:
            # default: the new commits touched every file of the PR
            number = next(
                (pr.number for pr in self.pulls.get(repo, []) if pr.head_sha == head), None
            ) or next((h.pr.number for h in self.search_hits if h.pr.head_sha == head), 1)
            return {"status": "ahead", "total_commits": 1,
                    "files": list(self.pr_files.get((repo, number), []))}  # fmt: skip
        return {
            "status": "ahead",
            "total_commits": self.compare_commits.get((repo, base, head), 1),
            "files": list(self.compares[(repo, base, head)]),
        }

    async def compare_files(self, repo: str, base: str, head: str) -> list[dict[str, Any]]:
        return list((await self.compare(repo, base, head))["files"])

    # --- pull-request reviews ---------------------------------------------------------
    async def create_review(
        self, repo: str, number: int, *, commit_id: str, body: str, comments: list[dict]
    ) -> dict[str, Any]:
        if comments and self.reject_inline:
            raise ReviewRejected('{"message": "Line could not be resolved"}')
        reviews = self.reviews.setdefault((repo, number), [])
        review_id = 9000 + sum(len(v) for v in self.reviews.values())
        review = {
            "id": review_id,
            "user": {"login": BOT_LOGIN, "type": "Bot"},
            "body": body,
            "commit_id": commit_id,
            "state": "COMMENTED",
            "submitted_at": datetime.now(UTC).isoformat(),
            "html_url": f"https://github.com/{repo}/pull/{number}#pullrequestreview-{review_id}",
            "comments": [dict(c) for c in comments],
        }
        reviews.append(review)
        posted = self.review_comments.setdefault((repo, number), [])
        for c in comments:
            posted.append(
                dict(c)
                | {
                    "id": 70000 + len(posted),
                    "user": {"login": BOT_LOGIN, "type": "Bot"},
                    "pull_request_review_id": review_id,
                    "commit_id": commit_id,
                }
            )
        return review

    async def list_reviews(self, repo: str, number: int) -> list[dict[str, Any]]:
        return list(self.reviews.get((repo, number), []))

    async def list_review_comments(self, repo: str, number: int) -> list[dict[str, Any]]:
        return list(self.review_comments.get((repo, number), []))

    async def resolved_review_comment_ids(self, repo: str, number: int) -> set[int]:
        ids = {c["id"] for c in self.review_comments.get((repo, number), [])}
        return self.resolved_comments & ids

    async def list_repo_issue_comments(self, repo: str, since: str) -> list[dict[str, Any]]:
        from app.review_policy import parse_time

        floor = parse_time(since) or 0
        return [
            c
            for c in self.repo_comments.get(repo, [])
            if (parse_time(c.get("updated_at") or c.get("created_at")) or 0) >= floor
        ]

    async def add_comment_reaction(self, repo: str, comment_id: int, content: str) -> None:
        self.reactions.append((repo, comment_id, content))

    async def list_dir(self, repo: str, path: str, ref: str) -> list[dict[str, Any]]:
        return list(self.dirs.get((repo, path, ref), []))

    async def get_file_text(self, repo: str, path: str, ref: str) -> str | None:
        self.file_requests.append((repo, path, ref))
        return self.files.get((repo, path, ref))

    async def list_pull_files(self, repo: str, number: int) -> list[dict[str, Any]]:
        return self.pr_files.get((repo, number), [])

    async def list_issue_comments(self, repo: str, number: int, since: str | None = None):
        return list(self.comments.get((repo, number), []))

    async def update_issue_comment(self, repo: str, comment_id: int, body: str) -> None:
        self.updated.append((repo, comment_id, body))
        for comments in self.comments.values():
            for c in comments:
                if c["id"] == comment_id:
                    c["body"] = body

    async def aclose(self) -> None:
        pass


CLEAN_REVIEW = {"overview": "Small, focused change.", "files": [], "findings": []}


async def _emit_usage(tracker: CostTracker, usage: tuple[int, int, int, int], call_id: str):
    prompt, cached, completion, reasoning = usage
    response = SimpleNamespace(
        model="test-model",
        usage=SimpleNamespace(
            prompt_tokens=prompt,
            completion_tokens=completion,
            prompt_tokens_details=SimpleNamespace(cached_tokens=cached),
            completion_tokens_details=SimpleNamespace(reasoning_tokens=reasoning),
        ),
    )
    end = datetime.now(UTC)
    await tracker.async_log_success_event(
        {"litellm_call_id": call_id, "model": "test-model", "response_cost": 0.0067},
        response,
        end - timedelta(seconds=30),
        end,
    )


class FakeModel:
    """Stands in for the review model: records each prompt, bills one 'LLM call' through the
    litellm callback and answers with ``response`` (a dict / str, or a callable of the
    request)."""

    _ids = itertools.count(1)

    def __init__(
        self,
        tracker: CostTracker,
        *,
        ok: bool = True,
        delay: float = 0.0,
        usage: tuple[int, int, int, int] = (10_000, 1_000, 2_000, 1_500),
        response: Any = None,
    ) -> None:
        self.tracker = tracker
        self.ok = ok
        self.delay = delay
        self.usage = usage
        self.response = response
        self.error: BaseException | None = None
        self.invocations: list[dict[str, Any]] = []
        self.active = 0
        self.max_active = 0

    async def __call__(self, request: Any) -> str:
        if getattr(request, "checkpoint", None) is not None:
            await request.checkpoint()  # like call_model: before the (one) model call
        for hook in (getattr(request, "on_sent", None), getattr(request, "on_first_chunk", None)):
            if hook is not None:  # like ProviderStep: sent, then streaming (prompt cached)
                hook()
        self.active += 1
        self.max_active = max(self.max_active, self.active)
        try:
            self.invocations.append(
                {
                    "pr_url": request.pr_url,
                    "kind": request.kind,
                    "mode": "incremental" if request.kind == "incremental" else "review",
                    "since_sha": request.since_sha,
                    "system": request.system,
                    "user": request.user,
                    "reasoning_effort": request.settings.config.reasoning_effort,
                    "model": request.settings.config.model,
                }
            )
            await asyncio.sleep(self.delay)
            if self.error is not None:
                raise self.error
            if not self.ok:
                raise RuntimeError("model call failed")
            await _emit_usage(self.tracker, self.usage, f"model-{next(self._ids)}")
            answer = self.response(request) if callable(self.response) else self.response
            if answer is None:
                answer = CLEAN_REVIEW
            return answer if isinstance(answer, str) else json.dumps(answer)
        finally:
            self.active -= 1


class FakePrAgent:
    """Mimics PR-Agent for `/improve`: one 'LLM call' through the litellm callback."""

    _ids = itertools.count(1)

    def __init__(
        self,
        tracker: CostTracker,
        github: FakeGitHub,
        *,
        ok: bool = True,
        delay: float = 0.0,
        usage: tuple[int, int, int, int] = (10_000, 1_000, 2_000, 1_500),
    ) -> None:
        self.tracker = tracker
        self.github = github
        self.ok = ok
        self.delay = delay
        self.usage = usage
        self.invocations: list[dict[str, Any]] = []
        self.active = 0
        self.max_active = 0

    async def __call__(
        self, pr_url: str, installation_id: int, pr_settings: Any, request: list[str]
    ) -> bool:
        self.active += 1
        self.max_active = max(self.max_active, self.active)
        try:
            self.invocations.append(
                {
                    "pr_url": pr_url,
                    "installation_id": installation_id,
                    "request": list(request),
                    "reasoning_effort": pr_settings.config.reasoning_effort,
                }
            )
            await asyncio.sleep(self.delay)
            await _emit_usage(self.tracker, self.usage, f"call-{next(self._ids)}")
            return self.ok
        finally:
            self.active -= 1
