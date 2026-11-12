"""GitHub App authentication and the handful of REST calls the poller needs.

Everything the poller and reviewer do on GitHub goes through this client: listing
installations and open PRs, fetching diffs and files, and posting the pull-request review.
"""

from __future__ import annotations

import asyncio
import base64
import binascii
import logging
import re
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import httpx
import jwt

logger = logging.getLogger(__name__)

_PEM_BLOCK_RE = re.compile(
    r"-----BEGIN (?P<label>[A-Z0-9 ]+)-----(?P<body>.*?)-----END (?P=label)-----", re.DOTALL
)
TOKEN_REFRESH_MARGIN_SECONDS = 300
INSTALLATIONS_TTL_SECONDS = 600
JWT_LIFETIME_SECONDS = 540  # GitHub caps app JWTs at 10 minutes
API_VERSION = "2022-11-28"
TARBALL_TIMEOUT_SECONDS = 120.0


def normalize_pem(raw: str) -> str:
    """Return a canonical PEM string from any of the shapes secrets arrive in.

    Accepts a raw PEM with real newlines, a PEM whose newlines were escaped to a literal
    ``\\n`` (or flattened to spaces), or a base64-encoded PEM. Raises ``ValueError`` when the
    value is none of these.
    """
    value = (raw or "").strip().strip('"').strip("'").strip()
    if not value:
        raise ValueError("GitHub App private key is empty")
    if "-----BEGIN" not in value:
        try:
            decoded = base64.b64decode("".join(value.split()), validate=True).decode("utf-8")
        except (binascii.Error, UnicodeDecodeError, ValueError) as exc:
            raise ValueError("GitHub App private key is neither PEM nor base64 PEM") from exc
        if "-----BEGIN" not in decoded:
            raise ValueError("GitHub App private key decoded from base64 is not a PEM")
        value = decoded.strip()
    value = value.replace("\\r\\n", "\n").replace("\\n", "\n").replace("\r\n", "\n")
    match = _PEM_BLOCK_RE.search(value)
    if not match:
        raise ValueError("GitHub App private key has no complete PEM block")
    label = match.group("label")
    body = "".join(match.group("body").split())
    if not body:
        raise ValueError("GitHub App private key PEM body is empty")
    lines = [body[i : i + 64] for i in range(0, len(body), 64)]
    return "\n".join([f"-----BEGIN {label}-----", *lines, f"-----END {label}-----"]) + "\n"


def mint_app_jwt(app_id: str, private_key_pem: str, *, now: float | None = None) -> str:
    """Mint a GitHub App JWT (RS256), backdated 60 s for clock skew."""
    issued = int(now if now is not None else time.time()) - 60
    payload = {"iat": issued, "exp": issued + 60 + JWT_LIFETIME_SECONDS, "iss": str(app_id)}
    return jwt.encode(payload, private_key_pem, algorithm="RS256")


def _parse_github_time(value: str) -> float:
    return datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp()


def _next_link(response: httpx.Response) -> str | None:
    nxt = response.links.get("next")
    return nxt.get("url") if nxt else None


@dataclass(frozen=True)
class PrComment:
    """A PR conversation comment, as needed to spot on-demand review requests."""

    id: int
    body: str
    created_at: str | None
    author_login: str
    author_type: str


@dataclass(frozen=True)
class PullRequest:
    repo: str
    number: int
    title: str
    draft: bool
    head_sha: str
    base_ref: str
    author_login: str
    author_type: str
    html_url: str
    api_url: str
    labels: tuple[str, ...] = ()
    created_at: str | None = None
    updated_at: str | None = None
    # Only known from GraphQL discovery (org mode); used for the go-live cutoff.
    head_committed_at: str | None = None
    ready_at: str | None = None
    force_pushed_at: str | None = None
    # Latest conversation comments (on-demand ``/review`` requests); GraphQL search carries
    # them, explicitly listed repos get them from the repo's recent issue comments.
    comments: tuple[PrComment, ...] = ()

    @classmethod
    def from_api(cls, repo: str, data: dict[str, Any]) -> PullRequest:
        user = data.get("user") or {}
        return cls(
            repo=repo,
            number=int(data["number"]),
            title=str(data.get("title") or ""),
            draft=bool(data.get("draft")),
            head_sha=str((data.get("head") or {}).get("sha") or ""),
            base_ref=str((data.get("base") or {}).get("ref") or ""),
            author_login=str(user.get("login") or ""),
            author_type=str(user.get("type") or ""),
            html_url=str(data.get("html_url") or ""),
            api_url=str(data.get("url") or ""),
            labels=tuple(str(lb.get("name") or "") for lb in data.get("labels") or []),
            created_at=data.get("created_at"),
            updated_at=data.get("updated_at"),
        )

    @classmethod
    def from_graphql(cls, node: dict[str, Any]) -> PullRequest:
        repo = str((node.get("repository") or {}).get("nameWithOwner") or "")
        author = node.get("author") or {}
        author_type = str(author.get("__typename") or "")
        login = str(author.get("login") or "")
        if author_type == "Bot" and login and not login.endswith("[bot]"):
            login = f"{login}[bot]"  # REST spelling, e.g. dependabot[bot]
        number = int(node["number"])

        def last(conn: str, *path: str) -> Any:
            nodes = (node.get(conn) or {}).get("nodes") or []
            value: Any = nodes[-1] if nodes else None
            for key in path:
                value = (value or {}).get(key)
            return value

        return cls(
            repo=repo,
            number=number,
            title=str(node.get("title") or ""),
            draft=bool(node.get("isDraft")),
            head_sha=str(node.get("headRefOid") or ""),
            base_ref=str(node.get("baseRefName") or ""),
            author_login=login,
            author_type=author_type,
            html_url=str(node.get("url") or ""),
            api_url=f"https://api.github.com/repos/{repo}/pulls/{number}",
            labels=tuple(
                str(lb.get("name") or "") for lb in (node.get("labels") or {}).get("nodes") or []
            ),
            created_at=node.get("createdAt"),
            updated_at=node.get("updatedAt"),
            head_committed_at=last("commits", "commit", "committedDate"),
            ready_at=last("readyEvents", "createdAt"),
            force_pushed_at=last("forcePushes", "createdAt"),
            comments=tuple(
                PrComment(
                    id=int(c.get("databaseId") or 0),
                    body=str(c.get("body") or ""),
                    created_at=c.get("createdAt"),
                    author_login=str((c.get("author") or {}).get("login") or ""),
                    author_type=str((c.get("author") or {}).get("__typename") or ""),
                )
                for c in (node.get("comments") or {}).get("nodes") or []
                if c and c.get("databaseId")
            ),
        )


REVIEW_THREADS_QUERY = """
query($owner: String!, $name: String!, $number: Int!, $after: String) {
  repository(owner: $owner, name: $name) {
    pullRequest(number: $number) {
      reviewThreads(first: 100, after: $after) {
        pageInfo { hasNextPage endCursor }
        nodes { isResolved comments(first: 1) { nodes { databaseId } } }
      }
    }
  }
}
"""

SEARCH_PULLS_QUERY = """
query($q: String!, $first: Int!, $after: String) {
  rateLimit { cost remaining limit resetAt }
  search(query: $q, type: ISSUE, first: $first, after: $after) {
    issueCount
    pageInfo { hasNextPage endCursor }
    nodes {
      ... on PullRequest {
        number title url isDraft createdAt updatedAt headRefOid baseRefName
        author { login __typename }
        repository { nameWithOwner isArchived isFork }
        labels(first: 20) { nodes { name } }
        commits(last: 1) { nodes { commit { committedDate } } }
        readyEvents: timelineItems(last: 1, itemTypes: [READY_FOR_REVIEW_EVENT]) {
          nodes { ... on ReadyForReviewEvent { createdAt } }
        }
        forcePushes: timelineItems(last: 1, itemTypes: [HEAD_REF_FORCE_PUSHED_EVENT]) {
          nodes { ... on HeadRefForcePushedEvent { createdAt } }
        }
        comments(last: 5) { nodes { databaseId body createdAt author { login __typename } } }
      }
    }
  }
}
"""


class ReviewRejected(Exception):
    """GitHub refused a review (422), usually an inline comment outside the diff."""


class TarballTooLarge(Exception):
    """A repository tarball is larger than the snapshot limit."""


@dataclass(frozen=True)
class SearchHit:
    pr: PullRequest
    archived: bool
    fork: bool


@dataclass
class SearchResult:
    hits: list[SearchHit]
    total: int
    truncated: bool
    pages: int


@dataclass(frozen=True)
class RepoInfo:
    full_name: str
    archived: bool
    fork: bool
    topics: tuple[str, ...]
    default_branch: str

    @classmethod
    def from_api(cls, data: dict[str, Any]) -> RepoInfo:
        return cls(
            full_name=str(data.get("full_name") or ""),
            archived=bool(data.get("archived")),
            fork=bool(data.get("fork")),
            topics=tuple(str(t) for t in data.get("topics") or []),
            default_branch=str(data.get("default_branch") or ""),
        )


class GitHubApp:
    """Async GitHub App client with installation-token caching."""

    def __init__(
        self,
        app_id: str,
        private_key: str,
        *,
        api_url: str = "https://api.github.com",
        client: httpx.AsyncClient | None = None,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self.app_id = str(app_id).strip()
        self.private_key = normalize_pem(private_key)
        self.api_url = api_url.rstrip("/")
        self._client = client or httpx.AsyncClient(timeout=httpx.Timeout(30.0))
        self._owns_client = client is None
        self._clock = clock
        self._jwt: tuple[str, float] | None = None
        self._tokens: dict[int, tuple[str, float]] = {}
        self._installations: tuple[list[dict[str, Any]], float] | None = None
        self._repo_installation: dict[str, int] = {}
        self._lock = asyncio.Lock()
        # Last rate-limit headers seen per resource (core / graphql / search).
        self.rate_limits: dict[str, dict[str, Any]] = {}

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.aclose()

    # --- app-level auth -------------------------------------------------------------
    def app_jwt(self) -> str:
        now = self._clock()
        if self._jwt and self._jwt[1] - 60 > now:
            return self._jwt[0]
        token = mint_app_jwt(self.app_id, self.private_key, now=now)
        self._jwt = (token, now + JWT_LIFETIME_SECONDS)
        return token

    def _headers(self, token: str, scheme: str = "Bearer") -> dict[str, str]:
        return {
            "Authorization": f"{scheme} {token}",
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": API_VERSION,
            "User-Agent": "gh-pr-reviewer",
        }

    async def _app_get(self, url: str) -> httpx.Response:
        response = await self._client.get(url, headers=self._headers(self.app_jwt()))
        response.raise_for_status()
        return response

    async def get_app(self) -> dict[str, Any]:
        return (await self._app_get(f"{self.api_url}/app")).json()

    async def list_installations(self) -> list[dict[str, Any]]:
        now = self._clock()
        if self._installations and self._installations[1] > now:
            return self._installations[0]
        items: list[dict[str, Any]] = []
        url: str | None = f"{self.api_url}/app/installations?per_page=100"
        while url:
            response = await self._app_get(url)
            items.extend(response.json())
            url = _next_link(response)
        self._installations = (items, now + INSTALLATIONS_TTL_SECONDS)
        return items

    async def installation_id_for_owner(self, owner: str) -> int | None:
        for inst in await self.list_installations():
            if str((inst.get("account") or {}).get("login", "")).lower() == owner.lower():
                return int(inst["id"])
        return None

    async def installation_id_for_repo(self, repo: str) -> int:
        key = repo.lower()
        if key in self._repo_installation:
            return self._repo_installation[key]
        inst_id = await self.installation_id_for_owner(key.split("/", 1)[0])
        if inst_id is not None:
            self._repo_installation[key] = inst_id
            return inst_id
        # Fall back to the per-repo lookup (e.g. user-level installs listed under another name).
        response = await self._app_get(f"{self.api_url}/repos/{repo}/installation")
        inst_id = int(response.json()["id"])
        self._repo_installation[key] = inst_id
        return inst_id

    async def installation_token(self, installation_id: int) -> str:
        async with self._lock:
            now = self._clock()
            cached = self._tokens.get(installation_id)
            if cached and cached[1] - TOKEN_REFRESH_MARGIN_SECONDS > now:
                return cached[0]
            response = await self._client.post(
                f"{self.api_url}/app/installations/{installation_id}/access_tokens",
                headers=self._headers(self.app_jwt()),
            )
            response.raise_for_status()
            data = response.json()
            expires_at = _parse_github_time(data["expires_at"])
            self._tokens[installation_id] = (data["token"], expires_at)
            return data["token"]

    # --- installation-scoped REST ---------------------------------------------------
    def _note_rate_limit(self, response: httpx.Response) -> None:
        headers = response.headers
        remaining = headers.get("x-ratelimit-remaining")
        if remaining is None:
            return
        resource = headers.get("x-ratelimit-resource") or "core"
        try:
            self.rate_limits[resource] = {
                "remaining": int(remaining),
                "limit": int(headers.get("x-ratelimit-limit") or 0),
                "reset": int(headers.get("x-ratelimit-reset") or 0),
            }
        except ValueError:
            pass

    async def _installation_request(
        self, method: str, installation_id: int, path_or_url: str, **kwargs: Any
    ) -> httpx.Response:
        token = await self.installation_token(installation_id)
        url = path_or_url if path_or_url.startswith("http") else f"{self.api_url}{path_or_url}"
        headers = self._headers(token, scheme="token") | kwargs.pop("headers", {})
        response = await self._client.request(method, url, headers=headers, **kwargs)
        self._note_rate_limit(response)
        return response

    async def _repo_request(
        self, method: str, repo: str, path_or_url: str, **kwargs: Any
    ) -> httpx.Response:
        installation_id = await self.installation_id_for_repo(repo)
        return await self._installation_request(method, installation_id, path_or_url, **kwargs)

    async def list_installation_repos(self, installation_id: int) -> list[RepoInfo]:
        """Every repo the installation can access (``GET /installation/repositories``)."""
        repos: list[RepoInfo] = []
        url: str | None = "/installation/repositories?per_page=100"
        while url:
            response = await self._installation_request("GET", installation_id, url)
            response.raise_for_status()
            repos.extend(RepoInfo.from_api(r) for r in response.json().get("repositories") or [])
            url = _next_link(response)
        return repos

    async def get_repo(self, repo: str) -> RepoInfo:
        response = await self._repo_request("GET", repo, f"/repos/{repo}")
        response.raise_for_status()
        return RepoInfo.from_api(response.json())

    async def search_pulls(
        self, installation_id: int, query: str, *, page_size: int = 50, max_pages: int = 10
    ) -> SearchResult:
        """Open PRs matching a search query, with everything the poller needs (GraphQL).

        One request per page returns head SHA, draft flag, labels, author type, the head
        commit's date and the last ready-for-review / force-push events, so discovery needs
        no per-PR REST call. Search returns at most 1000 results (``truncated``).
        """
        hits: list[SearchHit] = []
        after: str | None = None
        total, pages, truncated = 0, 0, False
        while True:
            response = await self._installation_request(
                "POST",
                installation_id,
                "/graphql",
                json={
                    "query": SEARCH_PULLS_QUERY,
                    "variables": {"q": query, "first": page_size, "after": after},
                },
            )
            response.raise_for_status()
            body = response.json()
            if body.get("errors"):
                raise RuntimeError(f"GraphQL search failed: {body['errors']!r}"[:500])
            data = body.get("data") or {}
            rate = data.get("rateLimit") or {}
            if rate:
                self.rate_limits["graphql"] = {
                    "remaining": rate.get("remaining"),
                    "limit": rate.get("limit"),
                    "reset_at": rate.get("resetAt"),
                    "last_cost": rate.get("cost"),
                }
            search = data.get("search") or {}
            total = int(search.get("issueCount") or 0)
            pages += 1
            for node in search.get("nodes") or []:
                if not node or "number" not in node:
                    continue  # not a PullRequest (or inaccessible)
                repo = node.get("repository") or {}
                hits.append(
                    SearchHit(
                        PullRequest.from_graphql(node),
                        archived=bool(repo.get("isArchived")),
                        fork=bool(repo.get("isFork")),
                    )
                )
            info = search.get("pageInfo") or {}
            if not info.get("hasNextPage"):
                break
            if pages >= max_pages:
                truncated = True
                break
            after = info.get("endCursor")
        return SearchResult(hits=hits, total=total, truncated=truncated, pages=pages)

    async def _paginate(self, repo: str, path: str) -> list[dict[str, Any]]:
        items: list[dict[str, Any]] = []
        url: str | None = path
        while url:
            response = await self._repo_request("GET", repo, url)
            response.raise_for_status()
            items.extend(response.json())
            url = _next_link(response)
        return items

    async def list_open_pulls(self, repo: str) -> list[PullRequest]:
        data = await self._paginate(repo, f"/repos/{repo}/pulls?state=open&per_page=100")
        return [PullRequest.from_api(repo, item) for item in data]

    async def get_pull(self, repo: str, number: int) -> dict[str, Any]:
        response = await self._repo_request("GET", repo, f"/repos/{repo}/pulls/{number}")
        response.raise_for_status()
        return response.json()

    async def compare(self, repo: str, base: str, head: str) -> dict[str, Any]:
        """``GET /compare/{base}...{head}``: ``files`` (with patches), ``total_commits``."""
        response = await self._repo_request("GET", repo, f"/repos/{repo}/compare/{base}...{head}")
        response.raise_for_status()
        return response.json()

    async def compare_files(self, repo: str, base: str, head: str) -> list[dict[str, Any]]:
        """Files changed between two commits (``GET /compare/{base}...{head}``)."""
        return list((await self.compare(repo, base, head)).get("files") or [])

    async def list_dir(self, repo: str, path: str, ref: str) -> list[dict[str, Any]]:
        """Entries of a directory at ``ref`` (``[]`` when it does not exist or is a file)."""
        url = f"/repos/{repo}/contents/{path}" if path else f"/repos/{repo}/contents"
        response = await self._repo_request("GET", repo, url, params={"ref": ref})
        if response.status_code == 404:
            return []
        response.raise_for_status()
        data = response.json()
        return data if isinstance(data, list) else []

    async def download_tarball(self, repo: str, ref: str, dest: Path, max_bytes: int) -> int:
        """Stream ``GET /repos/{repo}/tarball/{ref}`` (a redirect to codeload) to ``dest``.

        Raises ``TarballTooLarge`` past ``max_bytes``; returns the bytes written.
        """
        token = await self.installation_token(await self.installation_id_for_repo(repo))
        url = f"{self.api_url}/repos/{repo}/tarball/{ref}"
        size = 0
        # httpx drops Authorization on the cross-host redirect; the codeload URL is signed.
        async with self._client.stream(
            "GET", url, headers=self._headers(token, scheme="token"), follow_redirects=True,
            timeout=httpx.Timeout(TARBALL_TIMEOUT_SECONDS, connect=30.0),
        ) as response:  # fmt: skip
            response.raise_for_status()
            declared = int(response.headers.get("content-length") or 0)
            if declared > max_bytes:
                raise TarballTooLarge(f"{repo}@{ref[:12]} tarball is {declared} bytes")
            with dest.open("wb") as out:
                async for chunk in response.aiter_bytes(1 << 16):
                    size += len(chunk)
                    if size > max_bytes:
                        raise TarballTooLarge(f"{repo}@{ref[:12]} tarball over {max_bytes} bytes")
                    out.write(chunk)
        return size

    async def list_pull_files(self, repo: str, number: int) -> list[dict[str, Any]]:
        return await self._paginate(repo, f"/repos/{repo}/pulls/{number}/files?per_page=100")

    async def get_file_text(self, repo: str, path: str, ref: str) -> str | None:
        """Raw file content at ``ref``; ``None`` when the file does not exist."""
        response = await self._repo_request(
            "GET",
            repo,
            f"/repos/{repo}/contents/{path}",
            params={"ref": ref},
            headers={"Accept": "application/vnd.github.raw+json"},
        )
        if response.status_code == 404:
            return None
        response.raise_for_status()
        return response.text

    async def list_issue_comments(
        self, repo: str, number: int, since: str | None = None
    ) -> list[dict[str, Any]]:
        query = "per_page=100" + (f"&since={since}" if since else "")
        return await self._paginate(repo, f"/repos/{repo}/issues/{number}/comments?{query}")

    async def update_issue_comment(self, repo: str, comment_id: int, body: str) -> None:
        response = await self._repo_request(
            "PATCH", repo, f"/repos/{repo}/issues/comments/{comment_id}", json={"body": body}
        )
        response.raise_for_status()

    async def list_repo_issue_comments(self, repo: str, since: str) -> list[dict[str, Any]]:
        """Conversation comments on any issue / PR of the repo updated since ``since``."""
        return await self._paginate(
            repo,
            f"/repos/{repo}/issues/comments?sort=updated&direction=asc&per_page=100&since={since}",
        )

    async def add_comment_reaction(self, repo: str, comment_id: int, content: str) -> None:
        response = await self._repo_request(
            "POST",
            repo,
            f"/repos/{repo}/issues/comments/{comment_id}/reactions",
            json={"content": content},
        )
        response.raise_for_status()

    async def list_reviews(self, repo: str, number: int) -> list[dict[str, Any]]:
        return await self._paginate(repo, f"/repos/{repo}/pulls/{number}/reviews?per_page=100")

    async def list_review_comments(self, repo: str, number: int) -> list[dict[str, Any]]:
        """Inline review comments of the PR (every review, every author)."""
        return await self._paginate(repo, f"/repos/{repo}/pulls/{number}/comments?per_page=100")

    async def resolved_review_comment_ids(self, repo: str, number: int) -> set[int]:
        """REST ids of the first comment of every resolved review thread on the PR (GraphQL;
        at most 500 threads)."""
        owner, name = repo.split("/", 1)
        resolved: set[int] = set()
        after: str | None = None
        for _ in range(5):
            response = await self._repo_request(
                "POST",
                repo,
                "/graphql",
                json={
                    "query": REVIEW_THREADS_QUERY,
                    "variables": {"owner": owner, "name": name, "number": number, "after": after},
                },
            )
            response.raise_for_status()
            body = response.json()
            if body.get("errors"):
                raise RuntimeError(f"GraphQL reviewThreads failed: {body['errors']!r}"[:500])
            pull = ((body.get("data") or {}).get("repository") or {}).get("pullRequest") or {}
            threads = pull.get("reviewThreads") or {}
            for node in threads.get("nodes") or []:
                node = node or {}
                first = (node.get("comments") or {}).get("nodes") or [{}]
                if node.get("isResolved") and (first[0] or {}).get("databaseId"):
                    resolved.add(int(first[0]["databaseId"]))
            info = threads.get("pageInfo") or {}
            if not info.get("hasNextPage"):
                break
            after = info.get("endCursor")
        return resolved

    async def create_review(
        self,
        repo: str,
        number: int,
        *,
        commit_id: str,
        body: str,
        comments: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Submit a pull-request review. Always ``COMMENT``: the reviewer never approves or
        requests changes (it must not gate merges)."""
        response = await self._repo_request(
            "POST",
            repo,
            f"/repos/{repo}/pulls/{number}/reviews",
            json={"commit_id": commit_id, "body": body, "event": "COMMENT", "comments": comments},
        )
        if response.status_code == 422:
            raise ReviewRejected(response.text[:500])
        response.raise_for_status()
        return response.json()

    async def create_issue_comment(self, repo: str, number: int, body: str) -> dict[str, Any]:
        response = await self._repo_request(
            "POST", repo, f"/repos/{repo}/issues/{number}/comments", json={"body": body}
        )
        response.raise_for_status()
        return response.json()
