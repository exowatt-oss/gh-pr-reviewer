"""PEM normalisation, JWT minting, installation discovery and token caching."""

from __future__ import annotations

import asyncio
import base64
import json
from datetime import UTC, datetime

import httpx
import jwt
import pytest
from cryptography.hazmat.primitives import serialization

from app.github_app import GitHubApp, PullRequest, mint_app_jwt, normalize_pem


def _canonical(pem: str) -> str:
    return normalize_pem(pem)


def test_raw_pem_round_trips(rsa_pem: str):
    out = normalize_pem(rsa_pem)
    loaded = serialization.load_pem_private_key(out.encode(), password=None)
    assert loaded.private_numbers().public_numbers.e == 65537
    assert out.endswith("\n")


def test_literal_backslash_n_pem(rsa_pem: str):
    escaped = rsa_pem.strip().replace("\n", "\\n")
    assert "\n" not in escaped
    assert normalize_pem(escaped) == _canonical(rsa_pem)


def test_base64_pem(rsa_pem: str):
    encoded = base64.b64encode(rsa_pem.encode()).decode()
    assert normalize_pem(encoded) == _canonical(rsa_pem)
    # base64 wrapped across lines (e.g. `base64` without -w0) also works
    wrapped = "\n".join(encoded[i : i + 76] for i in range(0, len(encoded), 76))
    assert normalize_pem(wrapped) == _canonical(rsa_pem)


@pytest.mark.parametrize(
    "mangle",
    [
        lambda p: f'"{p.strip()}"',  # quoted
        lambda p: p.replace("\n", "\r\n"),  # CRLF
        lambda p: p.strip().replace("\n", " "),  # newlines flattened to spaces
        lambda p: "  \n" + p + "\n\n  ",  # surrounding whitespace
    ],
)
def test_mangled_pem_shapes(rsa_pem: str, mangle):
    assert normalize_pem(mangle(rsa_pem)) == _canonical(rsa_pem)


def test_pkcs8_label_preserved():
    from cryptography.hazmat.primitives.asymmetric import rsa

    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    pem = key.private_bytes(
        serialization.Encoding.PEM,
        serialization.PrivateFormat.PKCS8,
        serialization.NoEncryption(),
    ).decode()
    assert serialization.load_pem_private_key(normalize_pem(pem).encode(), None)


@pytest.mark.parametrize("bad", ["", "   ", "not a key", base64.b64encode(b"hello").decode()])
def test_invalid_keys_raise(bad: str):
    with pytest.raises(ValueError):
        normalize_pem(bad)


def test_app_jwt_claims(rsa_pem: str):
    now = 1_800_000_000
    token = mint_app_jwt("12345", rsa_pem, now=now)
    public = serialization.load_pem_private_key(rsa_pem.encode(), None).public_key()
    claims = jwt.decode(
        token, public, algorithms=["RS256"], options={"verify_exp": False, "verify_iat": False}
    )
    assert claims["iss"] == "12345"
    assert claims["iat"] == now - 60
    assert claims["exp"] - claims["iat"] <= 600


class FakeClock:
    def __init__(self, t: float) -> None:
        self.t = t

    def __call__(self) -> float:
        return self.t


def _github(rsa_pem: str, handler, clock) -> GitHubApp:
    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    return GitHubApp("42", rsa_pem, client=client, clock=clock)


def test_installation_token_cached_until_five_minutes_before_expiry(rsa_pem: str):
    clock = FakeClock(datetime(2026, 10, 7, 12, 0, tzinfo=UTC).timestamp())
    calls: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(f"{request.method} {request.url.path}")
        assert request.headers["Authorization"].startswith("Bearer ")
        expires = datetime.fromtimestamp(clock.t + 3600, UTC).isoformat().replace("+00:00", "Z")
        return httpx.Response(201, json={"token": f"tok-{len(calls)}", "expires_at": expires})

    gh = _github(rsa_pem, handler, clock)

    async def run() -> list[str]:
        tokens = [await gh.installation_token(7)]
        clock.t += 3600 - 301  # still > 5 min left
        tokens.append(await gh.installation_token(7))
        clock.t += 2  # now inside the 5-minute margin
        tokens.append(await gh.installation_token(7))
        return tokens

    assert asyncio.run(run()) == ["tok-1", "tok-1", "tok-2"]
    assert calls == ["POST /app/installations/7/access_tokens"] * 2


def test_installation_lookup_and_open_pulls(rsa_pem: str):
    clock = FakeClock(1_800_000_000)
    seen: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        seen.append(path)
        if path == "/app/installations":
            return httpx.Response(
                200,
                json=[
                    {"id": 1, "account": {"login": "other"}},
                    {"id": 9, "account": {"login": "sample-org"}},
                ],
            )
        if path.endswith("/access_tokens"):
            return httpx.Response(201, json={"token": "itok", "expires_at": "2099-01-01T00:00:00Z"})
        if path == "/repos/sample-org/example-project/pulls":
            assert request.headers["Authorization"] == "token itok"
            page = request.url.params.get("page")
            if page == "2":
                return httpx.Response(200, json=[_pr_json(2, draft=True)])
            nxt = "https://api.github.com/repos/sample-org/example-project/pulls?state=open&page=2"
            link = f'<{nxt}>; rel="next"'
            return httpx.Response(200, json=[_pr_json(1)], headers={"Link": link})
        return httpx.Response(404)

    gh = _github(rsa_pem, handler, clock)
    pulls = asyncio.run(gh.list_open_pulls("sample-org/example-project"))
    assert [p.number for p in pulls] == [1, 2]
    assert pulls[1].draft is True
    assert pulls[0].head_sha == "sha1"
    assert seen.count("/app/installations") == 1
    assert "/app/installations/9/access_tokens" in seen


def test_get_file_text_404_is_none(rsa_pem: str):
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/app/installations":
            return httpx.Response(200, json=[{"id": 9, "account": {"login": "sample-org"}}])
        if request.url.path.endswith("/access_tokens"):
            return httpx.Response(201, json={"token": "t", "expires_at": "2099-01-01T00:00:00Z"})
        if request.url.path.endswith("/AGENTS.md"):
            return httpx.Response(404, json={"message": "Not Found"})
        assert request.url.params["ref"] == "main"
        return httpx.Response(200, text="# rules")

    gh = _github(rsa_pem, handler, FakeClock(1_800_000_000))
    assert (
        asyncio.run(gh.get_file_text("sample-org/example-project", "CLAUDE.md", "main"))
        == "# rules"
    )
    assert asyncio.run(gh.get_file_text("sample-org/example-project", "AGENTS.md", "main")) is None


def _pr_json(number: int, draft: bool = False) -> dict:
    return json.loads(
        json.dumps(
            {
                "number": number,
                "title": f"PR {number}",
                "draft": draft,
                "head": {"sha": f"sha{number}"},
                "base": {"ref": "main"},
                "user": {"login": "sample-user", "type": "User"},
                "html_url": f"https://github.com/sample-org/example-project/pull/{number}",
                "url": f"https://api.github.com/repos/sample-org/example-project/pulls/{number}",
            }
        )
    )


def test_pull_request_from_api():
    pr = PullRequest.from_api("sample-org/example-project", _pr_json(5))
    assert (pr.number, pr.author_login, pr.author_type, pr.base_ref) == (
        5,
        "sample-user",
        "User",
        "main",
    )


def _graphql_node(number: int, **kw) -> dict:
    node = {
        "number": number,
        "title": f"PR {number}",
        "url": f"https://github.com/sample-org/x/pull/{number}",
        "isDraft": False,
        "createdAt": "2026-10-07T18:00:00Z",
        "updatedAt": "2026-10-07T18:05:00Z",
        "headRefOid": f"{number:040d}",
        "baseRefName": "main",
        "author": {"login": "sample-user", "__typename": "User"},
        "repository": {"nameWithOwner": "sample-org/x", "isArchived": False, "isFork": False},
        "labels": {"nodes": [{"name": "no-ai-review"}]},
        "commits": {"nodes": [{"commit": {"committedDate": "2026-10-07T17:59:00Z"}}]},
        "readyEvents": {"nodes": [{"createdAt": "2026-10-07T18:01:00Z"}]},
        "forcePushes": {"nodes": []},
    }
    return node | kw


def test_pull_request_from_graphql_maps_bots_and_dates():
    pr = PullRequest.from_graphql(
        _graphql_node(3, author={"login": "dependabot", "__typename": "Bot"})
    )
    assert pr.repo == "sample-org/x" and pr.number == 3 and pr.head_sha == f"{3:040d}"
    assert (pr.author_login, pr.author_type) == ("dependabot[bot]", "Bot")
    assert pr.labels == ("no-ai-review",)
    assert pr.head_committed_at == "2026-10-07T17:59:00Z"
    assert pr.ready_at == "2026-10-07T18:01:00Z" and pr.force_pushed_at is None
    assert pr.api_url == "https://api.github.com/repos/sample-org/x/pulls/3"


def test_search_pulls_paginates_and_records_graphql_rate_limit(rsa_pem: str):
    bodies: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/access_tokens"):
            return httpx.Response(201, json={"token": "t", "expires_at": "2099-01-01T00:00:00Z"})
        assert request.url.path == "/graphql" and request.headers["Authorization"] == "token t"
        body = json.loads(request.content)
        bodies.append(body)
        first = body["variables"]["after"] is None
        nodes = [_graphql_node(1), {}] if first else [_graphql_node(2, isDraft=True)]
        return httpx.Response(
            200,
            json={
                "data": {
                    "rateLimit": {"cost": 4, "remaining": 4990, "limit": 5000, "resetAt": "x"},
                    "search": {
                        "issueCount": 2,
                        "pageInfo": {"hasNextPage": first, "endCursor": "c1"},
                        "nodes": nodes,
                    },
                }
            },
        )

    gh = _github(rsa_pem, handler, FakeClock(1_800_000_000))
    result = asyncio.run(gh.search_pulls(9, "org:sample-org is:pr is:open", page_size=1))
    assert [h.pr.number for h in result.hits] == [1, 2]
    assert result.hits[1].pr.draft is True
    assert (result.total, result.pages, result.truncated) == (2, 2, False)
    assert bodies[0]["variables"] == {"q": "org:sample-org is:pr is:open", "first": 1,
                                      "after": None}  # fmt: skip
    assert bodies[1]["variables"]["after"] == "c1"
    assert gh.rate_limits["graphql"]["remaining"] == 4990


def test_search_pulls_raises_on_graphql_errors(rsa_pem: str):
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/access_tokens"):
            return httpx.Response(201, json={"token": "t", "expires_at": "2099-01-01T00:00:00Z"})
        return httpx.Response(200, json={"errors": [{"message": "bad query"}]})

    gh = _github(rsa_pem, handler, FakeClock(1_800_000_000))
    with pytest.raises(RuntimeError, match="bad query"):
        asyncio.run(gh.search_pulls(9, "org:x"))


def test_installation_repos_owner_lookup_and_core_rate_limit(rsa_pem: str):
    def handler(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if path == "/app/installations":
            return httpx.Response(200, json=[{"id": 9, "account": {"login": "sample-org"}}])
        if path.endswith("/access_tokens"):
            return httpx.Response(201, json={"token": "t", "expires_at": "2099-01-01T00:00:00Z"})
        assert path == "/installation/repositories"
        headers = {
            "x-ratelimit-remaining": "12000",
            "x-ratelimit-limit": "12500",
            "x-ratelimit-reset": "1",
            "x-ratelimit-resource": "core",
        }
        if request.url.params.get("page") == "2":
            repos = [{"full_name": "sample-org/b", "archived": True, "topics": ["x"]}]
            return httpx.Response(200, json={"repositories": repos}, headers=headers)
        link = '<https://api.github.com/installation/repositories?page=2>; rel="next"'
        repos = [{"full_name": "sample-org/a", "fork": True, "default_branch": "dev"}]
        return httpx.Response(200, json={"repositories": repos}, headers=headers | {"Link": link})

    gh = _github(rsa_pem, handler, FakeClock(1_800_000_000))

    async def run():
        inst = await gh.installation_id_for_owner("sample-org")
        return (
            inst,
            await gh.installation_id_for_owner("nobody"),
            await gh.list_installation_repos(9),
        )

    inst, missing, repos = asyncio.run(run())
    assert (inst, missing) == (9, None)
    assert [(r.full_name, r.archived, r.fork, r.topics, r.default_branch) for r in repos] == [
        ("sample-org/a", False, True, (), "dev"),
        ("sample-org/b", True, False, ("x",), ""),
    ]
    assert gh.rate_limits["core"] == {"remaining": 12000, "limit": 12500, "reset": 1}


def test_resolved_review_comment_ids_reads_review_threads(rsa_pem: str):
    bodies: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/access_tokens"):
            return httpx.Response(201, json={"token": "t", "expires_at": "2099-01-01T00:00:00Z"})
        if request.url.path == "/app/installations":
            return httpx.Response(200, json=[{"id": 9, "account": {"login": "o"}}])
        assert request.url.path == "/graphql"
        body = json.loads(request.content)
        bodies.append(body)
        first = body["variables"]["after"] is None
        nodes = (
            [{"isResolved": True, "comments": {"nodes": [{"databaseId": 11}]}},
             {"isResolved": False, "comments": {"nodes": [{"databaseId": 12}]}}]
            if first
            else [{"isResolved": True, "comments": {"nodes": [{"databaseId": 13}]}}, None]
        )  # fmt: skip
        threads = {"pageInfo": {"hasNextPage": first, "endCursor": "c1"}, "nodes": nodes}
        return httpx.Response(
            200, json={"data": {"repository": {"pullRequest": {"reviewThreads": threads}}}}
        )

    gh = _github(rsa_pem, handler, FakeClock(1_800_000_000))
    assert asyncio.run(gh.resolved_review_comment_ids("o/r", 7)) == {11, 13}
    assert bodies[0]["variables"] == {"owner": "o", "name": "r", "number": 7, "after": None}
    assert bodies[1]["variables"]["after"] == "c1"
