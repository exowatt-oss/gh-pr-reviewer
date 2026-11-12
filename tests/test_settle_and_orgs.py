"""Settle wait before reviewing a new head, merged/closed-while-waiting skips, multi-org."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from app.poller import ORG_MODE_SINCE_KEY
from app.review_policy import PrState, settle_update
from app.store import Store, iso
from tests.fakes import FakeGitHub, make_pr
from tests.test_org_mode import NEW, NOW, ORG, org_harness, org_pr, reviewed_urls, t
from tests.test_poller import REPO, Harness

OTHER_ORG = "sample-automation"
OTHER_REPO = "sample-automation/example-project"


@pytest.fixture
def store(tmp_path) -> Store:
    return Store(tmp_path / "settle.sqlite3")


def skips(store: Store) -> list[str]:
    return [r["skip_reason"] for r in store.recent_reviews(include_poller=True)]


# --- settle_update (pure) ----------------------------------------------------------------
def test_settle_anchor_first_sight_uses_latest_github_event():
    pr = make_pr(created_at=t(-600), head_committed_at=t(-120))
    assert settle_update(None, pr, NOW) == (pr.head_sha, False, NOW - 120)
    # no event at all (REST listing without dates): the wait starts now
    assert settle_update(None, make_pr(), NOW) == ("a" * 40, False, NOW)


def test_settle_anchor_kept_until_head_or_draft_changes():
    pr = make_pr(created_at=t(-600))
    state = PrState(settle_sha=pr.head_sha, settle_draft=False, head_ready_at=NOW - 600)
    assert settle_update(state, pr, NOW) is None
    # a push seen later: only events after the previous sighting count, else "now"
    seen = PrState(settle_sha="b" * 40, head_ready_at=NOW - 600, last_seen_at=NOW - 60)
    assert settle_update(seen, pr, NOW) == (pr.head_sha, False, NOW)
    pushed = make_pr(created_at=t(-600), head_committed_at=t(-30))
    assert settle_update(seen, pushed, NOW) == (pr.head_sha, False, NOW - 30)
    # marked ready for review: the ready event anchors the wait
    ready = make_pr(created_at=t(-600), ready_at=t(-10))
    drafted = PrState(settle_sha=pr.head_sha, settle_draft=True, head_ready_at=NOW - 600,
                      last_seen_at=NOW - 60)  # fmt: skip
    assert settle_update(drafted, ready, NOW) == (pr.head_sha, False, NOW - 10)


# --- poller ------------------------------------------------------------------------------
def test_review_waits_until_the_head_is_five_minutes_old(store: Store):
    gh = FakeGitHub({REPO: [make_pr(created_at=t(-60))]})
    h = Harness(store, gh, REVIEW_SETTLE_SECONDS=300)
    h.poll()
    assert h.agent.invocations == []
    status = h.poller.status()
    assert status["queue"]["settling"] == 1
    assert status["queued_jobs"][0]["ready_at"] == iso(datetime.fromtimestamp(NOW + 240, UTC))
    h.now[0] += 230
    h.poll()
    assert h.agent.invocations == []
    h.now[0] += 20
    h.poll()
    assert len(h.agent.invocations) == 1
    row = store.recent_reviews()[0]
    assert row["ready_at"] == iso(datetime.fromtimestamp(NOW + 240, UTC))
    assert row["kind"] == "full"


def test_new_push_during_the_wait_restarts_it(store: Store):
    gh = FakeGitHub()
    gh.add_repo(NEW)
    gh.add_hit(org_pr(1, created_at=t(10), updated_at=t(10), head_committed_at=t(5)))
    h = org_harness(store, gh, REVIEW_SETTLE_SECONDS=300)
    h.now[0] += 20
    h.poll()  # ready at NOW+310
    h.now[0] += 200
    gh.add_hit(
        org_pr(1, sha="b" * 40, created_at=t(10), updated_at=t(215), head_committed_at=t(210))
    )
    h.poll()  # new head: ready at NOW+510
    h.now[0] += 100  # NOW+320: the first head would have been due
    h.poll()
    assert h.agent.invocations == []
    h.now[0] += 200  # NOW+520
    h.poll()
    assert reviewed_urls(h) == [f"https://github.com/{NEW}/pull/1"]
    assert store.get_pr_state(NEW, 1).last_reviewed_sha == "b" * 40


@pytest.mark.parametrize(
    ("fresh", "reason"),
    [
        ({"state": "closed", "merged": True, "head": {"sha": "a" * 40}}, "merged_during_wait"),
        ({"state": "closed", "merged": False, "head": {"sha": "a" * 40}}, "closed"),
    ],
)
def test_pr_merged_or_closed_during_the_wait_is_recorded_not_reviewed(
    store: Store, fresh: dict, reason: str
):
    gh = FakeGitHub({REPO: [make_pr(created_at=t(0))]})
    h = Harness(store, gh, REVIEW_SETTLE_SECONDS=300)
    h.poll()
    gh.pulls[REPO] = []  # gone from the open list
    gh.pull_overrides[(REPO, 1)] = fresh
    h.now[0] += 310
    h.poll()
    assert h.agent.invocations == []
    assert skips(store) == [reason]
    row = store.recent_reviews(include_poller=True)[0]
    assert row["status"] == "skipped" and row["origin"] == "poller"
    assert row["ready_at"] == iso(datetime.fromtimestamp(NOW + 300, UTC))


def test_poller_skips_are_recorded_once_per_head(store: Store):
    gh = FakeGitHub(
        {
            REPO: [
                make_pr(number=1, draft=True),
                make_pr(number=2, author_login="dependabot[bot]", author_type="Bot"),
            ]
        }
    )
    h = Harness(store, gh)
    h.poll(times=3)
    assert sorted(skips(store)) == ["bot", "draft"]
    assert store.recent_reviews() == []  # the dashboard log shows review runs only


# --- multi-org ---------------------------------------------------------------------------
def test_org_not_installed_yet_activates_later_with_its_own_cutoff(store: Store):
    gh = FakeGitHub()
    gh.add_repo(NEW)
    h = org_harness(store, gh, REVIEW_REPOS=f"{ORG}/*,{OTHER_ORG}/*")
    h.poll()
    orgs = {o["org"]: o for o in h.poller.status()["orgs"]}
    assert orgs[OTHER_ORG] == {
        "org": OTHER_ORG,
        "state": "not_installed",
        "installation_id": None,
        "repos_covered": None,
        "cutoff": None,
    }
    assert orgs[ORG]["state"] == "ok" and orgs[ORG]["installation_id"] == 77
    assert not h.poller.status()["last_poll_errors"]

    # Installed ten minutes later: the open OTHER_ORG backlog is not reviewed, new work is.
    h.now[0] += 600
    gh.org_installations["sample-automation"] = 88
    gh.add_repo(OTHER_REPO, installation=88)
    gh.add_hit(org_pr(1, repo=OTHER_REPO, created_at=t(-86400), updated_at=t(500),
                      head_committed_at=t(-86400)))  # fmt: skip
    h.poll()
    assert h.agent.invocations == []
    orgs = {o["org"]: o for o in h.poller.status()["orgs"]}
    assert orgs[OTHER_ORG]["installation_id"] == 88 and orgs[OTHER_ORG]["repos_covered"] == 1
    assert orgs[OTHER_ORG]["cutoff"] == iso(datetime.fromtimestamp(NOW + 600, UTC))
    assert orgs[ORG]["cutoff"] == iso(datetime.fromtimestamp(NOW, UTC))
    assert store.get_meta(f"{ORG_MODE_SINCE_KEY}:{OTHER_ORG.lower()}") == orgs[OTHER_ORG]["cutoff"]

    h.now[0] += 60
    gh.add_hit(org_pr(2, repo=OTHER_REPO, created_at=t(650), updated_at=t(650)))
    h.poll()
    assert reviewed_urls(h) == [f"https://github.com/{OTHER_REPO}/pull/2"]
    assert len(gh.reviews[(OTHER_REPO, 2)]) == 1  # posted as a review on the OTHER_ORG repo


def test_legacy_single_org_cutoff_carries_over_only_to_orgs_it_covered(store: Store):
    legacy = iso(datetime.fromtimestamp(NOW - 1000, UTC))
    store.set_meta(ORG_MODE_SINCE_KEY, legacy)
    store.observe(NEW, 9, "c" * 40, False, NOW - 500, cutoff=NOW - 1000)
    gh = FakeGitHub()
    gh.org_installations["sample-automation"] = 88
    h = org_harness(store, gh, REVIEW_REPOS=f"{ORG}/*,{OTHER_ORG}/*")
    h.poll()
    orgs = {o["org"]: o for o in h.poller.status()["orgs"]}
    assert orgs[ORG]["cutoff"] == legacy
    assert orgs[OTHER_ORG]["cutoff"] == iso(datetime.fromtimestamp(NOW, UTC))
