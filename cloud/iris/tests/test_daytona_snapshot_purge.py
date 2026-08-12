"""Behavior tests for the pre-launch Daytona snapshot purge in cloud/iris/iris_backend.py.

Proves the purge rule that keeps the RL Daytona org under its snapshot quota: only
harbor-minted (``harbor__*``) snapshots idle past ``STALE_SNAPSHOT_MAX_AGE`` are deleted,
non-harbor (system) snapshots are never touched regardless of age, a null ``last_used_at``
falls back to ``created_at``, and pagination is followed to completion so late-page
candidates are still found. The Daytona SDK is faked via ``launcher._daytona_client`` —
no live Daytona org is exercised.

Run:
    python -m pytest cloud/iris/tests/test_daytona_snapshot_purge.py -v
"""

from __future__ import annotations

import datetime
import sys
from dataclasses import dataclass, field
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import cloud.iris.iris_backend as launcher  # noqa: E402


@dataclass
class _FakeSnapshot:
    name: str
    created_at: datetime.datetime
    last_used_at: datetime.datetime | None = None


@dataclass
class _FakePage:
    items: list[_FakeSnapshot]
    total: int
    total_pages: int


@dataclass
class _FakeSnapshotService:
    snapshots: list[_FakeSnapshot]
    page_size: int = 100
    deleted: list[str] = field(default_factory=list)

    def list(self, page: int, limit: int) -> _FakePage:
        start = (page - 1) * limit
        total = len(self.snapshots)
        total_pages = max(1, -(-total // limit))
        return _FakePage(items=self.snapshots[start : start + limit], total=total, total_pages=total_pages)

    def delete(self, snapshot: _FakeSnapshot) -> None:
        self.snapshots.remove(snapshot)
        self.deleted.append(snapshot.name)


class _FakeDaytona:
    def __init__(self, snapshots: list[_FakeSnapshot]) -> None:
        self.snapshot = _FakeSnapshotService(snapshots)


def _install_fake_client(monkeypatch, snapshots: list[_FakeSnapshot]) -> _FakeDaytona:
    fake = _FakeDaytona(snapshots)
    monkeypatch.setattr(launcher, "_daytona_client", lambda api_key: fake)
    return fake


def _now() -> datetime.datetime:
    return datetime.datetime.now(datetime.timezone.utc)


def test_purge_deletes_stale_harbor_snapshots_and_keeps_fresh_ones(monkeypatch, capsys):
    now = _now()
    snapshots = [
        _FakeSnapshot(
            name="harbor__stale",
            created_at=now - datetime.timedelta(hours=6),
            last_used_at=now - datetime.timedelta(hours=3),
        ),
        _FakeSnapshot(
            name="harbor__fresh",
            created_at=now - datetime.timedelta(hours=6),
            last_used_at=now - datetime.timedelta(hours=1),
        ),
    ]
    fake = _install_fake_client(monkeypatch, snapshots)

    launcher._purge_stale_daytona_snapshots("fake-key")

    assert fake.snapshot.deleted == ["harbor__stale"]
    assert [s.name for s in fake.snapshot.snapshots] == ["harbor__fresh"]
    out = capsys.readouterr().out
    assert "total=2 harbor=2 purged=1 kept=1" in out


def test_purge_never_deletes_non_harbor_snapshots_even_when_old(monkeypatch):
    now = _now()
    snapshots = [
        _FakeSnapshot(name="base-image-py312", created_at=now - datetime.timedelta(days=90), last_used_at=None),
    ]
    fake = _install_fake_client(monkeypatch, snapshots)

    launcher._purge_stale_daytona_snapshots("fake-key")

    assert fake.snapshot.deleted == []
    assert [s.name for s in fake.snapshot.snapshots] == ["base-image-py312"]


def test_purge_falls_back_to_created_at_when_last_used_at_is_null(monkeypatch):
    now = _now()
    snapshots = [
        _FakeSnapshot(name="harbor__never_used_stale", created_at=now - datetime.timedelta(hours=3), last_used_at=None),
        _FakeSnapshot(name="harbor__never_used_fresh", created_at=now - datetime.timedelta(hours=1), last_used_at=None),
    ]
    fake = _install_fake_client(monkeypatch, snapshots)

    launcher._purge_stale_daytona_snapshots("fake-key")

    assert fake.snapshot.deleted == ["harbor__never_used_stale"]
    assert [s.name for s in fake.snapshot.snapshots] == ["harbor__never_used_fresh"]


def test_purge_finds_stale_candidates_on_a_later_page(monkeypatch):
    now = _now()
    fresh_page_one = [
        _FakeSnapshot(
            name=f"harbor__fresh-{i}",
            created_at=now - datetime.timedelta(hours=6),
            last_used_at=now - datetime.timedelta(minutes=30),
        )
        for i in range(100)
    ]
    stale_page_two = _FakeSnapshot(
        name="harbor__stale-on-page-two",
        created_at=now - datetime.timedelta(hours=6),
        last_used_at=now - datetime.timedelta(hours=5),
    )
    snapshots = [*fresh_page_one, stale_page_two]
    fake = _install_fake_client(monkeypatch, snapshots)

    launcher._purge_stale_daytona_snapshots("fake-key")

    assert fake.snapshot.deleted == ["harbor__stale-on-page-two"]
    assert len(fake.snapshot.snapshots) == 100


def test_purge_skips_gracefully_when_daytona_sdk_is_unavailable(monkeypatch, capsys):
    """On a macOS launch host the daytona SDK is not installed (pyproject.toml gates it
    to ``sys_platform == 'linux'``). The purge must return without raising."""

    def _raise_import_error(_api_key):
        raise ImportError("No module named 'daytona'")

    monkeypatch.setattr(launcher, "_daytona_client", _raise_import_error)

    launcher._purge_stale_daytona_snapshots("fake-key")  # must not raise

    assert capsys.readouterr().out, "expected a warning when the purge is skipped"
