"""Unit tests for mandatory parent-IAP credential forwarding on CoreWeave."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import pytest

pytest.importorskip("rigging", reason="rigging is a private Marin-monorepo package, not installable in CI")
import rigging.auth  # noqa: E402

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import cloud.iris.iris_backend as launcher  # noqa: E402


def _args() -> argparse.Namespace:
    return argparse.Namespace(target_cluster="cw-rno2a", ingress_mode="controller")


def _record() -> dict[str, str]:
    return {
        "cluster": "marin",
        "endpoint": "https://iris.oa.dev",
        "edge_refresh_token": "refresh-token",
    }


def test_federated_parent_auth_is_not_needed_for_direct_ingress():
    assert (
        launcher.prepare_federated_parent_credentials(
            argparse.Namespace(target_cluster="cw-rno2a", ingress_mode="direct")
        )
        is None
    )


def test_federated_parent_auth_requires_the_marin_login_record(tmp_path, monkeypatch):
    monkeypatch.setattr(launcher, "MARIN_LOGIN_RECORD_PATH", tmp_path / "marin.json")
    monkeypatch.setattr(launcher, "_resolve_parent_cluster_config", lambda _config: None)
    with pytest.raises(SystemExit, match="cached Marin IAP login record.*or ambient service-account credentials"):
        launcher.prepare_federated_parent_credentials(_args())


def test_federated_parent_auth_forwards_ambient_service_account_iap_token(tmp_path, monkeypatch):
    import iris.cli.connect
    import iris.cluster.config
    from cloud.iris.ingress_utils import PARENT_IAP_TOKEN_ENV

    monkeypatch.setattr(launcher, "MARIN_LOGIN_RECORD_PATH", tmp_path / "marin.json")
    monkeypatch.setattr(launcher, "_resolve_parent_cluster_config", lambda _config: "/configs/marin.yaml")
    monkeypatch.setattr(iris.cluster.config, "load_config", lambda _path: object())
    monkeypatch.setattr(
        iris.cli.connect,
        "client_credentials",
        lambda _config, _cluster: argparse.Namespace(iap_provider=argparse.Namespace(get_token=lambda: "iap-token")),
    )

    assert launcher.prepare_federated_parent_credentials(_args()) is None
    assert os.environ[PARENT_IAP_TOKEN_ENV] == "iap-token"


def test_federated_parent_auth_validates_and_returns_the_record(tmp_path, monkeypatch):
    record_path = tmp_path / "marin.json"
    record_path.write_text(json.dumps(_record()))
    monkeypatch.setattr(launcher, "MARIN_LOGIN_RECORD_PATH", record_path)

    class FakeTokenProvider:
        def __init__(self, *args, **kwargs):
            pass

        def get_token(self):
            return "iap-token"

    monkeypatch.setattr(rigging.auth, "IapRefreshTokenProvider", FakeTokenProvider)
    assert json.loads(launcher.prepare_federated_parent_credentials(_args()) or "{}") == _record()


def test_federated_parent_auth_rejects_a_non_marin_record(tmp_path, monkeypatch):
    record_path = tmp_path / "marin.json"
    record_path.write_text(json.dumps({**_record(), "cluster": "another-cluster"}))
    monkeypatch.setattr(launcher, "MARIN_LOGIN_RECORD_PATH", record_path)
    with pytest.raises(SystemExit, match="not a Marin iris.oa.dev login record"):
        launcher.prepare_federated_parent_credentials(_args())
