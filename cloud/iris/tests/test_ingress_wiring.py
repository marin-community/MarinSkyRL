"""Unit tests for cloud/iris/ingress_utils.py — capability-URL wiring + federated parent-minting.

Proves the controller-ingress wiring WITHOUT a live controller: the pure helpers build
the ``/proxy/t/<token>/<name>/v1`` capability api_base, the worker-side token cache
mints/re-mints, and the FEDERATED path waits for the FederationSync mirror then mints at
the PARENT. All minters/resolvers are in-process fakes; no iris import or live controller
is exercised (iris is lazy-imported only by the production adapters).

Run:
    python -m pytest cloud/iris/tests/test_ingress_wiring.py -v
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from cloud.iris.ingress_utils import (  # noqa: E402
    DEFAULT_PARENT_INGRESS_HOST,
    DEFAULT_VLLM_PORT,
    TOKEN_REFRESH_MARGIN_SECONDS,
    CapabilityTokenCache,
    FederatedCapabilityTokenCache,
    build_capability_api_base,
    capability_api_base,
    controller_endpoint_name,
    controller_registration_plan,
    encode_endpoint_name,
    federated_capability_api_base,
    wait_for_endpoint_mirror,
)


class _FakeMinter:
    def __init__(self, expires_at: float = 10_000_000_000.0):
        self.expires_at = expires_at
        self.calls = 0

    def mint(self, endpoint_name, ttl_hours):
        self.calls += 1
        return f"TKN-{self.calls}", self.expires_at


class _FakeResolver:
    """Reports the endpoint mirrored after ``ready_after`` polls; can raise transiently."""

    def __init__(self, ready_after: int = 0, raise_first: int = 0):
        self.ready_after = ready_after
        self.raise_first = raise_first
        self.calls = 0

    def is_mirrored(self, endpoint_name):
        self.calls += 1
        if self.calls <= self.raise_first:
            raise RuntimeError("transient resolve error")
        return self.calls > self.ready_after


# --------------------------------------------------------------------------- #
# Pure helpers
# --------------------------------------------------------------------------- #


def test_controller_endpoint_name_sanitizes():
    assert controller_endpoint_name("rl-qwen3_8b/run.1") == "otagent-rl-qwen3_8b-run-1"
    assert controller_endpoint_name(None) == "otagent-job"
    name = controller_endpoint_name("a/b.c/d")
    assert "/" not in name and "." not in name
    assert encode_endpoint_name(name) == name


def test_build_capability_api_base_puts_token_in_path():
    assert (
        build_capability_api_base("ingress.example", "otagent-job1", "JWT.abc")
        == "https://ingress.example/proxy/t/JWT.abc/otagent-job1/v1"
    )
    assert build_capability_api_base("http://10.0.0.1:8443/", "ep", "TK") == "http://10.0.0.1:8443/proxy/t/TK/ep/v1"


def test_capability_api_base_uses_cached_token():
    class _Fixed:
        def mint(self, endpoint_name, ttl_hours):
            return "ABC", 10_000_000_000.0

    url = capability_api_base("ingress.example", "otagent-myjob", cache=CapabilityTokenCache(_Fixed()))
    assert url == "https://ingress.example/proxy/t/ABC/otagent-myjob/v1"


def test_capability_token_cache_reuses_until_margin_then_remints():
    minter = _FakeMinter(expires_at=1000.0)
    cache = CapabilityTokenCache(minter)
    assert cache.token_for("ep", now=0.0) == "TKN-1" and minter.calls == 1
    assert cache.token_for("ep", now=1000.0 - TOKEN_REFRESH_MARGIN_SECONDS - 1) == "TKN-1"
    assert minter.calls == 1
    assert cache.token_for("ep", now=1000.0 - TOKEN_REFRESH_MARGIN_SECONDS + 1) == "TKN-2"
    assert minter.calls == 2


def test_controller_registration_plan_picks_proxy_port_when_record_literal():
    name, addr = controller_registration_plan("job1", record_literal=True, proxy_port=8010, env={})
    assert name == "otagent-job1" and addr.endswith(":8010")
    name, addr = controller_registration_plan("job1", record_literal=False, proxy_port=8010, env={})
    assert addr.endswith(f":{DEFAULT_VLLM_PORT}")


# --------------------------------------------------------------------------- #
# Federated parent-minting
# --------------------------------------------------------------------------- #


def test_wait_for_endpoint_mirror_returns_when_ready():
    resolver = _FakeResolver(ready_after=2)
    slept = []
    wait_for_endpoint_mirror("otagent-x", resolver, timeout_s=100, interval_s=1, sleep=slept.append, now=lambda: 0.0)
    assert resolver.calls == 3 and slept == [1, 1]


def test_wait_for_endpoint_mirror_tolerates_transient_errors():
    resolver = _FakeResolver(ready_after=0, raise_first=2)
    wait_for_endpoint_mirror("otagent-x", resolver, timeout_s=100, interval_s=1, sleep=lambda _s: None, now=lambda: 0.0)
    assert resolver.calls == 3


def test_wait_for_endpoint_mirror_times_out():
    resolver = _FakeResolver(ready_after=999)
    clock = {"t": 0.0}
    with pytest.raises(TimeoutError, match="was not mirrored"):
        wait_for_endpoint_mirror(
            "otagent-x",
            resolver,
            timeout_s=5,
            interval_s=2,
            sleep=lambda s: clock.__setitem__("t", clock["t"] + s),
            now=lambda: clock["t"],
        )


def test_federated_cache_waits_for_mirror_then_mints_at_parent():
    minter, resolver = _FakeMinter(), _FakeResolver(ready_after=1)
    cache = FederatedCapabilityTokenCache(minter, resolver, mirror_interval_s=0, mirror_timeout_s=100)
    assert cache.token_for("otagent-fed", now=0.0) == "TKN-1"
    assert minter.calls == 1 and resolver.calls == 2
    # cached reuse — mirror wait NOT repeated
    assert cache.token_for("otagent-fed", now=1.0) == "TKN-1"
    assert minter.calls == 1 and resolver.calls == 2


def test_federated_cache_remint_does_not_repoll_mirror():
    minter, resolver = _FakeMinter(expires_at=1000.0), _FakeResolver(ready_after=0)
    cache = FederatedCapabilityTokenCache(minter, resolver, mirror_interval_s=0, mirror_timeout_s=100)
    cache.token_for("ep", now=0.0)
    assert minter.calls == 1 and resolver.calls == 1
    cache.token_for("ep", now=1000.0 - TOKEN_REFRESH_MARGIN_SECONDS + 1)
    assert minter.calls == 2 and resolver.calls == 1  # no re-poll


def test_federated_cache_propagates_mirror_timeout_and_never_mints():
    minter, resolver = _FakeMinter(), _FakeResolver(ready_after=999)
    cache = FederatedCapabilityTokenCache(minter, resolver, mirror_interval_s=0, mirror_timeout_s=0)
    with pytest.raises(TimeoutError):
        cache.token_for("ep", now=0.0)
    assert minter.calls == 0


def test_federated_capability_api_base_builds_parent_url():
    cache = FederatedCapabilityTokenCache(_FakeMinter(), _FakeResolver(ready_after=0), mirror_interval_s=0)
    url = federated_capability_api_base("otagent-fedjob", ingress_host="iris.oa.dev", cache=cache, now=0.0)
    assert url == "https://iris.oa.dev/proxy/t/TKN-1/otagent-fedjob/v1"


def test_default_parent_ingress_host_is_marin():
    assert DEFAULT_PARENT_INGRESS_HOST == "iris.oa.dev"


def test_materialize_parent_credentials_writes_forwarded_record(tmp_path, monkeypatch):
    """A forwarded login record is written to the path load_credentials reads (in-pod)."""
    import json
    from cloud.iris.ingress_utils import (
        PARENT_CREDENTIALS_JSON_ENV,
        materialize_parent_credentials,
    )

    monkeypatch.setenv("HOME", str(tmp_path))
    rec = json.dumps({"cluster": "marin", "endpoint": "https://iris.oa.dev", "edge_refresh_token": "RT"})
    monkeypatch.setenv(PARENT_CREDENTIALS_JSON_ENV, rec)
    dest = materialize_parent_credentials()
    assert dest is not None and dest.endswith("marin.json")
    written = json.loads((tmp_path / ".config" / "marin" / "credentials" / "marin.json").read_text())
    assert written["edge_refresh_token"] == "RT"


def test_materialize_parent_credentials_noop_without_env(tmp_path, monkeypatch):
    """No forwarded record => no-op (iris falls back to ambient service-account creds)."""
    from cloud.iris.ingress_utils import PARENT_CREDENTIALS_JSON_ENV, materialize_parent_credentials

    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.delenv(PARENT_CREDENTIALS_JSON_ENV, raising=False)
    assert materialize_parent_credentials() is None
    assert not (tmp_path / ".config" / "marin" / "credentials" / "marin.json").exists()


def test_materialize_parent_controller_config_writes_and_repoints(tmp_path, monkeypatch):
    """Forwarded marin.yaml content is written in-pod and PARENT_CONTROLLER_CONFIG_ENV repointed."""
    from cloud.iris.ingress_utils import (
        PARENT_CONTROLLER_CONFIG_ENV,
        PARENT_CONTROLLER_CONFIG_YAML_ENV,
        materialize_parent_controller_config,
    )

    monkeypatch.setenv("HOME", str(tmp_path))
    # The launcher forwards a launch-host path that does NOT resolve in-pod ...
    monkeypatch.setenv(PARENT_CONTROLLER_CONFIG_ENV, "/launch-host/marin.yaml")
    # ... plus the file content (write-from-env).
    monkeypatch.setenv(PARENT_CONTROLLER_CONFIG_YAML_ENV, "name: marin\ndashboard_url: https://iris.oa.dev\n")
    dest = materialize_parent_controller_config()
    assert dest is not None and dest.endswith("marin.yaml")
    written = (tmp_path / ".config" / "marin" / "marin.yaml").read_text()
    assert "dashboard_url: https://iris.oa.dev" in written
    # The env is repointed at the real in-pod file so load_config reads it.
    assert os.environ[PARENT_CONTROLLER_CONFIG_ENV] == dest


def test_materialize_parent_controller_config_noop_returns_existing_path(tmp_path, monkeypatch):
    """No content forwarded => returns the existing path unchanged (baked/synced marin.yaml)."""
    from cloud.iris.ingress_utils import (
        PARENT_CONTROLLER_CONFIG_ENV,
        PARENT_CONTROLLER_CONFIG_YAML_ENV,
        materialize_parent_controller_config,
    )

    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.delenv(PARENT_CONTROLLER_CONFIG_YAML_ENV, raising=False)
    monkeypatch.setenv(PARENT_CONTROLLER_CONFIG_ENV, "/in-pod/baked/marin.yaml")
    assert materialize_parent_controller_config() == "/in-pod/baked/marin.yaml"
    assert not (tmp_path / ".config" / "marin" / "marin.yaml").exists()
