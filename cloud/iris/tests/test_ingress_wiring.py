"""Unit tests for cloud/iris/ingress_utils.py — native capability-URL wiring.

Proves the native controller-ingress wiring (the SAME recipe datagen uses) WITHOUT a live
controller: the pure helpers build the ``/proxy/t/<token>/<name>/v1`` capability api_base,
the worker-side token cache mints/re-mints, and the registration plan picks the RecordProxy
vs. raw-vLLM port. All minters are in-process fakes; iris is lazy-imported only by the
production adapters, so nothing here touches a controller.

Run:
    python -m pytest cloud/iris/tests/test_ingress_wiring.py -v
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from cloud.iris.ingress_utils import (  # noqa: E402
    DEFAULT_VLLM_PORT,
    DUMMY_API_KEY,
    TOKEN_REFRESH_MARGIN_SECONDS,
    CapabilityTokenCache,
    build_capability_api_base,
    build_controller_endpoint_meta,
    capability_api_base,
    controller_endpoint_name,
    controller_registration_plan,
    encode_endpoint_name,
    inject_ingress_agent_key,
)


class _FakeMinter:
    def __init__(self, expires_at: float = 10_000_000_000.0):
        self.expires_at = expires_at
        self.calls = 0

    def mint(self, endpoint_name, ttl_hours):
        self.calls += 1
        return f"TKN-{self.calls}", self.expires_at


def _fixed_cache(token="ABC", expires_at=10_000_000_000.0):
    class _Fixed:
        def mint(self, endpoint_name, ttl_hours):
            return token, expires_at

    return CapabilityTokenCache(_Fixed())


def test_controller_endpoint_name_sanitizes():
    assert controller_endpoint_name("rl-qwen3_8b/run.1") == "otagent-rl-qwen3_8b-run-1"
    assert controller_endpoint_name(None) == "otagent-job"
    name = controller_endpoint_name("a/b.c/d")
    assert "/" not in name and "." not in name
    assert encode_endpoint_name(name) == name


def test_encode_endpoint_name_matches_rigging_scheme():
    assert encode_endpoint_name("/serve/foo") == "serve.foo"
    assert encode_endpoint_name("otagent-job") == "otagent-job"


def test_build_capability_api_base_puts_token_in_path():
    assert (
        build_capability_api_base("iris.oa.dev", "otagent-job1", "JWT.abc")
        == "https://iris.oa.dev/proxy/t/JWT.abc/otagent-job1/v1"
    )
    assert (
        build_capability_api_base("https://iris.oa.dev/", "ep", "TK")
        == "https://iris.oa.dev/proxy/t/TK/ep/v1"
    )


def test_capability_api_base_uses_cached_token():
    url = capability_api_base("https://iris.oa.dev", "otagent-myjob", cache=_fixed_cache())
    assert url == "https://iris.oa.dev/proxy/t/ABC/otagent-myjob/v1"


def test_capability_token_cache_reuses_until_margin_then_remints():
    minter = _FakeMinter(expires_at=1000.0)
    cache = CapabilityTokenCache(minter)
    assert cache.token_for("ep", now=0.0) == "TKN-1" and minter.calls == 1
    assert cache.token_for("ep", now=1000.0 - TOKEN_REFRESH_MARGIN_SECONDS - 1) == "TKN-1"
    assert minter.calls == 1
    assert cache.token_for("ep", now=1000.0 - TOKEN_REFRESH_MARGIN_SECONDS + 1) == "TKN-2"
    assert minter.calls == 2


def test_capability_token_cache_rejects_empty_token():
    class _EmptyMinter:
        def mint(self, endpoint_name, ttl_hours):
            return "", 10_000_000_000.0

    with pytest.raises(RuntimeError):
        CapabilityTokenCache(_EmptyMinter()).token_for("ep", now=0.0)


def test_controller_registration_plan_picks_proxy_port_when_record_literal():
    name, addr = controller_registration_plan("job1", record_literal=True, proxy_port=8010, env={})
    assert name == "otagent-job1" and addr.endswith(":8010")
    name, addr = controller_registration_plan("job1", record_literal=False, proxy_port=8010, env={})
    assert addr.endswith(f":{DEFAULT_VLLM_PORT}")


def test_build_controller_endpoint_meta_has_capability_url_and_dummy_key():
    meta = build_controller_endpoint_meta("https://iris.oa.dev", "otagent-job1", cache=_fixed_cache())
    assert meta["api_base"] == "https://iris.oa.dev/proxy/t/ABC/otagent-job1/v1"
    assert meta["api_key"] == DUMMY_API_KEY


def test_inject_ingress_agent_key_sets_dummy_without_clobbering_real_openai(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "real-judge-key")
    env = {"OPENAI_API_KEY": "real-judge-key"}
    assert inject_ingress_agent_key(env) is True
    assert env["OPENCODE_DUMMY_KEY"] == DUMMY_API_KEY
    assert env["OPENAI_API_KEY"] == "real-judge-key"  # real judge key preserved
