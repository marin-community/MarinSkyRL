"""Regression guard: controller-ingress must NOT overload OPENAI_BASE_URL.

training_driver._ingress_context publishes the co-located served-model URL as the harbor-
specific HARBOR_MODEL_ENDPOINT (which harbor's opencode config-writer reads). It must
NEVER write OPENAI_BASE_URL — that var is reserved for genuine OpenAI traffic (the
LLM-judge verifiers on the worker read it), so overloading it with the vLLM capability
URL would silently misroute every judge call to vLLM. This pins that contract (the
b77d80e6 band-aid that clobbered OPENAI_BASE_URL is the exact regression guarded here).

Run:
    python -m pytest cloud/iris/tests/test_training_driver_ingress.py -v
"""

from __future__ import annotations

import contextlib
import json
import os
import sys
from pathlib import Path

import pytest
from omegaconf import OmegaConf

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from cloud.iris import ingress_utils, literal_proxy_utils  # noqa: E402
from cloud.iris.training_driver import LocalRLConfig, LocalRLRunner  # noqa: E402

_FAKE_CAP_URL = "https://iris.oa.dev/proxy/t/faketoken/otagent-x/v1"


class _FakeRegistration:
    endpoint_id = "fake-id"

    def close(self) -> None:
        pass


def _patch_ingress(monkeypatch) -> None:
    """Stub the live ingress helpers so _ingress_context runs without iris/harbor/GPU."""
    monkeypatch.setattr(
        ingress_utils,
        "controller_registration_plan",
        lambda *a, **k: ("otagent-x", "http://10.0.0.1:8010"),
    )
    monkeypatch.setattr(ingress_utils, "register_controller_endpoint", lambda *a, **k: _FakeRegistration())
    monkeypatch.setattr(ingress_utils, "capability_api_base", lambda *a, **k: _FAKE_CAP_URL)
    monkeypatch.setattr(ingress_utils, "federated_capability_api_base", lambda *a, **k: _FAKE_CAP_URL)
    monkeypatch.setattr(ingress_utils, "inject_ingress_agent_key", lambda *a, **k: True)

    @contextlib.contextmanager
    def _null_proxy(*a, **k):
        yield "http://10.0.0.1:8010/v1"

    monkeypatch.setattr(literal_proxy_utils, "maybe_serve_literal_proxy", _null_proxy)


def _runner() -> LocalRLRunner:
    cfg = LocalRLConfig(
        job_name="test-job",
        model_path="Qwen/Qwen3-8B",
        ingress_mode="controller",
        ingress_host="iris.oa.dev",
        record_literal=True,
    )
    return LocalRLRunner(cfg)


def test_controller_ingress_sets_harbor_endpoint_and_never_touches_openai_base_url(monkeypatch):
    _patch_ingress(monkeypatch)
    monkeypatch.delenv("HARBOR_MODEL_ENDPOINT", raising=False)
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    runner = _runner()
    with runner._ingress_context():
        assert os.environ["HARBOR_MODEL_ENDPOINT"] == _FAKE_CAP_URL
        # THE GUARD: the harness must NOT overload OPENAI_BASE_URL with the vLLM URL.
        assert os.environ.get("OPENAI_BASE_URL") is None
        # The minted URL is ALSO captured for Hydra-cfg threading, so run() can inject
        # ++terminal_bench_config.agent_api_base=<url> — the env var alone never reaches
        # the pre-existing Ray workers where HarborTrajectoryRunner is built.
        assert runner._minted_agent_api_base == _FAKE_CAP_URL


def test_controller_ingress_preserves_a_real_openai_base_url(monkeypatch):
    _patch_ingress(monkeypatch)
    monkeypatch.setenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
    runner = _runner()
    with runner._ingress_context():
        assert os.environ["HARBOR_MODEL_ENDPOINT"] == _FAKE_CAP_URL
        # A real OpenAI base url (the LLM-judge verifiers' endpoint) survives untouched —
        # it is NOT clobbered to the vLLM capability URL.
        assert os.environ["OPENAI_BASE_URL"] == "https://api.openai.com/v1"


def test_direct_ingress_still_publishes_agent_dummy_key(monkeypatch):
    """Agent auth is DECOUPLED from controller-ingress: an installed agent (opencode) on
    ingress_mode=direct must still get the inert dummy key, or it refuses to start (zero
    requests -> silent empty rollouts). The dummy key is published BEFORE the direct-mode
    early-return, so it lands regardless of ingress_mode — no live controller stubbing
    needed here since the direct path calls no ingress helpers."""
    monkeypatch.delenv(ingress_utils.AGENT_DUMMY_KEY_VAR, raising=False)
    cfg = LocalRLConfig(
        job_name="test-job",
        model_path="Qwen/Qwen3-8B",
        ingress_mode="direct",
    )
    runner = LocalRLRunner(cfg)
    with runner._ingress_context():
        assert os.environ[ingress_utils.AGENT_DUMMY_KEY_VAR] == ingress_utils.DUMMY_API_KEY


def test_direct_ingress_never_clobbers_a_real_openai_api_key(monkeypatch):
    """The dummy-key injection only setdefaults OPENAI_API_KEY (real host key preserved)."""
    monkeypatch.setenv("OPENAI_API_KEY", "sk-real-host-key")
    cfg = LocalRLConfig(
        job_name="test-job",
        model_path="Qwen/Qwen3-8B",
        ingress_mode="direct",
    )
    runner = LocalRLRunner(cfg)
    with runner._ingress_context():
        assert os.environ["OPENAI_API_KEY"] == "sk-real-host-key"


def test_checkpoint_export_entrypoint_bypasses_rollout_environment(monkeypatch, tmp_path):
    launch_config = OmegaConf.create(
        {
            "run": {"mode": "checkpoint_export"},
            "runtime": {"entrypoint": "skyrl_train.entrypoints.checkpoint_export"},
            "skyrl": {"trainer": {"policy": {"model": {"path": "/tmp/policy"}}}},
        }
    )
    cfg = LocalRLConfig(
        job_name="checkpoint-export",
        model_path="Qwen/Qwen3-8B",
        train_data=[OmegaConf.create({"source": "unused-during-export"})],
        resolved_config_uri=(tmp_path / "resolved.json").as_uri(),
        gpus=4,
        launch_config=launch_config,
    )
    runner = LocalRLRunner(cfg)
    invocation = {}

    monkeypatch.setattr(
        runner,
        "_setup_environment",
        lambda _args: pytest.fail("checkpoint export must not configure the rollout runtime"),
    )
    monkeypatch.setattr(runner, "_run_skyrl", lambda config: invocation.update(run=config) or 0)

    assert runner.run() == 0
    assert invocation == {"run": launch_config}
    assert json.loads((tmp_path / "resolved.json").read_text()) == {
        "config": OmegaConf.to_container(launch_config, resolve=True),
        "train_data_sources": [{"source": "unused-during-export"}],
        "val_data_sources": [],
    }
