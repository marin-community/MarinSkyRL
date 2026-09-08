"""Actual streaming-copy identity checks for the bounded native fixture stage."""

import hashlib
import io

import pytest

from tests.gpu.diagnostics.stage_qwen_correction_fixture import copy_verified
from tests.correction_fixture_config import correction_actor_config, register_correction_reference


def test_actual_fixture_config_uses_native_token_mean_without_credentials(monkeypatch):
    from skyrl_train.utils.utils import validate_cfg

    monkeypatch.delenv("WANDB_API_KEY", raising=False)
    config = correction_actor_config("/tmp/immutable-qwen-fixture")
    assert config.trainer.logger == "console"
    assert config.trainer.algorithm.loss_reduction == "token_mean"
    assert config.trainer.policy.model.path == "/tmp/immutable-qwen-fixture"
    validate_cfg(config)


def test_correction_reference_survives_two_actual_ray_clusters():
    import ray
    from skyrl_train.utils.algorithm_registry import PolicyLossRegistry

    handles = []
    try:
        for mode in ("offpolicy", "m2"):
            finalizers = []
            ray.init(num_cpus=1, include_dashboard=False, log_to_driver=False)
            try:
                name = register_correction_reference(mode, finalizers.append)
                assert PolicyLossRegistry.get(name).keywords["mode"] == mode
                handles.append(PolicyLossRegistry._ray_actor._actor_id)
            finally:
                for finalize in reversed(finalizers):
                    finalize()
                ray.shutdown()
        assert len(handles) == 2 and handles[0] != handles[1]
    finally:
        ray.shutdown()
        PolicyLossRegistry.shutdown_actor()


@pytest.mark.parametrize("change", ["none", "wrong_hash", "truncated", "oversized"])
def test_streamed_artifact_requires_digest_and_size(tmp_path, change):
    payload = b"actual immutable fixture bytes"
    digest = hashlib.sha256(payload).hexdigest()
    incoming = payload[:-1] if change == "truncated" else payload + b"extra" if change == "oversized" else payload
    target = tmp_path / "model.safetensors"
    expected = "0" * 64 if change == "wrong_hash" else digest
    if change == "none":
        receipt = copy_verified(io.BytesIO(incoming), target, expected, expected_size=len(payload))
        assert receipt == {"sha256": digest, "bytes": len(payload)} and target.read_bytes() == payload
    else:
        with pytest.raises(ValueError):
            copy_verified(io.BytesIO(incoming), target, expected, expected_size=len(payload))
        assert not target.exists()
    assert not target.with_suffix(".safetensors.partial").exists()
