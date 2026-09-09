"""Actual zero-training lifecycle through the worker and receiver protocol."""

import json
from pathlib import Path

import pytest

from skyrl_train.weight_sync import bucket_qualification as qualification
from tests.cpu.weight_sync.test_megatron_bucket_protocol import gate as gate
from tests.cpu.weight_sync.test_readback_diagnostics import _geometry, _readbacks
from tests.cpu.weight_sync.test_worker_bucket_protocol import native_protocol as native_protocol


@pytest.fixture
def diagnostic(request, monkeypatch):
    case = request.getfixturevalue("gate")
    monkeypatch.setenv("IRIS_ATTEMPT_UID", "cpu-native-boundary")
    monkeypatch.setattr(qualification, "BUCKET_BYTES", 32)
    monkeypatch.setattr(qualification, "MAX_REPLAY_EXTRA_BYTES", 16)
    policy, receivers = _readbacks()
    policy[0]["expert_samples"] = [{"name": "source-expert"}]
    policy[0]["expert_layouts"] = [{"single_grouped_weight": False}]
    receivers[0][0].update(free_bytes=1000, layers=[{"backend": "TRITON"}])

    class NativeBoundary:
        def __init__(self):
            self.policy_model = self
            self.inference_engine_client = self
            self.calls = []
            self.readbacks = (policy, receivers)

        async def read_policy_environment(self):
            return {"rank": 0, "environment": policy[0]["environment"]}

        async def read_weight_sync_environment(self):
            return receivers

        def init_weight_sync_state(self):
            self.calls.append("init")

        async def async_sync_policy_weights_to_inference_engines(self):
            assert self.calls == ["init"]
            self.calls.append("reference-sync")

        async def _drain_policy_event_loops(self):
            self.calls.append("drain")

        async def read_policy(self):
            return policy[0]

        async def read_publication_receiver_state(self):
            assert self.calls == ["init", "reference-sync", "drain"]
            return receivers

        async def pause_generation(self):
            self.calls.append("pause")

        async def resume_generation(self):
            assert self.calls[-1] == "bucket"
            self.calls.append("resume")

        async def bucket(self):
            assert self.calls[-1] == "pause"
            self.calls.append("bucket")
            return await case.policy.diagnostic_bucket_install_and_replay(case.client)

        def async_run_ray_method(self, dispatch, method, *args):
            assert dispatch == "pass_through"
            if method == "diagnostic_bucket_install_and_replay":
                assert args == (self,)
                return [self.bucket()]
            return [
                {
                    "read_weight_sync_environment": self.read_policy_environment,
                    "read_weight_sync_policy_state": self.read_policy,
                }[method]()
            ]

        async def train(self):
            pytest.fail("The diagnostic must not enter training")

    return NativeBoundary()


@pytest.mark.asyncio
async def test_reference_then_one_frozen_install_and_replay_with_durable_evidence(diagnostic, tmp_path):
    result = await qualification.run_bucket_qualification(diagnostic, str(tmp_path), _geometry())
    assert (
        result["updates"] == 0 and result["initial_syncs"] == result["packed_installs"] == result["full_replays"] == 1
    )
    assert diagnostic.calls == ["init", "reference-sync", "drain", "pause", "bucket", "resume"]
    raw = json.loads(Path(result["bucket_native_durable"]["uri"]).read_text())
    assert raw["policy"][0]["phases"]["replay"]["receivers"][0]["mismatches"] == 0
    assert json.loads(Path(result["reference_durable"]["uri"]).read_text())["updates"] == 0
    assert result["policy"][0]["source_byte_coverage"] == 1.0
    with pytest.raises(ValueError, match="previous attempt"):
        qualification.mark_measurement_once(str(tmp_path))


@pytest.mark.asyncio
@pytest.mark.parametrize("missing", ["headroom", "backend", "source"])
async def test_live_precursor_failure_never_allocates_or_marks_bucket_measurement(diagnostic, tmp_path, missing):
    policy, receivers = diagnostic.readbacks
    if missing == "headroom":
        receivers[0][0]["free_bytes"] = 79
    elif missing == "backend":
        receivers[0][0]["layers"][0]["backend"] = "FLASHINFER"
    else:
        policy[0]["expert_samples"] = []
    with pytest.raises(ValueError, match="headroom|backend|precursor"):
        await qualification.run_bucket_qualification(diagnostic, str(tmp_path), _geometry())
    assert diagnostic.calls == ["init", "reference-sync", "drain"]
    assert not (tmp_path / "bucket-measurement-started.json").exists()


def test_marker_lookup_error_never_authorizes_measurement(tmp_path, monkeypatch):
    def failed_lookup(uri):
        raise OSError("native object lookup failed")

    monkeypatch.setattr(qualification, "exists", failed_lookup)
    with pytest.raises(OSError, match="lookup failed"):
        qualification.mark_measurement_once(str(tmp_path))
    assert not list(tmp_path.iterdir())
