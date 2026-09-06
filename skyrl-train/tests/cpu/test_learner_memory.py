"""CUDA boundary fakes with the actual Finelog event serializer."""

import asyncio
from dataclasses import dataclass
from types import SimpleNamespace

import pytest
from rigging.telemetry import serialization

from skyrl_train import learner_memory
from skyrl_train import telemetry as training_telemetry


@dataclass
class FakeCuda:
    allocated: int = 100
    reserved: int = 160
    peak_allocated: int = 900
    peak_reserved: int = 960
    failure: str | None = None
    backend: str = "native"

    def current_device(self):
        if self.failure == "identity":
            raise RuntimeError("CUDA context unavailable")
        return 2

    def get_allocator_backend(self):
        return self.backend

    def get_device_properties(self, device):
        assert device == 2
        return SimpleNamespace(uuid="GPU-physical-two")

    def reset_peak_memory_stats(self, device):
        assert device == 2
        if self.failure == "reset":
            raise RuntimeError("CUDA peak reset unavailable")
        self.peak_allocated = self.allocated
        self.peak_reserved = self.reserved

    def use_memory(self, allocated, reserved):
        self.allocated, self.reserved = allocated, reserved
        self.peak_allocated = max(self.peak_allocated, allocated)
        self.peak_reserved = max(self.peak_reserved, reserved)

    def memory_stats(self, device):
        assert device == 2
        if self.failure == "sample":
            raise RuntimeError("CUDA memory sample unavailable")
        return {
            "allocated_bytes.all.current": self.allocated,
            "reserved_bytes.all.current": self.reserved,
            "allocated_bytes.all.peak": self.peak_allocated,
            "reserved_bytes.all.peak": self.peak_reserved,
        }

    def mem_get_info(self, device):
        assert device == 2
        # Other processes and CUDA allocations account for device usage too.
        return 2000, 4096


@pytest.fixture
def observations(monkeypatch):
    cuda = FakeCuda()
    events = []

    def event(name, body, *, attributes):
        serialization.validate_attributes(attributes)
        fields = serialization.event_fields(body, budget=16_384)
        # Force the wire encoder too, retaining numeric (not string) byte values.
        serialization.json_bytes({"body": fields, "attributes": attributes})
        events.append({"name": name, "body": fields, "attributes": dict(attributes)})

    monkeypatch.setattr(learner_memory.torch, "cuda", cuda)
    monkeypatch.setattr(training_telemetry.telemetry, "event", event)
    return cuda, events


def test_phase_peaks_reset_between_intervals_and_preserve_current_device_semantics(observations):
    cuda, events = observations
    memory = learner_memory.LearnerMemory(enabled=True, rank=11)
    memory.snapshot("model_ready")
    with memory.span("learner_logprob_forward", step=7, step_kind="target_update"):
        cuda.use_memory(500, 600)
        cuda.use_memory(120, 200)
    with memory.span("weight_publication", step=7, step_kind="completed_update"):
        cuda.use_memory(300, 400)
        cuda.use_memory(100, 160)

    snapshot, forward_enter, forward_exit, publish_enter, publish_exit = events
    assert snapshot["body"] == {
        "allocated_bytes": 100,
        "reserved_bytes": 160,
        "device_free_bytes": 2000,
        "device_total_bytes": 4096,
    }
    assert snapshot["attributes"]["step_kind"] == "unknown"
    assert "step" not in snapshot["attributes"]
    assert forward_enter["attributes"]["outcome"] == "started"
    assert forward_exit["body"] == {
        "allocated_bytes": 120,
        "reserved_bytes": 200,
        "device_free_bytes": 2000,
        "device_total_bytes": 4096,
        "peak_allocated_bytes": 500,
        "peak_reserved_bytes": 600,
    }
    assert publish_enter["body"]["allocated_bytes"] == 120
    assert publish_exit["body"]["peak_allocated_bytes"] == 300
    assert publish_exit["body"]["peak_reserved_bytes"] == 400
    assert publish_exit["attributes"] == {
        "backend": "megatron",
        "role": "worker",
        "worker_role": "policy",
        "rank": "11",
        "cuda_device": "2",
        "gpu_uuid": "GPU-physical-two",
        "allocator_backend": "native",
        "phase": "weight_publication",
        "boundary": "exit",
        "outcome": "success",
        "step_kind": "completed_update",
        "step": "7",
    }
    assert all(row["name"] == "cuda_memory_observation" for row in events)


@pytest.mark.parametrize("error", [RuntimeError("training failed"), asyncio.CancelledError()])
def test_failed_training_preserves_exception_and_releases_peak_scope(observations, error):
    cuda, events = observations
    memory = learner_memory.LearnerMemory(enabled=True, rank=3)
    with pytest.raises(type(error)) as caught:
        with memory.span("ppo_forward_backward_update", step=4, step_kind="target_update"):
            cuda.use_memory(600, 700)
            raise error
    assert caught.value is error
    assert events[-1]["attributes"]["outcome"] == "failure"
    assert events[-1]["body"]["peak_allocated_bytes"] == 600
    with memory.span("weight_publication", step=None, step_kind="completed_update"):
        cuda.use_memory(700, 800)
    assert events[-1]["attributes"]["outcome"] == "success"
    assert events[-1]["attributes"]["step_kind"] == "unknown"
    assert events[-1]["body"]["peak_allocated_bytes"] == 700


def test_overlapping_collectors_and_snapshots_do_not_destroy_enclosing_peak(observations):
    cuda, events = observations
    outer = learner_memory.LearnerMemory(enabled=True, rank=3)
    inner = learner_memory.LearnerMemory(enabled=True, rank=3)
    with outer.span("ppo_forward_backward_update", step=4, step_kind="target_update"):
        cuda.use_memory(800, 850)
        cuda.use_memory(100, 160)
        inner.snapshot("model_ready")
        with inner.span("learner_logprob_forward", step=4, step_kind="target_update"):
            cuda.use_memory(200, 300)
    assert events[-1]["body"]["peak_allocated_bytes"] == 800
    assert [row["attributes"]["boundary"] for row in events] == ["enter", "snapshot", "exit"]
    with inner.span("weight_publication", step=4, step_kind="completed_update"):
        cuda.use_memory(300, 350)
    assert events[-1]["body"]["peak_allocated_bytes"] == 300


@pytest.mark.parametrize(
    ("failure", "at_exit"),
    [("identity", False), ("reset", False), ("sample", False), ("export", False), ("sample", True), ("export", True)],
)
def test_optional_observation_failure_does_not_replace_training_exception(observations, monkeypatch, failure, at_exit):
    cuda, events = observations
    memory = learner_memory.LearnerMemory(enabled=True, rank=3)
    training_error = RuntimeError("optimizer failed")
    event_emitter = training_telemetry.telemetry.event

    def fail_observation():
        if failure == "export":

            def failed_export(*args, **kwargs):
                raise RuntimeError("telemetry export failed")

            monkeypatch.setattr(training_telemetry.telemetry, "event", failed_export)
        else:
            cuda.failure = failure

    if not at_exit:
        fail_observation()
    with pytest.raises(RuntimeError) as caught:
        with memory.span("ppo_forward_backward_update", step=4, step_kind="target_update"):
            if at_exit:
                fail_observation()
            raise training_error
    assert caught.value is training_error
    # A separate observer can claim the device even after setup or exit fails.
    cuda.failure = None
    monkeypatch.setattr(training_telemetry.telemetry, "event", event_emitter)
    with learner_memory.LearnerMemory(enabled=True, rank=3).span(
        "weight_publication", step=4, step_kind="completed_update"
    ):
        cuda.use_memory(400, 500)
    assert events[-1]["body"]["peak_allocated_bytes"] == 400


@pytest.mark.parametrize("at_exit", [False, True])
def test_observation_failure_keeps_successful_training_and_disables_further_collection(observations, at_exit):
    cuda, events = observations
    memory = learner_memory.LearnerMemory(enabled=True, rank=3)
    trained = []
    if not at_exit:
        cuda.failure = "sample"
    with memory.span("ppo_forward_backward_update", step=4, step_kind="target_update"):
        trained.append(True)
        cuda.failure = "sample"
    cuda.failure = None
    memory.snapshot("model_ready")
    assert trained == [True]
    assert len(events) == (1 if at_exit else 0)


def test_disabled_observations_do_not_access_cuda_or_emit(observations):
    cuda, events = observations
    cuda.failure = "identity"
    memory = learner_memory.LearnerMemory(enabled=False, rank=3)
    body = []
    memory.snapshot("model_ready")
    with memory.span("ppo_forward_backward_update", step=4, step_kind="target_update"):
        body.append("trained")
    assert body == ["trained"]
    assert events == []


def test_unsupported_allocator_omits_misleading_peak_statistics(observations):
    cuda, events = observations
    cuda.backend = "cudaMallocAsync"
    memory = learner_memory.LearnerMemory(enabled=True, rank=3)
    trained = []
    with memory.span("ppo_forward_backward_update", step=4, step_kind="target_update"):
        trained.append(True)
    assert trained == [True]
    assert events == []
