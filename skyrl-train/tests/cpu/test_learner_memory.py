"""CUDA boundary fakes with the actual Finelog event serializer."""

import asyncio
from dataclasses import dataclass
from threading import Event, Thread
from types import SimpleNamespace

import pytest
import torch
from omegaconf import OmegaConf
from rigging.telemetry import serialization

from skyrl_train import learner_memory
from skyrl_train import telemetry as training_telemetry
from skyrl_train.training_batch import TrainingInputBatch
from skyrl_train.weight_sync import WeightChunk
from skyrl_train.workers.fsdp.fsdp_worker import FSDPPolicyWorkerBase


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

    def empty_cache(self):
        pass


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
    assert "peak_allocated_bytes" not in events[-1]["body"]
    assert "peak_reserved_bytes" not in events[-1]["body"]
    assert events[-1]["attributes"]["scope_overlap"] == "true"
    assert [row["attributes"]["boundary"] for row in events] == ["enter", "snapshot", "exit"]
    with inner.span("weight_publication", step=4, step_kind="completed_update"):
        cuda.use_memory(300, 350)
    assert events[-1]["body"]["peak_allocated_bytes"] == 300


def test_overlap_holds_reset_ownership_until_the_last_concurrent_scope_exits(observations):
    cuda, events = observations
    outer = learner_memory.LearnerMemory(enabled=True, rank=3, backend="fsdp2")
    inner = learner_memory.LearnerMemory(enabled=True, rank=3, backend="fsdp2")
    entered, release = Event(), Event()

    def concurrent_forward():
        with inner.span("learner_logprob_forward", step=4, step_kind="target_update"):
            cuda.use_memory(900, 950)
            entered.set()
            assert release.wait(timeout=10)

    with outer.span("weight_publication", step=3, step_kind="completed_update"):
        thread = Thread(target=concurrent_forward)
        thread.start()
        assert entered.wait(timeout=10)
    try:
        # The publication owner is gone, but its competing forward still runs.
        # A third phase cannot reset that forward's allocator interval.
        with outer.span("ppo_forward_backward_update", step=4, step_kind="target_update"):
            cuda.use_memory(100, 160)
        assert [row["attributes"]["boundary"] for row in events] == ["enter", "exit"]
        assert events[-1]["attributes"]["scope_overlap"] == "true"
        assert "peak_allocated_bytes" not in events[-1]["body"]
        assert cuda.peak_allocated == 900
    finally:
        release.set()
        thread.join(timeout=10)
    assert not thread.is_alive()
    with outer.span("ppo_forward_backward_update", step=5, step_kind="target_update"):
        cuda.use_memory(300, 350)
    assert events[-1]["body"]["peak_allocated_bytes"] == 300
    assert "scope_overlap" not in events[-1]["attributes"]
    assert all(row["attributes"]["backend"] == "fsdp2" for row in events)


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


@pytest.mark.parametrize("fail_publication", [False, True])
def test_fsdp_ppo_and_publication_measure_extraction_peak_and_keep_update_identity(
    observations, monkeypatch, fail_publication
):
    cuda, events = observations
    sent = []
    publication_error = RuntimeError("receiver rejected update")

    class CpuPolicy(FSDPPolicyWorkerBase):
        # Replace only the CUDA compute boundary; exercise the real PPO loop,
        # accumulation, status aggregation and FSDP publication path below.
        def training_step(self, experience, global_step, local_step, accumulation_steps):
            cuda.use_memory(600, 700)
            if (local_step + 1) % accumulation_steps == 0:
                self.model.model.add_(1)
            cuda.use_memory(150, 200)
            return {"policy_loss": 0.5, "response_length": 1, "policy_lr": 1e-6, "policy_entropy": 0.0}

    worker = object.__new__(CpuPolicy)
    worker.cfg = OmegaConf.create(
        {
            "trainer": {
                "strategy": "fsdp2",
                "micro_train_batch_size_per_gpu": 1,
                "update_epochs_per_batch": 1,
                "algorithm": {},
                "policy": {"grug_query_bias_update_mode": "frozen", "optimizer_config": {"max_grad_norm": 1.0}},
            },
            "generator": {
                "r3_transport": "driver",
                "enable_prefix_caching": False,
                "model_dtype": "float32",
                "fuse_weights": False,
            },
        }
    )
    worker._rank = 0
    worker._is_lora = False
    worker._completed_update = None
    worker._memory = learner_memory.LearnerMemory(enabled=True, rank=0, backend="fsdp2")
    worker.policy_mini_batch_size_per_gpu = 2
    worker.model = SimpleNamespace(model=torch.ones(3))
    worker.strategy = SimpleNamespace(is_rank_0=lambda: False, all_reduce=lambda status: status)
    worker.use_cuda_ipc = False
    worker._model_update_group = object()

    class Extractor:
        def extract_weights(self, dtype):
            # Gathering/conversion peaks before the communicator sees a tensor.
            cuda.use_memory(900, 1000)
            tensor = worker.model.model.to(dtype).clone()
            yield WeightChunk(names=["weight"], dtypes=["float32"], shapes=[list(tensor.shape)], tensors=[tensor])
            cuda.use_memory(150, 200)

    class Receiver:
        async def begin_weight_reload(self):
            pass

        async def update_named_weights(self, request):
            assert request["names"] == ["weight"]
            if fail_publication:
                raise publication_error

        async def finish_weight_reload(self):
            pass

    def broadcast(tensor, src, group):
        assert src == 0 and group is worker._model_update_group
        sent.append(tensor.clone())
        cuda.use_memory(400, 500)

    worker.weight_extractor = Extractor()
    monkeypatch.setattr(torch.distributed, "barrier", lambda: None)
    monkeypatch.setattr(torch.distributed, "get_rank", lambda: 0)
    monkeypatch.setattr(torch.distributed, "broadcast", broadcast)
    batch = TrainingInputBatch(
        {
            "sequences": torch.ones(2, 2, dtype=torch.long),
            "attention_mask": torch.ones(2, 2),
            **{
                key: torch.ones(2, 1)
                for key in (
                    "action_log_probs",
                    "base_action_log_probs",
                    "values",
                    "returns",
                    "advantages",
                    "loss_mask",
                    "response_mask",
                )
            },
            "rollout_logprobs": None,
        }
    )
    batch.metadata = {"global_step": 7, "response_length": 1}
    result = worker.ppo_train(batch)
    assert result.metadata["train_status"]["policy_update_steps"] == 1
    if fail_publication:
        with pytest.raises(RuntimeError) as caught:
            asyncio.run(worker.broadcast_to_inference_engines(Receiver()))
        assert caught.value is publication_error
    else:
        asyncio.run(worker.broadcast_to_inference_engines(Receiver()))
    torch.testing.assert_close(sent[0], torch.full((3,), 2.0), rtol=0, atol=0)
    ppo_enter, ppo_exit, publication_enter, publication_exit = events
    assert ppo_enter["attributes"]["step_kind"] == "target_update"
    assert ppo_exit["body"]["peak_allocated_bytes"] == 600
    assert ppo_exit["body"]["allocated_bytes"] == publication_enter["body"]["allocated_bytes"] == 150
    assert publication_exit["attributes"]["phase"] == "weight_publication"
    assert publication_exit["attributes"]["step_kind"] == "completed_update"
    assert publication_exit["attributes"]["step"] == "7"
    assert publication_exit["attributes"]["outcome"] == ("failure" if fail_publication else "success")
    assert publication_exit["body"]["peak_allocated_bytes"] == 900
    assert all(row["attributes"]["backend"] == "fsdp2" for row in events)
