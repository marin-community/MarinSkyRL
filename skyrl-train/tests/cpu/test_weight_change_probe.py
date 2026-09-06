"""Bitwise wire surveys checked against independent full-tensor populations."""

import random
import asyncio
import ast
import contextlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from omegaconf import OmegaConf
from rigging.telemetry.serialization import EventBody, event_fields, json_bytes
from skyrl_train.weight_change_probe import (
    ProbeStatus,
    WeightChangeProbe,
    WirePublicationObserver,
    validate_weight_change_probe_config,
)
from skyrl_train.trainer import RayPPOTrainer
from skyrl_train.fully_async_trainer import FullyAsyncRayPPOTrainer
from skyrl_train.weight_sync import WeightChunk
from skyrl_train.weight_sync.weight_extractor import weight_sync_dtype
from skyrl_train.utils.utils import str_to_torch_dtype


def test_startup_ack_roundtrips_the_actual_event_body_without_none(monkeypatch):
    bodies = []

    def event(name, body, **kwargs):
        bodies.append((name, json.loads(json_bytes(event_fields(body, budget=16_384)))))

    monkeypatch.setattr("skyrl_train.telemetry.telemetry.event", event)
    trainer = RayPPOTrainer.__new__(RayPPOTrainer)
    trainer._weight_change_probe_committed(
        {"publication_id": "resume-37", "base_update": None, "target_update": 37}, 0.125
    )
    assert bodies == [
        (
            "weight_change_probe_ack",
            {
                "publication_id": "resume-37",
                "target_update": 37,
                "acknowledgment_seconds": 0.125,
            },
        )
    ]


def _publication_classes():
    """Execute shipped method bodies without importing optional Megatron CUDA packages.

    All tensor conversion, observation, and publication control flow stays real;
    the test supplies only the external CUDA/collective/receiver interfaces.
    """
    source = Path(__file__).parents[2] / "skyrl_train/workers/megatron/megatron_worker.py"
    tree = ast.parse(source.read_text())
    methods = {
        "MegatronWeightExtractor": {"_wire_tensor", "extract_weights"},
        "MegatronPolicyWorkerBase": {
            "broadcast_to_inference_engines",
            "_broadcast_to_inference_engines",
            "finish_weight_change_probe",
        },
    }
    classes = []
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name in methods:
            node.bases = []
            node.keywords = []
            node.decorator_list = []
            node.body = [
                method
                for method in node.body
                if isinstance(method, (ast.FunctionDef, ast.AsyncFunctionDef)) and method.name in methods[node.name]
            ]
            classes.append(node)
    namespace = {
        "torch": torch,
        "asyncio": asyncio,
        "WirePublicationObserver": WirePublicationObserver,
        "WeightChunk": WeightChunk,
        "weight_sync_dtype": weight_sync_dtype,
        "str_to_torch_dtype": str_to_torch_dtype,
    }
    exec(compile(ast.Module(body=classes, type_ignores=[]), str(source), "exec"), namespace)
    return namespace["MegatronWeightExtractor"], namespace["MegatronPolicyWorkerBase"]


@pytest.mark.parametrize("rank", [0, 1])
@pytest.mark.parametrize("receiver_failure", [False, True])
def test_worker_probes_every_converted_wire_tensor_only_on_sender_and_commits_after_ack(
    monkeypatch, rank, receiver_failure
):
    extractor_type, worker_type = _publication_classes()
    wire = []
    events = []
    monkeypatch.setattr(torch.cuda, "current_device", lambda: "cpu")
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: events.append(("existing_cuda_sync", {})))
    monkeypatch.setattr(torch.distributed, "get_rank", lambda: rank)
    monkeypatch.setattr(torch.distributed, "barrier", lambda: None)
    monkeypatch.setattr(torch.distributed, "broadcast", lambda tensor, *args, **kwargs: wire.append(tensor.clone()))

    def event(name, body, **kwargs):
        events.append((name, json.loads(json_bytes(event_fields(body, budget=16_384)))))

    monkeypatch.setattr("skyrl_train.telemetry.telemetry.event", event)
    tensors = {
        "model.layers.0.self_attn.q_proj.weight": torch.tensor([1.00001, 2.0], dtype=torch.float32),
        "model.layers.0.mlp.router.bias": torch.tensor([1.00001], dtype=torch.float32),
    }
    extractor = extractor_type()
    extractor.enable_bucketing = False
    extractor.model_type = "grug_moe"
    extractor.actor_module = object()
    extractor.bridge = SimpleNamespace(export_hf_weights=lambda *args, **kwargs: iter(tensors.items()))
    worker = worker_type()
    worker.cfg = _probe_config()
    worker.cfg.trainer.seed = 17
    worker.cfg.generator.enable_prefix_caching = False
    worker.cfg.generator.model_dtype = "bfloat16"
    worker.use_cuda_ipc = False
    worker._completed_update = None
    worker._model_update_group = object()
    worker._memory = SimpleNamespace(span=lambda *args, **kwargs: contextlib.nullcontext())
    worker.weight_extractor = extractor
    updates = []

    async def begin():
        pass

    async def update(request):
        updates.append(request)

    async def finish():
        if receiver_failure:
            raise RuntimeError("receiver finalize failed")

    client = SimpleNamespace(begin_weight_reload=begin, update_named_weights=update, finish_weight_reload=finish)
    publication = {"publication_id": "resumed-37", "base_update": None, "target_update": 37}
    if receiver_failure and rank == 0:
        with pytest.raises(RuntimeError, match="receiver finalize failed"):
            asyncio.run(worker.broadcast_to_inference_engines(client, publication=publication))
        summary = next(body for name, body in events if name == "weight_change_probe")
        assert summary["status"] == ProbeStatus.FAILED
        assert "estimated_changed_elements" not in summary
        return
    asyncio.run(worker.broadcast_to_inference_engines(client, publication=publication))
    assert not any(name == "weight_change_probe" for name, _ in events)
    worker.finish_weight_change_probe("resumed-37")
    if rank == 0:
        assert [request["names"][0] for request in updates] == list(tensors)
        assert [tensor.dtype for tensor in wire] == [torch.bfloat16, torch.float32]
        torch.testing.assert_close(wire[0], tensors[next(iter(tensors))].to(torch.bfloat16), rtol=0, atol=0)
        torch.testing.assert_close(wire[1], tensors["model.layers.0.mlp.router.bias"], rtol=0, atol=0)
        summary = next(body for name, body in events if name == "weight_change_probe")
        assert summary["status"] == ProbeStatus.BASELINE
        assert summary["tensor_count"] == 2
        assert summary["dense_wire_bytes"] == 8
        assert summary["coverage_complete"] == 1
        assert summary["target_update"] == 37
        assert events.count(("existing_cuda_sync", {})) == 1
        assert events.index(("existing_cuda_sync", {})) < next(
            index for index, (name, _) in enumerate(events) if name == "weight_change_probe"
        )
    else:
        assert not wire and not updates
        assert not hasattr(worker, "_wire_publication_observer")


def _probe_config(enabled=True):
    return OmegaConf.create(
        {
            "trainer": {"weight_change_probe": enabled, "strategy": "megatron", "placement": {"colocate_all": False}},
            "generator": {"fuse_weights": False, "run_engines_locally": True},
        }
    )


@pytest.mark.parametrize(
    "field,value",
    [
        ("trainer.weight_change_probe", "yes"),
        ("trainer.strategy", "fsdp2"),
        ("trainer.placement.colocate_all", True),
        ("generator.fuse_weights", True),
        ("generator.run_engines_locally", False),
    ],
)
def test_probe_rejects_unsupported_config(field, value):
    cfg = _probe_config()
    OmegaConf.update(cfg, field, value)
    with pytest.raises(ValueError, match="weight_change_probe"):
        validate_weight_change_probe_config(cfg)


def test_disabled_probe_needs_no_transport_configuration():
    assert not validate_weight_change_probe_config(OmegaConf.create({"trainer": {}}))


def test_observer_requires_success_ack_after_completed_staging(monkeypatch):
    events = []
    monkeypatch.setattr(
        "skyrl_train.weight_change_probe.record_event", lambda name, fields: events.append((name, fields))
    )
    observer = WirePublicationObserver(seed=31)
    weights = torch.tensor([1.0, 2.0], dtype=torch.bfloat16)
    observer.begin(publication_id="initial", base_update=None, target_update=37)
    observer.capture("weights", weights)
    with pytest.raises(RuntimeError, match="staging"):
        observer.finish(publication_id="initial", success=True)
    observer.staging_complete()
    assert not events
    with pytest.raises(ValueError, match="acknowledgment"):
        observer.finish(publication_id="wrong", success=True)
    observer.finish(publication_id="initial", success=True)
    observer.begin(publication_id="next", base_update=37, target_update=39)
    observer.capture("weights", weights)
    observer.staging_complete()
    result = observer.finish(publication_id="next", success=True)
    assert result.summary["estimated_changed_elements"] == 0
    assert result.summary["base_update"] == 37
    assert result.summary["target_update"] == 39
    assert result.summary["retained_sample_bytes"] == weights.numel() * weights.element_size()
    assert result.summary["sample_cuda_milliseconds"] == 0
    assert torch.equal(weights, torch.tensor([1.0, 2.0], dtype=torch.bfloat16))


def test_observer_resource_failure_is_explicit_and_suppresses_estimates(monkeypatch):
    observer = WirePublicationObserver(seed=0)
    observer.begin(publication_id="failed-observation", base_update=None, target_update=0)

    def out_of_memory(*args, **kwargs):
        raise MemoryError("pinned staging allocation")

    monkeypatch.setattr(torch, "empty_like", out_of_memory)
    observer.capture("weights", torch.ones(4, dtype=torch.bfloat16))
    observer.capture("other_weights", torch.ones(8, dtype=torch.bfloat16))
    observer.staging_complete()
    result = observer.finish(publication_id="failed-observation", success=True)
    assert result.summary["status"] == ProbeStatus.INCOMPLETE
    assert "observer_resource_failure" in result.summary["reason"]
    assert "estimated_changed_elements" not in result.summary
    assert result.summary["tensor_count"] == 2
    assert result.summary["dense_wire_bytes"] == 24


def test_observer_does_not_hide_programming_errors(monkeypatch):
    observer = WirePublicationObserver(seed=0)
    observer.begin(publication_id="bad", base_update=None, target_update=0)

    def malformed(*args):
        raise ValueError("invalid observation")

    monkeypatch.setattr(observer.probe, "capture", malformed)
    with pytest.raises(ValueError, match="invalid observation"):
        observer.capture("weights", torch.ones(4, dtype=torch.bfloat16))


@pytest.mark.parametrize("asynchronous", [False, True])
def test_driver_commits_probe_only_after_all_workers_succeed_and_keeps_resume_identity(monkeypatch, asynchronous):
    trainer_type = FullyAsyncRayPPOTrainer if asynchronous else RayPPOTrainer
    trainer = trainer_type.__new__(trainer_type)
    trainer.cfg = _probe_config()
    trainer.global_step = 37
    trainer.all_timings = {}
    trainer.inference_engine_client = object()
    events = []
    fail = False

    def dispatch(mode, method, *args, **kwargs):
        return [(method, args, kwargs)]

    def collect(refs):
        for method, args, kwargs in refs:
            if method == "broadcast_to_inference_engines":
                events.append(("broadcast", kwargs["publication"].copy()))
                if fail:
                    raise RuntimeError("one policy worker failed")
                events.append(("all_workers_completed", None))
            else:
                assert events[-1][0] == "all_workers_completed"
                events.append(("ack", args[0]))
        return []

    async def async_dispatch(*args, **kwargs):
        return collect(dispatch(*args, **kwargs))

    async def drain():
        pass

    trainer.policy_model = SimpleNamespace(async_run_ray_method=dispatch, async_run_method=async_dispatch)
    trainer._drain_policy_event_loops = drain
    monkeypatch.setattr("skyrl_train.trainer.ray.get", collect)
    monkeypatch.setattr("skyrl_train.trainer.record_event", lambda *args, **kwargs: None)

    def publish_driver():
        if asynchronous:
            return asyncio.run(trainer.async_sync_policy_weights_to_inference_engines())
        return trainer.sync_policy_weights_to_inference_engines()

    publish_driver()
    assert trainer._wire_probe_published_update == 37
    assert events[0][1]["base_update"] is None
    assert events[0][1]["target_update"] == 37
    trainer.global_step = 39
    fail = True
    with pytest.raises(RuntimeError, match="one policy worker failed"):
        publish_driver()
    assert trainer._wire_probe_published_update == 37
    assert events[-1][0] == "broadcast"
    fail = False
    publish_driver()
    assert events[-3][1]["base_update"] == 37
    assert events[-3][1]["target_update"] == 39
    assert trainer._wire_probe_published_update == 39


@pytest.mark.parametrize("asynchronous", [False, True])
def test_disabled_probe_preserves_driver_dispatch_without_ack(monkeypatch, asynchronous):
    trainer_type = FullyAsyncRayPPOTrainer if asynchronous else RayPPOTrainer
    trainer = trainer_type.__new__(trainer_type)
    trainer.cfg = _probe_config(False)
    trainer.global_step = 0
    trainer.all_timings = {}
    trainer.inference_engine_client = object()
    calls = []
    sentinel = [object()]

    def dispatch(*args, **kwargs):
        calls.append((args, kwargs))
        return sentinel

    async def async_dispatch(*args, **kwargs):
        return dispatch(*args, **kwargs)

    async def drain():
        pass

    trainer.policy_model = SimpleNamespace(async_run_ray_method=dispatch, async_run_method=async_dispatch)
    trainer._drain_policy_event_loops = drain
    if asynchronous:
        result = asyncio.run(trainer.async_sync_policy_weights_to_inference_engines())
    else:
        result = trainer.sync_policy_weights_to_inference_engines()
    assert result is sentinel
    assert calls == [(("pass_through", "broadcast_to_inference_engines", trainer.inference_engine_client), {})]
    assert not hasattr(trainer, "_wire_probe_published_update")


def publish(probe, tensors, *, base=None, target=0, success=True):
    probe.begin(publication_id=f"publication-{target}", base_update=base, target_update=target)
    for name, tensor in tensors.items():
        sample = probe.capture(name, tensor)
        if sample is not None:
            probe.add_cpu_sample(sample, sample.bits)
    result = probe.finish(success=success)
    # Exercise the actual event contract, including typed numeric values.
    for fields in (result.summary, *result.groups):
        assert json_bytes(event_fields(EventBody(fields), budget=16_384))
    return result


def test_exact_wire_bits_include_signed_zero_nan_payloads_and_fp32_router_ulp():
    probe = WeightChangeProbe(seed=73)
    before = {
        "model.layers.0.self_attn.q_proj.weight": torch.tensor([0, 32705, 32706, 16256], dtype=torch.int16).view(
            torch.bfloat16
        ),
        "model.layers.0.mlp.router.bias": torch.tensor([1065353216, 1065353216], dtype=torch.int32).view(torch.float32),
    }
    after = {
        "model.layers.0.self_attn.q_proj.weight": torch.tensor([-32768, 32705, 32707, 16256], dtype=torch.int16).view(
            torch.bfloat16
        ),
        "model.layers.0.mlp.router.bias": torch.tensor([1065353217, 1065353216], dtype=torch.int32).view(torch.float32),
    }
    assert publish(probe, before).summary["status"] == ProbeStatus.BASELINE
    result = publish(probe, after, base=0, target=1)
    assert result.summary["estimated_changed_elements"] == 3
    assert result.summary["estimated_changed_value_bytes"] == 8
    assert result.summary["estimated_index32_value_bytes"] == 20
    assert result.summary["estimated_changed_element_fraction"] == 0.5
    assert result.summary["estimated_changed_value_fraction"] == 0.5
    router = next(row for row in result.groups if row["family"] == "router_bias")
    assert router["sampled_elements"] == 2
    assert router["sampled_changed_elements"] == 1


def test_bf16_master_changes_absorbed_before_the_wire_are_unchanged():
    master = torch.tensor([1.0, 2.0], dtype=torch.float32)
    before = master.to(torch.bfloat16)
    master += 1e-6
    after = master.to(torch.bfloat16)
    probe = WeightChangeProbe()
    publish(probe, {"weights": before})
    result = publish(probe, {"weights": after}, base=0, target=1)
    assert result.summary["estimated_changed_elements"] == 0


@pytest.mark.parametrize(("before", "after", "changed"), [(0, -32768, 1), (32705, 32705, 0), (32705, 32706, 1)])
def test_bit_equality_cannot_be_replaced_by_floating_equality(before, after, changed):
    probe = WeightChangeProbe()
    publish(probe, {"weights": torch.tensor([before], dtype=torch.int16).view(torch.bfloat16)})
    result = publish(
        probe, {"weights": torch.tensor([after], dtype=torch.int16).view(torch.bfloat16)}, base=0, target=1
    )
    assert result.summary["estimated_changed_elements"] == changed


def test_tensor_and_wire_width_weighting_match_full_population_not_pooled_samples():
    before = {
        "large": torch.zeros(1024, dtype=torch.bfloat16),
        "small": torch.zeros(4, dtype=torch.bfloat16),
        "model.layers.0.mlp.router.bias": torch.zeros(512, dtype=torch.float32),
    }
    after = {name: tensor.clone() for name, tensor in before.items()}
    after["large"].fill_(1)
    after["model.layers.0.mlp.router.bias"].fill_(1)
    probe = WeightChangeProbe()
    publish(probe, before)
    result = publish(probe, after, base=0, target=1)
    # Each changed tensor changes everywhere: any coordinate sample must produce
    # the independently known full population, despite unequal sampling fractions.
    assert result.summary["sampled_elements"] == 256 + 4 + 512
    assert result.summary["estimated_changed_elements"] == 1024 + 512
    assert result.summary["estimated_changed_value_bytes"] == 1024 * 2 + 512 * 4
    assert result.summary["estimated_index32_value_bytes"] == 1024 * 6 + 512 * 8
    assert result.summary["dense_wire_bytes"] == 1024 * 2 + 4 * 2 + 512 * 4


def test_every_stacked_expert_is_sampled_and_has_correct_population_weight():
    name = "model.layers.25.mlp.experts.down_proj.weight"
    before = torch.zeros((256, 8, 8), dtype=torch.bfloat16)
    after = before.clone()
    after[7].fill_(1)
    after[209].fill_(1)
    probe = WeightChangeProbe(seed=41)
    publish(probe, {name: before})
    result = publish(probe, {name: after}, base=0, target=2)
    assert result.groups[0]["strata"] == 256
    assert result.groups[0]["sampled_elements"] == 512
    assert result.groups[0]["sampled_changed_elements"] == 4
    assert result.summary["estimated_changed_elements"] == 2 * 8 * 8


def test_sampling_is_deterministic_across_order_and_does_not_change_training_rng():
    tensors = {
        "model.layers.0.self_attn.q_proj.weight": torch.arange(4096, dtype=torch.int16).view(torch.bfloat16),
        "model.layers.25.self_attn.q_proj.weight": torch.arange(8192, dtype=torch.int16).view(torch.bfloat16),
    }
    python_rng, torch_rng = random.getstate(), torch.random.get_rng_state().clone()
    captured = []
    for order in (list(tensors), list(reversed(tensors))):
        probe = WeightChangeProbe(seed=19)
        probe.begin(publication_id="initial", base_update=None, target_update=0)
        samples = {name: probe.capture(name, tensors[name]).bits for name in order}
        captured.append(samples)
        probe.finish(success=False)
    assert random.getstate() == python_rng
    assert torch.equal(torch.random.get_rng_state(), torch_rng)
    assert all(torch.equal(captured[0][name], captured[1][name]) for name in tensors)
    assert not torch.equal(captured[0][next(iter(tensors))], torch.arange(256, dtype=torch.int16))


def test_capture_and_committed_samples_do_not_alias_wire_or_caller_storage():
    tensor = torch.zeros(1024, dtype=torch.bfloat16)
    probe = WeightChangeProbe()
    probe.begin(publication_id="initial", base_update=None, target_update=0)
    sample = probe.capture("weights", tensor)
    assert sample.bits.untyped_storage().nbytes() == 256 * 2
    tensor.fill_(1)
    assert torch.count_nonzero(sample.bits) == 0
    probe.add_cpu_sample(sample, sample.bits)
    sample.bits.fill_(7)
    probe.finish(success=True)
    result = publish(probe, {"weights": torch.zeros_like(tensor)}, base=0, target=1)
    assert result.summary["estimated_changed_elements"] == 0


def test_capture_requires_no_device_read_and_cpu_receipt_never_transfers_implicitly():
    probe = WeightChangeProbe()
    probe.begin(publication_id="initial", base_update=None, target_update=0)
    sample = probe.capture("weights", torch.empty(4096, dtype=torch.bfloat16, device="meta"))
    # A meta tensor has no readable storage: a hidden .cpu()/.item() would fail
    # during capture. This is a CPU contract check, not CUDA runtime qualification.
    assert sample.bits.device.type == "meta"
    assert sample.bits.shape == (256,)
    with pytest.raises(ValueError, match="CPU"):
        probe.add_cpu_sample(sample, sample.bits)
    result = probe.finish(success=True)
    assert result.summary["status"] == ProbeStatus.INCOMPLETE
    assert "estimated_changed_elements" not in result.summary


def test_index32_estimate_is_omitted_when_tensor_local_indices_exceed_uint32():
    probe = WeightChangeProbe()
    population = 2**32 + 1
    for update in (0, 1):
        probe.begin(publication_id=f"pub-{update}", base_update=None if update == 0 else 0, target_update=update)
        # Simulate staging a constant huge wire tensor without allocating it.
        sample = probe.capture("weights", torch.empty(population, device="meta", dtype=torch.bfloat16))
        probe.add_cpu_sample(sample, torch.full(sample.bits.shape, update, dtype=torch.int16))
        result = probe.finish(success=True)
    assert result.summary["estimated_changed_elements"] == population
    assert result.summary["estimated_changed_value_bytes"] == population * 2
    assert "estimated_index32_value_bytes" not in result.summary
    assert result.groups[0]["index32_eligible_tensors"] == 0
    assert "estimated_index32_value_bytes" not in result.groups[0]


def test_budget_overflow_is_bounded_and_invalidates_next_installed_baseline():
    probe = WeightChangeProbe(max_sample_bytes=8)
    tensors = {"one": torch.zeros(4, dtype=torch.bfloat16)}
    publish(probe, tensors)
    probe.begin(publication_id="second", base_update=0, target_update=2)
    one = probe.capture("one", tensors["one"])
    probe.add_cpu_sample(one, one.bits)
    assert probe.retained_sample_bytes == 16
    assert probe.capture("two", tensors["one"]) is None
    assert probe.retained_sample_bytes == 16
    result = probe.finish(success=True)
    assert result.summary["status"] == ProbeStatus.INCOMPLETE
    assert result.summary["coverage_complete"] == 0
    assert "sample_budget" in result.summary["reason"]
    assert "estimated_changed_elements" not in result.summary
    assert probe.retained_sample_bytes == 0
    result = publish(probe, tensors, base=2, target=4)
    assert result.summary["status"] == ProbeStatus.BASELINE


def test_failed_publication_and_stale_receipt_do_not_advance_cadence_two_base():
    probe = WeightChangeProbe()
    zero = {"weights": torch.zeros(4, dtype=torch.bfloat16)}
    one = {"weights": torch.ones(4, dtype=torch.bfloat16)}
    publish(probe, zero)
    probe.begin(publication_id="attempt-two", base_update=0, target_update=2)
    stale = probe.capture("weights", one["weights"])
    probe.add_cpu_sample(stale, stale.bits)
    failed = probe.finish(success=False)
    assert failed.summary["status"] == ProbeStatus.FAILED
    assert "estimated_changed_elements" not in failed.summary
    probe.begin(publication_id="attempt-two", base_update=0, target_update=2)
    fresh = probe.capture("weights", zero["weights"])
    with pytest.raises(RuntimeError, match="pending publication"):
        probe.add_cpu_sample(stale, stale.bits)
    probe.add_cpu_sample(fresh, fresh.bits)
    recovered = probe.finish(success=True)
    assert recovered.summary["estimated_changed_elements"] == 0
    at_four = publish(probe, one, base=2, target=4)
    assert at_four.summary["base_update"] == 2
    assert at_four.summary["target_update"] == 4
    assert at_four.summary["estimated_changed_elements"] == 4
    final = publish(probe, one, base=4, target=5)
    assert final.summary["estimated_changed_elements"] == 0


@pytest.mark.parametrize("change", ["name", "shape", "strides", "dtype", "base"])
def test_incompatible_identity_establishes_new_baseline_instead_of_false_estimate(change):
    probe = WeightChangeProbe()
    publish(probe, {"weights": torch.zeros((2, 1, 2), dtype=torch.bfloat16)}, target=37)
    name = "renamed" if change == "name" else "weights"
    shape = (4,) if change == "shape" else (2, 1, 2)
    dtype = torch.float32 if change == "dtype" else torch.bfloat16
    base = 0 if change == "base" else 37
    tensors = {name: torch.ones(shape, dtype=dtype)}
    if change == "strides":
        tensors[name] = torch.empty_strided(shape, (2, 100, 1), dtype=dtype).fill_(1)
    result = publish(probe, tensors, base=base, target=38)
    assert result.summary["status"] == ProbeStatus.BASELINE
    assert "estimated_changed_elements" not in result.summary
    result = publish(probe, tensors, base=38, target=40)
    assert result.summary["estimated_changed_elements"] == 0


@pytest.mark.parametrize("failure", ["noncontiguous", "duplicate", "router_dtype", "missing"])
def test_invalid_or_missing_wire_population_never_produces_global_estimates(failure):
    probe = WeightChangeProbe()
    tensor = torch.zeros((2, 4), dtype=torch.bfloat16)
    publish(probe, {"weights": tensor})
    probe.begin(publication_id="second", base_update=0, target_update=1)
    name = "model.layers.0.mlp.router.bias" if failure == "router_dtype" else "weights"
    sample = probe.capture(name, tensor.t() if failure == "noncontiguous" else tensor)
    if sample is not None and failure != "missing":
        probe.add_cpu_sample(sample, sample.bits)
    if failure == "duplicate":
        assert probe.capture(name, tensor) is None
    result = probe.finish(success=True)
    assert result.summary["status"] == ProbeStatus.INCOMPLETE
    assert result.groups == ()
    assert "estimated_index32_value_bytes" not in result.summary
