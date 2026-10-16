import dataclasses
import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from omegaconf.errors import ConfigKeyError

from skyrl_train.benchmarks.learner.batch_artifact import LengthProfile
from skyrl_train.benchmarks.learner.driver import (
    ALL_PROFILES,
    SCHEMA_VERSION,
    AdamWSettings,
    BenchmarkResult,
    BenchmarkSpec,
    EnsembleEntry,
    Hardware,
    ModelDescription,
    MuonHSettings,
    ParallelLayout,
    ScheduledUpdate,
    WorkerObservation,
    build_policy_cfg,
    compose_base_config,
    parse_spec,
    select_entries,
    replay_routes,
    result_json,
    rows_per_update,
    schedule_updates,
    source_num_layers,
    summarize,
    update_record,
    worker_observation,
)
from skyrl_train.benchmarks.learner.truncate_checkpoint import TRUNCATION_PROVENANCE_FILENAME, TruncationProvenance
from skyrl_train.benchmarks.learner.flops import Accelerator, GrugFlopsModel, RecomputePolicy
from skyrl_train.dataset.routed_expert_batch import RoutedExpertRows
from skyrl_train.models.grug_moe import GrugMoeConfig
from skyrl_train.timing_observability import PPO_TRAIN_PHASE_SECONDS_KEY
from skyrl_train.training_batch import TrainingInputBatch

SHORT, LONG = LengthProfile.SHORT, LengthProfile.LONG_REASONING
LAYOUT = ParallelLayout(
    tensor_parallel=1,
    pipeline_parallel=2,
    expert_parallel=4,
    context_parallel=1,
    expert_tensor_parallel=1,
    num_nodes=1,
    gpus_per_node=8,
)


@pytest.fixture(scope="module")
def base_cfg():
    return compose_base_config()


def _spec(**overrides) -> BenchmarkSpec:
    fields = dict(
        run_name="bench",
        harness_revision="abc123",
        model_path="/models/hero-12l",
        layout=LAYOUT,
        optimizer=MuonHSettings(lr=1e-5, adam_lr=2e-6, offload_momentum=True),
        recompute=RecomputePolicy.FULL_LAYERS,
        micro_batch_size_per_gpu=2,
        sample_packing=True,
        flash_attn=False,
        ensemble_root="s3://bench/ensembles",
        ensemble_version="v0",
        profiles=(SHORT, LONG),
        routes_checkpoint="hero-step126000",
        warmup_updates=2,
        measured_updates=5,
        score_old_logprobs=False,
        profile_update=False,
        profiler_dir=None,
        output_path="/tmp/result.json",
        wandb_project=None,
    )
    return BenchmarkSpec(**{**fields, **overrides})


def _entries():
    return [
        EnsembleEntry("root/v0/short/batch_1", SHORT, 1, 64),
        EnsembleEntry("root/v0/short/batch_0", SHORT, 0, 64),
        EnsembleEntry("root/v0/long_reasoning/batch_0", LONG, 0, 60),
    ]


def test_schedule_interleaves_profiles_and_cycles_each_profile():
    schedule = schedule_updates(_entries(), (SHORT, LONG), warmup_updates=2, measured_updates=4, profile_update=True)
    assert [(u.entry.profile, u.entry.batch_index) for u in schedule] == [
        (SHORT, 0),
        (LONG, 0),
        (SHORT, 1),
        (LONG, 0),
        (SHORT, 0),
        (LONG, 0),
    ]
    assert [u.warmup for u in schedule] == [True, True, False, False, False, False]
    assert [u.profiled for u in schedule] == [False, False, True, False, False, False]


def test_select_entries_filters_by_profile_and_sequence_count():
    entries = [*_entries(), EnsembleEntry("root/v0/short/batch_2", SHORT, 2, 128)]
    assert [e.path for e in select_entries(entries, (SHORT,), 128)] == ["root/v0/short/batch_2"]
    assert len(select_entries(entries, (SHORT, LONG), None)) == 4
    assert select_entries(entries, (LONG,), 128) == []


def test_schedule_rejects_a_profile_without_batches():
    with pytest.raises(ValueError, match="no batches for profiles"):
        schedule_updates(
            _entries(), (LengthProfile.AGENTIC,), warmup_updates=0, measured_updates=1, profile_update=False
        )


def test_rows_per_update_pads_to_data_parallel_micro_batches():
    # DP = 8 / (TP1 * PP2 * CP1) = 4; with micro batch 2 rows pad to multiples of 8.
    assert rows_per_update(_entries(), LAYOUT, 2) == 64
    with pytest.raises(ValueError, match="pad to different sizes"):
        rows_per_update([*_entries(), EnsembleEntry("x", LONG, 1, 70)], LAYOUT, 2)
    # Padding clones leading rows, so 3 rows cannot reach the 8-row multiple.
    with pytest.raises(ValueError, match="cannot be padded"):
        rows_per_update([EnsembleEntry("x", LONG, 0, 3)], LAYOUT, 2)


def test_policy_cfg_trains_whole_batch_with_router_replay(base_cfg):
    cfg = build_policy_cfg(_spec(), 64, base_cfg)
    trainer = cfg.trainer
    assert (trainer.train_batch_size, trainer.policy_mini_batch_size, cfg.generator.n_samples_per_prompt) == (64, 64, 1)
    assert trainer.policy_train_spans and trainer.use_sample_packing and trainer.gradient_checkpointing
    assert not trainer.algorithm.use_kl_loss
    megatron = trainer.policy.megatron_config
    assert megatron.moe_router_replay
    assert (megatron.pipeline_model_parallel_size, megatron.expert_model_parallel_size) == (2, 4)
    assert megatron.transformer_config_kwargs.recompute_granularity == "full"
    optimizer = trainer.policy.optimizer_config
    assert optimizer.optimizer == "MuonH"
    assert optimizer.optimizer_kwargs == {"adam_lr": 2e-6, "offload_momentum": True}
    assert optimizer.max_grad_norm == 0.0 and optimizer.weight_decay == 0.0
    assert not megatron.torch_profiler_config.enable


def test_policy_cfg_profiles_the_first_measured_update(base_cfg):
    cfg = build_policy_cfg(_spec(profile_update=True, profiler_dir="/tmp/prof"), 64, base_cfg)
    profiler = cfg.trainer.policy.megatron_config.torch_profiler_config
    assert (profiler.enable, list(profiler.ranks), profiler.update_index) == (True, [0], 2)


def test_policy_cfg_without_recompute_and_with_offloaded_adamw(base_cfg):
    cfg = build_policy_cfg(
        _spec(recompute=RecomputePolicy.NONE, optimizer=AdamWSettings(lr=1e-6, offload_fraction=0.5)), 64, base_cfg
    )
    assert not cfg.trainer.gradient_checkpointing
    kwargs = cfg.trainer.policy.megatron_config.optimizer_config_kwargs
    assert kwargs.optimizer_cpu_offload and kwargs.optimizer_offload_fraction == 0.5
    assert kwargs.use_precision_aware_optimizer and not kwargs.store_param_remainders


def test_policy_cfg_applies_overrides_and_rejects_unknown_keys(base_cfg):
    cfg = build_policy_cfg(
        _spec(megatron_overrides=("trainer.policy.megatron_config.empty_cuda_cache=false",)), 64, base_cfg
    )
    assert cfg.trainer.policy.megatron_config.empty_cuda_cache is False
    with pytest.raises(ConfigKeyError):
        build_policy_cfg(_spec(megatron_overrides=("trainer.policy.megatron_config.no_such_knob=1",)), 64, base_cfg)


def _tiny_flops_model() -> GrugFlopsModel:
    config = GrugMoeConfig(
        hidden_size=64,
        intermediate_size=32,
        shared_expert_intermediate_size=32,
        num_local_experts=8,
        num_experts_per_tok=2,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        sliding_window=8,
        vocab_size=128,
    )
    return GrugFlopsModel.from_config(config)


def _batch(lengths: list[int], width: int = 12, response_width: int = 4) -> TrainingInputBatch:
    attention_mask = torch.tensor([[0] * (width - n) + [1] * n for n in lengths])
    response_mask = attention_mask[:, -response_width:].clone()
    return TrainingInputBatch({"attention_mask": attention_mask, "response_mask": response_mask})


def _observation(ppo_train_seconds: tuple[float, ...], scheduler_seconds: tuple[float, ...]) -> WorkerObservation:
    return WorkerObservation(
        train_status={"policy_loss": 0.5},
        phase_seconds_by_rank=tuple(
            {"ppo_train": total, "megatron_forward_backward_scheduler": scheduler}
            for total, scheduler in zip(ppo_train_seconds, scheduler_seconds, strict=True)
        ),
        peak_allocated_by_rank=(10, 30),
        peak_reserved_by_rank=(40, 20),
    )


def _record(update: ScheduledUpdate, wall_seconds: float):
    return update_record(
        update,
        _batch([12, 5, 3]),
        padded_rows=4,
        wall_seconds=wall_seconds,
        scoring_seconds=None,
        observation=_observation((wall_seconds - 0.5, wall_seconds - 0.2), (1.0, 2.0)),
        flops_model=_tiny_flops_model(),
        recompute=RecomputePolicy.FULL_LAYERS,
        num_gpus=2,
        accelerator=Accelerator.H100,
    )


def test_update_record_counts_unpadded_tokens_and_takes_rank_maxima():
    update = ScheduledUpdate(3, EnsembleEntry("p", SHORT, 0, 3), warmup=False, profiled=False)
    record = _record(update, wall_seconds=2.0)
    flops = _tiny_flops_model().batch_flops([12, 5, 3], RecomputePolicy.FULL_LAYERS)
    assert (record.tokens, record.response_tokens, record.max_sequence_tokens) == (20, 4 + 4 + 3, 12)
    assert record.tokens_per_second_per_gpu == pytest.approx(20 / 2.0 / 2)
    assert record.mfu == pytest.approx(flops.training_model / (2.0 * 2 * 989e12))
    assert record.hfu > record.mfu
    assert record.worker_seconds == pytest.approx(1.8)
    assert record.phase_seconds["megatron_forward_backward_scheduler"] == 2.0
    assert (record.peak_allocated_bytes, record.peak_reserved_bytes) == (30, 40)


def test_summaries_exclude_warmup_and_profiled_updates():
    walls = {0: 50.0, 1: 40.0, 2: 30.0, 3: 1.0, 4: 2.0, 5: 3.0, 6: 4.0}
    records = []
    for index, wall in walls.items():
        profile = SHORT if index % 2 == 0 else LONG
        update = ScheduledUpdate(
            index, EnsembleEntry(f"p{index}", profile, 0, 3), warmup=index < 2, profiled=index == 2
        )
        records.append(_record(update, wall))
    summaries = {summary.profile: summary for summary in summarize(records)}
    assert set(summaries) == {"short", "long_reasoning", ALL_PROFILES}
    assert summaries["short"].updates == 2  # indices 4 and 6
    assert summaries["short"].wall_seconds.median == pytest.approx(3.0)
    assert summaries[ALL_PROFILES].updates == 4
    assert summaries[ALL_PROFILES].wall_seconds.p10 == pytest.approx(1.3)
    assert summaries[ALL_PROFILES].wall_seconds.p90 == pytest.approx(3.7)


def test_result_record_serializes_to_json():
    update = ScheduledUpdate(0, EnsembleEntry("p", SHORT, 0, 3), warmup=False, profiled=False)
    record = _record(update, wall_seconds=2.0)
    result = BenchmarkResult(
        schema_version=SCHEMA_VERSION,
        created_at="2026-10-15T00:00:00+00:00",
        spec=_spec(),
        optimizer_name="MuonH",
        model=ModelDescription("/m", "grug_moe", 4, 64, 8, 2, None, 1.0),
        hardware=Hardware("NVIDIA H100 80GB HBM3", Accelerator.H100, 989e12, 2),
        rows_per_update=4,
        updates=(record,),
        summaries=summarize([record]),
    )
    payload = json.loads(result_json(result))
    assert payload["schema_version"] == SCHEMA_VERSION
    assert payload["optimizer_name"] == "MuonH"
    assert payload["spec"]["optimizer"] == {
        "lr": 1e-5,
        "adam_lr": 2e-6,
        "offload_momentum": True,
        "adam_betas": [0.9, 0.95],
    }
    assert payload["spec"]["profiles"] == ["short", "long_reasoning"]
    assert payload["updates"][0]["profile"] == "short"
    assert payload["summaries"][-1]["profile"] == ALL_PROFILES


def test_worker_observation_requires_phase_seconds():
    outputs = [SimpleNamespace(metadata={"train_status": {"policy_loss": 1.0}})]
    with pytest.raises(ValueError, match="policy_train_spans"):
        worker_observation(outputs, [{"peak_allocated": 1, "peak_reserved": 2}])
    outputs[0].metadata[PPO_TRAIN_PHASE_SECONDS_KEY] = {"ppo_train": 3.0}
    observation = worker_observation(outputs, [{"peak_allocated": 1, "peak_reserved": 2}])
    assert observation.phase_seconds_by_rank == ({"ppo_train": 3.0},)


REQUIRED_ARGS = [
    "--run-name=r",
    "--harness-revision=abc",
    "--model-path=/m",
    "--tp=1",
    "--pp=1",
    "--ep=8",
    "--cp=1",
    "--etp=1",
    "--num-nodes=1",
    "--gpus-per-node=8",
    "--lr=1e-5",
    "--recompute=full_layers",
    "--micro-batch-size-per-gpu=1",
    "--sample-packing",
    "--flash-attn",
    "--ensemble-root=/e",
    "--ensemble-version=v0",
    "--profiles",
    "short",
    "agentic",
    "--routes-checkpoint=hero",
    "--warmup-updates=2",
    "--measured-updates=6",
    "--output=/tmp/out.json",
]


def test_cli_builds_a_muonh_spec():
    spec = parse_spec([*REQUIRED_ARGS, "--optimizer=MuonH", "--adam-lr=1e-6", "--no-offload-momentum"])
    assert spec.optimizer == MuonHSettings(lr=1e-5, adam_lr=1e-6, offload_momentum=False)
    assert spec.profiles == (SHORT, LengthProfile.AGENTIC)
    assert spec.layout.data_parallel == 8
    assert not spec.profile_update


def test_cli_rejects_offload_fraction_for_muonh():
    with pytest.raises(SystemExit):
        parse_spec(
            [*REQUIRED_ARGS, "--optimizer=MuonH", "--adam-lr=1e-6", "--offload-momentum", "--offload-fraction=0.5"]
        )


def _routes(layers: int) -> RoutedExpertRows:
    rows = tuple(np.arange(tokens * layers * 2, dtype=np.uint8).reshape(tokens, layers, 2) % 64 for tokens in (3, 0, 2))
    return RoutedExpertRows(rows, response_len=4, num_experts=64)


def test_replay_routes_slice_full_model_routes_to_a_truncated_model():
    full = _routes(8)
    assert replay_routes(full, 8, 8) is full
    sliced = replay_routes(full, 4, 8)
    for full_row, row in zip(full.rows, sliced.rows, strict=True):
        np.testing.assert_array_equal(row, full_row[:, :4])
    # Routes recorded on the truncated model itself replay unchanged.
    assert replay_routes(_routes(4), 4, 8).rows[0].shape[1] == 4


@pytest.mark.parametrize("route_layers", [2, 6, 12])
def test_replay_routes_reject_layer_counts_of_neither_model(route_layers):
    with pytest.raises(ValueError, match="layers"):
        replay_routes(_routes(route_layers), 4, 8)


def test_source_num_layers_reads_truncation_provenance(tmp_path):
    assert source_num_layers(str(tmp_path), 48) == 48
    provenance = TruncationProvenance(
        source_path="s3://models/hero",
        source_config_sha256="abc",
        source_manifest_identity=None,
        source_num_layers=48,
        num_layers=8,
    )
    (tmp_path / TRUNCATION_PROVENANCE_FILENAME).write_text(json.dumps(dataclasses.asdict(provenance)))
    assert source_num_layers(str(tmp_path), 8) == 48
    with pytest.raises(ValueError, match="provenance"):
        source_num_layers(str(tmp_path), 12)
