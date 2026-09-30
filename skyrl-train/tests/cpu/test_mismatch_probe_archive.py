import hashlib
from types import SimpleNamespace

import numpy as np
import pytest
import ray
import torch
from finestore import mismatch_probe as mismatch
from tokenizers import Tokenizer, models
from transformers import PreTrainedTokenizerFast

from skyrl_train.callbacks.base import TrainerControl, TrainerState
from skyrl_train.config.utils import get_default_config
from skyrl_train.distributed.dispatch import MeshRank
from skyrl_train.group_admission import GroupAdvantageInvariant
from skyrl_train.inference_engines.vllm_teacher_oracle import tokenizer_vocabulary_fingerprint
from skyrl_train.mismatch_probe.archive import MismatchArchive, read_frozen_probe
from skyrl_train.mismatch_probe.collect import ProbeCollector
from skyrl_train.mismatch_probe.callback import MismatchProbeCallback
from skyrl_train.models.megatron_router_replay import MegatronRouterReplay
from skyrl_train.training_batch import TrainingOutputBatch
from skyrl_train.trainer import RayPPOTrainer


class _InferenceEndpoint:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    async def reset_prefix_cache(self):
        pass

    async def generate(self, request):
        probabilities = torch.log_softmax(torch.arange(32, dtype=torch.float32), dim=0)
        return {
            "prompt_logprobs": [
                [None] + [{token: float(probabilities[token])} for token in sequence[1:]]
                for sequence in request["prompt_token_ids"]
            ],
            "prefix_cache_hit_tokens": [0 for _ in request["prompt_token_ids"]],
            "student_topk_indices": [
                [[params["logprob_token_ids"][0]]] for params in request["sampling_params_per_prompt"]
            ],
            "behavior_topk_logprobs": [
                [[float(probabilities[params["logprob_token_ids"][0]])]]
                for params in request["sampling_params_per_prompt"]
            ],
        }


class _PolicyEndpoint:
    def __init__(self):
        self.prompt_routes_by_mode = {}
        self.actor_infos = [
            SimpleNamespace(rank=MeshRank(dp=dp, sp=0, tp=0, pp=0, world_size=2, dp_size=2, pp_size=1))
            for dp in range(2)
        ]

    def async_run_ray_method(self, dispatch, method, *, data=None):
        if method == "probe_weights_digest":
            return ["0" * 64]
        if method != "probe_forward":
            raise ValueError(method)
        width = data.metadata["response_length"]
        self.prompt_routes_by_mode[data.metadata["probe_mode"]] = data["rollout_prompt_routed_experts"].clone()
        probabilities = torch.log_softmax(torch.arange(32, dtype=torch.float32), dim=0)
        outputs = []
        for dp in range(2):
            start, end = dp * data.batch_size // 2, (dp + 1) * data.batch_size // 2
            targets = data["rollout_routed_experts"][start:end].reshape(-1, 2)
            positions = torch.stack(
                (
                    data["probe_row_indices"][start:end].repeat_interleave(width),
                    torch.arange(width).repeat(end - start),
                ),
                dim=-1,
            )
            controller = MegatronRouterReplay(local_layer_indices=[0], recompute_enabled=False)
            controller.begin_forward(
                {0: targets},
                targets.ne(0).any(-1),
                data["attention_mask"][start:end, -width:].bool().reshape(-1),
                record_recompute=False,
                probe_positions=positions,
            )
            router_scores = torch.arange(8, dtype=torch.float32).expand(len(targets), -1)
            controller.get_replay_topk(
                0, router_scores, 2, default_compute_topk=lambda scores, k, **kwargs: torch.topk(scores, k)
            )
            controller.end_forward()
            output = TrainingOutputBatch({"output": probabilities[data["sequences"][start:end, -width:]]})
            output.metadata = {"probe_routes": controller.take_probe_observations()}
            outputs.append(output)
        return outputs


@pytest.mark.asyncio
async def test_reuse_reads_completed_frozen_tokens_and_generation_scores(tmp_path, monkeypatch):
    uri = str(tmp_path / "probe")
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=Tokenizer(models.WordLevel({str(i): i for i in range(32)}, unk_token="0")),
        pad_token="0",
        unk_token="0",
    )
    cfg = get_default_config()
    cfg.trainer.mismatch_probe.archive_uri = uri
    cfg.trainer.mismatch_probe.seed = 17
    cfg.trainer.mismatch_probe.prompts.count = 3
    cfg.trainer.mismatch_probe.prompts.samples_per_prompt = 1
    cfg.trainer.policy.megatron_config.moe_router_replay = True
    cfg.trainer.mismatch_probe.extra_trainer_modes = ["router_replay", "router_replay_response"]
    cfg.trainer.algorithm.advantage_estimator = "uniform"
    cfg.trainer.algorithm.use_tis = False
    trainer = RayPPOTrainer.__new__(RayPPOTrainer)
    trainer.cfg = cfg
    trainer.group_advantage_invariant = GroupAdvantageInvariant.no_group_advantage(physical_group_size=1)
    trainer.tokenizer = tokenizer
    trainer.inference_engine_client = _InferenceEndpoint(tokenizer)
    trainer.policy_model = _PolicyEndpoint()
    monkeypatch.setattr(ray, "get", lambda results: results)
    trainer.critic_model = trainer.ref_model = None
    trainer.all_metrics = {}
    trainer._num_experts_cache = 8
    routes = np.asarray([[[1, 2]], [[0, 0]], [[1, 2]], [[0, 0]]], dtype=np.int32)
    trajectory = {
        "prompt_token_ids": [[3, 4], [5], [6, 7, 8]],
        "response_ids": [[7, 9, 11, 13], [9, 10, 11, 12], [15, 16, 17, 18]],
        "rewards": [1.0, 0.0, 0.5],
        "loss_masks": [[1] * 4 for _ in range(3)],
        "rollout_logprobs": [np.asarray([-0.125, -2.75, -1.5, -0.5], dtype=np.float32) for _ in range(3)],
        "rollout_routed_experts": [routes.copy() for _ in range(3)],
        "rollout_prompt_routed_experts": [
            np.asarray([[[3, 4]], [[5, 6]]], dtype=np.uint8),
            np.asarray([[[1, 7]]], dtype=np.uint8),
            np.asarray([[[2, 3]], [[4, 5]], [[6, 7]]], dtype=np.uint8),
        ],
    }
    prompt_ids = [f"prompt-{i}" for i in range(3)]
    sample_ids = [f"sample-{i}" for i in range(3)]
    collector = ProbeCollector(cfg)
    collector._collate_and_freeze(trainer, trajectory, prompt_ids, sample_ids, [91, 92, 93])
    probes = collector.probes
    assert collector.training_input.batch_size == 4
    assert not collector.training_input["loss_mask"][-1].any()
    for row in probes:
        assert row.route_valid_mask == [[True], [False], [True], [False]]
    for row, expected in zip(probes, trajectory["rollout_prompt_routed_experts"], strict=True):
        archived = np.frombuffer(row.prompt_routed_experts, dtype=row.prompt_routed_experts_dtype)
        np.testing.assert_array_equal(archived.reshape(row.prompt_routed_experts_shape), expected)
    weights_hash = "sha256:" + hashlib.sha256(("0" * 64).encode()).hexdigest()
    generation = [
        mismatch.ScoreRow(
            probe_hash=collector.probe_hash,
            sample_id=row.sample_id,
            scorer="vllm.generate",
            update=0,
            weights_hash=weights_hash,
            logprobs=values.tolist(),
        )
        for row, values in zip(probes, trajectory["rollout_logprobs"], strict=True)
    ]
    manifest = mismatch.ManifestRow(
        archive=uri,
        status=mismatch.ArchiveStatus.BUILDING,
        probe_hash=collector.probe_hash,
        starting_weights_hash=weights_hash,
        tokenizer_fingerprint=tokenizer_vocabulary_fingerprint(tokenizer),
        starting_global_step=1,
        scored_updates=[0],
        scored_global_steps=[1],
        architecture="tiny-grug",
        vllm_enforce_eager=False,
        optimizer_steps_per_update=0,
        seed=17,
        bootstrap_seed=18,
        created_at_utc="2026-09-26T00:00:00Z",
        config_json="{}",
        software_json="{}",
        hardware_json="{}",
        batch_layout_json="{}",
        timing_json="{}",
        step_metrics_json="{}",
    )
    archive = MismatchArchive(uri, writer_id="test")
    try:
        archive.write(probes=probes, scores=generation, manifest=manifest)
        with pytest.raises(ValueError, match="not a single complete"):
            read_frozen_probe(uri)
        archive.write(manifest=manifest.model_copy(update={"status": mismatch.ArchiveStatus.COMPLETE}))
    finally:
        archive.close()
    source = read_frozen_probe(uri)
    assert source.manifest.starting_weights_hash == weights_hash
    assert source.probes == probes
    assert list(source.generations.values()) == generation
    cfg.trainer.mismatch_probe.reuse_probe = uri
    reused = ProbeCollector(cfg)
    samples = reused._from_source()
    reused._collate_and_freeze(trainer, samples.trajectory, samples.prompt_ids, samples.sample_ids, samples.seeds)
    assert reused.probes == probes
    for original, restored in zip(trajectory["rollout_logprobs"], reused.generation_scores, strict=True):
        np.testing.assert_array_equal(restored, original)

    cfg.trainer.mismatch_probe.archive_uri = str(tmp_path / "reused")
    cfg.trainer.mismatch_probe.score_after_updates = [0, 1, 2]
    trainer.total_training_steps = 9
    trainer.global_step = 7
    trainer.colocate_all = False
    trainer.all_timings = {}
    callback = MismatchProbeCallback(cfg)
    control = TrainerControl()
    await callback.on_train_begin_async(TrainerState(7, 0, 9, 9), control, trainer=trainer)
    for step in (8, 9):
        trainer.global_step = step
        await callback.on_step_end_async(TrainerState(step, 0, 9, 9), control, trainer=trainer)
    callback.on_train_end(TrainerState(9, 0, 9, 9), control, trainer=trainer)
    chained = read_frozen_probe(cfg.trainer.mismatch_probe.archive_uri)
    assert chained.manifest.scored_updates == [0, 1, 2]
    assert chained.manifest.scored_global_steps == [7, 8, 9]
    assert chained.manifest.starting_global_step == 7
    assert chained.probes == probes
    assert chained.generations == source.generations
    # Prompt positions replay only in full router replay; native and response-only scorings route them natively.
    replayed = trainer.policy_model.prompt_routes_by_mode["router_replay"]
    assert replayed[0, :, 0].tolist() == [[0, 0], [3, 4], [5, 6]]
    assert replayed[1, :, 0].tolist() == [[0, 0], [0, 0], [1, 7]]
    for mode in ("native", "repeat", "router_replay_response"):
        assert not trainer.policy_model.prompt_routes_by_mode[mode].any()
