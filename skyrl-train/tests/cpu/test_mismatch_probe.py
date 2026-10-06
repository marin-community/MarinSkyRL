from types import SimpleNamespace
import json

import numpy as np
import pytest
import ray
import torch
from finestore.rl import mismatch_probe as mismatch
from omegaconf import OmegaConf
from skyrl_train.metric_names import TOKEN_PROVENANCE_RECONSTRUCTED_FRACTION_METRIC
from tokenizers import Tokenizer, models
from transformers import PreTrainedTokenizerFast

from skyrl_train.callbacks.base import TrainerControl, TrainerState
from skyrl_train.config.utils import get_default_config
from skyrl_train.distributed.dispatch import MeshRank
from skyrl_train.inference_engines.vllm_teacher_oracle import tokenizer_vocabulary_fingerprint
from skyrl_train.mismatch_probe.archive import MismatchArchive, read_frozen_probe
from skyrl_train.mismatch_probe.collect import ProbeCollector
from skyrl_train.callbacks.builtin import create_default_callbacks
from skyrl_train.callbacks.base import CallbackHandler
from skyrl_train.models.megatron_router_replay import MegatronRouterReplay
from skyrl_train.training_batch import TrainingOutputBatch


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
        self.actor_infos = [
            SimpleNamespace(rank=MeshRank(dp=dp, sp=0, tp=0, pp=0, world_size=2, dp_size=2, pp_size=1))
            for dp in range(2)
        ]

    def async_run_ray_method(self, dispatch, method, *, data=None):
        if method != "probe_forward":
            raise ValueError(method)
        width = data.metadata["response_length"]
        probabilities = torch.log_softmax(torch.arange(32, dtype=torch.float32), dim=0)
        outputs = []
        for dp in range(2):
            start, end = dp * data.batch_size // 2, (dp + 1) * data.batch_size // 2
            targets = data[start:end].routed_experts_tensor().reshape(-1, 2)
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
        return [ray.put(output) for output in outputs]


@pytest.mark.parametrize("explicit_callbacks", [False, True])
@pytest.mark.asyncio
async def test_reuse_reads_completed_frozen_tokens_and_generation_scores(
    tmp_path, driver_trainer_factory, explicit_callbacks
):
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
    cfg.trainer.algorithm.advantage_estimator = "uniform"
    cfg.trainer.algorithm.off_policy_correction = "none"
    cfg.trainer.train_batch_size = 4
    cfg.generator.n_samples_per_prompt = 1
    model = tmp_path / "model"
    model.mkdir()
    (model / "config.json").write_text(json.dumps({"model_type": "mixtral", "num_local_experts": 8}))
    cfg.trainer.policy.model.path = str(model)
    trainer = driver_trainer_factory(cfg, tokenizer=tokenizer, dp_size=2)
    cfg = trainer.cfg
    trainer.inference_engine_client = _InferenceEndpoint(tokenizer)
    trainer.policy_model = _PolicyEndpoint()
    routes = np.asarray([[[1, 2]], [[0, 0]], [[1, 2]], [[0, 0]]], dtype=np.int32)
    trajectory = {
        "prompt_token_ids": [[3, 4], [5], [6, 7, 8]],
        "response_ids": [[7, 9, 11, 13], [9, 10, 11, 12], [15, 16, 17, 18]],
        "rewards": [1.0, 0.0, 0.5],
        "loss_masks": [[1] * 4 for _ in range(3)],
        "rollout_logprobs": [np.asarray([-0.125, -2.75, -1.5, -0.5], dtype=np.float32) for _ in range(3)],
        "rollout_routed_experts": [routes.copy() for _ in range(3)],
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
    generation = [
        mismatch.ScoreRow(
            probe_hash=collector.probe_hash,
            sample_id=row.sample_id,
            scorer="vllm.generate",
            update=0,
            global_step=7,
            logprobs=values.tolist(),
        )
        for row, values in zip(probes, trajectory["rollout_logprobs"], strict=True)
    ]
    manifest = mismatch.ManifestRow(
        archive=uri,
        status=mismatch.ArchiveStatus.BUILDING,
        probe_hash=collector.probe_hash,
        checkpoint_path=str(cfg.trainer.policy.model.path),
        runtime_commit=None,
        tokenizer_fingerprint=tokenizer_vocabulary_fingerprint(tokenizer),
        starting_global_step=7,
        scored_updates=[0],
        scored_global_steps=[7],
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
    assert source.manifest.starting_global_step == 7
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
    cfg.trainer.mismatch_probe.updates = 2
    trainer.total_training_steps = 9
    trainer.global_step = 7
    trainer.colocate_all = False
    trainer.all_timings = {}
    cfg.trainer.mismatch_probe.enabled = True
    OmegaConf.update(cfg, "trainer.callbacks", [{"type": "logging"}] if explicit_callbacks else None, force_add=True)
    OmegaConf.update(cfg, "trainer.enable_db_registration", False, force_add=True)
    cfg.trainer.eval_interval = -1
    cfg.trainer.ckpt_interval = -1
    cfg.generator.inference_stats_interval = 0
    handler = CallbackHandler(create_default_callbacks(cfg))
    control = TrainerControl(step_limit=8)
    control.reset()
    control = await handler.call_event_async("on_train_begin", TrainerState(7, 0, 9, 9), control, trainer=trainer)
    for step in (8, 9):
        trainer.global_step = step
        await handler.call_event_async("on_step_end", TrainerState(step, 0, 9, 9), control, trainer=trainer)
    await handler.call_event_async("on_train_end", TrainerState(9, 0, 9, 9), control, trainer=trainer)
    chained = read_frozen_probe(cfg.trainer.mismatch_probe.archive_uri)
    assert control.step_limit == 9
    assert chained.manifest.scored_updates == [0, 1, 2]
    assert chained.manifest.scored_global_steps == [7, 8, 9]
    assert chained.manifest.starting_global_step == 7
    assert chained.probes == probes
    assert chained.generations == source.generations

    trainer.global_step = 7
    trainer.loaded_checkpoint_path = str(tmp_path / "another-checkpoint" / "global_step_7")
    cfg.trainer.mismatch_probe.archive_uri = str(tmp_path / "different-checkpoint")
    rejected = CallbackHandler(create_default_callbacks(cfg))
    with pytest.raises(ValueError, match="starting checkpoint, step or runtime differs"):
        await rejected.call_event_async("on_train_begin", TrainerState(7, 0, 9, 9), TrainerControl(), trainer=trainer)

    class ValidationPrompts:
        def __len__(self):
            return 3

        def __getitem__(self, index):
            return {"uid": f"prompt-{index}", "prompt": [], "env_class": None, "env_extras": {}}

        def collate_fn(self, prompts):
            return prompts

    class Runner:
        async def start_eval_session(self, **kwargs):
            pass

        async def stop_eval_session(self):
            pass

        async def run(self, request):
            batch = {key: value[:1] for key, value in trajectory.items()}
            batch["rollout_metrics"] = {TOKEN_PROVENANCE_RECONSTRUCTED_FRACTION_METRIC: 1.0}
            return batch

    trainer.eval_dataset = ValidationPrompts()
    trainer.trajectory_runner = Runner()
    with pytest.raises(ValueError, match="rejects re-tokenized responses"):
        await collector._generate(trainer)
