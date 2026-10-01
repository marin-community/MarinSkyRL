import json
import hashlib
from types import SimpleNamespace

import numpy as np
import pytest
import ray
import torch
from finestore import mismatch_probe as mismatch
from finestore.reader import ReadView
from tokenizers import Tokenizer, models
from transformers import PreTrainedTokenizerFast

from skyrl_train.callbacks.base import TrainerControl, TrainerState
from skyrl_train.config.utils import get_default_config
from skyrl_train.distributed.dispatch import MeshRank
from skyrl_train.group_admission import GroupAdvantageInvariant
from skyrl_train.inference_engines.vllm_teacher_oracle import tokenizer_vocabulary_fingerprint
from skyrl_train.mismatch_probe.archive import MismatchArchive, read_frozen_probe
from skyrl_train.mismatch_probe.collect import TIMING_REPETITIONS, ProbeCollector, token_digest
from skyrl_train.mismatch_probe.callback import MismatchProbeCallback
from skyrl_train.mismatch_probe.modes import KEPT_STACK_3
from skyrl_train.models.grug_inductor_kernels import VENDORED_KERNELS, source_digest
from skyrl_train.models.megatron_router_replay import MegatronRouterReplay
from skyrl_train.training_batch import TrainingOutputBatch
from skyrl_train.trainer import RayPPOTrainer


class _Engine:
    """One vLLM data-parallel rank's client engine; its worker logs each step as ``WorkerWrap`` does."""

    def __init__(self, endpoint, dp_rank):
        self.endpoint = endpoint
        self.dp_rank = dp_rank
        self.steps = None

    async def generate(self, request):
        # Every request runs in a step of its own, padded to a multiple of 8 rows; the other rank runs a dummy step.
        for prefix in request["prompt_token_ids"]:
            rows = -(-len(prefix) // 8) * 8
            if self.steps is not None:
                self.steps.append(
                    {
                        "requests": [
                            {
                                "id": f"r{len(self.steps)}",
                                "scheduled": len(prefix),
                                "computed": 0,
                                "prompt_tokens": len(prefix),
                                "prompt_sha256": token_digest(prefix),
                            }
                            for prefix in request["prompt_token_ids"]
                        ],
                        "scheduled_tokens": sum(len(prefix) for prefix in request["prompt_token_ids"]),
                        "dummy": False,
                        "rows": rows,
                        "tokens_across_dp": [rows, rows],
                    }
                )
            for engine in self.endpoint.engines:
                if engine is not self and engine.steps is not None:
                    engine.steps.append({"requests": [], "scheduled_tokens": 0, "dummy": True})
        output = await self.endpoint.generate(request)
        output.pop("engine_indices")
        return output


def _kernel_record(role, block):
    """A vLLM worker's loaded vendored kernel holding one launch config with ``block`` reduction columns."""
    return {
        "file": f"c{role}.py",
        "sha256": source_digest(VENDORED_KERNELS[role][1]),
        "kernel_name": VENDORED_KERNELS[role][0],
        "launches": [{"kwargs": {"XBLOCK": 1, "R0_BLOCK": block}, "num_warps": 16, "num_stages": 1}],
        "best_config": None,
    }


class _InferenceEndpoint:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.route_offset = 0
        # The post-attention norm's block the engines' autotuners chose.
        self.norm_block = 2048
        # Two vLLM data-parallel ranks, one client engine each; ``failover_engine`` serves every prompt.
        self.engines = [_Engine(self, 0), _Engine(self, 1)]
        self.failover_engine = None

    async def reset_prefix_cache(self):
        pass

    async def begin_probe_step_log(self):
        for engine in self.engines:
            engine.steps = []

    async def end_probe_step_log(self):
        logs = [[{"placement": {"dp_rank": engine.dp_rank}, "steps": engine.steps}] for engine in self.engines]
        for engine in self.engines:
            engine.steps = None
        return logs

    async def probe_numerics_provenance(self):
        worker = {
            "placement": {"dp_rank": 0, "ep_rank": 1},
            "parameter_sha256": "ab" * 32,
            "versions": {"torch": "test"},
            "inductor_output_code": {"q7/cq7kernel.py": "def call(args):\n    pass\n"},
            "inductor_kernels": [_kernel_record("rms_norm", self.norm_block)],
        }
        return [[worker]]

    async def generate(self, request):
        probabilities = torch.log_softmax(torch.arange(32, dtype=torch.float32), dim=0)
        prompts = len(request["prompt_token_ids"])
        per_engine = -(-prompts // len(self.engines))
        return {
            # The client's even split: prompt i goes to engine i // ceil(prompts / engines).
            "engine_indices": [
                index // per_engine if self.failover_engine is None else self.failover_engine
                for index in range(prompts)
            ],
            "prompt_logprobs": [
                [None] + [{token: float(probabilities[token])} for token in sequence[1:]]
                for sequence in request["prompt_token_ids"]
            ],
            "prefix_cache_hit_tokens": [0 for _ in request["prompt_token_ids"]],
            # Each input token t routes to experts (t + offset) % 8 and (t + offset + 1) % 8.
            "routed_experts": [
                np.asarray(
                    [[[(token + self.route_offset) % 8, (token + self.route_offset + 1) % 8]] for token in sequence],
                    dtype=np.int32,
                )
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
        self.prompt_routes_by_mode = {}
        self.response_routes_by_mode = {}
        self.dp_ranks_by_mode = {}
        self.steps_by_mode = {}
        self.kernel_configs_by_mode = {}
        self.timed_modes = []
        self.actor_infos = [
            SimpleNamespace(rank=MeshRank(dp=dp, sp=0, tp=0, pp=0, world_size=2, dp_size=2, pp_size=1))
            for dp in range(2)
        ]

    def async_run_ray_method(self, dispatch, method, *, data=None):
        if method == "probe_weights_digest":
            return ["0" * 64]
        if method == "probe_time_training_pass":
            self.timed_modes.append(data.metadata["probe_mode"])
            repetitions = data.metadata["probe_timing_repetitions"]
            return [{"seconds": [0.5 + dp] * repetitions, "peak_memory_bytes": 100 * (dp + 1)} for dp in range(2)]
        if method != "probe_forward":
            raise ValueError(method)
        width = data.metadata["response_length"]
        mode = data.metadata["probe_mode"]
        self.dp_ranks_by_mode[mode] = data.get("vllm_dp_rank")
        tokens, rows = data.get("vllm_step_tokens"), data.get("vllm_step_rows")
        self.steps_by_mode[mode] = None if tokens is None else list(zip(tokens.tolist(), rows.tolist(), strict=True))
        self.kernel_configs_by_mode[mode] = data.metadata["vllm_kernel_configs"]
        self.prompt_routes_by_mode[data.metadata["probe_mode"]] = data["rollout_prompt_routed_experts"].clone()
        self.response_routes_by_mode[data.metadata["probe_mode"]] = data["rollout_routed_experts"].clone()
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


def chained_scores(cfg) -> list[mismatch.ScoreRow]:
    view = ReadView(cfg.trainer.mismatch_probe.archive_uri)
    return [mismatch.ScoreRow.model_validate(row) for row in view.scan(mismatch.SCORES_TABLE).to_pylist()]


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
    step_mode = f"reread_replay+{KEPT_STACK_3}"
    cfg.trainer.mismatch_probe.extra_trainer_modes = [
        "router_replay",
        "router_replay_response",
        "reread_replay",
        step_mode,
    ]
    cfg.trainer.mismatch_probe.reread_again = True
    cfg.trainer.mismatch_probe.timing_modes = ["native", "reread_replay"]
    cfg.trainer.algorithm.advantage_estimator = "uniform"
    cfg.trainer.algorithm.use_tis = False
    cfg.generator.inference_engine_data_parallel_size = 2
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
    prompt_ids = [f"p{i}" for i in range(3)]
    sample_ids = [f"p{i}:0" for i in range(3)]
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

    # The cache-off re-read of sample 0 reads prefix [3, 4, 7, 9, 11]; re-read replay places its routes on
    # the left-padded prompt axis and on response inputs 0..2, leaving the final response token native.
    reread_prompt = trainer.policy_model.prompt_routes_by_mode["reread_replay"]
    reread_response = trainer.policy_model.response_routes_by_mode["reread_replay"]
    assert reread_prompt[0, :, 0].tolist() == [[0, 0], [3, 4], [4, 5]]
    assert reread_response[0, :, 0].tolist() == [[7, 0], [1, 2], [3, 4], [0, 0]]
    # Replay modes also carry the vLLM data-parallel rank that served their routes' source (the padding row
    # takes 0): rank 0, which re-read every prefix, or the engine each generation's session id hashes to.
    assert trainer.policy_model.dp_ranks_by_mode["reread_replay"].tolist() == [0, 0, 0, 0]
    sessions = [int.from_bytes(hashlib.sha256(f"p{i}_0".encode()).digest(), "big") % 2 for i in range(3)]
    assert sessions == [1, 1, 0]
    for mode in ("router_replay", "router_replay_response"):
        assert trainer.policy_model.dp_ranks_by_mode[mode].tolist() == [*sessions, 0]
    assert trainer.policy_model.dp_ranks_by_mode["native"] is None
    # The vllm_steps mode scores each sample as the logged step of its re-read ran it: its prefix (prompt and
    # response but the last token) and the step's rows; the padding row has no step. Other modes carry no steps.
    assert trainer.policy_model.steps_by_mode[step_mode] == [(5, 8), (4, 8), (6, 8), (0, 0)]
    assert trainer.policy_model.steps_by_mode["reread_replay"] is None
    # Every mode launches the vLLM kernels with the configs the re-read engine's autotuner chose.
    chosen = {"rms_norm": {"kwargs": {"R0_BLOCK": 2048, "XBLOCK": 1}, "num_warps": 16, "num_stages": 1}}
    assert all(configs == chosen for configs in trainer.policy_model.kernel_configs_by_mode.values())
    rereads = {row.sample_id: row for row in chained_scores(cfg) if row.scorer == "vllm.rescore" and row.update == 0}
    agains = [row for row in chained_scores(cfg) if row.scorer == "vllm.rescore_again" and row.update == 0]
    assert len(rereads) == len(agains) == 3
    assert rereads["p0:0"].expert_choices_shape == [5, 1, 2]

    # Reusing the chained archive freezes its re-read as the prefill reference and replay source.
    frozen = read_frozen_probe(cfg.trainer.mismatch_probe.archive_uri).rereads
    assert sorted(frozen) == sorted(row.sample_id for row in probes)
    assert frozen["p0:0"].logprobs == rereads["p0:0"].logprobs
    trainer.inference_engine_client.route_offset = 2
    trainer.inference_engine_client.failover_engine = 1
    trainer.inference_engine_client.norm_block = 4096
    cfg.trainer.mismatch_probe.reuse_probe = cfg.trainer.mismatch_probe.archive_uri
    cfg.trainer.mismatch_probe.archive_uri = str(tmp_path / "frozen-reference")
    cfg.trainer.mismatch_probe.score_after_updates = [0]
    trainer.global_step = 0
    callback = MismatchProbeCallback(cfg)
    await callback.on_train_begin_async(TrainerState(0, 0, 9, 9), TrainerControl(), trainer=trainer)
    callback.on_train_end(TrainerState(0, 0, 9, 9), TrainerControl(), trainer=trainer)
    code = tmp_path / "frozen-reference-inductor-output-code" / "engine-0" / "dp-0-ep-1" / "q7" / "cq7kernel.py"
    assert code.read_text() == "def call(args):\n    pass\n"
    vllm_workers = json.loads(read_frozen_probe(cfg.trainer.mismatch_probe.archive_uri).manifest.hardware_json)["vllm"]
    assert [worker["parameter_sha256"] for worker in vllm_workers] == ["ab" * 32]
    # The fresh re-read now routes differently and this job's engines autotuned another block, but re-read replay
    # still scores against the frozen reference: its routes, the rank, steps and kernel configs its source logged.
    assert trainer.policy_model.prompt_routes_by_mode["reread_replay"][0, :, 0].tolist() == [[0, 0], [3, 4], [4, 5]]
    assert trainer.policy_model.dp_ranks_by_mode["reread_replay"].tolist() == [0, 0, 0, 0]
    assert trainer.policy_model.steps_by_mode[step_mode] == [(5, 8), (4, 8), (6, 8), (0, 0)]
    assert trainer.policy_model.kernel_configs_by_mode[step_mode] == chosen
    assert trainer.policy_model.kernel_configs_by_mode["router_replay"]["rms_norm"]["kwargs"]["R0_BLOCK"] == 4096
    timing = json.loads(read_frozen_probe(cfg.trainer.mismatch_probe.archive_uri).manifest.timing_json)
    # The slowest data-parallel rank sets each repetition's pass time.
    assert timing["training_pass@0:reread_replay/seconds"] == [1.5] * TIMING_REPETITIONS
    assert timing["training_pass@0:native/peak_memory_bytes"] == 200
    frozen_rows = [row for row in chained_scores(cfg) if row.scorer == "vllm.rescore_frozen"]
    assert {row.sample_id: row.expert_choices for row in frozen_rows} == {
        sample: row.expert_choices for sample, row in rereads.items()
    }


def test_archive_write_splits_rows_across_transactions_under_the_size_limit(tmp_path):
    uri = str(tmp_path / "split")
    rows = [
        mismatch.ScoreRow(
            probe_hash="hash",
            sample_id=f"sample-{index}",
            scorer="trainer",
            mode="native",
            update=0,
            weights_hash="weights",
            logprobs=[-0.5] * 2000,
        )
        for index in range(6)
    ]
    archive = MismatchArchive(uri, writer_id="test", max_buffer_bytes=40_000)
    try:
        archive.write(scores=rows)
    finally:
        archive.close()
    stored = [mismatch.ScoreRow.model_validate(row) for row in ReadView(uri).scan(mismatch.SCORES_TABLE).to_pylist()]
    assert sorted(row.sample_id for row in stored) == [row.sample_id for row in rows]
