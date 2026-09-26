from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

# Direct execution from the repository root needs the test package on the path.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import ray
import torch
from omegaconf import OmegaConf
from ray.util.placement_group import placement_group, remove_placement_group
from skyrl_train.distributed.dispatch import concatenate_outputs_after_mesh_dispatch
from skyrl_train.training_batch import TrainingInputBatch
from skyrl_train.utils import get_ray_pg_ready_with_timeout
from skyrl_train.utils.utils import validate_cfg
from skyrl_train.workers.worker import PPORayActorGroup
from transformers import AutoConfig, AutoModelForCausalLM

from tests.gpu.utils import get_test_actor_config

WORLD_SIZE = 8
GLOBAL_SEQUENCES = 64
PROMPT_WIDTH = 32
RESPONSE_WIDTH = 16
MEASURED_STEPS = 8
WARMUP_STEPS = 2
SEED = 17


def _canonical_hash(batches: list[dict[str, torch.Tensor]]) -> str:
    digest = hashlib.sha256()
    for step, batch in enumerate(batches):
        digest.update(f"step={step}\n".encode())
        for key in sorted(batch):
            value = batch[key].contiguous().cpu()
            digest.update(f"{key}:{value.dtype}:{tuple(value.shape)}\n".encode())
            digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def _prepare(model_path: Path, output: Path) -> None:
    config = AutoConfig.from_pretrained(model_path)
    vocab_size = int(config.vocab_size)
    pad = int(config.eos_token_id or 0)
    generator = torch.Generator().manual_seed(SEED)
    model = (
        AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16, attn_implementation="eager")
        .eval()
        .to("cuda:0")
    )
    batches: list[dict[str, torch.Tensor]] = []
    for step in range(MEASURED_STEPS):
        lengths = torch.tensor([6 + (step + row * 7) % 11 for row in range(GLOBAL_SEQUENCES)])
        sequences = torch.randint(
            100, vocab_size, (GLOBAL_SEQUENCES, PROMPT_WIDTH + RESPONSE_WIDTH), generator=generator
        )
        attention_mask = torch.ones_like(sequences)
        response_mask = torch.zeros((GLOBAL_SEQUENCES, RESPONSE_WIDTH), dtype=torch.long)
        for row, length in enumerate(lengths.tolist()):
            response_mask[row, :length] = 1
            sequences[row, PROMPT_WIDTH + length :] = pad
            attention_mask[row, PROMPT_WIDTH + length :] = 0
        sequences_gpu = sequences.to("cuda:0")
        mask_gpu = attention_mask.to("cuda:0")
        position_ids = mask_gpu.long().cumsum(-1) - 1
        position_ids.masked_fill_(mask_gpu == 0, 1)
        with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            logits = model(sequences_gpu, attention_mask=mask_gpu, position_ids=position_ids).logits
            next_tokens = sequences_gpu.roll(-1, dims=1)
            token_logprobs = torch.log_softmax(logits.float(), dim=-1).gather(-1, next_tokens[..., None]).squeeze(-1)
            old_logprobs = token_logprobs[:, -RESPONSE_WIDTH - 1 : -1].cpu()
        # Alternating signed, nonzero sequence-level advantages emulate GRPO
        # while keeping the same frozen update inputs for every backend.
        advantages = (
            torch.tensor(
                [(-1.0 if (row + step) % 2 else 1.0) * (0.25 + (row % 5) * 0.15) for row in range(GLOBAL_SEQUENCES)],
                dtype=torch.float32,
            )[:, None]
            .expand(-1, RESPONSE_WIDTH)
            .clone()
        )
        zeros = torch.zeros_like(old_logprobs)
        batches.append(
            {
                "sequences": sequences,
                "attention_mask": attention_mask,
                "action_log_probs": old_logprobs,
                "base_action_log_probs": old_logprobs.clone(),
                "rollout_logprobs": old_logprobs.clone(),
                "values": zeros.clone(),
                "returns": zeros.clone(),
                "advantages": advantages,
                "loss_mask": response_mask.clone(),
                "response_mask": response_mask,
            }
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"batches": batches, "seed": SEED, "response_width": RESPONSE_WIDTH}, output)
    print(
        json.dumps(
            {
                "event": "fixture",
                "path": str(output),
                "sha256_tensors": _canonical_hash(batches),
                "sha256_file": hashlib.sha256(output.read_bytes()).hexdigest(),
                "steps": MEASURED_STEPS,
                "sequences_per_step": GLOBAL_SEQUENCES,
                "valid_tokens_per_step": [int(batch["response_mask"].sum()) for batch in batches],
            },
            sort_keys=True,
        )
    )


def _config(model_path: str, backend: str, *, fsdp_fp32_master: bool = False):
    cfg = get_test_actor_config()
    cfg.trainer.strategy = backend
    cfg.trainer.policy.model.path = model_path
    cfg.trainer.policy.model.tokenizer_path = model_path
    cfg.trainer.ref.model.path = model_path
    cfg.trainer.placement.colocate_all = False
    cfg.trainer.placement.colocate_policy_ref = True
    cfg.trainer.placement.policy_num_nodes = 1
    cfg.trainer.placement.policy_num_gpus_per_node = WORLD_SIZE
    cfg.trainer.placement.ref_num_nodes = 1
    cfg.trainer.placement.ref_num_gpus_per_node = WORLD_SIZE
    cfg.trainer.train_batch_size = 16
    cfg.trainer.policy_mini_batch_size = 16
    cfg.trainer.micro_train_batch_size_per_gpu = 1
    cfg.trainer.micro_forward_batch_size_per_gpu = 1
    cfg.trainer.update_epochs_per_batch = 1
    cfg.trainer.bf16 = True
    cfg.trainer.flash_attn = False
    cfg.trainer.use_sample_packing = False
    cfg.trainer.gradient_checkpointing = True
    cfg.trainer.policy.optimizer_config.lr = 2.0e-6
    cfg.trainer.policy.optimizer_config.adam_betas = [0.9, 0.999]
    cfg.trainer.policy.optimizer_config.weight_decay = 0.01
    cfg.trainer.policy.optimizer_config.max_grad_norm = 1.0
    cfg.trainer.policy.optimizer_config.offload_after_step = False
    cfg.trainer.policy.optimizer_config.num_warmup_steps = 0
    if fsdp_fp32_master:
        if backend != "fsdp2":
            raise ValueError("FP32 master comparison applies only to FSDP2")
        OmegaConf.update(cfg, "trainer.policy.optimizer_config.bf16_update_mode", "fp32_master", force_add=True)
    cfg.trainer.algorithm.advantage_estimator = "grpo"
    cfg.trainer.algorithm.policy_loss_type = "regular"
    cfg.trainer.algorithm.loss_reduction = "token_mean"
    cfg.trainer.algorithm.use_kl_loss = True
    cfg.trainer.algorithm.kl_loss_coef = 0.001
    cfg.trainer.algorithm.use_entropy_loss = False
    cfg.generator.n_samples_per_prompt = 4
    cfg.generator.num_inference_engines = WORLD_SIZE
    cfg.generator.inference_engine_tensor_parallel_size = 1
    cfg.generator.inference_engine_pipeline_parallel_size = 1
    cfg.generator.inference_engine_data_parallel_size = 1
    cfg.generator.inference_engine_expert_parallel_size = 1
    cfg.generator.sampling_params.temperature = 1.0
    cfg.trainer.seed = SEED
    if backend == "megatron":
        cfg.trainer.policy.megatron_config.tensor_model_parallel_size = 1
        cfg.trainer.policy.megatron_config.pipeline_model_parallel_size = 1
        cfg.trainer.policy.megatron_config.context_parallel_size = 1
        cfg.trainer.policy.megatron_config.expert_model_parallel_size = 1
        cfg.trainer.policy.megatron_config.expert_tensor_parallel_size = 1
        cfg.trainer.policy.megatron_config.empty_cuda_cache = False
    return cfg


def _actor_group(cfg, backend: str):
    if backend == "fsdp2":
        from skyrl_train.workers.fsdp.fsdp_worker import FSDPPolicyWorkerBase as Base
    else:
        from skyrl_train.workers.megatron.megatron_worker import MegatronPolicyWorkerBase as Base

    class BenchmarkPolicyWorker(Base):
        def benchmark_set_batch(self, batch):
            if not hasattr(self, "_benchmark_batches"):
                self._benchmark_batches = {}
            batch.to(torch.device("cuda", torch.cuda.current_device()))
            self._benchmark_batches[batch.metadata["global_step"]] = batch
            return self._rank

        def benchmark_step(self, index):
            torch.cuda.synchronize()
            torch.distributed.barrier()
            started = time.perf_counter()
            output = self.ppo_train(self._benchmark_batches[index])
            torch.cuda.synchronize()
            torch.distributed.barrier()
            return {
                "rank": self._rank,
                "elapsed_seconds": time.perf_counter() - started,
                "status": output.metadata["train_status"],
            }

    group = placement_group([{"GPU": WORLD_SIZE, "CPU": WORLD_SIZE}], strategy="PACK")
    get_ray_pg_ready_with_timeout(group, timeout=600)
    actors = PPORayActorGroup(
        cfg,
        num_nodes=1,
        num_gpus_per_node=WORLD_SIZE,
        ray_actor_type=ray.remote(num_gpus=1)(BenchmarkPolicyWorker),
        pg=group,
        num_gpus_per_actor=0.75,
        colocate_all=False,
        sequence_parallel_size=cfg.trainer.policy.sequence_parallel_size,
        record_memory=cfg.trainer.policy.record_memory,
    )
    ray.get(actors.async_init_model(cfg.trainer.policy.model.path))
    return actors, group


def _batch(raw: dict[str, torch.Tensor], step: int) -> TrainingInputBatch:
    batch = TrainingInputBatch({key: tensor.clone() for key, tensor in raw.items()})
    batch.metadata = {"response_length": RESPONSE_WIDTH, "global_step": step}
    return batch


def _probe(actors: PPORayActorGroup, raw: dict[str, torch.Tensor]) -> torch.Tensor:
    outputs = ray.get(actors.async_run_ray_method("mesh", "forward", data=_batch(raw, 0)))
    result = concatenate_outputs_after_mesh_dispatch(actors.actor_infos, outputs)
    return result["output"].cpu()


def _stage_batches(actors: PPORayActorGroup, batches: list[dict[str, torch.Tensor]]) -> None:
    for index, raw in enumerate(batches):
        ranks = ray.get(actors.async_run_ray_method("mesh", "benchmark_set_batch", _batch(raw, index)))
        assert len(ranks) == WORLD_SIZE


def _run(
    model_path: str,
    fixture: Path,
    backend: str,
    output: Path,
    repetition: int,
    measured_steps: int,
    fsdp_fp32_master: bool,
) -> None:
    payload = torch.load(fixture, map_location="cpu", weights_only=True)
    batches = payload["batches"]
    assert len(batches) == MEASURED_STEPS
    assert 1 <= measured_steps <= MEASURED_STEPS
    cfg = _config(model_path, backend, fsdp_fp32_master=fsdp_fp32_master)
    validate_cfg(cfg)
    ray_env = {
        "PYTHONPATH": os.environ.get("PYTHONPATH", ""),
        "CUDA_DEVICE_MAX_CONNECTIONS": "1",
        "NVTE_FUSED_ATTN": "0",
    }
    ray.init(runtime_env={"env_vars": ray_env})
    try:
        warm_actors, warm_group = _actor_group(cfg, backend)
        _stage_batches(warm_actors, batches[:WARMUP_STEPS])
        for warm_step in range(WARMUP_STEPS):
            ray.get(warm_actors.async_run_ray_method("pass_through", "benchmark_step", warm_step))
        warm_actors.kill_actors()
        remove_placement_group(warm_group)
        initial_attempts = []
        for _attempt in range(3):
            actors, group = _actor_group(cfg, backend)
            _stage_batches(actors, batches[:measured_steps])
            initial_probe = _probe(actors, batches[0])
            initial_mask = batches[0]["response_mask"].bool()
            initial_vs_frozen = (initial_probe[initial_mask] - batches[0]["action_log_probs"][initial_mask]).abs()
            initial_attempts.append(
                {"mean_abs": float(initial_vs_frozen.mean()), "max_abs": float(initial_vs_frozen.max())}
            )
            if initial_attempts[-1]["max_abs"] <= 0.05:
                break
            actors.kill_actors()
            remove_placement_group(group)
        else:
            raise RuntimeError(f"Initial policy differs from the fixed starting weights: {initial_attempts}")
        try:
            rows = []
            update_probes = []
            for step, raw in enumerate(batches[:measured_steps]):
                ranks = ray.get(actors.async_run_ray_method("pass_through", "benchmark_step", step))
                assert len(ranks) == WORLD_SIZE
                assert all(float(rank["status"]["policy_update_steps"]) == 1.0 for rank in ranks), ranks
                probe = _probe(actors, batches[0])
                update_probes.append(probe.tolist())
                rows.append(
                    {
                        "step": step + 1,
                        "max_rank_elapsed_seconds": max(float(row["elapsed_seconds"]) for row in ranks),
                        "rank_elapsed_seconds": [float(row["elapsed_seconds"]) for row in ranks],
                        "valid_tokens": int(raw["response_mask"].sum()),
                        "rank0_status": ranks[0]["status"],
                    }
                )
            final_probe = probe
            change = (final_probe[initial_mask] - initial_probe[initial_mask]).abs()
        finally:
            actors.kill_actors()
            remove_placement_group(group)
    finally:
        ray.shutdown()
    result = {
        "backend": backend,
        "fsdp_fp32_master": fsdp_fp32_master,
        "repetition": repetition,
        "model_path": model_path,
        "fixture_sha256_tensors": _canonical_hash(batches),
        "steps": measured_steps,
        "valid_tokens": sum(row["valid_tokens"] for row in rows),
        "total_train_seconds": sum(row["max_rank_elapsed_seconds"] for row in rows),
        "step_rows": rows,
        "initial_vs_frozen_logprob_mean_abs": float(initial_vs_frozen.mean()),
        "initial_vs_frozen_logprob_max_abs": float(initial_vs_frozen.max()),
        "initial_attempts": initial_attempts,
        "post_update_probe_mean_abs_change": float(change.mean()),
        "post_update_probe_max_abs_change": float(change.max()),
        "initial_probe": initial_probe.tolist(),
        "update_probes": update_probes,
        "final_probe": final_probe.tolist(),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, sort_keys=True, indent=2))
    print(
        json.dumps(
            {
                key: value
                for key, value in result.items()
                if key not in {"step_rows", "initial_probe", "update_probes", "final_probe"}
            },
            sort_keys=True,
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    prep = sub.add_parser("prepare")
    prep.add_argument("--model-path", type=Path, required=True)
    prep.add_argument("--output", type=Path, required=True)
    run = sub.add_parser("run")
    run.add_argument("--model-path", required=True)
    run.add_argument("--fixture", type=Path, required=True)
    run.add_argument("--backend", choices=("fsdp2", "megatron"), required=True)
    run.add_argument("--output", type=Path, required=True)
    run.add_argument("--repetition", type=int, required=True)
    run.add_argument("--measured-steps", type=int, default=MEASURED_STEPS)
    run.add_argument("--fsdp-fp32-master", action="store_true")
    args = parser.parse_args()
    if args.command == "prepare":
        _prepare(args.model_path, args.output)
    else:
        _run(
            args.model_path,
            args.fixture,
            args.backend,
            args.output,
            args.repetition,
            args.measured_steps,
            args.fsdp_fp32_master,
        )


if __name__ == "__main__":
    main()
