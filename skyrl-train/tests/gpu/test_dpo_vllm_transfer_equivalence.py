"""Training-to-vLLM transfer equivalence after a known DPO update (issue 969).

Takes exactly one DPO optimizer step through the production Ray policy worker at PP2 on a
fixed preference batch, snapshots the updated training tensors, broadcasts them into
disaggregated vLLM engines over the NCCL path, and then checks transfer equivalence twice:

1. Every named serving tensor matches the training snapshot byte for byte, accounting for
   the explicit layout and dtype conversions (fp32 router biases and router weight, bf16
   parameters, stacked-expert slices, vLLM vocabulary padding) via assert_engine_weights.
2. Token log-probabilities of the fixed sequences agree between the trainer's eval forward
   and vLLM under matched greedy scoring, both before the update (initial weights) and
   after the synchronization, within the bf16 parity tolerance used by the forward gates;
   the post-sync vLLM values must differ from the pre-update ones, proving the updated
   tensors are the ones installed.
"""

import asyncio
import math

import pytest
import ray
import torch
from omegaconf import OmegaConf
from transformers import AutoTokenizer

from skyrl_train.inference_engines.base import InferenceEngineInput
from skyrl_train.inference_engines.utils import get_sampling_params_for_backend
from skyrl_train.training_batch import TrainingInputBatch
from skyrl_train.utils import initialize_ray
from skyrl_train.utils.utils import validate_cfg
from tests.cpu.util import DPO_OVERRIDES
from tests.gpu.dpo_equivalence import fixed_pair_batch
from tests.gpu.grug_gpu_gates import require_hoppers
from tests.gpu.grug_serving import (
    GRUG_BIAS_NAMES,
    GRUG_PARAMETER_NAMES,
    GRUG_SERVING_EXPERT_INDEX_BY_NAME,
    assert_engine_weights,
    grug_engine_client,
    rank0_validation_snapshot,
)
from tests.gpu.grug_tiny_checkpoint import write_tiny_checkpoint
from tests.gpu.test_grug_megatron import (
    LOGPROB_MAX_ABS_TOLERANCE,
    LOGPROB_MEAN_ABS_TOLERANCE,
    _config,
    _megatron_response_logprobs,
)
from tests.gpu.utils import init_worker_with_type

POLICY_WORLD_SIZE = 2
ROLLOUT_WORLD_SIZE = 2


def _dpo_config(model_path: str):
    cfg = _config(model_path, world_size=POLICY_WORLD_SIZE, pp=2, ep=1)
    cfg.trainer.train_batch_size = 8
    cfg.trainer.policy_mini_batch_size = 8
    cfg.trainer.micro_train_batch_size_per_gpu = 2
    cfg.trainer.micro_forward_batch_size_per_gpu = 2
    cfg.trainer.update_epochs_per_batch = 1
    # The shipped DPO recipe, applied as one block so this experiment and the
    # reference in dpo_equivalence.py cannot drift apart field by field.
    cfg = OmegaConf.merge(cfg, OmegaConf.create(DPO_OVERRIDES))
    # No reference worker is created (base log-probs arrive in-batch); match its
    # sizing to policy so the colocated-recipe validation passes.
    cfg.trainer.placement.ref_num_nodes = 1
    cfg.trainer.placement.ref_num_gpus_per_node = POLICY_WORLD_SIZE
    validate_cfg(cfg)
    return cfg


def _pair_training_batch(batch, log_probs: torch.Tensor) -> TrainingInputBatch:
    zeros = torch.zeros_like(batch.loss_mask, dtype=torch.float32)
    rows = TrainingInputBatch(
        {
            "sequences": batch.sequences,
            "attention_mask": batch.attention_mask,
            "action_log_probs": log_probs.clone(),
            "base_action_log_probs": log_probs.clone(),
            "rollout_logprobs": log_probs.clone(),
            "values": zeros.clone(),
            "returns": zeros.clone(),
            "advantages": zeros.clone(),
            "loss_mask": batch.loss_mask.long(),
            "response_mask": batch.loss_mask.long(),
            "pair_roles": batch.pair_roles,
        }
    )
    rows.metadata = {"response_length": batch.num_actions, "global_step": 0}
    return rows


async def _vllm_logprobs(client, cfg, sequences: torch.Tensor, num_actions: int) -> torch.Tensor:
    """Score fixed sequences with vLLM and return each row's action-window log-probs."""
    sampling_params = get_sampling_params_for_backend("vllm", cfg.generator.sampling_params)
    sampling_params.update({"temperature": 0.0, "max_tokens": 1, "prompt_logprobs": 0})
    request = InferenceEngineInput(prompt_token_ids=sequences.tolist(), sampling_params=sampling_params)
    result = await client.generate(request)
    rows = []
    for sequence, per_token in zip(sequences.tolist(), result["prompt_logprobs"], strict=True):
        window = per_token[-num_actions:]
        rows.append([entry[token] for entry, token in zip(window, sequence[-num_actions:], strict=True)])
    return torch.tensor(rows, dtype=torch.float32)


def _assert_logprob_parity(label: str, actual: torch.Tensor, expected: torch.Tensor, mask: torch.Tensor) -> None:
    valid = mask.bool()
    difference = (actual[valid] - expected[valid]).abs()
    print(
        f"{label} logprob parity over {valid.sum().item()} tokens: "
        f"max abs {difference.max().item():.4f}, mean abs {difference.mean().item():.4f}",
        flush=True,
    )
    assert difference.max().item() < LOGPROB_MAX_ABS_TOLERANCE, difference.max()
    assert difference.mean().item() < LOGPROB_MEAN_ABS_TOLERANCE, difference.mean()


def _dpo_train_step(policy, batch: TrainingInputBatch) -> dict[str, float]:
    """Run one training step and assert the DPO loss row is finite.

    Mirrors the shared PPO `_train_step`, whose `policy_loss` row does not exist for
    the DPO objective (it reports `dpo/*` rows instead).
    """
    train_output = ray.get(policy.async_run_ray_method("mesh", "ppo_train", batch))[0]
    status = train_output.metadata["train_status"]
    assert math.isfinite(status["dpo/loss"]), status
    return status


@pytest.mark.vllm
def test_dpo_update_transfers_to_vllm_tensors_and_logprobs(tmp_path):
    require_hoppers(POLICY_WORLD_SIZE + ROLLOUT_WORLD_SIZE)
    model_path = tmp_path / "model"
    model_path.mkdir()
    write_tiny_checkpoint(model_path)
    cfg = _dpo_config(str(model_path))
    pad_token_id = AutoTokenizer.from_pretrained(model_path).pad_token_id
    batch = fixed_pair_batch(pad_token_id)
    print(f"input batch sha256={batch.sha256()}", flush=True)

    initialize_ray(cfg)
    client = grug_engine_client(cfg, str(model_path))
    try:
        policy = init_worker_with_type(
            "policy", shared_pg=None, colocate_all=False, num_gpus_per_node=POLICY_WORLD_SIZE, num_nodes=1, cfg=cfg
        )
        names = [*GRUG_PARAMETER_NAMES, *GRUG_BIAS_NAMES]
        before = rank0_validation_snapshot(policy, names)
        trainer_before = _megatron_response_logprobs(
            policy, _pair_training_batch(batch, torch.zeros_like(batch.loss_mask))
        )
        served_before = asyncio.run(_vllm_logprobs(client, cfg, batch.sequences, batch.num_actions))
        _assert_logprob_parity("pre-update", served_before, trainer_before, batch.loss_mask)

        _dpo_train_step(policy, _pair_training_batch(batch, trainer_before.cpu()))
        training = rank0_validation_snapshot(policy, names)
        changed = [name for name in GRUG_PARAMETER_NAMES if not torch.equal(training[name], before[name])]
        assert changed, "the DPO step did not move any training tensors"
        print(f"known update moved {len(changed)} of {len(GRUG_PARAMETER_NAMES)} parameter tensors", flush=True)

        ray.get(policy.async_run_ray_method("pass_through", "init_weight_sync_state", client))
        ray.get(policy.async_run_ray_method("pass_through", "broadcast_to_inference_engines", client))
        assert_engine_weights(client, names, training, GRUG_BIAS_NAMES, GRUG_SERVING_EXPERT_INDEX_BY_NAME)

        asyncio.run(client.reset_prefix_cache())
        trainer_after = _megatron_response_logprobs(
            policy, _pair_training_batch(batch, torch.zeros_like(batch.loss_mask))
        )
        served_after = asyncio.run(_vllm_logprobs(client, cfg, batch.sequences, batch.num_actions))
        _assert_logprob_parity("post-sync", served_after, trainer_after, batch.loss_mask)
        drift = (served_after - served_before).abs()[batch.loss_mask.bool()]
        assert drift.max().item() > 1e-7, drift.max()
    finally:
        ray.shutdown()
