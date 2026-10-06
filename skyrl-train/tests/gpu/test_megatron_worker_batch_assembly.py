import asyncio
import copy
import sys
from hashlib import sha256

import numpy as np
from omegaconf import open_dict
import pytest
import ray
from ray.util.placement_group import placement_group
from safetensors import safe_open
import torch
from transformers import AutoTokenizer

from skyrl_train.dataset.routed_expert_batch import RoutedExpertRows
from skyrl_train.rollouts.buffer import RolloutGroup
from skyrl_train.rollouts.context import TrainingContext
from skyrl_train.trainer import RayPPOTrainer
from skyrl_train.utils import initialize_ray, validate_cfg
from tests.gpu.grug_gpu_gates import require_hoppers
from tests.gpu.grug_serving import rank0_validation_snapshot
from tests.gpu.router_replay_fixtures import random_unique_routes
from tests.gpu.test_grug_megatron import NUM_EXPERTS, NUM_LAYERS, TOY_SHAPE, _config, _write_tiny_checkpoint
from tests.gpu.utils import import_worker
from tests.rollout_fixtures import FixedPromptDataset, FixedRolloutRunner


@pytest.mark.slow
@pytest.mark.parametrize("corrupt_routes", [False, True])
def test_megatron_pp_cp_worker_batches_match_driver_update(tmp_path, corrupt_routes):
    require_hoppers(8)
    model_path = tmp_path / "model"
    model_path.mkdir()
    _write_tiny_checkpoint(model_path, num_experts_per_tok=4, shape={**TOY_SHAPE, "num_key_value_heads": 2})
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    with safe_open(model_path / "model.safetensors", framework="pt", device="cpu") as checkpoint:
        parameter_names = list(checkpoint.keys())
    cfg = _config(str(model_path), world_size=8, pp=2, ep=1)
    cfg.trainer.policy.megatron_config.context_parallel_size = 2
    with open_dict(cfg.trainer.policy.megatron_config.transformer_config_kwargs):
        cfg.trainer.policy.megatron_config.transformer_config_kwargs.cp_comm_type = "a2a"
    cfg.trainer.policy.megatron_config.moe_router_replay = True
    cfg.trainer.use_sample_packing = True
    cfg.trainer.train_batch_size = cfg.trainer.policy_mini_batch_size = 3
    cfg.trainer.micro_train_batch_size_per_gpu = cfg.trainer.micro_forward_batch_size_per_gpu = 1
    cfg.trainer.algorithm.advantage_estimator = "rloo_n"
    cfg.trainer.algorithm.group_advantage_min_size = 2
    cfg.trainer.algorithm.off_policy_correction = "tis"
    cfg.generator.n_samples_per_prompt = 4
    validate_cfg(cfg)
    cfg.trainer.training_metrics = True
    cfg.trainer.rollout_buffer.max_staleness_steps = 0
    cfg.trainer.rollout_buffer.max_in_flight = 3
    cfg.generator.trajectory_retention.enabled = False
    cfg.data.shuffle = False
    generator = torch.Generator().manual_seed(888)
    groups = []
    for index in range(3):
        lengths = [17, 22, 25, 32 if index == 2 else 27]
        batch = {
            "prompt_token_ids": [[10 + index] * (11 + index)] * 4,
            "response_ids": [[20 + row] * length for row, length in enumerate(lengths)],
            "loss_masks": [[1] * length for length in lengths],
            "rewards": [0.0, 1.0, 3.0, 2.0],
            "rollout_logprobs": [np.full(length, -3.0, dtype=np.float32) for length in lengths],
            "rollout_routed_experts": [
                random_unique_routes((length, NUM_LAYERS, 4), NUM_EXPERTS, generator=generator).numpy().astype(np.uint8)
                for length in lengths
            ],
            "is_last_step": [True] * 4,
            "rollout_metrics": {},
        }
        groups.append(RolloutGroup(batch, f"g{index}", 1, {"uid": f"g{index}"}))
    initialize_ray(cfg)
    shared_pg = placement_group([{"GPU": 1, "CPU": 1}] * 8, strategy="PACK")
    results = {}
    trainer = None
    try:
        ray.get(shared_pg.ready(), timeout=120)
        for builder in ("driver", "verify"):
            current = copy.deepcopy(cfg)
            current.trainer.batch_builder = builder
            validate_cfg(current)
            dataset = FixedPromptDataset([group.uid for group in groups])
            runner = FixedRolloutRunner(groups)
            trainer = RayPPOTrainer(
                cfg=current,
                tracker=None,
                tokenizer=tokenizer,
                train_dataset=dataset,
                inference_engine_client=None,
                trajectory_runner=runner,
                context=TrainingContext.from_config(current, dataset, runner),
                callbacks=[],
            )
            trainer.build_models(import_worker("megatron", "policy"), None, None, policy_pg=shared_pg)
            trainer.global_step = 1
            policy = trainer.policy_model
            initial = {
                name: sha256(value.contiguous().view(torch.uint8).numpy()).hexdigest()
                for name, value in rank0_validation_snapshot(policy, parameter_names).items()
            }
            if builder == "verify":
                assert initial == results["driver"]["initial"]

            async def run_admitted_batch():
                context = trainer.context
                source = trainer.batch_source
                context.start()
                try:
                    await context.publish(1)
                    await source.admit(stall_timeout=30, diagnostics=trainer.batch_diagnostics())
                    await source.prepare(global_step=1, step_wall=None)
                    reference = source.driver if builder == "verify" else source
                    if builder == "verify" and corrupt_routes:
                        routes = reference.batch.routed_expert_rows
                        rows = list(routes.rows)
                        rows[0] = rows[0].copy()
                        route = rows[0][0, 0]
                        route[0] = next(expert for expert in range(NUM_EXPERTS) if expert not in route)
                        reference.batch.routed_expert_rows = RoutedExpertRows(
                            tuple(rows), routes.response_len, routes.num_experts
                        )
                        with pytest.raises(ValueError, match="worker forward input digest mismatch"):
                            await source.forward(step_wall=None)
                        unchanged = {
                            name: sha256(value.contiguous().view(torch.uint8).numpy()).hexdigest()
                            for name, value in rank0_validation_snapshot(policy, parameter_names).items()
                        }
                        assert unchanged == initial
                        return None
                    await source.forward(step_wall=None)
                    await source.finalize(step_wall=None)
                    status = await source.train(step_wall=None)
                    assert status["router_replay/hit_fraction"] == 1.0
                    parameters = {
                        name: sha256(value.contiguous().view(torch.uint8).numpy()).hexdigest()
                        for name, value in rank0_validation_snapshot(policy, parameter_names).items()
                    }
                    assert parameters.keys() == initial.keys() == set(parameter_names)
                    assert parameters != initial
                    return {
                        "initial": initial,
                        "parameters": parameters,
                        "status": status,
                        "logprobs": reference.batch["action_log_probs"].clone(),
                    }
                finally:
                    try:
                        await source.release(sys.exception())
                    finally:
                        await context.close()

            result = asyncio.run(run_admitted_batch())
            trainer.cleanup_ray_actors()
            trainer = None
            if result is None:
                return
            results[builder] = result
        torch.testing.assert_close(results["verify"]["logprobs"], results["driver"]["logprobs"], rtol=0, atol=0)
        assert (
            results["verify"]["logprobs"].view(torch.uint8).numpy().tobytes()
            == results["driver"]["logprobs"].view(torch.uint8).numpy().tobytes()
        )
        for key in ("policy_loss", "policy_entropy", "raw_grad_norm"):
            assert results["verify"]["status"][key] == results["driver"]["status"][key], key
        assert results["verify"]["parameters"] == results["driver"]["parameters"]
    finally:
        if trainer is not None:
            trainer.cleanup_ray_actors()
        ray.util.remove_placement_group(shared_pg)
        ray.shutdown()
