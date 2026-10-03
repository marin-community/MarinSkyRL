import asyncio
import copy
from hashlib import sha256
from types import SimpleNamespace

import numpy as np
import pytest
import ray
from ray.util.placement_group import placement_group
from safetensors import safe_open
import torch
from transformers import AutoTokenizer

from skyrl_train.batch_assembly import plan_batch
from skyrl_train.dynamic_sampling import GroupSelectionPolicy
from skyrl_train.distributed.dispatch import WorkerGroupTaskError
from skyrl_train.group_admission import GroupAdmissionPolicy, GroupAdvantageInvariant
from skyrl_train.rollouts.buffer import (
    BatchPolicy,
    RolloutBuffer,
    RolloutBufferConfig,
    RolloutContentPolicy,
    RolloutGroup,
)
from skyrl_train.rollouts.context import RolloutBatchMetadata, RolloutReader
from skyrl_train.rollouts.payloads import MemoryPayloads
from skyrl_train.trainer import RayPPOTrainer
from skyrl_train.utils import initialize_ray, validate_cfg
from tests.gpu.grug_gpu_gates import require_hoppers
from tests.gpu.grug_serving import rank0_validation_snapshot
from tests.gpu.router_replay_fixtures import random_unique_routes
from tests.gpu.test_grug_megatron import NUM_EXPERTS, NUM_LAYERS, TOY_SHAPE, _config, _write_tiny_checkpoint
from tests.gpu.utils import init_worker_with_type


@pytest.mark.slow
@pytest.mark.parametrize("corrupt_routes", [False, True])
def test_megatron_pp_cp_worker_batches_match_driver_update(tmp_path, corrupt_routes):
    require_hoppers(8)
    model_path = tmp_path / "model"
    model_path.mkdir()
    _write_tiny_checkpoint(model_path, num_experts_per_tok=4, shape={**TOY_SHAPE, "global_every": 1})
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    with safe_open(model_path / "model.safetensors", framework="pt", device="cpu") as checkpoint:
        parameter_names = list(checkpoint.keys())
    cfg = _config(str(model_path), world_size=8, pp=2, ep=1)
    cfg.trainer.policy.megatron_config.context_parallel_size = 2
    cfg.trainer.policy.megatron_config.moe_router_replay = True
    cfg.trainer.use_sample_packing = True
    cfg.trainer.train_batch_size = cfg.trainer.policy_mini_batch_size = 3
    cfg.trainer.micro_train_batch_size_per_gpu = cfg.trainer.micro_forward_batch_size_per_gpu = 1
    cfg.trainer.algorithm.advantage_estimator = "rloo_n"
    cfg.trainer.algorithm.group_advantage_min_size = 2
    cfg.trainer.algorithm.off_policy_correction = "tis"
    cfg.generator.n_samples_per_prompt = 4
    validate_cfg(cfg)
    contract = GroupAdvantageInvariant.from_config(cfg.trainer.algorithm.resolved_group_advantage)
    content = RolloutContentPolicy(
        GroupAdmissionPolicy(contract, rollout_logprobs_required=True), GroupSelectionPolicy(None)
    )
    buffer_config = RolloutBufferConfig(3, 3, 0, BatchPolicy.FULL_BATCH, None, None)
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
    ray.get(shared_pg.ready(), timeout=120)
    results = {}
    policy = None
    try:
        for builder in ("driver", "verify"):
            current = copy.deepcopy(cfg)
            current.trainer.batch_builder = builder
            validate_cfg(current)
            policy = init_worker_with_type(
                "policy",
                shared_pg=shared_pg,
                colocate_all=False,
                num_gpus_per_node=8,
                num_nodes=1,
                cfg=current,
                num_training_steps=1,
            )
            policy.run_method("pass_through", "_set_pad_token_id", tokenizer.pad_token_id)
            initial = {
                name: sha256(value.contiguous().view(torch.uint8).numpy()).hexdigest()
                for name, value in rank0_validation_snapshot(policy, parameter_names).items()
            }
            if builder == "verify":
                assert initial == results["driver"]["initial"]
            trainer = RayPPOTrainer.__new__(RayPPOTrainer)
            trainer.cfg = current
            trainer.tokenizer = tokenizer
            trainer.policy_model = policy
            trainer.critic_model = trainer.ref_model = trainer.trajectory_selector = None
            trainer.distillation_plan = trainer._distillation_runtime = None
            trainer.context = SimpleNamespace(config=buffer_config)
            trainer.group_advantage_invariant = contract
            trainer.colocate_all = False
            trainer._training_metrics_enabled = True
            trainer._num_experts_cache = NUM_EXPERTS
            trainer.global_step = 1
            trainer.all_metrics, trainer.all_timings = {}, {}
            driver_batch = trainer.convert_rollout_groups_to_training_input(groups)
            buffer = None
            try:
                if builder == "verify":
                    buffer = ray.remote(num_cpus=0)(RolloutBuffer).remote(buffer_config)
                    ray.get(buffer.publish.remote(1))
                    payloads = MemoryPayloads()
                    writer = payloads.writer(buffer, content)
                    worker_groups = copy.deepcopy(groups) if corrupt_routes else groups
                    if corrupt_routes:
                        route = worker_groups[0].trajectory_batch["rollout_routed_experts"][0][0, 0]
                        route[0] = next(expert for expert in range(NUM_EXPERTS) if expert not in route)
                    for group in worker_groups:
                        lease = ray.get(buffer.acquire_lease.remote())
                        asyncio.run(writer.write_rollout(lease, group))
                    admission = ray.get(buffer.admit.remote(30))
                    assert admission.selection is not None and len(admission.admitted) == 3
                    metadata = RolloutBatchMetadata(
                        1, admission.admitted, {}, moe_router_replay=True, num_experts=NUM_EXPERTS
                    )
                    plan = plan_batch(metadata, dp_size=2, algorithm=current.trainer.algorithm)
                    reader = RolloutReader(buffer, payloads, metadata.groups, 30)
                    ray.get(policy.async_run_ray_method("pass_through", "load_batch", ray.put(plan), reader))
                    if corrupt_routes:
                        with pytest.raises(WorkerGroupTaskError) as failure:
                            asyncio.run(trainer._run_training(plan, driver_batch=driver_batch))
                        assert "worker forward input digest mismatch" in str(failure.value.__cause__)
                        policy = None
                        return
                    status = asyncio.run(trainer._run_training(plan, driver_batch=driver_batch))
                else:
                    status = asyncio.run(trainer._run_training(driver_batch))
                assert status["router_replay/hit_fraction"] == 1.0
                parameters = {
                    name: sha256(value.contiguous().view(torch.uint8).numpy()).hexdigest()
                    for name, value in rank0_validation_snapshot(policy, parameter_names).items()
                }
                assert parameters.keys() == initial.keys() == set(parameter_names)
                assert parameters != initial
                results[builder] = {
                    "initial": initial,
                    "parameters": parameters,
                    "status": status,
                    "logprobs": driver_batch["action_log_probs"].clone(),
                }
            finally:
                if builder == "verify" and policy is not None:
                    policy.run_method("pass_through", "unload_batch", 1)
                if buffer is not None:
                    ray.kill(buffer)
            policy.kill_actors()
            policy = None
        torch.testing.assert_close(results["verify"]["logprobs"], results["driver"]["logprobs"], rtol=0, atol=0)
        for key in ("policy_loss", "policy_entropy", "raw_grad_norm"):
            assert results["verify"]["status"][key] == results["driver"]["status"][key], key
        assert results["verify"]["parameters"] == results["driver"]["parameters"]
    finally:
        if policy is not None:
            policy.kill_actors()
        ray.util.remove_placement_group(shared_pg)
        ray.shutdown()
