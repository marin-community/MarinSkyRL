import asyncio
from types import SimpleNamespace
import numpy as np
import pytest
import torch
from omegaconf import open_dict

from skyrl_train.batch_assembly import plan_batch
from skyrl_train.distributed.dispatch import MeshRank
from skyrl_train.workers.worker import PolicyWorkerBase
from skyrl_train.config.utils import get_default_config
from skyrl_train.dynamic_sampling import GroupSelectionPolicy
from skyrl_train.group_admission import GroupAdmissionPolicy, GroupAdvantageInvariant, resolve_group_advantage_invariant
from skyrl_train.rollouts.buffer import AdmittedRollout, RolloutContentPolicy, RolloutGroup
from skyrl_train.rollouts.context import RolloutBatchMetadata
from skyrl_train.trainer import RayPPOTrainer
from skyrl_train.trajectory_runners.trajectory_processing import concatenate_trajectory_batches, get_outcome_rewards


@pytest.mark.parametrize("dp_size", [1, 2, 3, 4, 8])
@pytest.mark.parametrize("estimator", ["rloo", "rloo_n", "grpo"])
def test_dp_slices_with_fields_and_longest_rows_on_other_ranks_match_driver(dp_size, estimator):
    cfg = get_default_config()
    cfg.generator.n_samples_per_prompt = 4
    cfg.trainer.algorithm.advantage_estimator = estimator
    cfg.trainer.algorithm.group_advantage_min_size = 2 if estimator == "rloo_n" else None
    cfg.trainer.algorithm.use_kl_loss = False
    cfg.trainer.policy.megatron_config.moe_router_replay = True
    cfg.trainer.algorithm.enable_token_reward_channel = True
    invariant = resolve_group_advantage_invariant(
        advantage_estimator=estimator,
        physical_group_size=4,
        minimum_group_size=cfg.trainer.algorithm.group_advantage_min_size,
    )
    with open_dict(cfg.trainer.algorithm):
        cfg.trainer.algorithm.resolved_group_advantage = invariant.to_config()
    policy = RolloutContentPolicy(
        GroupAdmissionPolicy(
            GroupAdvantageInvariant.from_config(cfg.trainer.algorithm.resolved_group_advantage),
            rollout_logprobs_required=False,
        ),
        GroupSelectionPolicy(None),
    )
    groups, admitted = [], []
    for index in range(6):
        lengths = [1, 2, 3, 9 if index == 5 else 4]
        batch = {
            "prompt_token_ids": [[index + 1] * (7 if index == 5 else 2) for _ in lengths],
            "response_ids": [[10 + index, *([20 + row] * (length - 1))] for row, length in enumerate(lengths)],
            "loss_masks": [[1] * length for length in lengths],
            "rewards": [0.0, 1.0, 3.0, 2.0],
            "is_last_step": [True] * 4,
            "stop_reasons": ["stop", "length", None, "stop"],
            "loop_advantages": [[0.0] * length for length in lengths],
        }
        if index % 2 == 0:
            batch["rewards"] = [[0.0] * (length - 1) + [reward] for length, reward in zip(lengths, batch["rewards"])]
        if index == 0:
            batch["rewards"][1] = [1.0, 0.0]
        if index >= 4:
            batch["rollout_logprobs"] = [np.full(length, -1.25, dtype=np.float32) for length in lengths]
            batch["rollout_routed_experts"] = [np.full((length, 3, 2), index, dtype=np.uint8) for length in lengths]
            batch["token_level_shaping"] = [[-0.5] * length for length in lengths]
            batch["response_span_tags"] = [[2] * length for length in lengths]
            batch["exclude_from_baseline"] = [False, False, True, False]
        group = RolloutGroup(batch, f"g{index}", index % 2, {"uid": f"g{index}"})
        verdict = policy.verdict(group)
        assert verdict.trainable
        groups.append(group)
        admitted.append(AdmittedRollout(index, group.uid, group.policy_step, 4, sum(lengths), verdict.row_facts))
    plan = plan_batch(
        RolloutBatchMetadata(2, tuple(admitted), {}, moe_router_replay=True, num_experts=128),
        dp_size=dp_size,
        algorithm=cfg.trainer.algorithm,
    )
    trainer = RayPPOTrainer.__new__(RayPPOTrainer)
    trainer.cfg = cfg
    trainer.tokenizer = SimpleNamespace(pad_token_id=0)
    trainer.group_advantage_invariant = invariant
    trainer.all_metrics = {}
    trainer._num_experts_cache = 128
    trainer.policy_model = SimpleNamespace(actor_infos=[SimpleNamespace(rank=SimpleNamespace(dp_size=dp_size))])
    trainer.critic_model = trainer.ref_model = None
    uids = list(plan.uids)
    trajectory = concatenate_trajectory_batches(
        [group.trajectory_batch for group in groups],
        tis_lcs_alert_threshold=float(cfg.trainer.algorithm.tis_lcs_alert_threshold),
    )
    trajectory = trainer.postprocess_trajectory_batch(trajectory, uids)
    whole = trainer.convert_to_training_input(
        trajectory,
        uids,
        rollout_staleness=[2 - group.policy_step for group in groups for _ in group.trajectory_batch["response_ids"]],
    )
    whole["values"] = None
    whole = trainer.compute_advantages_and_returns(whole)
    whole = trainer.apply_loop_advantages(whole)
    whole.pop("values")
    whole.metadata.pop("metrics")
    for rank, expected in enumerate(whole.chunk(whole.batch_size // dp_size)):

        async def read(batch_id, indices):
            return [groups[index] for index in indices]

        worker = SimpleNamespace(
            cfg=cfg, mesh_rank=MeshRank(rank, 0, 0, 0, dp_size, dp_size, 1), _pad_token_id=0, _loaded_batches={}
        )
        output = asyncio.run(PolicyWorkerBase.load_batch(worker, plan, SimpleNamespace(selected=admitted, read=read)))
        for index, observation in output.metadata["batch_observations"]:
            original = groups[index].trajectory_batch
            assert get_outcome_rewards(observation) == get_outcome_rewards(original)
            assert [sum(reward) if isinstance(reward, list) else reward for reward in observation["rewards"]] == [
                sum(reward) if isinstance(reward, list) else reward for reward in original["rewards"]
            ]
            assert all(len(reward) <= 1 for reward in observation["rewards"] if isinstance(reward, list))
        actual = worker._loaded_batches[plan.batch_id]
        actual = trainer.apply_loop_advantages(actual)
        assert actual.keys() == expected.keys()
        for key, value in expected.items():
            if value is None:
                assert actual[key] is None
            else:
                assert actual[key] is not None
                assert actual[key].dtype == value.dtype
                torch.testing.assert_close(actual[key], value, rtol=0, atol=0)
                if value.is_floating_point():
                    assert actual[key].numpy().tobytes() == value.numpy().tobytes()
        assert actual.metadata.keys() == expected.metadata.keys()
        for key, value in expected.metadata.items():
            if isinstance(value, np.ndarray):
                np.testing.assert_array_equal(actual.metadata[key], value)
            else:
                assert actual.metadata[key] == value
        assert actual.routed_expert_rows.response_len == expected.routed_expert_rows.response_len
        assert actual.routed_expert_rows.num_experts == expected.routed_expert_rows.num_experts
        for row, reference in zip(actual.routed_expert_rows.rows, expected.routed_expert_rows.rows, strict=True):
            assert row.dtype == reference.dtype
            np.testing.assert_array_equal(row, reference)
        torch.testing.assert_close(actual.routed_experts_tensor(), expected.routed_experts_tensor(), rtol=0, atol=0)
