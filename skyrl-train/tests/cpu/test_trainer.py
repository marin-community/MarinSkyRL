"""
uv  run --isolated --group dev --extra cpu pytest tests/cpu/test_trainer.py
"""

import contextlib
import asyncio
import gc
import weakref
from types import SimpleNamespace

import torch
import pytest
from jaxtyping import Float, Integer
from omegaconf import DictConfig, OmegaConf
from pytest import approx
from unittest.mock import MagicMock, patch


from skyrl_train.distributed.dispatch import MeshRank
from skyrl_train.group_admission import GroupAdvantageInvariant
import skyrl_train.trainer as trainer_module
from skyrl_train.trainer import RayPPOTrainer
from skyrl_train.utils.trainer_utils import ResumeMode
from skyrl_train.utils.policy_losses import ppo_policy_loss
from skyrl_train.training_batch import TrainingBatchIterator, TrainingInputBatch, TrainingOutputBatch
from skyrl_train.models.grug_moe import GrugMoeForCausalLM
from skyrl_train.model_wrapper import HFModelWrapper
from skyrl_train.models.grug_query_bias import (
    GrugQueryBiasCapturePlan,
    GrugQueryBiasShardLayout,
    GrugQueryBiasWindow,
    next_query_bias,
)
import numpy as np
from skyrl_train.workers.worker import CriticWorkerBase, PolicyWorkerBase
from skyrl_train.utils.utils import validate_batch_sizes
from skyrl_train.config.utils import get_default_config
from tests.cpu.util import example_dummy_config
from tests.grug_training_parity import ORACLE_FIXTURE_DIR


@pytest.fixture
def dummy_config():
    return example_dummy_config()


class DummyDataset:
    def __len__(self):
        return 1

    def __getitem__(self, idx):
        return "dummy"

    def collate_fn(self, batch):
        return batch


class _CapturingPolicyGroup:
    def __init__(self):
        self.actor_infos = [SimpleNamespace(rank=SimpleNamespace(dp_size=2)) for _ in range(4)]
        self.training_batch = None

    def async_run_ray_method(self, dispatch_type, method_name, *args):
        del dispatch_type
        if method_name == "ppo_train":
            self.training_batch = args[0]
            return [object()]
        if method_name == "empty_cache":
            return []
        raise AssertionError(f"Unexpected policy method: {method_name}")


class _ResidencyPolicyGroup:
    def __init__(self):
        self.model_on_gpu = False
        self.optimizer_on_gpu = False

    def backload_to_gpu(self, backload_optimizer=True, backload_model=True):
        self.optimizer_on_gpu |= backload_optimizer
        self.model_on_gpu |= backload_model

    def offload_to_cpu(self, offload_optimizer=True, offload_model=True):
        if offload_optimizer:
            self.optimizer_on_gpu = False
        if offload_model:
            self.model_on_gpu = False


class _ResidencyInferenceClient:
    def __init__(self):
        self.awake = True
        self.wake_tags = []

    async def sleep(self):
        self.awake = False

    async def wake_up(self, tags):
        self.wake_tags.append(tags)
        self.awake = True


@pytest.mark.parametrize("save_error", [None, RuntimeError("storage of size 0")])
def test_colocated_checkpoint_temporarily_backloads_policy_and_restores_rollout_residency(save_error, monkeypatch):
    trainer = RayPPOTrainer.__new__(RayPPOTrainer)
    trainer.colocate_all = True
    trainer.policy_model = _ResidencyPolicyGroup()
    trainer.inference_engine_client = _ResidencyInferenceClient()
    trainer.sync_policy_weights_to_inference_engines = lambda: []
    trainer.all_timings = {}
    monkeypatch.setattr(trainer_module.ray, "get", lambda refs: refs)
    save_observations = []

    def save_checkpoints():
        save_observations.append(
            (
                trainer.policy_model.model_on_gpu,
                trainer.policy_model.optimizer_on_gpu,
                trainer.inference_engine_client.awake,
            )
        )
        if save_error is not None:
            raise save_error

    trainer.save_checkpoints = save_checkpoints

    if save_error is None:
        asyncio.run(trainer._save_checkpoints_with_residency())
    else:
        with pytest.raises(RuntimeError, match="storage of size 0"):
            asyncio.run(trainer._save_checkpoints_with_residency())

    assert save_observations == [(True, True, False)]
    assert not trainer.policy_model.model_on_gpu
    assert not trainer.policy_model.optimizer_on_gpu
    assert trainer.inference_engine_client.awake
    assert trainer.inference_engine_client.wake_tags == [["weights"], ["kv_cache"]]


def test_sync_trainer_attaches_global_loss_denominator_before_dispatch(monkeypatch):
    trainer = object.__new__(RayPPOTrainer)
    trainer.cfg = OmegaConf.create(
        {"trainer": {"algorithm": {"loss_reduction": "seq_mean_token_sum_norm_global", "max_seq_len": 8}}}
    )
    trainer.global_step = 3
    trainer.all_metrics = {}
    trainer.all_timings = {}
    trainer.colocate_all = False
    trainer.critic_model = None
    trainer.policy_model = _CapturingPolicyGroup()

    status = TrainingOutputBatch()
    status.metadata = {"train_status": {}}
    monkeypatch.setattr(trainer_module, "collect_actor_results", lambda *args, **kwargs: [status])
    monkeypatch.setattr(trainer_module.ray, "get", lambda refs: refs)

    batch = TrainingInputBatch({"advantages": torch.tensor([[1.0, 0.0], [0.0, 2.0]])})
    batch.metadata = {}

    trainer.train_critic_and_policy(batch)

    assert trainer.policy_model.training_batch.metadata["global_loss_denom"] == 32.0


@pytest.fixture
def dummy_tokenizer():
    mock_tokenizer = MagicMock()
    mock_tokenizer.pad_token_id = 0
    mock_tokenizer.eos_token_id = 2

    # encode("abc") -> [97, 98, 99]
    mock_tokenizer.encode.side_effect = lambda x: [ord(c) for c in x]

    # tokenizer("abc") -> {"input_ids": [...], "attention_mask": [...]}
    def fake_tokenizer_call(text, **kwargs):
        ids = [ord(c) for c in text]
        return {
            "input_ids": ids,
            "attention_mask": [1] * len(ids),
        }

    mock_tokenizer.side_effect = fake_tokenizer_call

    return mock_tokenizer


@pytest.fixture
def dummy_trajectory_runner():
    return MagicMock()


class _ObservableGrugCausalLM(GrugMoeForCausalLM):
    def __init__(self):
        self.config = SimpleNamespace(
            num_experts_per_tok=2,
            num_local_experts=4,
            num_hidden_layers=1,
        )
        self.query_bias = torch.tensor([[3.0, -3.0]])

    def set_query_bias(self, query_bias):
        self.query_bias = query_bias.clone()


class _FixedQueryBiasAccumulator:
    def __init__(self, betas):
        self.betas = betas

    def finalize_betas(self):
        return self.betas


def _window_with_grug_query_bias_accumulator(accumulator):
    causal_lm = _ObservableGrugCausalLM()
    shard_layout = GrugQueryBiasShardLayout(micro_batch_size=1, accumulation_steps=1, ep_size=1, ep_rank=0)
    capture_plan = GrugQueryBiasCapturePlan.build(torch.ones((1, 1)), shard_layout)
    window = GrugQueryBiasWindow(causal_lm, valid_tokens=1, capture_plan=capture_plan)
    window.accumulator = accumulator
    return window, causal_lm


def _grug_ppo_worker_and_batch(
    cfg: DictConfig,
    causal_lm: GrugMoeForCausalLM,
    sequences: torch.Tensor,
) -> tuple[PolicyWorkerBase, TrainingInputBatch]:
    batch_size = sequences.shape[0]
    batch = TrainingInputBatch(
        {
            "sequences": sequences,
            "attention_mask": torch.ones_like(sequences),
            "action_log_probs": torch.zeros(batch_size, 2),
            "base_action_log_probs": torch.zeros(batch_size, 2),
            "values": torch.zeros(batch_size, 2),
            "returns": torch.zeros(batch_size, 2),
            "advantages": torch.ones(batch_size, 2),
            "loss_mask": torch.ones(batch_size, 2),
            "response_mask": torch.ones(batch_size, 2),
            "rollout_logprobs": None,
        }
    )
    batch.metadata = {"global_step": 0, "response_length": 2}

    worker = PolicyWorkerBase(
        cfg=cfg,
        world_size=1,
        rank=0,
        local_rank=0,
        master_addr="localhost",
        master_port=12345,
        sequence_parallel_size=1,
    )
    worker.strategy = MagicMock(fsdp_strategy="fsdp2")
    worker.strategy.is_rank_0.return_value = False
    worker.strategy.all_reduce.side_effect = lambda status: status
    worker.model = SimpleNamespace(model=causal_lm)
    return worker, batch


def _run_grug_ppo_train(worker: PolicyWorkerBase, batch: TrainingInputBatch) -> None:
    with (
        patch("torch.cuda.empty_cache"),
        patch("torch.cuda.current_device", return_value="cpu"),
        patch("torch.autocast", side_effect=lambda *args, **kwargs: contextlib.nullcontext()),
        patch("torch.distributed.barrier"),
        patch("tqdm.tqdm", side_effect=lambda iterator, **kwargs: iterator),
    ):
        worker.ppo_train(batch)


class _CpuPolicyStrategy:
    """Exercise the policy worker while replacing only its distributed/CUDA adapter."""

    device_mesh = None
    ep_size = 1
    last_optimizer_step_succeeded = True

    def is_rank_0(self):
        return False

    def all_reduce(self, value, op="mean"):
        return value

    def backward(self, loss, model, optimizer):
        loss.backward()

    def optimizer_step(self, optimizer, model, scheduler, **kwargs):
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)
        return torch.tensor(0.0)


def _enable_cpu_policy_training(worker: PolicyWorkerBase, causal_lm: GrugMoeForCausalLM) -> None:
    worker.model = HFModelWrapper(causal_lm, bf16=False, training_strategy="fsdp2")
    worker.strategy = _CpuPolicyStrategy()
    worker.optimizer = torch.optim.AdamW(worker.model.parameters(), lr=1e-4)
    worker.scheduler = torch.optim.lr_scheduler.LambdaLR(worker.optimizer, lambda _: 1.0)


def test_failed_optimizer_step_discards_grug_query_bias_window():
    accumulator = _FixedQueryBiasAccumulator(torch.tensor([[1.0, -2.0]]))
    window, causal_lm = _window_with_grug_query_bias_accumulator(accumulator)
    previous_bias = causal_lm.query_bias.clone()

    window.finish(optimizer_step_succeeded=False)
    window.finish(optimizer_step_succeeded=True)

    torch.testing.assert_close(causal_lm.query_bias, previous_bias)


def test_successful_step_applies_grug_query_bias_once():
    betas = torch.tensor([[1.0, -2.0]])
    accumulator = _FixedQueryBiasAccumulator(betas)
    window, causal_lm = _window_with_grug_query_bias_accumulator(accumulator)

    window.finish(optimizer_step_succeeded=True)

    torch.testing.assert_close(causal_lm.query_bias, next_query_bias(betas))
    causal_lm.query_bias.fill_(17)
    window.finish(optimizer_step_succeeded=True)
    torch.testing.assert_close(causal_lm.query_bias, torch.full_like(causal_lm.query_bias, 17))


def test_grug_query_bias_virtual_shards_partition_optimizer_window():
    attention_mask = torch.tensor(
        [
            [1, 1, 0],
            [1, 0, 0],
            [1, 1, 1],
            [0, 1, 1],
        ]
    )
    microbatches = attention_mask.split(2)

    rank_masks = []
    for ep_rank in range(2):
        shard_layout = GrugQueryBiasShardLayout(
            micro_batch_size=2,
            accumulation_steps=2,
            ep_size=2,
            ep_rank=ep_rank,
        )
        capture_plan = GrugQueryBiasCapturePlan.build(attention_mask, shard_layout)
        assert capture_plan.valid_token_counts == ((3, 0), (0, 5))[ep_rank]
        rank_masks.append(
            torch.cat([shard_layout.mask_for(mask, local_step) for local_step, mask in enumerate(microbatches)])
        )

    torch.testing.assert_close(rank_masks[0].logical_xor(rank_masks[1]), attention_mask.bool())
    assert not torch.logical_and(rank_masks[0], rank_masks[1]).any()
    assert rank_masks[0].sum().item() == 3
    assert rank_masks[1].sum().item() == 5
    single_rank_layout = GrugQueryBiasShardLayout(
        micro_batch_size=4,
        accumulation_steps=1,
        ep_size=1,
        ep_rank=0,
    )
    torch.testing.assert_close(
        single_rank_layout.mask_for(attention_mask, local_step=0),
        attention_mask.bool(),
    )


def _get_test_data(trainer: RayPPOTrainer):
    trainer.critic_model = MagicMock()  # pretend we're using a critic

    batch_size = 2
    total_seq_len = 5
    action_len = 3

    # Create test data
    ret_sequences: Float[torch.Tensor, "batch_size total_seq_len"] = torch.randint(0, 1000, (batch_size, total_seq_len))
    ret_attention_masks: Float[torch.Tensor, "batch_size total_seq_len"] = torch.ones((batch_size, total_seq_len))
    ret_loss_masks: Integer[torch.Tensor, "batch_size total_seq_len"] = torch.stack(
        [torch.tensor([1, 1, 0, 0, 0], dtype=torch.int32), torch.tensor([1, 1, 1, 0, 0], dtype=torch.int32)], dim=0
    )
    base_log_probs: Float[torch.Tensor, "batch_size total_seq_len"] = torch.log(
        torch.tensor([[0.1, 0.2, 0.3, 0.2, 0.2], [0.25, 0.25, 0.25, 0.15, 0.10]])
    )
    action_log_probs: Float[torch.Tensor, "batch_size total_seq_len"] = torch.log(
        torch.tensor([[0.1, 0.3, 0.2, 0.2, 0.2], [0.3, 0.3, 0.2, 0.1, 0.1]])
    )
    action_masks: Integer[torch.Tensor, "batch_size total_seq_len"] = torch.stack(
        [torch.tensor([1, 1, 1, 0, 0], dtype=torch.int32), torch.tensor([1, 1, 1, 1, 1], dtype=torch.int32)], dim=0
    )
    actual_response_lengths: Float[torch.Tensor, "batch_size"] = action_masks.sum(dim=-1).to(float)
    rewards_all: Float[torch.Tensor, "batch_size total_seq_len"] = torch.stack(
        [torch.tensor([0.0, 1.0, 0.0, 0.0, 0.0]), torch.tensor([0.0, 0.0, 1.0, 0.0, 0.0])], dim=0
    )
    values: Float[torch.Tensor, "batch_size action_len"] = torch.randn(batch_size, action_len)
    uids: np.ndarray[str] = np.array(["0", "0"])

    # Run method
    data = TrainingInputBatch(
        {
            "sequences": ret_sequences,
            "attention_mask": ret_attention_masks,
            "loss_mask": ret_loss_masks,
            "base_action_log_probs": base_log_probs,
            "action_log_probs": action_log_probs,
            "response_mask": action_masks,
            "rewards": rewards_all,
            "values": values,
        },
    )
    data.metadata = {
        "uids": uids,
        "response_length": action_len,
        "avg_response_length": actual_response_lengths.mean().item(),
    }
    data = trainer.apply_reward_kl_penalty(data)

    return data


def test_load_checkpoints_preserves_cloud_resume_uri(dummy_config):
    resume_path = "s3://marin-us-east-02a/iris/test/checkpoints/global_step_12"
    dummy_config.trainer.resume_path = resume_path

    trainer = RayPPOTrainer.__new__(RayPPOTrainer)
    trainer.cfg = dummy_config
    trainer.resume_mode = ResumeMode.FROM_PATH

    with patch("skyrl_train.trainer.io.exists", return_value=False) as exists:
        with pytest.raises(FileNotFoundError, match="Checkpoint path not found"):
            trainer.load_checkpoints()

    exists.assert_called_once_with(resume_path)


def test_load_checkpoints_accepts_trailing_slash_resume_path(dummy_config):
    resume_path = "s3://marin-us-east-02a/iris/test/checkpoints/global_step_12/"
    dummy_config.trainer.resume_path = resume_path

    trainer = RayPPOTrainer.__new__(RayPPOTrainer)
    trainer.cfg = dummy_config
    trainer.resume_mode = ResumeMode.FROM_PATH

    with patch("skyrl_train.trainer.io.exists", return_value=False) as exists:
        with pytest.raises(FileNotFoundError, match="Checkpoint path not found"):
            trainer.load_checkpoints()

    exists.assert_called_once_with(resume_path.rstrip("/"))


def test_calculate_kl_create_experience_batched(dummy_config, dummy_trajectory_runner):
    trainer = RayPPOTrainer(
        cfg=dummy_config,
        tracker=None,
        tokenizer=None,
        train_dataset=DummyDataset(),
        eval_dataset=DummyDataset(),
        inference_engine_client=None,
        trajectory_runner=dummy_trajectory_runner,
    )
    data = _get_test_data(trainer)
    # Assertions
    metrics = data.metadata["metrics"]
    assert metrics["avg_kl_max"] == approx(0.3143, abs=1e-4)
    # Note; the raw KL mean is 0.054, but then the masked mean is different.
    assert metrics["avg_kl"] == approx(0.1249, abs=1e-4)


@patch("skyrl_train.trainer.compute_advantages_and_returns", new_callable=MagicMock)
def test_calc_advantages_and_returns(mock_compute_adv_and_ret, dummy_config, dummy_trajectory_runner):
    trainer = RayPPOTrainer(
        cfg=dummy_config,
        tracker=None,
        tokenizer=None,
        train_dataset=DummyDataset(),
        eval_dataset=DummyDataset(),
        inference_engine_client=None,
        trajectory_runner=dummy_trajectory_runner,
    )
    data = _get_test_data(trainer)

    # Mocked return values
    mock_advantages = torch.tensor([[0.1, 0.2, 0.3, 0.4, 0.5], [0.6, 0.7, 0.8, 0.9, 1.0]])
    mock_returns = torch.tensor([[0.6, 0.7, 0.8, 0.9, 1.0], [1.1, 1.2, 1.3, 1.4, 1.5]])

    # Set up mocks
    mock_compute_adv_and_ret.return_value = (mock_advantages, mock_returns)

    # Run the method
    data = trainer.compute_advantages_and_returns(data)
    metrics = data.metadata["metrics"]

    # Assertions
    assert torch.allclose(data["advantages"], mock_advantages)
    assert torch.allclose(data["returns"], mock_returns)
    assert isinstance(metrics, dict)
    assert "avg_final_rewards" in metrics
    assert "avg_response_length" in metrics
    assert "avg_advantages_abs" in metrics
    assert metrics["avg_advantages"] == approx(
        torch.masked_select(mock_advantages, data["response_mask"].bool()).mean().item(), rel=1e-5
    )


@pytest.mark.parametrize(
    ("loss_reduction", "expected_policy_loss"),
    [("token_mean", 0.05), ("sequence_mean", 0.04375)],
)
def test_grpo_loop_credit_is_token_local_when_every_group_member_has_the_same_outcome(
    loss_reduction, expected_policy_loss
):
    response_length = 48
    trainer = RayPPOTrainer.__new__(RayPPOTrainer)
    trainer.cfg = OmegaConf.create(
        {
            "trainer": {
                "step_wise_training": False,
                "algorithm": {
                    "advantage_estimator": "grpo",
                    "gamma": 1.0,
                    "lambd": 1.0,
                    "grpo_norm_by_std": True,
                    "policy_loss_type": "regular",
                    "loss_reduction": loss_reduction,
                    "eps_clip_low": 0.2,
                    "eps_clip_high": 0.2,
                    "use_tis": False,
                    "max_seq_len": response_length,
                },
            }
        }
    )
    trainer.group_advantage_invariant = GroupAdvantageInvariant.exact_physical(physical_group_size=4)
    trainer.all_metrics = {}
    loop_start = 18
    response_mask = torch.ones(4, response_length)
    response_mask[2:, 24:] = 0
    loop_advantages = torch.zeros(4, response_length)
    loop_advantages[:, loop_start:] = -0.1
    loop_advantages *= response_mask
    data = TrainingInputBatch(
        {
            "rewards": torch.zeros(4, response_length),
            "response_mask": response_mask,
            "values": None,
            "loop_advantages": loop_advantages,
        }
    )
    data.metadata = {
        "uids": ["same-group"] * 4,
        "avg_response_length": float(response_length),
    }

    result = trainer.compute_advantages_and_returns(data)
    result = trainer_module.normalize_advantages_dict(result)
    result = trainer.apply_loop_advantages(result)

    assert torch.equal(result["advantages"], loop_advantages)
    assert torch.equal(result["returns"], torch.zeros(4, response_length))
    policy_loss, _ = ppo_policy_loss(
        torch.zeros_like(loop_advantages),
        torch.zeros_like(loop_advantages),
        result["advantages"],
        config=trainer.cfg.trainer.algorithm,
        loss_mask=response_mask,
    )
    assert policy_loss.item() == pytest.approx(expected_policy_loss)


def test_loop_advantages_are_collated_with_response_tokens(dummy_config, dummy_tokenizer):
    trainer = RayPPOTrainer.__new__(RayPPOTrainer)
    trainer.cfg = dummy_config
    trainer.group_advantage_invariant = GroupAdvantageInvariant.no_group_advantage(physical_group_size=1)
    trainer.tokenizer = dummy_tokenizer
    trainer.pad_batch = lambda batch: batch
    trajectory_batch = {
        "prompt_token_ids": [[1, 2], [3]],
        "response_ids": [[4, 5, 6], [7]],
        "rewards": [[0.0, 0.0, 0.0], [0.0]],
        "loss_masks": [[1, 1, 1], [1]],
        "rollout_logprobs": None,
        "loop_advantages": [[0.0, -0.1, -0.1], [-0.2]],
    }

    batch = trainer.convert_to_training_input(trajectory_batch, ["a", "b"])

    torch.testing.assert_close(
        batch["loop_advantages"],
        torch.tensor([[0.0, -0.1, -0.1], [-0.2, 0.0, 0.0]]),
    )


def test_normalize_mini_batch_size():
    """Test the _normalize_mini_batch_size method with various configurations."""

    # Create minimal worker instances for testing
    class TestPolicyWorker(PolicyWorkerBase):
        def init_model(self, *args, **kwargs):
            pass

        def offload_to_cpu(self, pin_memory=True, non_blocking=True):
            pass

        def backload_to_gpu(self, non_blocking=True):
            pass

        def _forward_micro_batch(self, micro_batch):
            pass

    class TestCriticWorker(CriticWorkerBase):
        def init_model(self, *args, **kwargs):
            pass

        def offload_to_cpu(self, pin_memory=True, non_blocking=True):
            pass

        def backload_to_gpu(self, non_blocking=True):
            pass

        def _forward_micro_batch(self, micro_batch):
            pass

    def create_policy_worker_with_config(
        train_batch_size, policy_mini_batch_size, micro_train_batch_size_per_gpu, n_samples_per_prompt, dp_size
    ):
        """Helper to create policy worker with specific config."""
        cfg = OmegaConf.create(
            {
                "trainer": {
                    "train_batch_size": train_batch_size,
                    "policy_mini_batch_size": policy_mini_batch_size,
                    "micro_train_batch_size_per_gpu": micro_train_batch_size_per_gpu,
                    "algorithm": {
                        "batch_invariant": False,
                        "policy_loss_type": "regular",
                    },
                },
                "generator": {
                    "n_samples_per_prompt": n_samples_per_prompt,
                },
            }
        )

        worker = TestPolicyWorker(
            cfg=cfg,
            world_size=dp_size,
            rank=0,
            local_rank=0,
            master_addr="localhost",
            master_port=12345,
            sequence_parallel_size=1,
        )

        # Mock mesh_rank
        worker.mesh_rank = MeshRank(dp=0, sp=0, tp=0, pp=0, world_size=dp_size, dp_size=dp_size, pp_size=1)

        return worker

    def create_critic_worker_with_config(
        train_batch_size, critic_mini_batch_size, micro_train_batch_size_per_gpu, n_samples_per_prompt, dp_size
    ):
        """Helper to create critic worker with specific config."""
        cfg = OmegaConf.create(
            {
                "trainer": {
                    "train_batch_size": train_batch_size,
                    "critic_mini_batch_size": critic_mini_batch_size,
                    "micro_train_batch_size_per_gpu": micro_train_batch_size_per_gpu,
                    "algorithm": {"batch_invariant": False},
                },
                "generator": {
                    "n_samples_per_prompt": n_samples_per_prompt,
                },
            }
        )

        worker = TestCriticWorker(
            cfg=cfg,
            world_size=dp_size,
            rank=0,
            local_rank=0,
            master_addr="localhost",
            master_port=12345,
            sequence_parallel_size=1,
        )

        # Mock mesh_rank
        worker.mesh_rank = MeshRank(dp=0, sp=0, tp=0, pp=0, world_size=dp_size, dp_size=dp_size, pp_size=1)

        return worker

    # Test Case 1: Basic valid configuration for PolicyWorker
    policy_worker = create_policy_worker_with_config(
        train_batch_size=128,
        policy_mini_batch_size=16,
        micro_train_batch_size_per_gpu=2,
        n_samples_per_prompt=2,
        dp_size=4,
    )
    policy_worker._normalize_mini_batch_size()

    expected_policy_mini_batch_size_per_gpu = (16 * 2) // 4  # 8
    assert policy_worker.policy_mini_batch_size_per_gpu == expected_policy_mini_batch_size_per_gpu

    # Test Case 2: Basic valid configuration for CriticWorker
    critic_worker = create_critic_worker_with_config(
        train_batch_size=128,
        critic_mini_batch_size=8,
        micro_train_batch_size_per_gpu=2,
        n_samples_per_prompt=2,
        dp_size=4,
    )
    critic_worker._normalize_mini_batch_size()

    expected_critic_mini_batch_size_per_gpu = (8 * 2) // 4  # 4
    assert critic_worker.critic_mini_batch_size_per_gpu == expected_critic_mini_batch_size_per_gpu

    # Test Case 3: Single GPU (dp_size=1) for PolicyWorker
    policy_worker = create_policy_worker_with_config(
        train_batch_size=32,
        policy_mini_batch_size=8,
        micro_train_batch_size_per_gpu=4,
        n_samples_per_prompt=1,
        dp_size=1,
    )
    policy_worker._normalize_mini_batch_size()

    expected_policy_mini_batch_size_per_gpu = (8 * 1) // 1  # 8
    assert policy_worker.policy_mini_batch_size_per_gpu == expected_policy_mini_batch_size_per_gpu

    # Test Case 4: High n_samples_per_prompt for CriticWorker
    critic_worker = create_critic_worker_with_config(
        train_batch_size=256,
        critic_mini_batch_size=32,
        micro_train_batch_size_per_gpu=8,
        n_samples_per_prompt=4,
        dp_size=2,
    )
    critic_worker._normalize_mini_batch_size()

    expected_critic_mini_batch_size_per_gpu = (32 * 4) // 2  # 64
    assert critic_worker.critic_mini_batch_size_per_gpu == expected_critic_mini_batch_size_per_gpu

    # Test Case 5: Error case - mesh_rank not initialized
    policy_worker_no_mesh = create_policy_worker_with_config(
        train_batch_size=128,
        policy_mini_batch_size=16,
        micro_train_batch_size_per_gpu=2,
        n_samples_per_prompt=1,
        dp_size=4,
    )
    policy_worker_no_mesh.mesh_rank = None

    with pytest.raises(RuntimeError, match="mesh_rank must be initialized"):
        policy_worker_no_mesh._normalize_mini_batch_size()


def test_validate_batch_sizes():
    """Test the validate_batch_sizes function with various configurations to trigger all error cases."""

    def create_test_config(
        train_batch_size=128,
        policy_mini_batch_size=16,
        critic_mini_batch_size=8,
        micro_train_batch_size_per_gpu=2,
        micro_forward_batch_size_per_gpu=4,
        n_samples_per_prompt=2,
        policy_num_nodes=1,
        policy_num_gpus_per_node=4,
        critic_num_nodes=1,
        critic_num_gpus_per_node=4,
        policy_sequence_parallel_size=1,
        critic_sequence_parallel_size=1,
        critic_model_path=None,
    ):
        """Helper to create config for validation testing."""
        cfg = get_default_config()
        cfg.trainer.train_batch_size = train_batch_size
        cfg.trainer.policy_mini_batch_size = policy_mini_batch_size
        cfg.trainer.critic_mini_batch_size = critic_mini_batch_size
        cfg.trainer.micro_train_batch_size_per_gpu = micro_train_batch_size_per_gpu
        cfg.trainer.micro_forward_batch_size_per_gpu = micro_forward_batch_size_per_gpu
        cfg.trainer.placement.policy_num_nodes = policy_num_nodes
        cfg.trainer.placement.policy_num_gpus_per_node = policy_num_gpus_per_node
        cfg.trainer.placement.critic_num_nodes = critic_num_nodes
        cfg.trainer.placement.critic_num_gpus_per_node = critic_num_gpus_per_node
        cfg.trainer.policy.sequence_parallel_size = policy_sequence_parallel_size
        cfg.trainer.critic.model.path = critic_model_path
        cfg.trainer.critic.sequence_parallel_size = critic_sequence_parallel_size
        cfg.trainer.algorithm.use_kl_loss = False
        cfg.trainer.algorithm.use_kl_in_reward = False
        cfg.generator.n_samples_per_prompt = n_samples_per_prompt
        return cfg

    # Test Case 1: Valid configuration
    cfg = create_test_config()
    validate_batch_sizes(cfg)  # Should not raise any exceptions

    # Test Case 2: Error case - train_batch_size < policy_mini_batch_size
    cfg = create_test_config(train_batch_size=8, policy_mini_batch_size=16)
    with pytest.raises(AssertionError):
        validate_batch_sizes(cfg)

    # Test Case 3: Error case - train_batch_size < critic_mini_batch_size
    cfg = create_test_config(train_batch_size=4, critic_mini_batch_size=8)
    with pytest.raises(AssertionError):
        validate_batch_sizes(cfg)

    # Test Case 4: Error case - policy_mini_batch_size = 0
    cfg = create_test_config(policy_mini_batch_size=0)
    with pytest.raises(AssertionError, match="policy_mini_batch_size must be greater than 0"):
        validate_batch_sizes(cfg)

    # Test Case 5: Error case - critic_mini_batch_size = 0
    cfg = create_test_config(critic_mini_batch_size=0, critic_model_path="test")
    with pytest.raises(AssertionError, match="critic_mini_batch_size must be greater than 0"):
        validate_batch_sizes(cfg)

    # Test Case 6: Error case - micro_train_batch_size_per_gpu = 0
    cfg = create_test_config(micro_train_batch_size_per_gpu=0)
    with pytest.raises(AssertionError, match="micro_train_batch_size_per_gpu must be greater than 0"):
        validate_batch_sizes(cfg)

    # Test Case 7: Error case - micro_forward_batch_size_per_gpu = 0
    cfg = create_test_config(micro_forward_batch_size_per_gpu=0)
    with pytest.raises(AssertionError, match="micro_forward_batch_size_per_gpu must be greater than 0"):
        validate_batch_sizes(cfg)

    # Test Case 8: Error case - train_batch_size not divisible by (policy_mini_batch_size * policy_dp_size)
    cfg = create_test_config(train_batch_size=100, policy_mini_batch_size=16, policy_num_gpus_per_node=4)
    # Should fail because train_batch_size is not evenly divisible by policy batch requirements
    with pytest.raises(AssertionError, match="train_batch_size .* should be divisible by policy_mini_batch_size"):
        validate_batch_sizes(cfg)

    # Test Case 9: Error case - train_batch_size not divisible by (critic_mini_batch_size * critic_dp_size)
    cfg = create_test_config(
        train_batch_size=100,
        policy_mini_batch_size=5,
        critic_mini_batch_size=16,
        critic_num_gpus_per_node=4,
        critic_model_path="test",
    )
    # Should fail because train_batch_size is not evenly divisible by critic batch requirements
    with pytest.raises(AssertionError, match="train_batch_size .* should be divisible by critic_mini_batch_size"):
        validate_batch_sizes(cfg)

    # Test Case 10: Error case - policy_mini_batch_size_per_gpu not divisible by micro_train_batch_size_per_gpu
    cfg = create_test_config(
        policy_mini_batch_size=8, n_samples_per_prompt=1, policy_num_gpus_per_node=1, micro_train_batch_size_per_gpu=3
    )
    # Should fail because policy mini batch per GPU is not evenly divisible by micro batch size
    with pytest.raises(
        AssertionError,
        match="normalized policy_mini_batch_size_per_gpu .* should be divisible by micro_train_batch_size_per_gpu",
    ):
        validate_batch_sizes(cfg)

    # Test Case 11: Error case - critic_mini_batch_size_per_gpu not divisible by micro_train_batch_size_per_gpu
    cfg = create_test_config(
        train_batch_size=144,
        policy_mini_batch_size=12,  # Policy validation passes
        critic_mini_batch_size=8,  # Critic micro batch divisibility fails
        n_samples_per_prompt=1,
        critic_num_gpus_per_node=1,
        micro_train_batch_size_per_gpu=3,
        critic_model_path="test",
    )
    # Should fail because critic mini batch per GPU is not evenly divisible by micro batch size
    with pytest.raises(
        AssertionError,
        match="normalized critic_mini_batch_size_per_gpu .* should be divisible by micro_train_batch_size_per_gpu",
    ):
        validate_batch_sizes(cfg)

    # Test Case 12: Valid configuration with sequence parallelism
    cfg = create_test_config(
        policy_sequence_parallel_size=2,
        critic_sequence_parallel_size=2,
        policy_num_gpus_per_node=8,
        critic_num_gpus_per_node=8,
    )
    validate_batch_sizes(cfg)  # Should not raise any exceptions

    # Test Case 13: Valid configuration - train_batch_size not divisible by (critic_mini_batch_size * critic_dp_size), but critic model path is None
    cfg = create_test_config(
        train_batch_size=100,
        policy_mini_batch_size=5,
        critic_mini_batch_size=16,
        critic_num_gpus_per_node=4,
        critic_model_path=None,
    )
    validate_batch_sizes(cfg)

    # Test Case 14: Valid configuration - critic_mini_batch_size is invalid but critic model is not specified
    cfg = create_test_config(critic_mini_batch_size=0, critic_model_path=None)
    validate_batch_sizes(cfg)

    # Test Case 15: Error case - train_batch_size_per_gpu not divisible by policy_mini_batch_size_per_gpu
    cfg = create_test_config(
        train_batch_size=10,
        policy_mini_batch_size=5,
        policy_num_gpus_per_node=2,
        micro_train_batch_size_per_gpu=1,
        n_samples_per_prompt=1,
    )
    with pytest.raises(
        AssertionError, match="policy_train_batch_size_per_gpu .* should be divisible by policy_mini_batch_size_per_gpu"
    ):
        validate_batch_sizes(cfg)

    # Test Case 16: Error case - train_batch_size_per_gpu not divisible by critic_mini_batch_size_per_gpu
    cfg = create_test_config(
        train_batch_size=10,
        policy_mini_batch_size=10,
        policy_num_gpus_per_node=1,
        critic_mini_batch_size=5,
        critic_num_gpus_per_node=2,
        micro_train_batch_size_per_gpu=1,
        n_samples_per_prompt=1,
        critic_model_path="test",
    )
    with pytest.raises(
        AssertionError, match="critic_train_batch_size_per_gpu .* should be divisible by critic_mini_batch_size_per_gpu"
    ):
        validate_batch_sizes(cfg)


def test_ppo_train_batch_calculations():
    """Test the key batch calculations and control flow in ppo_train methods."""

    # Create test configuration
    cfg = OmegaConf.create(
        {
            "trainer": {
                "micro_train_batch_size_per_gpu": 2,
                "update_epochs_per_batch": 1,
                "policy": {
                    "grug_query_bias_update_mode": "frozen",
                    "optimizer_config": {"max_grad_norm": 1.0},
                },
                "algorithm": {
                    "batch_invariant": False,
                    "policy_loss_type": "regular",
                    "loss_reduction": "token_mean",
                },
            },
            "generator": {
                "sampling_params": {
                    "temperature": 1.0,
                },
            },
        }
    )

    # Create dummy databatch with known size
    batch_size = 12  # This will create 6 micro batches with micro_train_batch_size_per_gpu=2
    response_length = 4  # number of actions
    dummy_databatch = TrainingInputBatch(
        {
            "sequences": torch.randint(0, 100, (batch_size, 10)),  # dummy token sequences
            "attention_mask": torch.ones(batch_size, 10),
            "action_log_probs": torch.randn(batch_size, response_length),
            "base_action_log_probs": torch.randn(batch_size, response_length),
            "values": torch.randn(batch_size, response_length),
            "returns": torch.randn(batch_size, response_length),
            "advantages": torch.randn(batch_size, response_length),
            "loss_mask": torch.ones(batch_size, response_length),
            "response_mask": torch.ones(batch_size, response_length),
            "rollout_logprobs": None,
        },
    )
    dummy_databatch.metadata = {"global_step": 0, "response_length": response_length}

    # Helper function to create worker with minimal setup
    def create_test_worker(worker_class):
        worker = worker_class(
            cfg=cfg,
            world_size=1,
            rank=0,
            local_rank=0,
            master_addr="localhost",
            master_port=12345,
            sequence_parallel_size=1,
        )
        # Set appropriate mini batch size per gpu based on worker type
        if worker_class == PolicyWorkerBase:
            worker.policy_mini_batch_size_per_gpu = 6  # Should result in 3 micro batches per mini batch
        elif worker_class == CriticWorkerBase:
            worker.critic_mini_batch_size_per_gpu = 6  # Should result in 3 micro batches per mini batch

        # Mock dependencies
        worker.strategy = MagicMock()
        worker.strategy.is_rank_0.return_value = False  # Disable progress bars
        worker.strategy.all_reduce.side_effect = lambda status, *args, **kwargs: status

        # Always set model for all worker types (policy/critic need this for ppo_train)
        worker.model = MagicMock()

        return worker

    # Test PolicyWorkerBase
    policy_worker = create_test_worker(PolicyWorkerBase)

    # Mock training_step to track calls and verify accumulation behavior
    policy_training_calls = []

    def mock_policy_training_step(experience, global_step, local_step, accumulation_steps):
        policy_training_calls.append({"local_step": local_step, "accumulation_steps": accumulation_steps})
        return {
            "policy_loss": 0.5,
            "policy_lr": 1e-4,
            "policy_entropy": 2.0,
            "response_length": response_length,
        }

    policy_worker.training_step = mock_policy_training_step

    # Calculate expected values based on new accumulation logic
    dataloader = TrainingBatchIterator(dummy_databatch, cfg.trainer.micro_train_batch_size_per_gpu)
    total_micro_batches = len(dataloader)  # Should be 6
    micro_batches_per_mini_batch = (
        policy_worker.policy_mini_batch_size_per_gpu // cfg.trainer.micro_train_batch_size_per_gpu
    )  # 6 // 2 = 3
    # New logic: accumulation_steps = micro_batches_per_mini_batch (accumulate within mini-batch)
    expected_accumulation_steps = micro_batches_per_mini_batch  # Should be 3

    # Run policy ppo_train with minimal mocking
    with (
        patch("torch.distributed.barrier"),
        patch("tqdm.tqdm", side_effect=lambda x, **kwargs: x),
    ):  # Disable progress bar
        result = policy_worker.ppo_train(dummy_databatch)

    # Verify Policy Worker Results
    assert len(policy_training_calls) == total_micro_batches, (
        f"PolicyWorker: Expected {total_micro_batches} training_step calls, got {len(policy_training_calls)}"
    )

    # Verify accumulation_steps are consistent (should equal micro_batches_per_mini_batch)
    for call in policy_training_calls:
        assert call["accumulation_steps"] == expected_accumulation_steps, (
            f"PolicyWorker: Expected accumulation_steps={expected_accumulation_steps}, got {call['accumulation_steps']}"
        )

    # Verify no early termination (all micro batches processed)
    expected_local_steps = list(range(total_micro_batches))
    actual_local_steps = [call["local_step"] for call in policy_training_calls]
    assert actual_local_steps == expected_local_steps, (
        f"PolicyWorker: Expected local_steps {expected_local_steps}, got {actual_local_steps}"
    )

    # Verify result structure
    assert "train_status" in result.metadata
    train_status = result.metadata["train_status"]
    assert "policy_update_steps" in train_status

    # Verify policy_update_steps calculation (should be total_calls / accumulation_steps)
    expected_policy_update_steps_normalized = len(policy_training_calls) / expected_accumulation_steps
    assert train_status["policy_update_steps"] == expected_policy_update_steps_normalized

    # Test CriticWorkerBase with same accumulation logic
    critic_worker = create_test_worker(CriticWorkerBase)

    critic_training_calls = []

    def mock_critic_training_step(experience, global_step, local_step, accumulation_steps):
        critic_training_calls.append({"local_step": local_step, "accumulation_steps": accumulation_steps})
        return {"critic_loss": 0.3, "values": 1.0, "critic_lr": 1e-4}

    critic_worker.training_step = mock_critic_training_step

    # Run critic ppo_train
    with (
        patch("torch.distributed.barrier"),
        patch("tqdm.tqdm", side_effect=lambda x, **kwargs: x),
        patch("torch.cuda.empty_cache"),
    ):
        result = critic_worker.ppo_train(dummy_databatch)

    # Verify Critic Worker Results
    assert len(critic_training_calls) == total_micro_batches, (
        f"CriticWorker: Expected {total_micro_batches} training_step calls, got {len(critic_training_calls)}"
    )

    # Verify accumulation_steps are consistent for critic (should equal micro_batches_per_mini_batch)
    for call in critic_training_calls:
        assert call["accumulation_steps"] == expected_accumulation_steps, (
            f"CriticWorker: Expected accumulation_steps={expected_accumulation_steps}, got {call['accumulation_steps']}"
        )

    # Verify no early termination for critic
    actual_local_steps = [call["local_step"] for call in critic_training_calls]
    assert actual_local_steps == expected_local_steps, (
        f"CriticWorker: Expected local_steps {expected_local_steps}, got {actual_local_steps}"
    )

    # Verify result structure for critic
    assert "train_status" in result.metadata
    train_status = result.metadata["train_status"]
    assert "critic_update_steps" in train_status
    assert train_status["critic_update_steps"] == len(critic_training_calls) / expected_accumulation_steps


def test_grug_ppo_train_does_not_retain_consumed_microbatches():
    """The policy releases each consumed Experience before loading the next one."""

    cfg = OmegaConf.create(
        {
            "trainer": {
                "micro_train_batch_size_per_gpu": 1,
                "update_epochs_per_batch": 1,
                "policy": {
                    "grug_query_bias_update_mode": "frozen",
                    "optimizer_config": {"max_grad_norm": 1.0},
                },
                "algorithm": {
                    "batch_invariant": False,
                    "policy_loss_type": "regular",
                    "loss_reduction": "token_mean",
                },
            },
            "generator": {"sampling_params": {"temperature": 1.0}},
        }
    )
    worker, batch = _grug_ppo_worker_and_batch(
        cfg,
        _ObservableGrugCausalLM(),
        torch.ones(4, 4, dtype=torch.long),
    )
    worker.policy_mini_batch_size_per_gpu = 2
    worker.strategy.ep_size = 1
    previous_experience = None
    prior_microbatch_was_released = []

    def training_step(experience, _global_step, _local_step, _accumulation_steps):
        nonlocal previous_experience
        if previous_experience is not None:
            gc.collect()
            prior_microbatch_was_released.append(previous_experience() is None)
        previous_experience = weakref.ref(experience)
        return {"policy_loss": 0.5, "policy_lr": 1e-4, "policy_entropy": 0.1, "response_length": 2}

    worker.training_step = training_step
    _run_grug_ppo_train(worker, batch)

    assert prior_microbatch_was_released == [True, True, True]


def test_default_grug_ppo_train_keeps_query_bias_exact_across_optimizer_steps():
    causal_lm = GrugMoeForCausalLM.from_pretrained(
        ORACLE_FIXTURE_DIR,
        local_files_only=True,
        attn_implementation="eager",
        dtype=torch.float32,
    )
    causal_lm.train()
    frozen_bias = torch.linspace(
        -0.3,
        0.3,
        steps=causal_lm.config.num_hidden_layers * causal_lm.config.num_local_experts,
    ).reshape(causal_lm.config.num_hidden_layers, causal_lm.config.num_local_experts)
    frozen_bias -= frozen_bias.mean(dim=-1, keepdim=True)
    causal_lm.set_query_bias(frozen_bias)

    cfg = get_default_config()
    cfg.trainer.micro_train_batch_size_per_gpu = 1
    cfg.trainer.update_epochs_per_batch = 1
    cfg.trainer.algorithm.loss_reduction = "token_mean"
    OmegaConf.update(cfg, "trainer.algorithm.max_seq_len", 6, force_add=True)
    batch_size = 5
    sequences = torch.arange(batch_size * 6).reshape(batch_size, 6) % causal_lm.config.vocab_size
    worker, batch = _grug_ppo_worker_and_batch(
        cfg,
        causal_lm,
        sequences,
    )
    worker.policy_mini_batch_size_per_gpu = 1
    _enable_cpu_policy_training(worker, causal_lm)
    initial_bias = torch.stack([layer.mlp.router.bias for layer in causal_lm.model.layers]).clone()
    initial_lm_head = causal_lm.lm_head.weight.detach().clone()
    _run_grug_ppo_train(worker, batch)

    actual_bias = torch.stack([layer.mlp.router.bias for layer in causal_lm.model.layers])
    torch.testing.assert_close(actual_bias, initial_bias, rtol=0, atol=0)
    assert not torch.equal(causal_lm.lm_head.weight, initial_lm_head)


def test_replace_mode_updates_grug_query_bias_through_policy_training():
    causal_lm = GrugMoeForCausalLM.from_pretrained(
        ORACLE_FIXTURE_DIR,
        local_files_only=True,
        attn_implementation="eager",
        dtype=torch.float32,
    )
    causal_lm.train()
    initial_bias = torch.stack([layer.mlp.router.bias for layer in causal_lm.model.layers]).clone()

    cfg = get_default_config()
    cfg.trainer.policy.grug_query_bias_update_mode = "replace"
    cfg.trainer.micro_train_batch_size_per_gpu = 1
    cfg.trainer.update_epochs_per_batch = 1
    cfg.trainer.algorithm.loss_reduction = "token_mean"
    OmegaConf.update(cfg, "trainer.algorithm.max_seq_len", 6, force_add=True)
    sequences = torch.arange(6).reshape(1, 6) % causal_lm.config.vocab_size
    worker, batch = _grug_ppo_worker_and_batch(cfg, causal_lm, sequences)
    worker.policy_mini_batch_size_per_gpu = 1
    _enable_cpu_policy_training(worker, causal_lm)

    _run_grug_ppo_train(worker, batch)

    actual_bias = torch.stack([layer.mlp.router.bias for layer in causal_lm.model.layers])
    assert not torch.equal(actual_bias, initial_bias)


def test_validate_batch_sizes_lcm_dp_requirement():
    """Ensure train_batch_size is >= lcm(policy_dp, ref_dp) when ref is used; else >= policy_dp."""

    def create_config(train_batch_size, policy_dp, ref_dp, include_ref=True):
        cfg = get_default_config()
        cfg.trainer.train_batch_size = train_batch_size
        cfg.trainer.policy_mini_batch_size = train_batch_size
        cfg.trainer.critic_mini_batch_size = 1
        cfg.trainer.micro_train_batch_size_per_gpu = 1
        cfg.trainer.micro_forward_batch_size_per_gpu = 1
        cfg.trainer.placement.policy_num_nodes = 1
        cfg.trainer.placement.policy_num_gpus_per_node = policy_dp
        cfg.trainer.placement.ref_num_nodes = 1
        cfg.trainer.placement.ref_num_gpus_per_node = ref_dp if include_ref else 1
        cfg.trainer.placement.critic_num_nodes = 1
        cfg.trainer.placement.critic_num_gpus_per_node = 1
        cfg.trainer.policy.sequence_parallel_size = 1
        cfg.trainer.ref.sequence_parallel_size = 1
        cfg.trainer.critic.model.path = None
        cfg.trainer.critic.sequence_parallel_size = 1
        cfg.trainer.algorithm.use_kl_loss = include_ref
        cfg.trainer.algorithm.use_kl_in_reward = False
        cfg.trainer.algorithm.policy_loss_type = "regular"
        return cfg

    # Fail: lcm(2, 3) = 6, but train_batch_size = 5 when ref is used
    cfg = create_config(train_batch_size=5, policy_dp=2, ref_dp=3, include_ref=True)
    with pytest.raises(
        AssertionError,
        match=r"least common multiple of the data parallel sizes",
    ):
        validate_batch_sizes(cfg)

    # Pass: train_batch_size equals lcm(2, 3) = 6 when ref is used
    cfg = create_config(train_batch_size=6, policy_dp=2, ref_dp=3, include_ref=True)
    validate_batch_sizes(cfg)

    # Pass: ref disabled -> requirement reduces to policy_dp. With policy_dp=2, tbs=2 is valid.
    cfg = create_config(train_batch_size=2, policy_dp=2, ref_dp=3, include_ref=False)
    validate_batch_sizes(cfg)
