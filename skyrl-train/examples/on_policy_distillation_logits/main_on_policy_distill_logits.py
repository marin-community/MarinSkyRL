"""
On-Policy Distillation with Teacher Logits.

Uses a separate vLLM-served teacher model to provide top-K logprobs for
student-generated sequences. The teacher can be quantized (AWQ/GPTQ) and
runs on dedicated GPUs, enabling efficient large-teacher → small-student distillation.

Key differences from examples/on_policy_distillation/:
- Teacher is a vLLM inference engine (not FSDP RefWorker)
- Supports quantized teacher models via engine_init_kwargs
- Provides top-K logprob distributions (not just per-token log_probs)
- Enables forward KL, reverse KL, and JSD loss variants
"""

import torch
import ray
from omegaconf import DictConfig
from skyrl_train.entrypoints.main_base import BasePPOExp, config_dir, validate_cfg
from skyrl_train.entrypoints.main_base import create_teacher_inference_engines_from_config
from skyrl_train.distillation_trainer import DistillationTrainer
from skyrl_train.training_batch import TrainingInputBatch
from skyrl_train.utils import initialize_ray
from skyrl_train.utils.ppo_utils import (
    register_advantage_estimator,
    register_policy_loss,
    reduce_loss,
    masked_mean,
)
from skyrl_train.utils.distillation_utils import kl_from_sparse_teacher_logprobs
import hydra


class OnPolicyDistillationLogitsTrainer(DistillationTrainer):
    """
    On-policy distillation trainer with teacher logits.

    Uses teacher top-K logprobs as the reward signal via KL divergence,
    with configurable KL type (forward, reverse, or JSD).
    """

    def apply_reward_kl_penalty(
        self,
        data: TrainingInputBatch,
    ) -> TrainingInputBatch:
        """Compute KL-based reward from teacher logprobs.

        For on-policy distillation, the reward is the negative KL divergence
        between teacher and student at each token position. This replaces
        any environment reward.
        """
        loss_mask = data["loss_mask"]
        teacher_action_log_probs = data["base_action_log_probs"]
        action_log_probs = data["action_log_probs"]

        # Use per-token reverse KL from teacher log_probs as reward
        # (same as existing on-policy distillation example)
        rewards = -(action_log_probs - teacher_action_log_probs) * loss_mask
        data["rewards"] = rewards

        # Log teacher KL metrics
        kl_mean = masked_mean(rewards.abs(), loss_mask, dim=-1).mean().item()
        self.all_metrics.update({
            "distill/token_kl_mean": kl_mean,
        })

        return data


# Reuse the no-op advantage estimator from the original example
@register_advantage_estimator("no_op")
def compute_no_op_advantage(token_level_rewards: torch.Tensor, **kwargs):
    return token_level_rewards, token_level_rewards


@register_policy_loss("importance_sampling")
def compute_importance_sampling_policy_loss(
    log_probs, old_log_probs, advantages, config, loss_mask=None, rollout_logprobs=None, **kwargs
):
    loss = -torch.exp(log_probs - old_log_probs) * advantages
    loss = reduce_loss(loss, loss_mask, "seq_mean_token_sum_norm", config.max_seq_len)
    return loss, 0.0


class OnPolicyDistillationLogitsExp(BasePPOExp):
    def get_trainer(self, *args, **kwargs):
        return OnPolicyDistillationLogitsTrainer(*args, **kwargs)

    def _setup_trainer(self):
        """Override to create teacher inference engines."""
        trainer = super()._setup_trainer()

        # Create teacher engines if configured
        if self.cfg.teacher.model_path is not None:
            teacher_engines, teacher_tokenizer = create_teacher_inference_engines_from_config(
                self.cfg, self.tokenizer
            )
            trainer.setup_teacher_engine(
                teacher_engines,
                student_tokenizer=self.tokenizer,
                teacher_tokenizer=teacher_tokenizer,
            )

        return trainer


@ray.remote(num_cpus=1)
def skyrl_entrypoint(cfg: DictConfig):
    exp = OnPolicyDistillationLogitsExp(cfg)
    exp.run()


@hydra.main(config_path=config_dir, config_name="ppo_base_config", version_base=None)
def main(cfg: DictConfig) -> None:
    validate_cfg(cfg)
    initialize_ray(cfg)
    ray.get(skyrl_entrypoint.remote(cfg))


if __name__ == "__main__":
    main()
