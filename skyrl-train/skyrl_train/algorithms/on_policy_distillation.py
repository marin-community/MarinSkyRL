"""On-policy distillation trainer and registered loss functions."""

import torch

from skyrl_train.trainer import RayPPOTrainer
from skyrl_train.training_batch import TrainingInputBatch
from skyrl_train.utils.algorithm_registry import NoGroupAdvantage, register_advantage_estimator, register_policy_loss
from skyrl_train.utils.loss_reduction import reduce_loss


class OnPolicyDistillationTrainer(RayPPOTrainer):
    """Train against a teacher by replacing task rewards with the teacher KL penalty."""

    def apply_reward_kl_penalty(self, data: TrainingInputBatch) -> TrainingInputBatch:
        data["rewards"] = compute_reverse_kl_rewards(data)
        return data


def compute_reverse_kl_rewards(data: TrainingInputBatch) -> torch.Tensor:
    """Return token rewards for reverse KL from student to teacher."""
    loss_masks: torch.Tensor = data["loss_mask"]
    teacher_log_probs: torch.Tensor = data["base_action_log_probs"]
    action_log_probs: torch.Tensor = data["action_log_probs"]
    return -(action_log_probs - teacher_log_probs) * loss_masks


@register_advantage_estimator("no_op", group_contract=NoGroupAdvantage())
def compute_no_op_advantage(token_level_rewards: torch.Tensor, **kwargs):
    """Use token rewards directly as returns and advantages."""
    return token_level_rewards, token_level_rewards


@register_policy_loss("importance_sampling")
def compute_importance_sampling_policy_loss(
    log_probs, old_log_probs, advantages, config, loss_mask=None, rollout_logprobs=None, **kwargs
):
    """Compute the on-policy distillation importance-sampling objective."""
    loss = -torch.exp(log_probs - old_log_probs) * advantages
    loss = reduce_loss(loss, loss_mask, "seq_mean_token_sum_norm", config.max_seq_len)
    return loss, {}
