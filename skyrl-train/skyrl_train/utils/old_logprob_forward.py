"""Skip the policy's pre-update old-log-prob forward when its importance ratio is identically one."""

from __future__ import annotations

from loguru import logger
from omegaconf import DictConfig

from skyrl_train.utils.algorithm_registry import PolicyLossType

OLD_LOGPROBS_FROM_TRAINING_FORWARD_KEY = "old_logprobs_from_training_forward"


def old_logprob_forward_skip_problems(cfg: DictConfig) -> list[str]:
    """Every reason the policy's old-log-prob forward must still run; empty when it can be skipped.

    With one optimizer step per training batch, the weights the pre-update forward would read are
    the weights of the training forward itself, so the PPO ratio is exactly one and the clipped
    surrogate is the on-policy policy gradient. The forward then only feeds its other consumers,
    and each of them keeps it: every objective but the regular PPO loss, the reference KL, TIS,
    distillation, the train/eval parity probe, the batch dump, and any backend but Megatron. The
    consume-time mismatch diagnostics move into the policy workers, which pool them over
    data-parallel ranks.
    """
    algorithm = cfg.trainer.algorithm
    if not bool(algorithm.get(OLD_LOGPROBS_FROM_TRAINING_FORWARD_KEY, False)):
        return [f"trainer.algorithm.{OLD_LOGPROBS_FROM_TRAINING_FORWARD_KEY} is off"]
    trainer = cfg.trainer
    problems = []
    if trainer.strategy != "megatron":
        problems.append(f"trainer.strategy must be megatron, got {trainer.strategy}")
    if algorithm.policy_loss_type != PolicyLossType.REGULAR:
        problems.append(f"trainer.algorithm.policy_loss_type must be regular, got {algorithm.policy_loss_type}")
    if trainer.update_epochs_per_batch != 1:
        problems.append(f"trainer.update_epochs_per_batch must be 1, got {trainer.update_epochs_per_batch}")
    if trainer.train_batch_size != trainer.policy_mini_batch_size:
        problems.append(
            "trainer.train_batch_size must equal trainer.policy_mini_batch_size so every batch takes one "
            f"optimizer step, got {trainer.train_batch_size} and {trainer.policy_mini_batch_size}"
        )
    if trainer.critic.model.path or algorithm.advantage_estimator == "gae":
        problems.append("trainer.critic.model.path must be empty and the advantage estimator must not be gae")
    if algorithm.use_kl_loss or algorithm.use_kl_in_reward:
        problems.append("trainer.algorithm.use_kl_loss and use_kl_in_reward must be off; the reference KL reads them")
    if algorithm.use_tis:
        problems.append("trainer.algorithm.use_tis must be off; its importance ratio reads them")
    if algorithm.get("distillation") is not None:
        problems.append("trainer.algorithm.distillation must be unset; its objective reads them")
    if trainer.policy.megatron_config.check_train_eval_parity:
        problems.append("trainer.policy.megatron_config.check_train_eval_parity must be off; the probe reads them")
    if trainer.dump_data_batch:
        problems.append("trainer.dump_data_batch must be off; the dump records them")
    ratio_diagnostics = algorithm.get("ratio_diagnostics") or {}
    if ratio_diagnostics.get("pooled") is not True:
        problems.append(
            "trainer.algorithm.ratio_diagnostics.pooled must be on; the workers pool the mismatch diagnostics"
        )
    return problems


def skip_old_logprob_forward(cfg: DictConfig) -> bool:
    """Whether the policy's old-log-prob forward is skipped and the training forward supplies its values."""
    return not old_logprob_forward_skip_problems(cfg)


def log_old_logprob_forward_choice(cfg: DictConfig) -> None:
    """Log once whether the old-log-prob forward runs, and what keeps it when it does."""
    problems = old_logprob_forward_skip_problems(cfg)
    if problems:
        logger.info("The policy's old-log-prob forward runs every step; skipping it requires: " + "; ".join(problems))
        return
    logger.info(
        "The policy's old-log-prob forward is skipped: one optimizer step per batch makes the PPO ratio exactly "
        "one, so the training forward supplies the old log-probs and the mismatch diagnostics"
    )
