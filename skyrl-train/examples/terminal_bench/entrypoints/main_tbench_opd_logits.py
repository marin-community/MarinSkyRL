"""
Main entrypoint for on-policy distillation with teacher logits on terminal bench tasks.

Combines the TerminalBenchGenerator (Harbor/Daytona agent environment) with the
DistillationTrainer (vLLM-based teacher scoring for top-K logprobs).

The teacher model is served via a separate vLLM engine (supports AWQ/GPTQ quantization)
and provides top-K log-probability distributions for student-generated sequences.
"""

import signal
import sys
import torch
import ray
import hydra
from loguru import logger
from omegaconf import DictConfig
from skyrl_train.entrypoints.main_base import config_dir, create_teacher_inference_engines_from_config
from skyrl_train.utils import validate_cfg
from skyrl_train.utils.utils import initialize_ray
from skyrl_train.utils.ppo_utils import (
    register_advantage_estimator,
    register_policy_loss,
    reduce_loss,
    masked_mean,
)
from skyrl_train.distillation_trainer import DistillationTrainer
from skyrl_train.training_batch import TrainingInputBatch
from examples.terminal_bench.entrypoints.main_tbench import TerminalBenchExp


class OnPolicyDistillationLogitsTerminalBenchTrainer(DistillationTrainer):
    """
    On-policy distillation trainer with teacher logits for terminal bench.

    Uses teacher top-K logprobs as the reward signal, replacing environment
    rewards with KL divergence between teacher and student.
    """

    def apply_reward_kl_penalty(
        self,
        data: TrainingInputBatch,
    ) -> TrainingInputBatch:
        """Compute KL-based reward from teacher/ref logprobs."""
        loss_mask = data["loss_mask"]
        teacher_action_log_probs = data["base_action_log_probs"]
        action_log_probs = data["action_log_probs"]

        # Reverse KL as reward: -(student_logprobs - teacher_logprobs)
        rewards = -(action_log_probs - teacher_action_log_probs) * loss_mask
        data["rewards"] = rewards

        kl_mean = masked_mean(rewards.abs(), loss_mask, dim=-1).mean().item()
        self.all_metrics.update({"distill/token_kl_mean": kl_mean})

        return data


# Register custom advantage estimator and policy loss for distillation
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


class OnPolicyDistillationLogitsTerminalBenchExp(TerminalBenchExp):
    """Terminal bench experiment with on-policy distillation + teacher logits."""

    def get_trainer(self, *args, **kwargs):
        return OnPolicyDistillationLogitsTerminalBenchTrainer(*args, **kwargs)

    def _setup_trainer(self):
        """Override to create teacher vLLM inference engines."""
        trainer = super()._setup_trainer()

        # Create teacher engines if configured
        if hasattr(self.cfg, "teacher") and self.cfg.teacher.model_path is not None:
            teacher_engines = create_teacher_inference_engines_from_config(
                self.cfg, self.tokenizer
            )
            trainer.setup_teacher_engine(teacher_engines)
            logger.info(f"Teacher engine created for {self.cfg.teacher.model_path}")
        else:
            logger.warning(
                "No teacher.model_path configured. Running without teacher logits. "
                "Set teacher.model_path to enable teacher scoring."
            )

        return trainer


@ray.remote(num_cpus=1, max_retries=0)
def skyrl_entrypoint(cfg: DictConfig):
    exp = OnPolicyDistillationLogitsTerminalBenchExp(cfg)
    exp.run()


@hydra.main(config_path=config_dir, config_name="ppo_base_config", version_base=None)
def main(cfg: DictConfig) -> None:
    validate_cfg(cfg)
    initialize_ray(cfg)

    def _sigterm_handler(signum, frame):
        logger.warning("Received SIGTERM on head node, shutting down Ray...")
        ray.shutdown()
        sys.exit(1)

    signal.signal(signal.SIGTERM, _sigterm_handler)

    try:
        ray.get(skyrl_entrypoint.remote(cfg))
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise
    finally:
        logger.info("Shutting down Ray on head node...")
        ray.shutdown()


if __name__ == "__main__":
    main()
