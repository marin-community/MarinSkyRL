"""
Main entrypoint for on-policy distillation with teacher logits on terminal bench tasks.

Combines the HarborTrajectoryRunner (Harbor/Daytona agent environment) with the
DistillationTrainer (vLLM-based teacher scoring for top-K logprobs).

The teacher model is served via a separate vLLM engine (supports AWQ/GPTQ quantization)
and provides top-K log-probability distributions for student-generated sequences.
"""

import ray
import hydra
from loguru import logger
from omegaconf import DictConfig
from skyrl_train.entrypoints.main_base import config_dir, create_teacher_inference_engines_from_config, run_ray_driver
from skyrl_train.algorithms.on_policy_distillation import (
    compute_importance_sampling_policy_loss as compute_importance_sampling_policy_loss,
    compute_no_op_advantage as compute_no_op_advantage,
    compute_reverse_kl_rewards,
)
from skyrl_train.utils.policy_math import masked_mean
from skyrl_train.distillation_trainer import DistillationTrainer
from skyrl_train.training_batch import TrainingInputBatch
from skyrl_train.entrypoints.terminal_bench import TerminalBenchExp


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
        rewards = compute_reverse_kl_rewards(data)
        data["rewards"] = rewards

        kl_mean = masked_mean(rewards.abs(), loss_mask, dim=-1).mean().item()
        self.all_metrics.update({"distill/token_kl_mean": kl_mean})

        return data


class OnPolicyDistillationLogitsTerminalBenchExp(TerminalBenchExp):
    """Terminal bench experiment with on-policy distillation + teacher logits."""

    def get_trainer(self, *args, **kwargs):
        return OnPolicyDistillationLogitsTerminalBenchTrainer(*args, **kwargs)

    def _setup_trainer(self):
        """Override to create teacher vLLM inference engines."""
        trainer = super()._setup_trainer()

        # Create teacher engines if configured
        if hasattr(self.cfg, "teacher") and self.cfg.teacher.model_path is not None:
            teacher_engines, teacher_tokenizer = create_teacher_inference_engines_from_config(self.cfg, self.tokenizer)
            trainer.setup_teacher_engine(
                teacher_engines,
                student_tokenizer=self.tokenizer,
                teacher_tokenizer=teacher_tokenizer,
            )
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
    run_ray_driver(cfg, skyrl_entrypoint)


if __name__ == "__main__":
    main()
