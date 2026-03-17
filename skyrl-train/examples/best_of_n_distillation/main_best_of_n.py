"""
Best-of-N Distillation.

Generates N completions per prompt from the student, selects the best
completion by reward, and trains the student on selected completions.

Supports two modes:
- **Without teacher logits**: Pure rejection sampling + SFT on best completion.
- **With teacher logits**: Additionally uses teacher top-K logprobs for KL-based
  auxiliary loss on the selected completions.

Config:
- generator.n_samples_per_prompt=N controls how many completions per prompt.
- trainer.algorithm.policy_loss_type=sft for pure SFT on best completion.
- teacher.model_path for optional teacher logit scoring.
"""

import numpy as np
import torch
import ray
from omegaconf import DictConfig
from skyrl_train.entrypoints.main_base import BasePPOExp, config_dir, validate_cfg
from skyrl_train.entrypoints.main_base import create_teacher_inference_engines_from_config
from skyrl_train.distillation_trainer import DistillationTrainer
from skyrl_train.trainer import RayPPOTrainer
from skyrl_train.generators.base import GeneratorOutput
from skyrl_train.training_batch import TrainingInputBatch
from skyrl_train.utils import initialize_ray
from skyrl_train.utils.ppo_utils import (
    register_advantage_estimator,
    register_policy_loss,
    reduce_loss,
)
from skyrl_train.utils.distillation_utils import best_of_n_select
import hydra
from typing import List


class BestOfNDistillationTrainer(DistillationTrainer):
    """
    Best-of-N distillation trainer.

    Generates N completions per prompt, selects the best by reward,
    and trains on the selected completions.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._best_indices: List[int] = []

    def postprocess_generator_output(self, generator_output: GeneratorOutput, uids: List[str]) -> GeneratorOutput:
        """Override to select best-of-N completions after reward computation."""
        # First, run parent postprocessing (computes rewards, pass@N metrics, etc.)
        generator_output = super().postprocess_generator_output(generator_output, uids)

        N = self.cfg.generator.n_samples_per_prompt
        if N <= 1:
            self._best_indices = list(range(len(generator_output["response_ids"])))
            return generator_output

        # Get response-level rewards for selection
        rewards = generator_output["rewards"]
        if isinstance(rewards[0], list):
            # Token-level rewards → sum to get response-level
            response_rewards = [sum(r) for r in rewards]
        else:
            response_rewards = list(rewards)

        # Select best completion per prompt group
        self._best_indices = best_of_n_select(response_rewards, N)

        best_rewards = [response_rewards[i] for i in self._best_indices]
        mean_rewards = [
            np.mean(response_rewards[g * N : (g + 1) * N])
            for g in range(len(self._best_indices))
        ]
        self.all_metrics.update({
            "best_of_n/best_reward_mean": np.mean(best_rewards),
            "best_of_n/group_reward_mean": np.mean(mean_rewards),
            "best_of_n/reward_improvement": np.mean(best_rewards) - np.mean(mean_rewards),
            "best_of_n/n_samples_per_prompt": N,
        })

        # Filter generator_output to keep only best completions
        for key in ["prompt_token_ids", "response_ids", "rewards", "loss_masks",
                     "stop_reasons", "rollout_logprobs", "is_last_step",
                     "exclude_from_baseline"]:
            if key in generator_output and generator_output[key] is not None:
                val = generator_output[key]
                if isinstance(val, list) and len(val) == len(response_rewards):
                    generator_output[key] = [val[i] for i in self._best_indices]

        # Filter trajectory_ids if present
        if generator_output.get("trajectory_ids") is not None:
            tids = generator_output["trajectory_ids"]
            if len(tids) == len(response_rewards):
                generator_output["trajectory_ids"] = [tids[i] for i in self._best_indices]

        return generator_output

    def convert_to_training_input(self, generator_output: GeneratorOutput, uids: List[str]) -> TrainingInputBatch:
        """Override to deduplicate UIDs after best-of-N selection."""
        N = self.cfg.generator.n_samples_per_prompt
        if N > 1 and self._best_indices:
            # Deduplicate UIDs: each prompt's N samples share the same UID,
            # and we've already filtered generator_output to 1 per group
            filtered_uids = [uids[i] for i in self._best_indices]
            return super().convert_to_training_input(generator_output, filtered_uids)
        return super().convert_to_training_input(generator_output, uids)


# SFT loss: simple negative log-likelihood on selected completions
@register_policy_loss("sft")
def sft_loss(
    log_probs, old_log_probs, advantages, config, loss_mask=None, rollout_logprobs=None, **kwargs
):
    """SFT loss — maximize log probability of selected tokens."""
    loss = -log_probs
    loss = reduce_loss(loss, loss_mask, config.loss_reduction, config.max_seq_len)
    return loss, 0.0


# Uniform advantage estimator: all advantages = 1.0 (for SFT mode)
@register_advantage_estimator("uniform")
def compute_uniform_advantage(token_level_rewards: torch.Tensor, **kwargs):
    ones = torch.ones_like(token_level_rewards)
    return ones, ones


class BestOfNDistillationExp(BasePPOExp):
    def get_trainer(self, *args, **kwargs):
        return BestOfNDistillationTrainer(*args, **kwargs)

    def _setup_trainer(self):
        """Override to create teacher engines if configured."""
        trainer = super()._setup_trainer()

        if self.cfg.teacher.model_path is not None:
            teacher_engines = create_teacher_inference_engines_from_config(
                self.cfg, self.tokenizer
            )
            trainer.setup_teacher_engine(teacher_engines)

        return trainer


@ray.remote(num_cpus=1)
def skyrl_entrypoint(cfg: DictConfig):
    exp = BestOfNDistillationExp(cfg)
    exp.run()


@hydra.main(config_path=config_dir, config_name="ppo_base_config", version_base=None)
def main(cfg: DictConfig) -> None:
    validate_cfg(cfg)
    initialize_ray(cfg)
    ray.get(skyrl_entrypoint.remote(cfg))


if __name__ == "__main__":
    main()
