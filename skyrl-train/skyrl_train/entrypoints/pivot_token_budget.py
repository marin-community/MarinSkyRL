"""Pivot training with a shared token budget and consumed-trajectory accounting."""

import json

import hydra
import ray
from omegaconf import DictConfig

from skyrl_train.batch_sampling import filter_trajectory_batch
from skyrl_train.config.trajectory_runner_capabilities import TrajectoryRunnerMode
from skyrl_train.entrypoints.main_base import BasePPOExp, config_dir, run_ray_driver
from skyrl_train.io import io
from skyrl_train.pivot_token_budget import TOKEN_MATCH_TOLERANCE, select_token_budget_groups
from skyrl_train.trainer import RayPPOTrainer


class PivotTokenTrainer(RayPPOTrainer):
    def _configure_training_schedule(self):
        super()._configure_training_schedule()
        self.token_budget = self.cfg.trainer.pivot_token_budget
        if self.token_budget is None or self.token_budget <= 0 or self.cfg.trainer.resume_mode != "none":
            raise ValueError("Pivot token matching requires a positive budget and a fresh run")
        if self.cfg.trainer.update_epochs_per_batch != 1:
            raise ValueError("Pivot token accounting requires one pass through each training batch")
        self.consumed_tokens = 0
        self.budget_steps = []

    def select_trajectories(self, trajectory_batch, uids):
        trajectory_batch, uids = super().select_trajectories(trajectory_batch, uids)
        lengths = [
            len(prompt) + len(response)
            for prompt, response in zip(
                trajectory_batch["prompt_token_ids"], trajectory_batch["response_ids"], strict=True
            )
        ]
        selection = select_token_budget_groups(
            lengths, uids, self.token_budget - self.consumed_tokens, self.cfg.generator.n_samples_per_prompt
        )
        if len(selection.indices) != len(uids):
            trajectory_batch = filter_trajectory_batch(trajectory_batch, selection.indices)
            uids = [uids[index] for index in selection.indices]
        self.selected_tokens = selection.input_tokens
        self.selected_ids = [[item.instance_id, item.repetition_id] for item in trajectory_batch["trajectory_ids"]]
        return trajectory_batch, uids

    def train_critic_and_policy(self, data):
        status = super().train_critic_and_policy(data)
        if status["policy_update_steps"] != 1:
            raise RuntimeError("Token-matched training requires one optimizer update, including partial batches")
        real_rows = data.batch_size - data.metadata.get("pad_size", 0)
        input_tokens = int(data["attention_mask"][:real_rows].sum().item())
        if input_tokens != self.selected_tokens:
            raise RuntimeError("Learner token count differs from the admitted trajectory count")
        self.consumed_tokens += input_tokens
        self.budget_steps.append(
            {
                "step": self.global_step,
                "input_tokens": input_tokens,
                "response_tokens": int(data["response_mask"][:real_rows].sum().item()),
                "loss_tokens": int(data["loss_mask"][:real_rows].sum().item()),
                "trajectory_ids": self.selected_ids,
            }
        )
        self.all_metrics.update(
            {
                "budget/input_tokens": self.consumed_tokens,
                "budget/target_tokens": self.token_budget,
                "budget/remaining_tokens": self.token_budget - self.consumed_tokens,
            }
        )
        io.write_bytes_atomic(
            f"{self.cfg.trainer.export_path}/learner-token-budget.json",
            json.dumps(
                {"target_tokens": self.token_budget, "input_tokens": self.consumed_tokens, "steps": self.budget_steps}
            ).encode(),
        )
        if self.token_budget - self.consumed_tokens <= TOKEN_MATCH_TOLERANCE * self.token_budget:
            self.total_training_steps = self.global_step
        return status

    async def _finalize_training(self, *, completed_step, epoch):
        if self.token_budget - self.consumed_tokens > TOKEN_MATCH_TOLERANCE * self.token_budget:
            raise RuntimeError("Safety step limit reached before consuming the learner token budget")
        await super()._finalize_training(completed_step=completed_step, epoch=epoch)


class PivotTokenExp(BasePPOExp):
    def get_trainer(
        self,
        cfg,
        tracker,
        tokenizer,
        train_dataset,
        eval_dataset,
        inference_engine_client,
        trajectory_runner,
        colocate_pg,
    ):
        return PivotTokenTrainer(
            cfg,
            tracker,
            tokenizer,
            train_dataset,
            inference_engine_client,
            trajectory_runner,
            colocate_pg=colocate_pg,
            eval_dataset=eval_dataset,
        )


@ray.remote(num_cpus=1, max_retries=0)
def pivot_entrypoint(cfg: DictConfig):
    PivotTokenExp(cfg).run()


def run(cfg: DictConfig) -> None:
    run_ray_driver(cfg, pivot_entrypoint, TrajectoryRunnerMode.SKYRL_GYM)


@hydra.main(config_path=config_dir, config_name="ppo_base_config", version_base=None)
def main(cfg: DictConfig) -> None:
    run(cfg)


if __name__ == "__main__":
    main()
