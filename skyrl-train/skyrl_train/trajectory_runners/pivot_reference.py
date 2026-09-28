"""Teacher-forced SWE reference actions for SFT, with generated held-out evaluation."""

from infra.rl_data.pivot_swe import reference_action_tokens
from skyrl_train.trajectory_runners.base import TrajectoryRunner, propagate_data_sources
from skyrl_train.trajectory_runners.types import TrajectoryBatch, TrajectoryRequestBatch


class PivotReferenceRunner(TrajectoryRunner):
    def __init__(self, evaluation_runner: TrajectoryRunner, tokenizer, generator_config):
        self.evaluation_runner = evaluation_runner
        self.tokenizer = tokenizer
        self.trajectory_runner_cfg = generator_config

    async def startup(self) -> None:
        await self.evaluation_runner.startup()

    async def shutdown(self) -> None:
        await self.evaluation_runner.shutdown()

    async def _run(self, input_batch: TrajectoryRequestBatch, disable_tqdm: bool = False) -> TrajectoryBatch:
        metadata = input_batch["batch_metadata"]
        if metadata is None:
            raise ValueError("SFT requires an explicit training phase")
        if metadata.training_phase == "eval":
            return await self.evaluation_runner.run(input_batch, disable_tqdm=disable_tqdm)
        prompts, responses = [], []
        for prompt, extras in zip(input_batch["prompts"], input_batch["env_extras"], strict=True):
            prefix, completion = reference_action_tokens(prompt, extras["extra_info"], self.tokenizer)
            if (
                len(prefix) > self.trajectory_runner_cfg.max_input_length
                or len(completion) > self.trajectory_runner_cfg.sampling_params.max_generate_length
            ):
                raise ValueError("SFT reference exceeds the shared experiment token budget")
            prompts.append(prefix)
            responses.append(completion)
        count = len(responses)
        batch = TrajectoryBatch(
            prompt_token_ids=prompts,
            response_ids=responses,
            rewards=[0.0] * count,
            loss_masks=[[1] * len(tokens) for tokens in responses],
            stop_reasons=["reference"] * count,
            rollout_logprobs=None,
            rollout_metrics={"sft/reference_examples": count, "sft/reference_tokens": sum(map(len, responses))},
            trajectory_ids=input_batch["trajectory_ids"],
        )
        propagate_data_sources(input_batch, batch)
        return batch
