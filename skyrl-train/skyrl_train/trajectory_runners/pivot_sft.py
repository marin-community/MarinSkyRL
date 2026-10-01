"""Supply released next actions to the SFT loss; sample normally during evaluation."""

import json
from collections.abc import Mapping
from typing import Any

from skyrl_gym.verification import VerificationResult
from skyrl_train.trajectory_runners.base import TrajectoryRunner, propagate_data_sources
from skyrl_train.trajectory_runners.types import TrajectoryBatch, TrajectoryRequestBatch


def demonstration_tokens(prompt, extras: dict[str, Any], tokenizer) -> tuple[list[int], list[int]]:
    """Render only the next demonstrated action as supervision, preserving tool context."""
    ultra = extras["extra_info"]["nemotron_ultra"]
    record, options = json.loads(ultra["record_json"]), json.loads(ultra["request_json"])
    if "expected_answer" in record:
        target = {"role": "assistant", "content": record["expected_answer"]}
    else:
        action = record["expected_action"]
        if action["type"] == "message":
            target = {"role": "assistant", "content": action["content"]}
        elif action["type"] == "function_call":
            args = action["arguments"]
            target = {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "demonstration",
                        "type": "function",
                        "function": {
                            "name": action["name"],
                            "arguments": json.loads(args) if isinstance(args, str) else args,
                        },
                    }
                ],
            }
        else:
            raise ValueError(f"Unsupported demonstration action: {action['type']}")
    tools = [
        {"type": "function", "function": {k: v for k, v in tool.items() if k != "type"}}
        for tool in options.get("tools", [])
    ]
    kwargs = {**options.get("chat_template_kwargs", {}), "tools": tools or None, "tokenize": True}
    prefix = tokenizer.apply_chat_template(prompt, add_generation_prompt=True, **kwargs)
    complete = tokenizer.apply_chat_template([*prompt, target], add_generation_prompt=False, **kwargs)
    if isinstance(prefix, Mapping):
        prefix = prefix["input_ids"]
    if isinstance(complete, Mapping):
        complete = complete["input_ids"]
    prefix, complete = list(prefix), list(complete)
    if complete[: len(prefix)] != prefix or len(complete) <= len(prefix):
        raise ValueError("The demonstration must extend the exact generation prefix")
    return prefix, complete[len(prefix) :]


class PivotSFTRunner(TrajectoryRunner):
    def __init__(self, sampled_runner: TrajectoryRunner, tokenizer, config):
        self.sampled_runner = sampled_runner
        self.tokenizer = tokenizer
        self.trajectory_runner_cfg = config

    async def startup(self) -> None:
        await self.sampled_runner.startup()

    async def shutdown(self) -> None:
        await self.sampled_runner.shutdown()

    async def _run(self, input_batch: TrajectoryRequestBatch, disable_tqdm: bool = False) -> TrajectoryBatch:
        phase = input_batch.get("batch_metadata")
        if phase is None:
            raise ValueError("Teacher forcing requires an explicit training/evaluation phase")
        if phase.training_phase == "eval":
            return await self.sampled_runner.run(input_batch, disable_tqdm=disable_tqdm)
        prompts, actions = [], []
        for prompt, extras in zip(input_batch["prompts"], input_batch["env_extras"], strict=True):
            prefix, target = demonstration_tokens(prompt, extras, self.tokenizer)
            if len(prefix) + len(target) > self.trajectory_runner_cfg.engine_init_kwargs.max_model_len:
                raise ValueError("Demonstration exceeds the context window; do not truncate SFT targets")
            prompts.append(prefix)
            actions.append(target)
        batch = TrajectoryBatch(
            prompt_token_ids=prompts,
            response_ids=actions,
            rewards=[0.0] * len(actions),
            loss_masks=[[1] * len(action) for action in actions],
            rollout_logprobs=None,
            verification_results=[
                VerificationResult.skipped(
                    "teacher-forced demonstration", diagnostics={"record_kind": "teacher_forced"}
                )
                for _ in actions
            ],
            stop_reasons=["stop"] * len(actions),
            rollout_metrics={"sft/examples": len(actions), "sft/target_tokens": sum(map(len, actions))},
        )
        propagate_data_sources(input_batch, batch)
        return batch
