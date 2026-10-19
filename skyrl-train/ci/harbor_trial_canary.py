"""Exercise a full Harbor trial with synthetic model turns and no training GPUs."""

import argparse
import asyncio
import json
from contextlib import AsyncExitStack
from pathlib import Path

from omegaconf import OmegaConf
from rolloutengine.engine import RolloutEngine
from rolloutengine.models import ModelTurn, TaskRolloutSpec
from taskcompendium.grading_result import Outcome
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from transformers import PreTrainedTokenizerFast

from skyrl_train.rollouts.agent_backend import AgentInferenceBackend
from skyrl_train.rollouts.agent_runtime import InstalledAgentRuntime, InstalledAgentSettings
from skyrl_train.rollouts.harbor_tasks import HarborTaskSettings
from skyrl_train.rollouts.training_record import training_rollout


async def run(config_path: Path) -> None:
    config = json.loads(config_path.read_text())
    lowered = TaskRolloutSpec.model_validate_json(Path(config["task_path"]).read_text())
    settings = InstalledAgentSettings.model_validate(config["installed_agent"])
    harbor = HarborTaskSettings.from_config(OmegaConf.create({"harbor": config["harbor"]}))
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=Tokenizer(WordLevel({"[UNK]": 0, "canary": 1})))

    async def tokenize(_payload):
        return {"tokens": [1], "count": 1, "max_model_len": config["max_model_len"]}

    runtime = InstalledAgentRuntime(
        settings,
        AgentInferenceBackend("synthetic-canary", config["max_model_len"], tokenize, tokenizer),
        config["max_output_tokens"],
    )
    async with AsyncExitStack() as cleanup:
        await runtime.startup()
        cleanup.push_async_callback(runtime.shutdown)
        harness = runtime.original_harness(lowered, config["session_id"], harbor, config["phase"])
        turns = 0

        async def model(request):
            nonlocal turns
            if turns >= len(config["commands"]):
                raise ValueError("The installed agent requested more turns than the canary supplied")
            command = config["commands"][turns]
            message = {
                "role": "assistant",
                "content": "Synthetic transport canary; not model-generated output.",
                "tool_calls": [
                    {
                        "id": f"canary-{turns}",
                        "type": "function",
                        "function": {"name": "bash", "arguments": json.dumps({"command": command})},
                    }
                ],
            }
            prefix = (*request.continuation.served_token_ids, 90, 91) if request.continuation else (1,)
            turns += 1
            return ModelTurn(message, prefix, (1,), (-0.25,), "stop")

        engine = RolloutEngine({}, {lowered.harness: harness})
        result = training_rollout(await engine.run_task(lowered, model, run_id=config["session_id"]), lowered)
        grade = result.grade
        result = {
            "synthetic_model": True,
            "turns": turns,
            "grade_status": grade.status,
            "reward": grade.reward,
            "diagnostics": grade.diagnostics,
        }
        Path(config["result_path"]).write_text(json.dumps(result, indent=2) + "\n")
        if grade.status != Outcome.GRADED:
            raise RuntimeError("The original verifier did not return a grade")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    asyncio.run(run(args.config))


if __name__ == "__main__":
    main()
