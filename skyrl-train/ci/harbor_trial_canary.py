"""Exercise a full Harbor trial with synthetic model turns and no training GPUs."""

import argparse
import asyncio
import json
from contextlib import AsyncExitStack
from pathlib import Path

from omegaconf import OmegaConf
from rolloutengine.contracts import ModelTurn
from rolloutengine.spec import LoweredTaskSpec
from taskcompendium.grading_result import Outcome
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from transformers import PreTrainedTokenizerFast

from skyrl_train.rollouts.agent_backend import AgentInferenceBackend
from skyrl_train.rollouts.agent_runtime import InstalledAgentRuntime, InstalledAgentSettings
from skyrl_train.rollouts.harbor_tasks import HarborTaskSettings


async def run(config_path: Path) -> None:
    config = json.loads(config_path.read_text())
    lowered = LoweredTaskSpec.model_validate_json(Path(config["task_path"]).read_text())
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
        session = runtime.original_session(lowered, config["session_id"], harbor, config["phase"])
        cleanup.push_async_callback(session.close)
        request = await session.prepare()
        if not request.messages:
            raise ValueError("The installed agent did not request a model response")
        turns = 0
        for command in config["commands"]:
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
            transition = await session.advance(ModelTurn(message, (1,), (1,), (-0.25,), "stop"))
            turns += 1
            if transition.done:
                break
        else:
            raise RuntimeError("The installed agent did not finish within the configured synthetic turns")
        grade = await session.grade((), stop_reason="stop", outputs={})
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
