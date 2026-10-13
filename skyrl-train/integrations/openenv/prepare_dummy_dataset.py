# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Prepare OpenEnv TaskSpecs with an explicit server image."""

import argparse
import json
import os
from typing import List
import math
from pathlib import Path

from taskcompendium.models import EnvironmentRequirements
from marinskyrl.task_preparation import source_row_task, write_tasks


def example_question(env_name: str) -> List[str]:
    """
    Return simple example prompts for each environment.
    Each prompt instructs the LLM to generate an <action>...</action> response.
    """

    # Shared instruction appended to each example
    instruction = "Wrap the action between <action> and </action> tags.\nFor example: <action>ACTION_HERE</action>."

    if env_name == "echo_env":
        # Echo environment simply echoes the text back.
        examples = [
            "Send a short test message to the echo environment.",
        ]
    elif env_name == "coding_env":
        # Coding environment executes code and returns stdout.
        examples = [
            "Print 'I love RL!' using Python.",
            "Provide a function fib(n) that can compute the n-th Fibonacci number",
            "Using a for loop, print numbers 1 to 5 and their squares.",
        ]
        instruction = (
            "Write the python code inside <action>...</action> tags. Example: <action>print('Hello, World!')</action>"
        )
    else:
        # NOTE: we only provide dummy train set examples for echo_env and coding_env
        # For other environments, you need to provide your own train / validation set examples.
        raise ValueError(f"Unknown environment name: {env_name}")

    return [f"{ex}\n\n{instruction}" for ex in examples]


def build_dataset(
    env_name: str, output_dir: str, *, environment: EnvironmentRequirements, config: dict, target_size=32
):
    """Prepare bare TaskSpecs for one OpenEnv server image."""
    base_questions = example_question(env_name)
    repeat_factor = math.ceil(target_size / len(base_questions))
    questions = (base_questions * repeat_factor)[:target_size]

    # Split the questions into train and validation sets
    split_index = target_size // 2
    train_questions = questions[:split_index]
    val_questions = questions[split_index:]

    os.makedirs(output_dir, exist_ok=True)
    for split, questions, filename in (
        ("train", train_questions, "train.parquet"),
        ("validation", val_questions, "validation.parquet"),
    ):
        write_tasks(
            Path(output_dir) / filename,
            (
                source_row_task(
                    {
                        "prompt": [{"role": "user", "content": question}],
                        "env_name": env_name,
                    },
                    index,
                    source_name=f"openenv/{env_name}/{split}",
                    config=config,
                    environment=environment,
                )
                for index, question in enumerate(questions)
            ),
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", default="~/data/openenv")
    parser.add_argument("--env_name", choices=("echo_env", "coding_env"), required=True)
    parser.add_argument("--environment", type=Path, required=True, help="EnvironmentRequirements JSON")
    parser.add_argument("--config", type=Path, required=True, help="Private server parameters in JSON")
    args = parser.parse_args()

    args.output_dir = os.path.expanduser(args.output_dir)
    build_dataset(
        args.env_name,
        args.output_dir,
        environment=EnvironmentRequirements.model_validate_json(args.environment.read_text()),
        config=json.loads(args.config.read_text()),
    )
