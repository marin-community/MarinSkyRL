# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Policy-aware jailbreak reward from NVIDIA's released verifier templates."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from importlib.resources import files
from typing import Any

import yaml

from skyrl_gym.envs.nemotron_ultra.judge import OpenAIJudge


def _configuration() -> dict[str, Any]:
    resource = files(__package__).joinpath("jailbreak_verifiers.yaml")
    value = yaml.safe_load(resource.read_text())
    if not isinstance(value, dict):
        raise RuntimeError("NVIDIA jailbreak verifier configuration is invalid")
    return value


def grade_jailbreak(
    response: str,
    record: dict[str, Any],
    judge: OpenAIJudge,
) -> tuple[float, dict[str, Any]]:
    config = _configuration()
    policy = record.get("response_policy_mapped")
    policy_map = config.pop("policy_verifier_map")
    if policy not in policy_map:
        raise ValueError(f"unsupported NVIDIA jailbreak response policy {policy!r}")
    verifier_names = policy_map[policy]["verifiers"]
    adversarial_prompt = record.get("adversarial_prompt", "")

    def evaluate(name: str) -> tuple[str, str | None, float]:
        verifier = config[name]
        prompt = verifier["prompt_template"].format(
            adversarial_prompt=adversarial_prompt,
            model_response=response,
        )
        try:
            output = judge.generate([{"role": "user", "content": prompt}])
        except Exception:  # noqa: BLE001 - NVIDIA maps all judge transport/parsing failures to "unclear".
            return name, None, float(verifier["rewards"].get("unclear", 0.0))
        positive = verifier["labels"]["positive"]
        negative = verifier["labels"]["negative"]
        positive_position = output.find(positive)
        negative_position = output.find(negative)
        if positive_position >= 0 and (negative_position < 0 or positive_position < negative_position):
            return name, positive, float(verifier["rewards"].get("positive", 1.0))
        if negative_position >= 0:
            return name, negative, float(verifier["rewards"].get("negative", 0.0))
        return name, None, float(verifier["rewards"].get("unclear", 0.0))

    with ThreadPoolExecutor(max_workers=len(verifier_names)) as executor:
        results = list(executor.map(evaluate, verifier_names))
    rewards = {name: reward for name, _, reward in results}
    labels = {name: label for name, label, _ in results}
    combination = policy_map[policy].get("reward_combination", "product")
    if combination == "product":
        reward = 1.0
        for value in rewards.values():
            reward *= value
    elif combination == "average":
        reward = sum(rewards.values()) / max(len(rewards), 1)
    else:
        reward = next(iter(rewards.values()), 0.0)
    return reward, {"response_policy": policy, "verifier_rewards": rewards, "verifier_labels": labels}
