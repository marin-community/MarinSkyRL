# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Exact answer-contract metrics for explicitly tagged frozen math evaluations."""

import hashlib
import math
from collections import defaultdict

from skyrl_gym.envs.data_contracts import get_data_contract
from skyrl_gym.envs.reasoning_gym.scoring import score_response
from skyrl_train.trajectory_runners.trajectory_reward_shaping import DEFAULT_ACCEPTED_STOP_REASONS

CONTRACTS = {
    "gsm8k": "gsm8k-first-hash-v1",
    "aime": "aime-last-answer-300-v1",
    "reasoning_gym": "reasoning-gym-0.1.25-last-answer-v1",
}


def evaluation_contract_metrics(envs, extras, responses, rewards, stops):
    """Keep native reward means separate from exact correctness and stop-gated correctness.

    Legacy evaluations without an explicit contract tag emit their original metrics.
    A tagged battery must have valid contracts on every row. All emitted means use
    all response sequences in the named population, including wrong/truncated rows.
    """
    tags = [extra.get("extra_info", {}).get("contract") for extra in extras]
    if not any(tags):
        return {}
    count = len(extras)
    if not all(len(values) == count for values in (envs, responses, rewards, stops)):
        raise ValueError("Contract metrics require aligned finalized evaluation rows")
    populations = defaultdict(list)
    for env, extra, response, reward, stop, tag in zip(envs, extras, responses, rewards, stops, tags, strict=True):
        if env not in CONTRACTS or tag != CONTRACTS[env]:
            raise ValueError("Tagged evaluation contains missing or unknown answer contracts")
        gold = extra["reward_model"]["ground_truth"]
        if gold != extra["reward_spec"]["ground_truth"]:
            raise ValueError("Evaluation ground-truth channels disagree")
        correct = get_data_contract(env).is_correct(response, gold)
        native = float(correct)
        if env == "aime":
            native = 1.0 if correct else -1.0
        elif env == "reasoning_gym":
            native = score_response(response, gold)
            correct = native >= 1.0
        raw = sum(reward) if isinstance(reward, list) else reward
        if not math.isfinite(raw) or not math.isclose(raw, native, abs_tol=1e-12, rel_tol=0):
            raise ValueError(
                "Evaluation reward differs from its task-native answer contract: "
                f"env={env}, prompt_sha256={extra.get('extra_info', {}).get('prompt_sha256')}, "
                f"raw={raw}, native={native}, response_sha256={hashlib.sha256(response.encode()).hexdigest()}"
            )
        values = (float(correct), float(correct and stop in DEFAULT_ACCEPTED_STOP_REASONS))
        source = (extra.get("data_source") or "unknown").replace("/", "_")
        if source == "all":
            raise ValueError("Dataset name all collides with the aggregate metric namespace")
        populations["all"].append(values)
        populations[source].append(values)
    return {
        f"eval/{source}/{metric}": sum(row[index] for row in rows) / len(rows)
        for source, rows in populations.items()
        for index, metric in enumerate(("contract_correct", "contract_completed"))
    }
