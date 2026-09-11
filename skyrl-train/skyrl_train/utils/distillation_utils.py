"""
Distillation utility functions.

Provides Best-of-N selection logic.
"""

from typing import List


def best_of_n_select(
    rewards: List[float],
    n_samples_per_prompt: int,
) -> List[int]:
    """Select the best sample per prompt group based on reward.

    Args:
        rewards: Flat list of per-sample rewards (response-level, not token-level).
            Length must be divisible by n_samples_per_prompt.
        n_samples_per_prompt: Number of samples generated per prompt.

    Returns:
        List of indices (into the flat rewards list) of the best sample per group.
    """
    assert len(rewards) % n_samples_per_prompt == 0, (
        f"Number of rewards ({len(rewards)}) must be divisible by n_samples_per_prompt ({n_samples_per_prompt})"
    )
    num_prompts = len(rewards) // n_samples_per_prompt
    best_indices = []
    for i in range(num_prompts):
        start = i * n_samples_per_prompt
        end = start + n_samples_per_prompt
        group_rewards = rewards[start:end]
        best_local = max(range(len(group_rewards)), key=lambda j: group_rewards[j])
        best_indices.append(start + best_local)
    return best_indices
