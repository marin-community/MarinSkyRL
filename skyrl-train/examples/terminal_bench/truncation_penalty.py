"""Standalone truncation penalty logic.

Extracted from ``terminal_bench_generator.py`` so it can be unit-tested without
pulling in the full generator / skyrl_gym import chain.
"""

from typing import Tuple


def apply_truncation_penalty(
    reward: float,
    original_reward: float,
    stop_reason: str,
    truncation_penalty: float,
) -> Tuple[float, bool]:
    """Penalize a cap-truncated trial so it is distinguishable from an honest wrong answer.

    A trial whose generation terminated at ``max_generate_length`` (``stop_reason == "length"``)
    and failed its verifier (``original_reward == 0.0``) is scored at
    ``-truncation_penalty`` — below the zero floor — so RLOO assigns it a negative
    advantage instead of lumping it with honest failures at zero.

    Returns ``(new_reward, penalized)``.
    """
    if truncation_penalty > 0.0 and stop_reason == "length" and original_reward == 0.0:
        return -truncation_penalty, True
    return reward, False
