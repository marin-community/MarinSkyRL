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

    A trial whose generation terminated at ``max_generate_length``
    (``stop_reason == "length"``) and failed its verifier (``original_reward == 0.0``)
    has ``truncation_penalty`` SUBTRACTED from its reward, taking it below the zero
    floor so RLOO assigns it a negative advantage instead of lumping it with honest
    failures at zero.

    The penalty subtracts rather than overwrites, which matters as soon as a
    partial-credit shaper (``pass_ratio``) is enabled. Overwriting would score every
    truncated trial at the same value regardless of how many tests it passed,
    flattening the partial-credit signal on exactly the population the penalty is
    aimed at — the same "indistinguishable outcomes" defect the penalty exists to fix,
    relocated rather than removed. Subtracting keeps trials ordered by how much of the
    task they actually completed while still ranking each one below its untruncated
    equivalent.

    With no shaper the two are identical: ``reward == original_reward == 0.0``, so
    subtracting yields ``-truncation_penalty`` either way.

    Returns ``(new_reward, penalized)``.
    """
    if truncation_penalty > 0.0 and stop_reason == "length" and original_reward == 0.0:
        return reward - truncation_penalty, True
    return reward, False
