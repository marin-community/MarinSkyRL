"""Mild per-think-token cost (Stage D / M4).

Mechanism 4 of the loop-behavior reward plan: apply gentle *active* pressure
against thinking-faff. F7's loss down-weighting (``build_think_weighted_loss_mask``
in ``ppo_utils.py``) is the *passive* half — it stops RL from *reinforcing*
think-token growth (credit no longer flows full-strength to ``<think>`` tokens).
M4 is the *active* half — a small negative token reward on ``<think>`` tokens that
*penalizes* faffing, ``capped per turn`` so it can never dominate the outcome.

Design (mirrors ``pbs_shaping.compute_pbs_token_shaping``):
  * Writes a small negative value (``think_token_cost``, default 0.0 = OFF) onto
    each ``SPAN_THINK``-tagged response token.
  * **Capped per turn**: the total cost for any single assistant turn's think span
    is clamped to ``-max_cost_per_turn`` (default 0.05), so a single long ramble
    can't swamp the trajectory. Within a turn the (capped) total is scattered
    uniformly across that turn's THINK tokens.
  * Rides the SAME additive ``token_level_shaping`` channel + ``rloo_n_pbs`` seam
    as Stage C's PBS credit. PBS writes EDIT tokens; M4 writes THINK tokens — the
    spans are **non-overlapping**, so the two can be summed into one channel
    (``compute_think_token_cost`` is additive-friendly: it only touches THINK
    positions, leaving any pre-existing PBS/EDIT entries intact).

BYTE-IDENTICAL CONTRACT: ``think_token_cost == 0.0`` ⇒ an all-zeros vector ⇒ the
channel is unchanged ⇒ the additive seam is a no-op ⇒ advantages bit-identical.

Pure / CPU-only; no torch dependency. Unit-testable in isolation.
"""

from __future__ import annotations

from typing import List, Optional

# SPAN_THINK == 1 (mirrors skyrl_train.utils.span_tagger.SPAN_THINK). Defined
# locally so this module stays torch- and generator-free (span_tagger imports the
# generator stack at module load, which we don't need just for the constant).
SPAN_THINK: int = 1

# Default M4 hyperparameters (override via reward_shaping config).
DEFAULT_THINK_TOKEN_COST: float = 0.0  # OFF by default (per-think-token negative)
DEFAULT_MAX_COST_PER_TURN: float = 0.05  # |per-turn total| ceiling (kept mild)


def _think_turn_token_ranges(response_span_tags: List[int]) -> List[List[int]]:
    """Group maximal runs of consecutive THINK positions into per-turn ranges.

    A single assistant turn's ``<think>`` span is a contiguous run of THINK tags
    (the tagger emits THINK 1:1 with the think tokens, bounded by ACTION/EDIT/
    OTHER on either side). Returns, in trajectory order, the token-index list for
    each such THINK run.
    """
    ranges: List[List[int]] = []
    cur: List[int] = []
    for j, tag in enumerate(response_span_tags):
        if tag == SPAN_THINK:
            cur.append(j)
        elif cur:
            ranges.append(cur)
            cur = []
    if cur:
        ranges.append(cur)
    return ranges


def compute_think_token_cost(
    response_span_tags: List[int],
    *,
    think_token_cost: float = DEFAULT_THINK_TOKEN_COST,
    max_cost_per_turn: float = DEFAULT_MAX_COST_PER_TURN,
    base: Optional[List[float]] = None,
) -> List[float]:
    """Per-token think-cost vector for one trajectory (M4).

    Args:
        response_span_tags: the F4 per-token span tags (SPAN_THINK==1), 1:1 with
            ``response_ids``. Its length defines the output length.
        think_token_cost: NEGATIVE-magnitude cost applied per THINK token (a
            positive number here is interpreted as the magnitude; the written
            value is ``-abs(think_token_cost)`` per token). ``0.0`` ⇒ no-op.
        max_cost_per_turn: ceiling on the |total| cost charged to any single
            think span (default 0.05). The per-turn total is clamped to this
            magnitude before being scattered uniformly across the turn's THINK
            tokens (keeps a single long ramble from dominating).
        base: optional pre-existing channel vector (e.g. Stage C's PBS EDIT
            credit) to ADD the think-cost onto. When given, it is copied and the
            think-cost is summed at THINK positions only (non-overlapping with
            EDIT). When None, a fresh zeros vector is used.

    Returns:
        A list of floats, len == len(response_span_tags). Non-positive entries
        land ONLY on THINK tokens (plus any ``base`` entries elsewhere). All-zero
        (or == ``base``) when ``think_token_cost == 0.0`` — the flag-off path.
    """
    n = len(response_span_tags)
    if base is not None:
        assert len(base) == n, "base channel must match response_span_tags length"
        out = list(base)
    else:
        out = [0.0] * n

    # Flag-off / no-think fast paths leave the channel exactly as `base`.
    if think_token_cost == 0.0 or n == 0:
        return out

    cost_mag = abs(float(think_token_cost))
    cap = abs(float(max_cost_per_turn))

    for think_toks in _think_turn_token_ranges(response_span_tags):
        if not think_toks:
            continue
        # Per-turn total cost = (#think tokens) * per-token cost, clamped to the cap.
        turn_total = cost_mag * len(think_toks)
        if turn_total > cap:
            turn_total = cap
        per_tok = -turn_total / len(think_toks)
        for j in think_toks:
            out[j] += per_tok

    return out
