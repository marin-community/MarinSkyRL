"""Admit complete prompt groups against a learner token budget."""

from dataclasses import dataclass

TOKEN_MATCH_TOLERANCE = 0.001


@dataclass(frozen=True)
class TokenBudgetSelection:
    indices: list[int]
    input_tokens: int


def select_token_budget_groups(
    sequence_lengths: list[int], uids: list[str], remaining_tokens: int, group_size: int
) -> TokenBudgetSelection:
    """Keep whole groups that fit, preserving order and complete reference actions."""
    groups: dict[str, list[int]] = {}
    for index, uid in enumerate(uids):
        groups.setdefault(uid, []).append(index)
    if len(sequence_lengths) != len(uids) or any(len(group) != group_size for group in groups.values()):
        raise ValueError("Token budgeting requires complete, aligned prompt groups")
    indices = []
    consumed = 0
    for group in groups.values():
        cost = sum(sequence_lengths[index] for index in group)
        if cost <= 0:
            raise ValueError("A training group must contain tokens")
        if consumed + cost <= remaining_tokens:
            indices.extend(group)
            consumed += cost
    if not indices:
        raise ValueError("Remaining learner token budget cannot fit a complete prompt group")
    return TokenBudgetSelection(sorted(indices), consumed)
