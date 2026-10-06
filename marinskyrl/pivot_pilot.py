"""Pilot scheduling and diagnostic reductions for reported pivot verifiers."""

from collections import defaultdict
from collections.abc import Iterable
from typing import Any


def evaluation_kind(arm: str, step: int, loss_total: int = 0, loss_step: int = 0) -> str | None:
    """Choose one generation pass, giving full evaluation precedence over quick."""
    if step == 0:
        return "full"
    if arm.startswith("rl_"):
        if step in (5, 10, 15, 20):
            return "full"
        return "quick" if step <= 20 and step % 2 == 0 else None
    return "full" if loss_total // 250000 > (loss_total - loss_step) // 250000 else None


def quick_evaluation_batches(
    prompt_batches: Iterable[list[dict[str, Any]]], source_ids: Iterable[str], batch_size: int
) -> list[list[dict[str, Any]]]:
    """Pack the fixed quick-evaluation rows in validation order."""
    quick_ids = set(source_ids)
    prompts = [
        prompt
        for batch in prompt_batches
        for prompt in batch
        if prompt["env_extras"]["extra_info"]["source_id"] in quick_ids
    ]
    if (
        len(prompts) != len(quick_ids)
        or {prompt["env_extras"]["extra_info"]["source_id"] for prompt in prompts} != quick_ids
    ):
        raise ValueError("Quick-evaluation source IDs must appear exactly once in validation")
    return [prompts[start : start + batch_size] for start in range(0, len(prompts), batch_size)]


def diagnostic_metrics(batch, *, prefix: str, indices: list[int] | None = None) -> dict[str, float]:
    """Reduce diagnostic grades without touching optimization rewards or masks."""
    verdicts = batch.get("verification_results")
    if verdicts is None:
        return {}
    indices = list(range(len(verdicts))) if indices is None else indices
    grades = [(index, verdicts[index].diagnostics.get("pivot")) for index in indices if verdicts[index] is not None]
    usable = [(index, grade) for index, grade in grades if grade is not None]
    if not usable:
        return {}
    result = {f"{prefix}/graded_fraction": len(usable) / len(indices)}
    verifiers = sorted({verifier for _, grade in usable for verifier in grade["scores"]})
    for verifier in verifiers:
        observed = [(index, grade) for index, grade in usable if verifier in grade["scores"]]
        result[f"{prefix}/{verifier}/accuracy"] = sum(g["scores"][verifier] for _, g in observed) / len(observed)
        groups = defaultdict(list)
        for index, grade in observed:
            groups[batch["trajectory_ids"][index].instance_id].append(grade["scores"][verifier])
        complete = [scores for scores in groups.values() if len(scores) == 8]
        if complete:
            result[f"{prefix}/{verifier}/mixed_group_fraction"] = sum(0 < sum(g) < 8 for g in complete) / len(complete)
    for field in ("malformed_tool_calls", "extra_tool_calls"):
        observed = [grade[field] for _, grade in usable if field in grade]
        if observed:
            result[f"{prefix}/{field}"] = sum(observed) / len(observed)
    return result
