"""Pure scheduling and diagnostic reductions for the SWE pilot."""

from collections import defaultdict

VERIFIERS = ("tool_name", "nemo", "exact")


def evaluation_kind(arm: str, step: int, loss_total: int = 0, loss_step: int = 0) -> str | None:
    """Choose one generation pass, giving full evaluation precedence over quick."""
    if step == 0:
        return "full"
    if arm.startswith("rl_"):
        if step in (5, 10, 15, 20):
            return "full"
        return "quick" if step <= 20 and step % 2 == 0 else None
    return "full" if loss_total // 250000 > (loss_total - loss_step) // 250000 else None


def diagnostic_metrics(batch, *, prefix: str, indices: list[int] | None = None) -> dict[str, float]:
    """Reduce diagnostic grades without touching optimization rewards or masks."""
    verdicts = batch.get("verification_results")
    if verdicts is None:
        return {}
    indices = list(range(len(verdicts))) if indices is None else indices
    grades = [(index, verdicts[index].diagnostics.get("pivot")) for index in indices]
    usable = [(index, grade) for index, grade in grades if grade is not None]
    if not usable:
        return {}
    result = {f"{prefix}/graded_fraction": len(usable) / len(indices)}
    for verifier in VERIFIERS:
        result[f"{prefix}/{verifier}/accuracy"] = sum(g["scores"][verifier] for _, g in usable) / len(usable)
        groups = defaultdict(list)
        for index, grade in usable:
            groups[batch["trajectory_ids"][index].instance_id].append(grade["scores"][verifier])
        complete = [scores for scores in groups.values() if len(scores) == 8]
        if complete:
            result[f"{prefix}/{verifier}/mixed_group_fraction"] = sum(0 < sum(g) < 8 for g in complete) / len(complete)
    for field in ("malformed_tool_calls", "extra_tool_calls"):
        result[f"{prefix}/{field}"] = sum(grade[field] for _, grade in usable) / len(usable)
    return result
