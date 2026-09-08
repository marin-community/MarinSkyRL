"""Map synchronous optimizer ages to actual contiguous DP-shard minibatches."""

import torch


def consumed_update_age_counts(
    response_mask: torch.Tensor, *, dp_size: int, mini_batch_sequences: int, samples_per_prompt: int, epochs: int
) -> list[dict[str, int]]:
    """Count the exact rows consumed by each local-minibatch optimizer update.

    MeshDispatch first divides all response rows into contiguous DP shards.
    TrainingBatchIterator then traverses contiguous minibatches within each
    shard. Thus update k uses one strip per DP rank, not one global contiguous
    minibatch. Replicated TP/CP/EP/SP ranks must not increase ``dp_size``.
    """
    rows = response_mask.shape[0]
    if any(
        type(value) is not int or value <= 0 for value in (dp_size, mini_batch_sequences, samples_per_prompt, epochs)
    ):
        raise ValueError("Type-C geometry requires positive integer counts")
    if rows <= 0 or rows % mini_batch_sequences or mini_batch_sequences % dp_size or rows % dp_size:
        raise ValueError("Type-C rows and minibatches must divide exactly across DP shards")
    if mini_batch_sequences % samples_per_prompt:
        raise ValueError("Type-C minibatches must contain whole response groups")
    updates_per_batch = rows // mini_batch_sequences
    per_rank_rows = rows // dp_size
    per_rank_mini = mini_batch_sequences // dp_size
    if per_rank_mini % samples_per_prompt:
        raise ValueError("Type-C DP minibatches must retain complete response groups")
    lengths = response_mask.detach().sum(-1).reshape(dp_size, per_rank_rows)
    results = []
    for epoch in range(epochs):
        for index in range(updates_per_batch):
            tokens = lengths[:, index * per_rank_mini : (index + 1) * per_rank_mini].sum().item()
            results.append(
                {
                    "age": epoch * updates_per_batch + index,
                    "groups": mini_batch_sequences // samples_per_prompt,
                    "sequences": mini_batch_sequences,
                    "response_tokens": int(tokens),
                }
            )
    return results


def optimizer_success_counts(updates: list[dict[str, float]]) -> dict[str, float]:
    """Count updates whose reduced native optimizer-success flag is exactly true."""
    flags = [row.get("optimizer_step_succeeded") for row in updates]
    if not flags or any(flag not in (0.0, 1.0) for flag in flags):
        return {"policy_successful_update_steps_valid": 0.0}
    return {"policy_successful_update_steps": float(sum(flags)), "policy_successful_update_steps_valid": 1.0}
