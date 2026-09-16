import torch


def shard_dense_sequence_for_context_parallel(
    tensor: torch.Tensor, *, cp_size: int, cp_rank: int, sequence_dim: int = 1
) -> torch.Tensor:
    """Select Megatron's two load-balanced causal chunks for one CP rank."""
    if cp_size < 1:
        raise ValueError(f"cp_size must be positive, got {cp_size}")
    if not 0 <= cp_rank < cp_size:
        raise ValueError(f"cp_rank must be in [0, {cp_size}), got {cp_rank}")
    if cp_size == 1:
        return tensor

    sequence_length = tensor.shape[sequence_dim]
    chunk_count = 2 * cp_size
    if sequence_length % chunk_count != 0:
        raise ValueError(f"sequence length {sequence_length} must be divisible by 2 * cp_size ({chunk_count})")

    chunks = torch.chunk(tensor, chunk_count, dim=sequence_dim)
    return torch.cat((chunks[cp_rank], chunks[chunk_count - cp_rank - 1]), dim=sequence_dim)
