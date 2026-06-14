"""Independent zigzag-offset oracle for the Stage-5 CP unshard parity test.

This is a CPU-only port of slime's context-parallel index math
(`slime/backends/megatron_utils/cp_utils.py:9-50` `get_logits_and_tokens_offset_with_cp`
and `:299-340` `slice_with_cp`), with the Megatron MPU dependency removed (we pass
`cp_rank` / `cp_size` explicitly). It computes, for a given `(cp_size, seq_len)`,
exactly which natural-order token indices each CP rank's local shard holds under
torch's built-in 2-chunk zigzag load balancer, so we can *independently* verify
that `torch.distributed.tensor.experimental._attention.context_parallel_unshard`
returns tokens in natural order.

Both torch's `context_parallel` load balancer and slime's `slice_with_cp` use the
SAME 2-chunk round-robin (a.k.a. "zigzag") scheme: with `chunk_size = seq_len /
(2*cp)`, rank `r` holds chunk `r` (indices `[r*c, (r+1)*c)`) followed by chunk
`(2*cp - r - 1)` (indices `[(2*cp-r-1)*c, (2*cp-r)*c)`). `seq_len` must be a
multiple of `2*cp` (G4 — the test only feeds divisible lengths to the oracle).
"""

from typing import List, Tuple

import torch


def cp_shard_indices(cp_rank: int, cp_size: int, seq_len: int) -> List[int]:
    """Natural-order token indices held by `cp_rank` under the 2-chunk zigzag shard.

    Returns the list of natural-order indices in the ON-SHARD order (chunk_0 then
    chunk_1), i.e. `local_tensor[j]` corresponds to natural index `out[j]`.

    Mirrors slime `slice_with_cp:327-340` (thd path) exactly:
        chunk_size = ceil(seq_len / (2*cp))   # divisible here -> exact
        start_1, end_1 = c*r,            c*(r+1)
        start_2, end_2 = c*(2*cp-r-1),   c*(2*cp-r)
        local = cat(tokens[start_1:end_1], tokens[start_2:end_2])
    """
    assert cp_size > 1, "oracle is for cp_size > 1"
    assert seq_len % (2 * cp_size) == 0, f"seq_len {seq_len} must be divisible by 2*cp={2 * cp_size}"
    c = seq_len // (2 * cp_size)
    start_1, end_1 = c * cp_rank, c * (cp_rank + 1)
    start_2, end_2 = c * (2 * cp_size - cp_rank - 1), c * (2 * cp_size - cp_rank)
    return list(range(start_1, end_1)) + list(range(start_2, end_2))


def cp_unshard_permutation(cp_size: int, seq_len: int) -> List[int]:
    """The permutation `context_parallel_unshard` must implement.

    If you concatenate every rank's local shard in rank order (rank 0's chunk_0,
    chunk_1, rank 1's chunk_0, chunk_1, ...), entry `k` of that concatenation holds
    natural-order token index `perm[k]`. A correct unshard scatters `perm[k] -> k`
    so the gathered tensor is in natural order `[0, 1, 2, ..., seq_len-1]`.

    This returns `perm` (sharded-concat position -> natural index). The test asserts
    that gathering the per-rank shards and inverting `perm` reproduces natural order
    (equivalently: `context_parallel_unshard` output indexed back through the shard
    layout equals identity).
    """
    perm: List[int] = []
    for r in range(cp_size):
        perm.extend(cp_shard_indices(r, cp_size, seq_len))
    assert sorted(perm) == list(range(seq_len)), "zigzag permutation is not a bijection over [0, seq_len)"
    return perm


def build_sharded_then_unshard_check(cp_size: int, seq_len: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """Helper for a pure-CPU sanity check of the oracle itself.

    Returns (sharded_concat, expected_natural) where `sharded_concat[k]` is a
    value tag = natural index `perm[k]`, and `expected_natural[i] == i`. Inverting
    `perm` on `sharded_concat` must give `expected_natural`. Used to self-test the
    oracle without any distributed/torch-CP machinery.
    """
    perm = cp_unshard_permutation(cp_size, seq_len)
    perm_t = torch.tensor(perm, dtype=torch.long)
    sharded_concat = perm_t.clone()  # value at concat-pos k is its natural index
    inv = torch.empty_like(perm_t)
    inv[perm_t] = torch.arange(seq_len, dtype=torch.long)
    natural = sharded_concat[inv]  # gather natural-order
    expected_natural = torch.arange(seq_len, dtype=torch.long)
    return natural, expected_natural
