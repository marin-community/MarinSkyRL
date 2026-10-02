import pytest
import torch
from vllm.vllm_flash_attn import flash_attn_varlen_func

from skyrl_train.models.grug_fa3_invariant import (
    FA3_BLOCK_M,
    FA3_DYNAMIC_SPLIT_MAX_BATCH,
    FA3_INVARIANT_SPLITS,
    fa3_fixed_split_metadata,
    fa3_invariant_requests,
    fa3_request_calls,
    fa3_window_start_rows,
)
from skyrl_train.models.grug_vllm_kernels import fa3_attention_sbhd
from tests.gpu.grug_gpu_gates import require_hoppers

HEADS, KV_HEADS, HEAD_DIM, BLOCK, WINDOW = 20, 5, 128, 16, 2048
GROUP = HEADS // KV_HEADS
SCALE = HEAD_DIM**-0.5
LENGTHS = (700, 2049, 3100)


def _paged_calls(query, caches, query_start, key_lengths, tables, window, *, causal_rows):
    """FA3 calls as the engine issues them: paged keys, fixed-split metadata, at most 992 requests per call, and with
    ``causal_rows`` on a sliding-window layer the one-row requests in calls of their own on FA3's causal kernel from
    each row's window start; without it every request on the local kernel."""
    output = torch.empty_like(query)
    calls = fa3_request_calls(
        query_start, one_row_calls=causal_rows and window is not None, max_requests=FA3_DYNAMIC_SPLIT_MAX_BATCH
    )
    for begin, end, one_row in calls:
        rows = slice(query_start[begin], query_start[end])
        starts = torch.tensor([s - query_start[begin] for s in query_start[begin : end + 1]], dtype=torch.int32)
        starts = starts.cuda()
        lengths = torch.tensor(key_lengths[begin:end], dtype=torch.int32, device="cuda")
        metadata = fa3_fixed_split_metadata(starts, GROUP, KV_HEADS)
        if one_row:
            fa3_window_start_rows(
                query[rows],
                caches[0],
                caches[1],
                output[rows],
                cu_seqlens_q=starts,
                seqused_k=lengths,
                leftpad_k=torch.clamp(lengths - window, min=0),
                max_seqlen_k=max(key_lengths[begin:end]),
                block_table=tables[begin:end],
                softmax_scale=SCALE,
                scheduler_metadata=metadata,
                num_splits=FA3_INVARIANT_SPLITS,
            )
            continue
        flash_attn_varlen_func(
            q=query[rows],
            k=caches[0],
            v=caches[1],
            out=output[rows],
            cu_seqlens_q=starts,
            max_seqlen_q=int((starts[1:] - starts[:-1]).max()),
            seqused_k=lengths,
            max_seqlen_k=max(key_lengths[begin:end]),
            softmax_scale=SCALE,
            causal=True,
            window_size=None if window is None else [window - 1, 0],
            block_table=tables[begin:end],
            scheduler_metadata=metadata,
            fa_version=3,
            num_splits=FA3_INVARIANT_SPLITS,
        )
    return output


def _sequences(generator):
    blocks = [-(-length // BLOCK) for length in LENGTHS]
    caches = [torch.zeros(sum(blocks), BLOCK, KV_HEADS, HEAD_DIM, dtype=torch.bfloat16, device="cuda") for _ in "kv"]
    order = torch.randperm(sum(blocks), generator=generator).to(torch.int32).cuda()
    tables, sequences, used = [], [], 0
    for length, count in zip(LENGTHS, blocks, strict=True):
        q = (torch.randn(length, HEADS, HEAD_DIM, generator=generator) * 2).to(torch.bfloat16).cuda()
        k = (torch.randn(length, KV_HEADS, HEAD_DIM, generator=generator) * 2).to(torch.bfloat16).cuda()
        v = torch.randn(length, KV_HEADS, HEAD_DIM, generator=generator).to(torch.bfloat16).cuda()
        table = order[used : used + count]
        used += count
        for cache, values in zip(caches, (k, v), strict=True):
            padded = torch.zeros(count * BLOCK, KV_HEADS, HEAD_DIM, dtype=torch.bfloat16, device="cuda")
            padded[:length] = values
            cache[table.long()] = padded.view(count, BLOCK, KV_HEADS, HEAD_DIM)
        tables.append(table)
        sequences.append((q, k, v))
    width = max(blocks)
    table_rows = torch.stack([torch.nn.functional.pad(table, (0, width - table.numel())) for table in tables])
    return sequences, caches, table_rows


def _same_rows(left, right) -> bool:
    return torch.equal(left.contiguous().view(torch.int16), right.contiguous().view(torch.int16))


@pytest.mark.parametrize("window", [None, WINDOW], ids=["full", "sliding_window"])
def test_engine_steps_and_trainer_rows_equal_rows_decoded_alone(window):
    require_hoppers(1)
    sequences, caches, tables = _sequences(torch.Generator().manual_seed(0))
    for index, (length, (q, k, v)) in enumerate(zip(LENGTHS, sequences, strict=True)):
        decode = (q, caches, list(range(length + 1)), list(range(1, length + 1)), tables[[index] * length], window)
        decoded = _paged_calls(*decode, causal_rows=True)
        # The local kernel gives every row decoded alone the same bytes.
        assert _same_rows(decoded, _paged_calls(*decode, causal_rows=False))
        # A prefill from a 16-token cached prefix, through the engine's request plan (its rows before position 32
        # run as their own request; on a sliding-window layer each row past the window alone).
        planned = fa3_invariant_requests([0, length - 16], [length], window, FA3_BLOCK_M // GROUP)
        prefill = _paged_calls(
            q[16:],
            caches,
            planned.query_start.tolist(),
            planned.key_lengths.tolist(),
            tables[[index] * planned.key_lengths.numel()],
            window,
            causal_rows=True,
        )
        assert _same_rows(prefill, decoded[16:])

        trainer = fa3_attention_sbhd(q[:, None], k[:, None], v[:, None], window=window, scale=SCALE)
        assert _same_rows(trainer.view(length, HEADS, HEAD_DIM), decoded)
