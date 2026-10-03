from types import SimpleNamespace

import pytest
import torch
from vllm.v1.attention.backends.flash_attn import FlashAttentionImpl, FlashAttentionMetadata

from skyrl_train.inference_engines.vllm import decode_invariant
from skyrl_train.models.grug_fa3_invariant import (
    FA3_DYNAMIC_SPLIT_MAX_BATCH,
    FA3_INVARIANT_SPLITS,
    fa3_fixed_split_metadata,
)
from skyrl_train.models.grug_vllm_kernels import fa3_attention_sbhd
from tests.gpu.grug_gpu_gates import require_hoppers

HEADS, KV_HEADS, HEAD_DIM, BLOCK, WINDOW = 20, 5, 128, 16, 2048
GROUP = HEADS // KV_HEADS
SCALE = HEAD_DIM**-0.5
LENGTHS = (700, 2049, 3100)
VLLM_FORWARD = FlashAttentionImpl.forward


def _step(query_start, key_lengths, block_table) -> FlashAttentionMetadata:
    """A step's FA3 metadata as the decode-invariant engine's metadata builder writes it."""
    query_start_loc = torch.tensor(query_start, dtype=torch.int32, device="cuda")
    return FlashAttentionMetadata(
        num_actual_tokens=query_start[-1],
        max_query_len=max(end - begin for begin, end in zip(query_start, query_start[1:])),
        query_start_loc=query_start_loc,
        max_seq_len=max(key_lengths),
        seq_lens=torch.tensor(key_lengths, dtype=torch.int32, device="cuda"),
        block_table=block_table,
        slot_mapping=torch.empty(0, dtype=torch.int64, device="cuda"),
        use_cascade=False,
        common_prefix_len=0,
        cu_prefix_query_lens=None,
        prefix_kv_lens=None,
        suffix_kv_lens=None,
        scheduler_metadata=fa3_fixed_split_metadata(query_start_loc, GROUP, KV_HEADS),
        max_num_splits=FA3_INVARIANT_SPLITS,
    )


def _attend(forward, impl, query, kv_cache, query_start, key_lengths, block_table):
    layer = SimpleNamespace(**{name: torch.ones((), device="cuda") for name in ("_q_scale", "_k_scale", "_v_scale")})
    output = torch.empty_like(query)
    forward(impl, layer, query, None, None, kv_cache, _step(query_start, key_lengths, block_table), output)
    return output


def _decoded_alone(impl, query, kv_cache, table):
    """Each row of ``query`` as a one-row request of a step of at most ``FA3_DYNAMIC_SPLIT_MAX_BATCH`` requests, run by
    vLLM's FA3 forward."""
    output = torch.empty_like(query)
    for begin in range(0, len(query), FA3_DYNAMIC_SPLIT_MAX_BATCH):
        end = min(begin + FA3_DYNAMIC_SPLIT_MAX_BATCH, len(query))
        output[begin:end] = _attend(
            VLLM_FORWARD,
            impl,
            query[begin:end],
            kv_cache,
            list(range(end - begin + 1)),
            list(range(begin + 1, end + 1)),
            table.repeat(end - begin, 1),
        )
    return output


def _sequences(generator):
    blocks = [-(-length // BLOCK) for length in LENGTHS]
    kv_cache = torch.zeros(sum(blocks), KV_HEADS, BLOCK, 2 * HEAD_DIM, dtype=torch.bfloat16, device="cuda")
    order = torch.randperm(sum(blocks), generator=generator).cuda()
    queries, tables, used = [], [], 0
    for length, count in zip(LENGTHS, blocks, strict=True):
        queries.append((torch.randn(length, HEADS, HEAD_DIM, generator=generator) * 2).to(torch.bfloat16).cuda())
        keys = torch.randn(length, KV_HEADS, HEAD_DIM, generator=generator) * 2
        values = torch.randn(length, KV_HEADS, HEAD_DIM, generator=generator)
        padded = torch.zeros(count * BLOCK, KV_HEADS, 2 * HEAD_DIM)
        padded[:length] = torch.cat([keys, values], dim=-1)
        table = order[used : used + count]
        used += count
        kv_cache[table] = padded.view(count, BLOCK, KV_HEADS, 2 * HEAD_DIM).transpose(1, 2).to(torch.bfloat16).cuda()
        tables.append(table.to(torch.int32))
    width = max(blocks)
    table_rows = torch.stack([torch.nn.functional.pad(table, (0, width - table.numel())) for table in tables])
    return queries, kv_cache, table_rows


def _sequence_keys_values(kv_cache, table, length):
    """The ``[length, KV_HEADS, HEAD_DIM]`` keys and values of the sequence whose blocks ``table`` lists."""
    blocks = kv_cache[table[: -(-length // BLOCK)].long()].transpose(1, 2).reshape(-1, KV_HEADS, 2 * HEAD_DIM)
    return blocks[:length, :, :HEAD_DIM].contiguous(), blocks[:length, :, HEAD_DIM:].contiguous()


def _same_rows(left, right) -> bool:
    return torch.equal(left.contiguous().view(torch.int16), right.contiguous().view(torch.int16))


@pytest.mark.parametrize("window", [None, WINDOW], ids=["full", "sliding_window"])
def test_engine_steps_and_trainer_rows_equal_rows_decoded_alone(window):
    require_hoppers(1)
    decode_invariant.install()
    impl = FlashAttentionImpl(HEADS, HEAD_DIM, SCALE, KV_HEADS, None, window, "auto")
    queries, kv_cache, tables = _sequences(torch.Generator().manual_seed(0))
    for length, query, table in zip(LENGTHS, queries, tables, strict=True):
        table = table[None]
        decoded = _decoded_alone(impl, query, kv_cache, table)

        # Decode steps of every row and of the last FA3_DYNAMIC_SPLIT_MAX_BATCH rows.
        for first in (0, max(length - FA3_DYNAMIC_SPLIT_MAX_BATCH, 0)):
            count = length - first
            step = (list(range(count + 1)), list(range(first + 1, length + 1)), table.repeat(count, 1))
            assert _same_rows(
                _attend(FlashAttentionImpl.forward, impl, query[first:], kv_cache, *step), decoded[first:]
            )
        # A prefill of the rows after a 16-token cached prefix.
        prefill = _attend(FlashAttentionImpl.forward, impl, query[16:], kv_cache, [0, length - 16], [length], table)
        assert _same_rows(prefill, decoded[16:])

        keys, values = _sequence_keys_values(kv_cache, table[0], length)
        trainer = fa3_attention_sbhd(query[:, None], keys[:, None], values[:, None], window=window, scale=SCALE)
        assert _same_rows(trainer.view(length, HEADS, HEAD_DIM), decoded)
