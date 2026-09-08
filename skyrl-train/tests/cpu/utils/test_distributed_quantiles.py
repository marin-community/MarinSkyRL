"""Compare actual distributed selection to the concatenated token oracle."""

from concurrent.futures import ThreadPoolExecutor
from threading import Barrier

import pytest
import torch

from skyrl_train.utils.distributed_quantiles import AbsoluteQuantileBuffer


def pooled(buffers, ownership=None):
    barrier = Barrier(len(buffers))
    inputs = [None] * len(buffers)
    calls = [0] * len(buffers)

    def run(rank):
        def reduce(value):
            inputs[rank] = value.clone()
            barrier.wait(timeout=10)
            result = torch.stack(inputs).sum(0)
            barrier.wait(timeout=10)
            calls[rank] += 1
            return result

        return buffers[rank].quantiles(reduce, owns_tokens=ownership[rank] if ownership else True)

    with ThreadPoolExecutor(max_workers=len(buffers)) as executor:
        outputs = list(executor.map(run, range(len(buffers))))
    assert all(output == outputs[0] for output in outputs)
    assert len(set(calls)) == 1
    return outputs[0], calls[0]


def buffer(*chunks, capacity=4_194_304):
    result = AbsoluteQuantileBuffer(torch.device("cpu"), capacity)
    for chunk in chunks:
        result.add(chunk)
    return result


@pytest.mark.parametrize("seed", [1, 17, 431])
def test_unequal_shards_and_microbatches_match_exact_float32_oracle(seed):
    generator = torch.Generator().manual_seed(seed)
    values = [torch.randn(size, generator=generator) * (rank + 1) for rank, size in enumerate((0, 1, 77, 10003))]
    buffers = [buffer(*value.split(127)) for value in values]
    actual, calls = pooled(buffers)
    oracle = torch.quantile(torch.cat(values).abs().double(), torch.tensor([0.5, 0.95], dtype=torch.float64))
    assert actual["abs_log_ratio_p50"] == pytest.approx(oracle[0].item(), abs=1e-12)
    assert actual["abs_log_ratio_p95"] == pytest.approx(oracle[1].item(), abs=1e-12)
    assert actual["quantiles_valid"] == 1
    assert calls == 5
    assert sum(item.retained_bytes for item in buffers) == sum(value.numel() for value in values) * 4


def test_nonfinite_coverage_ties_extremes_and_replica_ownership():
    values = torch.tensor([0, -0.0, 1e-30, -1e-20, 1, 1, 1e30, float("nan"), float("inf")])
    actual, calls = pooled([buffer(values), buffer(torch.ones(30)), buffer(values)], [True, True, False])
    oracle = torch.quantile(torch.cat([values[torch.isfinite(values)].abs(), torch.ones(30)]).double(), 0.95)
    assert actual["selected_tokens"] == 39
    assert actual["finite_tokens"] == 37
    assert actual["finite_fraction"] == 37 / 39
    assert actual["abs_log_ratio_p95"] == pytest.approx(oracle.item(), rel=1e-12)
    assert actual["abs_log_ratio_p50"] == 1
    assert calls == 5


def test_overflow_is_global_invalid_and_retention_is_bounded():
    small = buffer(torch.ones(4), capacity=4)
    assert small.retained_bytes == 16
    small.add(torch.ones(1))
    assert small.retained_bytes == 0
    actual, calls = pooled([small, buffer(torch.ones(10)), buffer(torch.empty(0))])
    assert actual["quantiles_valid"] == 0
    assert actual["quantiles_overflow"] == 1
    assert actual["finite_tokens"] == 15
    assert calls == 1


def test_empty_shards_and_single_value():
    actual, calls = pooled([buffer(torch.empty(0)), buffer(torch.empty(0))])
    assert actual["quantiles_valid"] == 0
    assert calls == 1
    actual, calls = pooled([buffer(torch.tensor([-3.25])), buffer(torch.empty(0))])
    assert actual["abs_log_ratio_p50"] == actual["abs_log_ratio_p95"] == 3.25
    assert actual["quantiles_valid"] == 1
    assert calls == 5


def test_full_bound_owns_compact_storage_and_scan_chunks_are_bounded():
    values = torch.ones(4_194_304)
    retained = buffer(values)
    assert retained.retained_bytes == 16 * 1024**2
    assert max(chunk.numel() for chunk in retained.chunks) <= 65_536
    assert all(chunk.untyped_storage().nbytes() == chunk.numel() * 4 for chunk in retained.chunks)
    values.zero_()
    assert all(chunk.min().item() == 1 for chunk in retained.chunks)


def test_float32_conversion_overflow_is_explicitly_invalid():
    actual, calls = pooled([buffer(torch.tensor([1e100], dtype=torch.float64)), buffer(torch.ones(2))])
    assert actual["finite_tokens"] == 3
    assert actual["finite_fraction"] == 1
    assert actual["quantiles_valid"] == 0
    assert actual["quantiles_nonrepresentable_tokens"] == 1
    assert calls == 1
