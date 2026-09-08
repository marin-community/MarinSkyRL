from concurrent.futures import ThreadPoolExecutor
from threading import Barrier

import pytest
import torch

from skyrl_train.utils.gradient_direction import GradientDirectionTracker


class SumCollective:
    """In-memory collective that waits for every independently computed shard."""

    def __init__(self, ranks):
        self.values = [None] * ranks
        self.barrier = Barrier(ranks, timeout=10)

    def reducer(self, rank):
        def reduce(value):
            self.values[rank] = value.clone()
            self.barrier.wait()
            result = torch.stack(self.values).sum(0)
            self.barrier.wait()
            return result

        return reduce


def test_cosine_identical_opposite_and_owned_snapshot():
    tracker = GradientDirectionTracker("gpu_fp32", torch.device("cpu"))
    grad = torch.tensor([3.0, 4.0])
    first = tracker.observe([grad])
    assert first["grad_cosine_valid"] == 0 and first["grad_norm_reduced"] == 5
    assert tracker.observe([grad])["grad_cosine"] == pytest.approx(1.0, abs=1e-6)
    grad.neg_()  # The retained previous gradient must not alias optimizer storage.
    opposite = tracker.observe([grad])
    assert opposite["grad_cosine_valid"] == 1
    assert opposite["grad_cosine"] == pytest.approx(-1.0, abs=1e-6)


@pytest.mark.parametrize("store", ["gpu_fp32", "cpu_bf16"])
def test_uneven_shards_with_empty_rank_match_flat_reference(store):
    generator = torch.Generator().manual_seed(1741)
    vectors = [torch.randn(103, generator=generator) for _ in range(3)]
    collective = SumCollective(3)
    trackers = [GradientDirectionTracker(store, torch.device("cpu"), reduce_fn=collective.reducer(r)) for r in range(3)]
    with ThreadPoolExecutor(3) as executor:
        for step, vector in enumerate(vectors):
            shards = [[vector[:1], vector[1:3]], [vector[3:10], vector[10:65], vector[65:]], []]
            results = list(executor.map(lambda pair: pair[0].observe(pair[1]), zip(trackers, shards, strict=True)))
            for result in results:
                assert result["grad_norm_reduced"] == pytest.approx(vector.norm().item(), rel=1e-6)
                assert result["grad_cosine_valid"] == int(step > 0)
                if step:
                    expected = torch.nn.functional.cosine_similarity(vector, vectors[step - 1], dim=0).item()
                    assert result["grad_cosine"] == pytest.approx(expected, abs=1e-6 if store == "gpu_fp32" else 1e-2)


@pytest.mark.parametrize("invalid", ["skipped", "nan", "inf"])
def test_invalid_rank_resets_every_rank_until_next_complete_pair(invalid):
    collective = SumCollective(2)
    trackers = [
        GradientDirectionTracker("gpu_fp32", torch.device("cpu"), reduce_fn=collective.reducer(r)) for r in range(2)
    ]
    with ThreadPoolExecutor(2) as executor:
        for step in range(5):
            bad = step == 2
            gradients = [torch.tensor([2.0]), torch.tensor([float(invalid) if bad and invalid != "skipped" else 3.0])]

            def observe(rank):
                return trackers[rank].observe(
                    [gradients[rank]], successful=not (bad and rank == 1 and invalid == "skipped")
                )

            results = list(executor.map(observe, range(2)))
            assert [r["grad_cosine_valid"] for r in results] == [float(step in {1, 4})] * 2
            assert [r["grad_norm_valid"] for r in results] == [float(not bad)] * 2


def test_chunked_gradient_matches_flat_reference():
    generator = torch.Generator().manual_seed(83)
    first, second = [torch.randn((1 << 20) + 17, generator=generator) for _ in range(2)]
    tracker = GradientDirectionTracker("gpu_fp32", torch.device("cpu"))
    tracker.observe([first])
    result = tracker.observe([second])
    expected = torch.nn.functional.cosine_similarity(first, second, dim=0).item()
    assert result["grad_cosine"] == pytest.approx(expected, abs=1e-6)
    assert result["grad_dot"] == pytest.approx(torch.dot(first, second).item(), abs=1e-3)


def test_zero_norm_and_shape_change_invalidate_comparison():
    tracker = GradientDirectionTracker("cpu_bf16", torch.device("cpu"))
    tracker.observe([torch.ones(3)])
    zero = tracker.observe([torch.zeros(3)])
    assert zero["grad_norm_reduced"] == 0 and zero["grad_cosine_valid"] == 0
    assert tracker.observe([torch.ones(3)])["grad_cosine_valid"] == 0
    assert tracker.observe([torch.ones(4)])["grad_cosine_valid"] == 0
    assert tracker.observe([torch.ones(4)])["grad_cosine_valid"] == 1
