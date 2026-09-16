"""Exact weighted route mixtures for multi-teacher training batches."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterator, Mapping
import math
import random
from typing import Any

import datasets
from torch.utils.data import Sampler

ROUTE_COLUMN = "teacher_route"


def weighted_quotas(total: int, weights: Mapping[str, float], batch_index: int) -> dict[str, int]:
    """Allocate an exact batch with stable, rotating ties."""
    if total <= 0 or not weights or any(not math.isfinite(weight) or weight <= 0 for weight in weights.values()):
        raise ValueError("weighted route quotas require a positive batch size and positive weights")
    names = tuple(weights)
    scale = sum(weights.values())
    raw = {name: total * weights[name] / scale for name in names}
    quotas = {name: math.floor(raw[name]) for name in names}
    remainder = total - sum(quotas.values())
    priority = sorted(
        names,
        key=lambda name: (-(raw[name] - quotas[name]), (names.index(name) - batch_index) % len(names)),
    )
    for name in priority[:remainder]:
        quotas[name] += 1
    return quotas


class DomainWeightedSampler(Sampler[int]):
    """Cycle shuffled route pools with an exact per-batch weighted mixture.

    The sampler and iterator expose torchdata's stateful protocol, so a
    checkpoint can restore the same route and row sequence mid-epoch.
    """

    def __init__(self, dataframe: datasets.Dataset, weights: Mapping[str, float], seed: int, batch_size: int):
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if ROUTE_COLUMN not in dataframe.column_names:
            raise ValueError(f"Domain-weighted sampling requires a top-level {ROUTE_COLUMN!r} column")
        self.weights = dict(weights)
        self.names = tuple(self.weights)
        self.batch_size = batch_size
        self.seed = seed
        self.epoch = 0
        self.num_samples = len(dataframe) // batch_size * batch_size
        if self.num_samples == 0:
            raise ValueError(f"Dataset has fewer than {batch_size} rows")
        pools: dict[str, list[int]] = defaultdict(list)
        for index, route in enumerate(dataframe[ROUTE_COLUMN]):
            pools[str(route)].append(index)
        unknown = set(pools) - set(self.names)
        if unknown:
            raise ValueError(f"Dataset has unknown teacher routes: {sorted(unknown)}")
        self.pools = {name: pools[name] for name in self.names}
        max_quotas = {
            name: max(weighted_quotas(batch_size, self.weights, batch)[name] for batch in range(len(self.names)))
            for name in self.names
        }
        underfilled = {name: quota for name, quota in max_quotas.items() if len(self.pools[name]) < quota}
        if underfilled:
            raise ValueError(f"Domain-weighted route pools cannot fill a unique batch: {underfilled}")

    def __len__(self) -> int:
        return self.num_samples

    def __iter__(self) -> DomainWeightedSamplerIterator:
        iterator = DomainWeightedSamplerIterator(self, epoch=self.epoch)
        self.epoch += 1
        return iterator

    def state_dict(self) -> dict[str, int]:
        return {"epoch": self.epoch}

    def load_state_dict(self, state_dict: Mapping[str, int]) -> None:
        self.epoch = state_dict["epoch"]


class DomainWeightedSamplerIterator(Iterator[int]):
    """Stateful draw stream for one weighted-sampler epoch."""

    def __init__(self, sampler: DomainWeightedSampler, epoch: int):
        self.sampler = sampler
        self.rng = random.Random(sampler.seed + epoch)
        self.pools = {name: list(rows) for name, rows in sampler.pools.items()}
        for rows in self.pools.values():
            self.rng.shuffle(rows)
        self.cursors = {name: 0 for name in sampler.names}
        self.yielded = 0
        self.batch_index = 0
        self.batch: list[int] = []
        self.batch_position = 0

    def __iter__(self) -> DomainWeightedSamplerIterator:
        return self

    def __next__(self) -> int:
        if self.yielded >= len(self.sampler):
            raise StopIteration
        if self.batch_position == len(self.batch):
            self._next_batch()
        index = self.batch[self.batch_position]
        self.batch_position += 1
        self.yielded += 1
        return index

    def _take(self, route: str, count: int) -> list[int]:
        selected: list[int] = []
        seen: set[int] = set()
        while len(selected) < count:
            cursor = self.cursors[route]
            pool = self.pools[route]
            row = pool[cursor]
            cursor += 1
            if cursor == len(pool):
                self.rng.shuffle(pool)
                cursor = 0
            self.cursors[route] = cursor
            if row not in seen:
                selected.append(row)
                seen.add(row)
        return selected

    def _next_batch(self) -> None:
        quotas = weighted_quotas(self.sampler.batch_size, self.sampler.weights, self.batch_index)
        self.batch = [row for name in self.sampler.names for row in self._take(name, quotas[name])]
        self.rng.shuffle(self.batch)
        self.batch_position = 0
        self.batch_index += 1

    def state_dict(self) -> dict[str, Any]:
        return {
            "rng_state": self.rng.getstate(),
            "pools": {name: list(rows) for name, rows in self.pools.items()},
            "cursors": dict(self.cursors),
            "yielded": self.yielded,
            "batch_index": self.batch_index,
            "batch": list(self.batch),
            "batch_position": self.batch_position,
        }

    def load_state_dict(self, state_dict: Mapping[str, Any]) -> None:
        self.rng.setstate(state_dict["rng_state"])
        self.pools = {name: list(rows) for name, rows in state_dict["pools"].items()}
        self.cursors = dict(state_dict["cursors"])
        self.yielded = state_dict["yielded"]
        self.batch_index = state_dict["batch_index"]
        self.batch = list(state_dict["batch"])
        self.batch_position = state_dict["batch_position"]
