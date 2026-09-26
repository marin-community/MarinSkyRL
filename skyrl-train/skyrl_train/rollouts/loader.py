"""The training dataset as an endless sequence of prompt groups."""

from __future__ import annotations

import collections
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol

import torch


class PromptGroupDataset(Protocol):
    def __len__(self) -> int: ...

    def __getitem__(self, index: int) -> Any: ...

    def collate_fn(self, items: list[Any]) -> list[dict]: ...


@dataclass(frozen=True)
class JudgedGroup:
    """A prompt group that batch selection judged in one step, kept or discarded, with its per-sample rewards.

    ``uid`` is the dataset row the group was generated from.
    """

    uid: str
    rewards: tuple[float, ...]


class PromptOrder(Protocol):
    """The dataset rows the loader offers, in order; an adaptive order learns from each step's judged groups."""

    def next_index(self) -> int: ...

    def observe(self, groups: Sequence[JudgedGroup]) -> dict[str, float]:
        """Fold one training step's judged groups into the order and return its metrics."""
        ...

    def state_dict(self) -> dict[str, Any]: ...

    def load_state_dict(self, state: Mapping[str, Any]) -> None: ...


class SeededPasses:
    """Endless passes over the dataset, each in a fresh seeded order."""

    def __init__(self, num_rows: int, *, seed: int, shuffle: bool):
        self._num_rows = num_rows
        self._seed = seed
        self._shuffle = shuffle
        self._epoch = 0
        self._position = 0
        self._order = self._pass_order(0)

    def next_index(self) -> int:
        if self._position == self._num_rows:
            self._epoch += 1
            self._order = self._pass_order(self._epoch)
            self._position = 0
        index = self._order[self._position]
        self._position += 1
        return index

    def observe(self, groups: Sequence[JudgedGroup]) -> dict[str, float]:
        return {}

    def state_dict(self) -> dict[str, Any]:
        return {"epoch": self._epoch, "position": self._position}

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        self._epoch = state["epoch"]
        self._order = self._pass_order(self._epoch)
        self._position = state["position"]

    def _pass_order(self, epoch: int) -> list[int]:
        if not self._shuffle:
            return list(range(self._num_rows))
        generator = torch.Generator()
        generator.manual_seed(self._seed + epoch)
        return torch.randperm(self._num_rows, generator=generator).tolist()


@dataclass(frozen=True)
class GroupLoaderState:
    """The prompt order's position and the prompts awaiting regeneration."""

    order: dict[str, Any]
    retries: list[dict]


class GroupLoader:
    """Yield one prompt group at a time, re-offering prompts whose rollouts must be regenerated first.

    The sequence never ends; the trainer decides when to stop.
    """

    def __init__(self, dataset: PromptGroupDataset, order: PromptOrder):
        if len(dataset) == 0:
            raise ValueError("the training dataset is empty")
        self._dataset = dataset
        self._order = order
        self._retries: collections.deque[dict] = collections.deque()

    def next_group(self) -> dict:
        if self._retries:
            return self._retries.popleft()
        (prompt,) = self._dataset.collate_fn([self._dataset[self._order.next_index()]])
        return prompt

    def retry(self, prompt: dict) -> None:
        self._retries.append(prompt)

    def observe(self, groups: Sequence[JudgedGroup]) -> dict[str, float]:
        """Report one training step's judged groups to the prompt order and return its metrics."""
        return self._order.observe(groups)

    def state_dict(self) -> GroupLoaderState:
        return GroupLoaderState(self._order.state_dict(), list(self._retries))

    def load_state_dict(self, state: GroupLoaderState) -> None:
        self._order.load_state_dict(state.order)
        self._retries = collections.deque(state.retries)
