"""The training dataset as an endless sequence of prompt groups."""

from __future__ import annotations

import collections
from dataclasses import dataclass
from typing import Any, Protocol

import torch


class PromptGroupDataset(Protocol):
    def __len__(self) -> int: ...

    def __getitem__(self, index: int) -> Any: ...

    def collate_fn(self, items: list[Any]) -> list[dict]: ...


@dataclass(frozen=True)
class GroupLoaderState:
    """Position in the current dataset pass and prompts awaiting regeneration."""

    epoch: int
    position: int
    retries: list[dict]


class GroupLoader:
    """Yield one prompt group at a time, re-offering prompts whose rollouts must be regenerated first.

    Passes over the dataset repeat without end, each in a fresh seeded order; the trainer decides when to stop.
    """

    def __init__(self, dataset: PromptGroupDataset, *, seed: int, shuffle: bool):
        if len(dataset) == 0:
            raise ValueError("the training dataset is empty")
        self._dataset = dataset
        self._seed = seed
        self._shuffle = shuffle
        self._epoch = 0
        self._position = 0
        self._order = self._pass_order(0)
        self._retries: collections.deque[dict] = collections.deque()

    def next_group(self) -> dict:
        if self._retries:
            return self._retries.popleft()
        if self._position == len(self._order):
            self._epoch += 1
            self._order = self._pass_order(self._epoch)
            self._position = 0
        index = self._order[self._position]
        self._position += 1
        (prompt,) = self._dataset.collate_fn([self._dataset[index]])
        return prompt

    def retry(self, prompt: dict) -> None:
        self._retries.append(prompt)

    def state_dict(self) -> GroupLoaderState:
        return GroupLoaderState(self._epoch, self._position, list(self._retries))

    def load_state_dict(self, state: GroupLoaderState) -> None:
        self._epoch = state.epoch
        self._order = self._pass_order(state.epoch)
        self._position = state.position
        self._retries = collections.deque(state.retries)

    def _pass_order(self, epoch: int) -> list[int]:
        if not self._shuffle:
            return list(range(len(self._dataset)))
        generator = torch.Generator()
        generator.manual_seed(self._seed + epoch)
        return torch.randperm(len(self._dataset), generator=generator).tolist()
