from collections.abc import Container

import pytest

from skyrl_train.rollouts.loader import PromptLoader, SeededPasses

NOTHING_LIVE: frozenset[str] = frozenset()


class _Prompts:
    def __init__(self, uids: list[str]):
        self._uids = uids

    def __len__(self) -> int:
        return len(self._uids)

    def __getitem__(self, index: int) -> str:
        return self._uids[index]

    def uid(self, index: int) -> str:
        return self._uids[index]

    def collate_fn(self, items: list[str]) -> list[dict]:
        return [{"uid": uid} for uid in items]


class _FirstRowOnly:
    """An order that keeps drawing row 0, as an adaptive order can when one row dominates its weights."""

    def next_index(self) -> int:
        return 0

    def observe(self, groups):
        return {}

    def state_dict(self):
        return {}

    def load_state_dict(self, state) -> None:
        pass


def _uids(count: int) -> list[str]:
    return [str(index) for index in range(count)]


def _loader(count: int, *, seed: int = 0, shuffle: bool = False) -> PromptLoader:
    return PromptLoader(_Prompts(_uids(count)), SeededPasses(count, seed=seed, shuffle=shuffle), batch_size=1)


def _take(loader: PromptLoader, count: int, live: Container[str] = NOTHING_LIVE) -> list[str]:
    return [loader.next_prompt(live)["uid"] for _ in range(count)]


def test_each_pass_visits_every_prompt_in_a_fresh_seeded_order():
    uids = _take(_loader(8, seed=3, shuffle=True), 16)

    assert sorted(uids[:8]) == sorted(uids[8:]) == _uids(8)
    assert uids[:8] != uids[8:]
    assert _take(_loader(8, seed=3, shuffle=True), 16) == uids
    assert _take(_loader(8, seed=4, shuffle=True), 16) != uids


def test_unshuffled_passes_follow_dataset_order():
    assert _take(_loader(3), 6) == ["0", "1", "2", "0", "1", "2"]


def test_retries_come_first_and_resume_continues_the_same_sequence():
    loader = _loader(4, seed=0, shuffle=True)
    _take(loader, 3)
    loader.retry({"uid": "retry"})
    state = loader.state_dict()
    expected = _take(loader, 6)

    resumed = _loader(4, seed=0, shuffle=True)
    resumed.load_state_dict(state)

    assert expected[0] == "retry"
    assert _take(resumed, 6) == expected


def test_live_uids_are_skipped_without_losing_their_retries():
    loader = _loader(4)
    loader.retry({"uid": "1"})

    assert _take(loader, 3, live={"1"}) == ["0", "2", "3"]
    assert _take(loader, 2) == ["1", "0"]


def test_waiting_for_a_free_uid_does_not_advance_the_order():
    loader = _loader(3)

    assert loader.next_prompt({"0", "1", "2"}) is None
    assert _take(loader, 3) == ["0", "1", "2"]


def test_an_order_that_only_draws_live_rows_falls_back_to_a_free_row():
    loader = PromptLoader(_Prompts(_uids(3)), _FirstRowOnly(), batch_size=1)

    assert loader.next_prompt({"0"})["uid"] == "1"
    assert loader.next_prompt({"0", "1"})["uid"] == "2"
    assert loader.next_prompt({"0", "1", "2"}) is None


def test_a_dataset_needs_a_batch_of_distinct_uids():
    uids = ["a", "a", "b"]
    with pytest.raises(ValueError, match="2 distinct prompt uids, fewer than one batch of 3"):
        PromptLoader(_Prompts(uids), SeededPasses(len(uids), seed=0, shuffle=False), batch_size=3)
