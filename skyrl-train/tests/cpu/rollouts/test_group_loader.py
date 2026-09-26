from skyrl_train.rollouts.loader import GroupLoader


class _Prompts:
    def __init__(self, count: int):
        self._uids = [str(index) for index in range(count)]

    def __len__(self) -> int:
        return len(self._uids)

    def __getitem__(self, index: int) -> str:
        return self._uids[index]

    def collate_fn(self, items: list[str]) -> list[dict]:
        return [{"uid": uid} for uid in items]


def _take(loader: GroupLoader, count: int) -> list[str]:
    return [loader.next_group()["uid"] for _ in range(count)]


def test_each_pass_visits_every_prompt_in_a_fresh_seeded_order():
    uids = _take(GroupLoader(_Prompts(8), seed=3, shuffle=True), 16)

    assert sorted(uids[:8]) == sorted(uids[8:]) == [str(index) for index in range(8)]
    assert uids[:8] != uids[8:]
    assert _take(GroupLoader(_Prompts(8), seed=3, shuffle=True), 16) == uids


def test_retries_come_first_and_resume_continues_the_same_sequence():
    loader = GroupLoader(_Prompts(4), seed=0, shuffle=True)
    _take(loader, 3)
    loader.retry({"uid": "retry"})
    state = loader.state_dict()
    expected = _take(loader, 6)

    resumed = GroupLoader(_Prompts(4), seed=0, shuffle=True)
    resumed.load_state_dict(state)

    assert expected[0] == "retry"
    assert _take(resumed, 6) == expected
