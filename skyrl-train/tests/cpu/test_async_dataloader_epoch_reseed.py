"""_AsyncDataloader.reset_at_epoch_end draws a new permutation per epoch (seed + epoch) instead of replaying epoch 0."""

import asyncio

import torch
from torchdata.stateful_dataloader import StatefulDataLoader

from skyrl_train.fully_async_trainer import _AsyncDataloader


class _Rows(torch.utils.data.Dataset):
    def __init__(self, n):
        self.n = n

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        return {"uid": f"u{i}"}

    @staticmethod
    def collate_fn(batch):
        return batch


class _Tracker:
    def get_consumed_uids_in_epoch(self):
        return set()


def _loader(seed, n=40):
    g = torch.Generator()
    g.manual_seed(seed)
    return StatefulDataLoader(_Rows(n), batch_size=1, shuffle=True, collate_fn=_Rows.collate_fn, num_workers=0, drop_last=True, generator=g)


async def _drain(dl):
    out = []
    while True:
        rows = await dl.get_next_non_consumed_data()
        if rows is None:
            return out
        out.append(rows[0]["uid"])


def _epoch_orders(seed, mini_batch=8, epochs=3):
    dl = _AsyncDataloader(_loader(seed), mini_batch, _Tracker())
    orders = []

    async def run():
        for e in range(epochs):
            orders.append(await _drain(dl))
            await dl.reset_at_epoch_end(next_epoch=e + 1)

    asyncio.run(run())
    return orders


def test_each_epoch_is_a_new_permutation_and_tail_rows_rotate():
    orders = _epoch_orders(seed=1234)
    assert all(len(o) == 40 for o in orders)  # 40 // 8 * 8: no rows dropped here; the point is the permutation
    assert orders[0] != orders[1] and orders[1] != orders[2]
    assert set(orders[0]) == set(orders[1]) == {f"u{i}" for i in range(40)}


def test_dropped_tail_rotates_when_length_is_not_a_multiple():
    # 43 rows, mini-batch 8 -> effective length 40: the 3 rows cut each epoch must differ across epochs
    dl = _AsyncDataloader(_loader(7, n=43), 8, _Tracker())
    seen = []

    async def run():
        for e in range(4):
            seen.append(set(await _drain(dl)))
            await dl.reset_at_epoch_end(next_epoch=e + 1)

    asyncio.run(run())
    assert all(len(s) == 40 for s in seen)
    assert len(set.union(*seen)) == 43  # every row sampled in some epoch


def test_reseed_is_deterministic_in_seed_and_epoch():
    a = _epoch_orders(seed=99)
    b = _epoch_orders(seed=99)
    assert a == b


def test_legacy_reset_without_epoch_replays_epoch_zero():
    dl = _AsyncDataloader(_loader(5), 8, _Tracker())
    orders = []

    async def run():
        for _ in range(2):
            orders.append(await _drain(dl))
            await dl.reset_at_epoch_end()  # no epoch -> old behaviour (restore initial state)

    asyncio.run(run())
    assert orders[0] == orders[1]
