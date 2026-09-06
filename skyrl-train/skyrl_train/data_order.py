"""Explicit, checkpoint-validated prompt source order for matched training runs."""

import hashlib
import json
from dataclasses import asdict, dataclass
from itertools import islice

from omegaconf import OmegaConf
from torch.utils.data.distributed import DistributedSampler


def epoch_seeded_shuffle_enabled(cfg) -> bool:
    return OmegaConf.select(cfg, "data.epoch_seeded_shuffle", default=False)


def validate_epoch_seeded_shuffle(cfg) -> None:
    enabled = epoch_seeded_shuffle_enabled(cfg)
    if not isinstance(enabled, bool):
        raise ValueError("data.epoch_seeded_shuffle must be a boolean")
    if not enabled:
        return
    seed = cfg.trainer.seed
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2**32:
        raise ValueError("Epoch-seeded source order requires an integer trainer.seed in [0, 2**32)")
    batch = cfg.trainer.train_batch_size
    if isinstance(batch, bool) or not isinstance(batch, int) or batch <= 0:
        raise ValueError("Epoch-seeded source order requires a positive integer train_batch_size")
    if (
        cfg.data.sampling.kind is not None
        or cfg.trainer.algorithm.dynamic_sampling.type is not None
        or cfg.trainer.step_wise_training
        or cfg.trainer.train_batch_size != cfg.trainer.policy_mini_batch_size
    ):
        raise ValueError(
            "data.epoch_seeded_shuffle requires ordinary per-prompt sampling, no curriculum or dynamic sampling, "
            "and equal train_batch_size/policy_mini_batch_size"
        )


@dataclass(frozen=True)
class SourceOrderContract:
    algorithm: str
    seed: int
    dataset_sha256: str
    rows: int
    prompts_per_step: int

    @property
    def steps_per_epoch(self) -> int:
        return self.rows // self.prompts_per_step


class EpochSeededSampler(DistributedSampler):
    """A single source stream; distributed policy sharding happens after generation."""

    def __init__(self, dataset, *, seed: int, prompts_per_step: int):
        digest = hashlib.sha256()
        uids = set()
        for index in range(len(dataset)):
            row = dataset.collate_fn([dataset[index]])[0]
            uid = row["uid"]
            if not isinstance(uid, str) or uid in uids:
                raise ValueError("Epoch-seeded source order requires unique string prompt UIDs")
            uids.add(uid)
            # Include prompts/reward inputs, not just positional UIDs. Hash one row
            # at a time so dataset validation does not retain a second dataset copy.
            digest.update(json.dumps(row, sort_keys=True, separators=(",", ":"), allow_nan=False).encode())
            digest.update(b"\n")
        self.contract = SourceOrderContract(
            "torch-randperm-seed-plus-epoch-v1", seed, digest.hexdigest(), len(dataset), prompts_per_step
        )
        if self.contract.steps_per_epoch < 1:
            raise ValueError("Epoch-seeded source order requires at least one complete training batch")
        super().__init__(dataset, num_replicas=1, rank=0, shuffle=True, seed=seed, drop_last=False)

    def epoch_uids(self, epoch: int, *, include_tail: bool = False) -> set[str]:
        """UIDs for an epoch, optionally including its dropped tail; used only on resume."""
        previous = self.epoch
        self.set_epoch(epoch)
        try:
            count = (
                self.contract.rows if include_tail else self.contract.steps_per_epoch * self.contract.prompts_per_step
            )
            return {self.dataset.collate_fn([self.dataset[index]])[0]["uid"] for index in islice(iter(self), count)}
        finally:
            self.set_epoch(previous)


def set_source_epoch(dataloader, epoch: int) -> None:
    if isinstance(dataloader.sampler, EpochSeededSampler):
        dataloader.sampler.set_epoch(epoch)


def source_order_checkpoint(dataloader, completed_step: int) -> dict | None:
    if not isinstance(dataloader.sampler, EpochSeededSampler):
        return None
    contract = dataloader.sampler.contract
    epoch, step_in_epoch = divmod(completed_step, contract.steps_per_epoch)
    return {
        "contract": asdict(contract),
        "loader_batch_size": dataloader.batch_size,
        "loader_workers": dataloader.num_workers,
        "completed_step": completed_step,
        "epoch": epoch,
        "step_in_epoch": step_in_epoch,
    }


def validate_source_order_checkpoint(dataloader, saved: dict | None, completed_step: int) -> bool:
    """Validate the source contract and select the epoch before loader restoration.

    Returns whether an enabled checkpoint is at an epoch boundary, where a fresh
    iterator replaces the preceding epoch's cursor and asynchronous buffers.
    """
    expected = source_order_checkpoint(dataloader, completed_step)
    if saved != expected:
        raise ValueError("Checkpoint source-order contract or epoch position differs from the requested run")
    if expected is None:
        return False
    set_source_epoch(dataloader, expected["epoch"])
    return expected["step_in_epoch"] == 0


async def normalize_async_source_epoch(dataloader, tracker, completed_step: int, pending_uids: set[str]) -> bool:
    """Validate consumed work and advance a saved, completed epoch exactly once."""
    sampler = dataloader.sampler
    if not isinstance(sampler, EpochSeededSampler):
        return False
    contract = sampler.contract
    epoch, position = divmod(completed_step, contract.steps_per_epoch)
    if tracker.total_samples_consumed != completed_step * contract.prompts_per_step:
        raise ValueError("Checkpoint source-order total consumed prompt count is inconsistent")
    consumed = tracker.get_consumed_uids_in_epoch()
    if position:
        if not (consumed | pending_uids) <= sampler.epoch_uids(epoch):
            raise ValueError("Checkpoint source-order UIDs do not belong to the saved epoch")
        if tracker.current_epoch != epoch or tracker.consumed_in_epoch_count != position * contract.prompts_per_step:
            raise ValueError("Checkpoint source-order consumed epoch position is inconsistent")
        return False
    if tracker.current_epoch == epoch and not consumed:
        if pending_uids:
            raise ValueError("Checkpoint has pending prompts after source epoch cleanup")
        return True
    if tracker.current_epoch != epoch - 1 or consumed != sampler.epoch_uids(epoch - 1):
        raise ValueError("Checkpoint source-order epoch boundary is inconsistent")
    if not pending_uids <= sampler.epoch_uids(epoch - 1, include_tail=True):
        raise ValueError("Checkpoint source-order UIDs do not belong to the saved epoch")
    await tracker.on_epoch_end()
    return True
