"""Exercise source ordering through real loaders and persisted trainer checkpoints."""

from copy import deepcopy
import asyncio
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from datasets import Dataset
from omegaconf import OmegaConf

from skyrl_train.config.utils import get_default_config
from skyrl_train.data_order import (
    normalize_async_source_epoch,
    set_source_epoch,
    source_order_checkpoint,
    validate_source_order_checkpoint,
)
from skyrl_train.fully_async_trainer import _AsyncDataloader
from skyrl_train.dataset import PromptDataset
from skyrl_train.trainer import RayPPOTrainer
from skyrl_train.utils.data_tracker import DataConsumptionTracker
from skyrl_train.utils.trainer_utils import ResumeMode, build_dataloader
from skyrl_train.utils.utils import validate_cfg
from tests.cpu.test_fully_async_publication_cadence import (
    DriverWithCpuLearner,
    PromptRows,
    Runner,
    StartupLearnerService,
    make_driver,
)


class Rows:
    def __init__(self, count=19, answer="2"):
        self.count = count
        self.answer = answer

    def __len__(self):
        return self.count

    def __getitem__(self, index):
        return {"uid": str(index), "prompt": [{"role": "user", "content": "1+1?"}], "answer": self.answer}

    @staticmethod
    def collate_fn(rows):
        return rows


def config(workers=0, *, enabled=True):
    cfg = get_default_config()
    cfg.data.epoch_seeded_shuffle = enabled
    cfg.trainer.seed = 17
    cfg.trainer.train_batch_size = cfg.trainer.policy_mini_batch_size = 4
    cfg.generator.enable_http_endpoint = workers == 0
    return cfg


def loader(cfg, *, asynchronous=False, dataset=None):
    result = build_dataloader(cfg, dataset or Rows(), is_train=True, is_fully_async=asynchronous)
    assert result.num_workers == (0 if cfg.generator.enable_http_endpoint else 8)
    return result


async def drain(source):
    rows = []
    while (batch := await source.get_next_non_consumed_data()) is not None:
        rows.append(batch[0]["uid"])
    return rows


@pytest.mark.asyncio
@pytest.mark.parametrize("workers", [0, 8])
@pytest.mark.parametrize("enabled", [False, True])
async def test_multiple_epochs_and_partial_tail_share_source_order_only_when_enabled(workers, enabled):
    cfg = config(workers, enabled=enabled)
    sync = loader(cfg)
    raw_async = loader(cfg, asynchronous=True)
    tracker = DataConsumptionTracker(4, 4)
    asynchronous = _AsyncDataloader(raw_async, 4, tracker)
    sync_orders, async_orders = [], []
    for epoch in range(3):
        set_source_epoch(sync, epoch)
        sync_order = [row["uid"] for batch in sync for row in batch]
        async_order = await drain(asynchronous)
        assert len(sync_order) == len(set(sync_order)) == len(async_order) == len(set(async_order)) == 16
        sync_orders.append(sync_order)
        async_orders.append(async_order)
        await tracker.mark_consumed(async_order)
        await tracker.on_epoch_end()
        await asynchronous.reset_at_epoch_end()
    assert sync_orders[0] != sync_orders[1] != sync_orders[2]
    if enabled:
        assert sync_orders == async_orders
        # Contract fixture: first eight source UIDs for seed17, epoch0 (19 rows).
        assert sync_orders[0][:8] == ["11", "16", "1", "9", "3", "18", "8", "6"]
    else:
        assert sync_orders[0] == async_orders[0] == async_orders[1] == async_orders[2]
        assert sync_orders[1] != async_orders[1]


@pytest.mark.asyncio
@pytest.mark.parametrize("workers", [0, 8])
async def test_async_mid_epoch_resume_replays_only_unconsumed_unreserved_prompts(workers, tmp_path):
    cfg = config(workers)
    source = loader(cfg, asynchronous=True)
    set_source_epoch(source, 1)
    reference = _AsyncDataloader(source, 4, DataConsumptionTracker(4, 4))
    order = await drain(reference)
    tracker = DataConsumptionTracker(4, 4)
    await tracker.mark_consumed([str(i) for i in range(16)])
    await tracker.on_epoch_end()
    await tracker.mark_consumed(order[::2])  # eight completed groups, deliberately out of source order
    # Completed/retry ownership can include a late duplicate of consumed work;
    # existing admission discards that duplicate, and source replay skips its UID.
    pending = {order[0], order[1], order[5]}
    checkpoint = tmp_path / "checkpoint.pt"
    torch.save({"tracker": tracker.get_state(), "source_order": source_order_checkpoint(source, 6)}, checkpoint)
    saved = torch.load(checkpoint, weights_only=False)
    restored_tracker = DataConsumptionTracker(4, 4)
    restored_tracker.load_state(saved["tracker"])
    restored = loader(cfg, asynchronous=True)
    resumed = _AsyncDataloader(restored, 4, restored_tracker)
    assert not validate_source_order_checkpoint(restored, saved["source_order"], 6)
    assert not await normalize_async_source_epoch(restored, restored_tracker, 6, pending)
    resumed.load_state_from_checkpoint()
    resumed.reserve_pending_uids(pending)
    assert await drain(resumed) == [uid for uid in order if uid not in set(order[::2]) | pending]


@pytest.mark.asyncio
@pytest.mark.parametrize("after_cleanup", [False, True])
async def test_boundary_resume_advances_once_and_never_reuses_preceding_epoch_buffers(after_cleanup):
    cfg = config()
    source = loader(cfg, asynchronous=True)
    tracker = DataConsumptionTracker(4, 4)
    asynchronous = _AsyncDataloader(source, 4, tracker)
    order = await drain(asynchronous)
    await tracker.mark_consumed(order)
    # A persisted buffer may include an unused row from the dropped final batch.
    pending = {order[0]} | (source.sampler.epoch_uids(0, include_tail=True) - set(order))
    if after_cleanup:
        await tracker.on_epoch_end()
        pending = set()
    assert await normalize_async_source_epoch(source, tracker, 4, pending)
    assert tracker.current_epoch == 1 and tracker.consumed_in_epoch_count == 0
    assert await normalize_async_source_epoch(source, tracker, 4, set())
    assert tracker.current_epoch == 1
    asynchronous.load_state_from_checkpoint()
    expected = loader(cfg)
    set_source_epoch(expected, 1)
    assert await drain(asynchronous) == [row["uid"] for batch in expected for row in batch]


@pytest.mark.asyncio
async def test_boundary_resume_rejects_pending_work_from_another_epoch():
    source = loader(config(), asynchronous=True)
    tracker = DataConsumptionTracker(4, 4)
    await tracker.mark_consumed(source.sampler.epoch_uids(0))
    with pytest.raises(ValueError, match="source-order UIDs"):
        await normalize_async_source_epoch(source, tracker, 4, {"not-in-completed-epoch"})
    assert tracker.current_epoch == 0 and tracker.consumed_in_epoch_count == 16


def checkpoint_trainer(cfg, dataloader, path, *, step):
    trainer = RayPPOTrainer.__new__(RayPPOTrainer)
    trainer.cfg = deepcopy(cfg)
    trainer.cfg.trainer.ckpt_path = str(path)
    trainer.cfg.trainer.resume_path = str(path / f"global_step_{step}")
    trainer.resume_mode = ResumeMode.FROM_PATH
    trainer.train_dataloader = dataloader
    trainer.global_step = step
    trainer.policy_model = SimpleNamespace(async_run_ray_method=lambda *args, **kwargs: None)
    trainer.critic_model = None
    trainer.colocate_all = False
    trainer.tokenizer = None
    trainer.all_timings = {}
    return trainer


@pytest.mark.parametrize("workers", [0, 8])
@pytest.mark.parametrize("step", [2, 4, 6, 8])
def test_real_trainer_checkpoint_restores_sync_cursor_or_next_epoch(workers, step, tmp_path, monkeypatch):
    monkeypatch.setattr("skyrl_train.trainer.ray.get", lambda result: result)
    cfg = config(workers)
    source = loader(cfg)
    epoch = (step - 1) // 4
    set_source_epoch(source, epoch)
    iterator = iter(source)
    for _ in range((step - 1) % 4 + 1):
        next(iterator)
    original = checkpoint_trainer(cfg, source, tmp_path, step=step)
    original.save_checkpoints()
    restored = checkpoint_trainer(cfg, loader(cfg), tmp_path, step=step)
    assert restored.load_checkpoints()[0] == step
    expected = loader(cfg)
    set_source_epoch(expected, step // 4)
    order = [row["uid"] for batch in expected for row in batch]
    remaining = [row["uid"] for batch in restored.train_dataloader for row in batch]
    assert remaining == order[(step % 4) * 4 :]


@pytest.mark.parametrize("change", ["content", "seed", "rows", "batch", "disabled", "missing"])
def test_changed_source_contract_fails_before_resuming(change):
    cfg = config()
    saved = source_order_checkpoint(loader(cfg), 2)
    data = Rows()
    if change == "content":
        data.answer = "different reward input"
    elif change == "seed":
        cfg.trainer.seed += 1
    elif change == "rows":
        data.count += 1
    elif change == "batch":
        cfg.trainer.train_batch_size = cfg.trainer.policy_mini_batch_size = 2
    elif change == "disabled":
        cfg.data.epoch_seeded_shuffle = False
    else:
        saved = None
    with pytest.raises(ValueError, match="source-order contract"):
        validate_source_order_checkpoint(loader(cfg, dataset=data), saved, 2)


@pytest.mark.parametrize(
    "key,value",
    [
        ("data.sampling.kind", "naive"),
        ("trainer.algorithm.dynamic_sampling.type", "filter"),
        ("trainer.step_wise_training", True),
        ("trainer.policy_mini_batch_size", 2),
    ],
)
def test_unsupported_order_controls_fail_during_config_validation(key, value):
    cfg = config()
    OmegaConf.update(cfg, key, value)
    with pytest.raises(ValueError, match="epoch_seeded_shuffle"):
        validate_cfg(cfg)


class SourceOrderDriver(DriverWithCpuLearner):
    """Keep the existing CPU learner fixture and persist its source contract."""

    def __init__(self, *args, **kwargs):
        kwargs["cfg"].data.epoch_seeded_shuffle = True
        super().__init__(*args, **kwargs)

    def save_checkpoints(self):
        super().save_checkpoints()
        checkpoint = Path(self.cfg.trainer.ckpt_path) / f"global_step_{self.global_step}"
        (checkpoint / "source_order.json").write_text(
            json.dumps(source_order_checkpoint(self.train_dataloader, self.global_step))
        )

    def load_checkpoints(self):
        step, checkpoint = super().load_checkpoints()
        validate_source_order_checkpoint(
            self.train_dataloader, json.loads((Path(checkpoint) / "source_order.json").read_text()), step
        )
        return step, checkpoint


class PartialSourceOrderDriver(SourceOrderDriver):
    def __init__(self, *args, **kwargs):
        kwargs["train_dataset"] = PromptRows(5)  # two complete batches plus one dropped tail prompt
        super().__init__(*args, **kwargs)


@pytest.mark.asyncio
@pytest.mark.parametrize("stop", [1, 2, 3, 4])
@pytest.mark.parametrize("driver_type", [SourceOrderDriver, PartialSourceOrderDriver])
async def test_actual_async_driver_resumes_mid_epoch_boundary_and_completed_run(stop, driver_type, tmp_path):
    original = make_driver(
        steps=4, epochs=2, steps_per_epoch=2, stop_step=stop, save_step=stop, driver_type=driver_type
    )
    original.cfg.trainer.ckpt_path = str(tmp_path)
    await asyncio.wait_for(original._train_loop(), timeout=10)
    if driver_type is PartialSourceOrderDriver and stop == 2:
        # Exercise the full resume path with persisted, unconsumed tail work.
        # Current loader prefetch does not submit tail rows to generation; this
        # also covers a checkpoint producer that saved such work before cleanup.
        tail_uid = (
            original.train_dataloader.sampler.epoch_uids(0, include_tail=True)
            - original.train_dataloader.sampler.epoch_uids(0)
        ).pop()
        buffer_path = tmp_path / "global_step_2/generation_buffer_state.pt"
        buffer = (
            torch.load(buffer_path, weights_only=False)
            if buffer_path.exists()
            else {"completed_groups": [], "retry_prompts": []}
        )
        buffer["retry_prompts"].append([original.train_dataset[int(tail_uid)]])
        torch.save(buffer, buffer_path)
    resumed = make_driver(steps=4, epochs=2, steps_per_epoch=2, driver_type=driver_type)
    resumed.resume_mode = ResumeMode.FROM_PATH
    resumed.cfg.trainer.resume_path = str(tmp_path / f"global_step_{stop}")
    await asyncio.wait_for(resumed._train_loop(), timeout=10)
    assert [step for step, _ in resumed.policy_model.consumed] == list(range(stop + 1, 5))
    assert resumed.data_tracker.total_samples_consumed == 8
    # Every source UID occurs once in each epoch, regardless of admission order.
    uninterrupted = make_driver(steps=4, epochs=2, steps_per_epoch=2, driver_type=driver_type)
    await asyncio.wait_for(uninterrupted._train_loop(), timeout=10)
    assert uninterrupted.data_tracker.total_samples_consumed == resumed.data_tracker.total_samples_consumed


class CountingTokenizer:
    def apply_chat_template(self, messages, **kwargs):
        return list(range(len(messages[0]["content"])))


class SourceRecordingRunner(Runner):
    async def run(self, request, **kwargs):
        if request["batch_metadata"].training_phase == "train":
            self.source_uids.extend(dict.fromkeys(item.instance_id for item in request["trajectory_ids"]))
        return await super().run(request, **kwargs)


class SyncSourceOrderDriver(RayPPOTrainer):
    def __init__(self, *args, **kwargs):
        kwargs["cfg"].data.epoch_seeded_shuffle = True
        kwargs["cfg"].trainer.algorithm.use_kl_loss = False
        super().__init__(*args, **kwargs)

    def init_weight_sync_state(self):
        pass

    def fwd_logprobs_values_reward(self, batch):
        batch["action_log_probs"] = torch.full_like(batch["loss_mask"], -0.5)
        batch["values"] = None
        return batch

    def train_critic_and_policy(self, batch):
        self.policy_model.completed_update += 1
        return {"learner/update": self.policy_model.completed_update}


@pytest.mark.asyncio
async def test_actual_sync_driver_advances_the_shared_source_epoch(monkeypatch):
    trainer = make_driver(
        steps=6, epochs=3, steps_per_epoch=2, driver_type=SyncSourceOrderDriver, runner_type=SourceRecordingRunner
    )
    trainer.trajectory_runner.source_uids = []
    trainer.policy_model = StartupLearnerService()
    monkeypatch.setattr("skyrl_train.trainer.ray.get", lambda refs: refs)
    await asyncio.wait_for(trainer._train_loop(), timeout=10)
    expected = []
    for epoch in range(3):
        source = loader(trainer.cfg, dataset=PromptRows(4))
        set_source_epoch(source, epoch)
        expected.extend(row["uid"] for batch in source for row in batch)
    assert trainer.policy_model.completed_update == 6
    assert trainer.trajectory_runner.source_uids == expected


def test_real_parquet_prompt_dataset_hashes_collated_content_and_positional_uids(tmp_path):
    path = tmp_path / "prompts.parquet"
    rows = [
        {
            "prompt": [{"role": "user", "content": f"{i}+1?"}],
            "env_class": "gsm8k",
            "reward_model": {"ground_truth": str(i + 1)},
            "optional": None,
            "input_ids": [1, 2, i],
        }
        for i in range(5)
    ]
    Dataset.from_list(rows).to_parquet(str(path))
    dataset = PromptDataset(str(path), CountingTokenizer(), max_prompt_length=32, num_workers=1)
    original = loader(config(), dataset=dataset)
    saved = source_order_checkpoint(original, 1)
    reread = PromptDataset(str(path), CountingTokenizer(), max_prompt_length=32, num_workers=1)
    assert {row["uid"] for batch in loader(config(), dataset=reread) for row in batch} <= {str(i) for i in range(5)}
    assert source_order_checkpoint(loader(config(), dataset=reread), 1) == saved
    rows[0]["reward_model"]["ground_truth"] = "different reward"
    changed_path = tmp_path / "changed.parquet"
    Dataset.from_list(rows).to_parquet(str(changed_path))
    changed = PromptDataset(str(changed_path), CountingTokenizer(), max_prompt_length=32, num_workers=1)
    with pytest.raises(ValueError, match="source-order contract"):
        validate_source_order_checkpoint(loader(config(), dataset=changed), saved, 1)
