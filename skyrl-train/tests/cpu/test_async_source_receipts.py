"""Consumed-source receipts follow the actual async learner input across epochs."""

import asyncio
import json

import pytest
from rigging.telemetry.serialization import EventBody, event_fields

from skyrl_train.data_order import consumed_uid_digest
from tests.cpu.test_fully_async_publication_cadence import DriverWithCpuLearner, PromptRows, make_driver


class SourceRecordingLearner(DriverWithCpuLearner):
    def __init__(self, *args, **kwargs):
        self.input_uids = []
        self.fail_step = None
        super().__init__(*args, **kwargs)

    async def _run_training(self, training_input):
        if self.global_step == self.fail_step:
            raise RuntimeError("learner failed before consumption")
        self.input_uids.append(list(dict.fromkeys(training_input.metadata["uids"])))
        return await super()._run_training(training_input)


@pytest.mark.asyncio
@pytest.mark.parametrize("enabled", [False, True])
async def test_actual_async_source_receipts_match_learner_input_across_chunks_and_epochs(monkeypatch, enabled):
    events = []
    monkeypatch.setattr(
        "skyrl_train.fully_async_trainer.record_event",
        lambda name, body, **kwargs: events.append((name, body, kwargs["attributes"])),
    )

    def driver(**kwargs):
        cfg = kwargs["cfg"]
        cfg.trainer.training_metrics = enabled
        cfg.trainer.train_batch_size = cfg.trainer.policy_mini_batch_size = 65
        cfg.trainer.fully_async.num_parallel_generation_workers = 65
        cfg.data.epoch_seeded_shuffle = True
        kwargs["train_dataset"] = PromptRows(65)
        return SourceRecordingLearner(**kwargs)

    trainer = make_driver(interval=1, age=0, steps=2, epochs=2, driver_type=driver)
    await asyncio.wait_for(trainer._train_loop(), timeout=15)
    assert len(trainer.input_uids) == 2
    assert all(len(uids) == len(set(uids)) == 65 for uids in trainer.input_uids)
    assert trainer.input_uids[0] != trainer.input_uids[1]
    assert trainer.inference_engine_client.publications == [0, 1, 2]
    receipts = [(body, attrs) for name, body, attrs in events if name == "consumed_source_order"]
    if not enabled:
        assert receipts == []
        return
    assert len(receipts) == 4
    assert [body["prompt_offset"] for body, _ in receipts] == [0, 64, 65, 129]
    assert [body["epoch"] for body, _ in receipts] == [0, 0, 1, 1]
    for step in (1, 2):
        chunks = [body for body, attrs in receipts if attrs == {"role": "trainer", "step": str(step)}]
        actual = [uid for chunk in chunks for uid in json.loads(chunk["uids_json"])]
        assert actual == trainer.input_uids[step - 1]
        metric = next(
            row for logged_step, row in trainer.tracker.rows if logged_step == step and "consumed/uid_digest_u52" in row
        )
        assert metric["consumed/uid_digest_u52"] == consumed_uid_digest(actual)
        for chunk in chunks:
            assert event_fields(EventBody(chunk), budget=16_384) == chunk


@pytest.mark.asyncio
async def test_failed_async_learner_does_not_emit_a_consumed_source_receipt(monkeypatch):
    receipts = []
    monkeypatch.setattr(
        "skyrl_train.fully_async_trainer.record_event",
        lambda name, body, **kwargs: receipts.append((name, kwargs["attributes"]["step"])),
    )
    trainer = make_driver(interval=1, age=0, steps=2, driver_type=SourceRecordingLearner)
    trainer.fail_step = 2
    with pytest.raises(RuntimeError, match="learner failed before consumption"):
        await asyncio.wait_for(trainer._train_loop(), timeout=15)
    assert [step for name, step in receipts if name == "consumed_source_order"] == ["1"]
    assert len(trainer.input_uids) == 1
