"""Check age populations against actual dispatch and worker iterator slicing."""

import json
from types import SimpleNamespace

import pytest
from rigging.telemetry.serialization import EventBody, event_fields

from skyrl_train.config.utils import get_default_config
from skyrl_train.trainer import RayPPOTrainer
from skyrl_train.fully_async_trainer import FullyAsyncRayPPOTrainer
from tests.cpu.test_fully_async_staleness import _generated_group
import torch

from skyrl_train.training_batch import TrainingInputBatch, TrainingBatchIterator
from skyrl_train.utils.type_c_staleness import consumed_update_age_counts


def test_age_counts_follow_real_dp_chunk_and_training_iterator_order():
    mask = torch.arange(1, 17).unsqueeze(-1) > torch.arange(16).unsqueeze(0)
    actual = consumed_update_age_counts(mask, dp_size=2, mini_batch_sequences=4, samples_per_prompt=2, epochs=2)
    batch = TrainingInputBatch({"response_mask": mask})
    shards = batch.chunk(8)  # MeshDispatch's exact contiguous shard operation.
    # The real iterator preconstructs this same TensorBatch chunk sequence.
    iterators = [TrainingBatchIterator(shard, 2) for shard in shards]
    tokens = [
        sum(iterator._chunks[index]["response_mask"].sum().item() for iterator in iterators) for index in range(4)
    ]
    assert [row["response_tokens"] for row in actual] == tokens * 2
    assert tokens == [22, 30, 38, 46]
    assert [row["age"] for row in actual] == list(range(8))
    assert all(row["groups"] == 2 and row["sequences"] == 4 for row in actual)
    assert sum(row["response_tokens"] for row in actual) == 2 * mask.sum().item()


@pytest.mark.parametrize("dp,mini,samples,epochs", [(3, 4, 2, 1), (2, 3, 1, 1), (2, 4, 3, 1), (2, 4, 2, 0)])
def test_invalid_age_geometry_rejects(dp, mini, samples, epochs):
    with pytest.raises(ValueError):
        consumed_update_age_counts(
            torch.ones(16, 2), dp_size=dp, mini_batch_sequences=mini, samples_per_prompt=samples, epochs=epochs
        )


def test_sync_driver_receipts_preserve_source_order_and_serialize(monkeypatch):
    trainer = RayPPOTrainer.__new__(RayPPOTrainer)
    trainer.cfg = get_default_config()
    trainer.cfg.data.epoch_seeded_shuffle = True
    trainer.cfg.trainer.train_batch_size = 8
    trainer.cfg.trainer.policy_mini_batch_size = 2
    trainer.cfg.trainer.update_epochs_per_batch = 1
    trainer.cfg.generator.n_samples_per_prompt = 2
    trainer.global_step = 2
    trainer._training_metrics_enabled = True
    trainer.policy_model = SimpleNamespace(actor_infos=[SimpleNamespace(rank=SimpleNamespace(dp_size=2))])
    events, ages = [], []

    def capture(name, body, **kwargs):
        assert event_fields(EventBody(body), budget=100_000) == body
        events.append((name, body, kwargs))

    monkeypatch.setattr("skyrl_train.trainer.record_event", capture)
    monkeypatch.setattr("skyrl_train.trainer.record_rollout_staleness", lambda values, step: ages.extend(values))
    batch = TrainingInputBatch({"response_mask": torch.ones(16, 3)})
    uids = [f"prompt-{index}" for index in range(8) for _ in range(2)]
    trainer._record_sync_update_ages(batch, uids)
    assert ages == [0, 0, 1, 1, 2, 2, 3, 3]
    assert [event[1]["response_tokens"] for event in events[:4]] == [12] * 4
    assert events[-1][0] == "consumed_source_order"
    assert events[-1][1]["prompt_offset"] == 8
    assert json.loads(events[-1][1]["uids_json"]) == uids[::2]


def test_async_driver_counts_accepted_group_tokens_before_padding(monkeypatch):
    trainer = FullyAsyncRayPPOTrainer.__new__(FullyAsyncRayPPOTrainer)
    trainer.cfg = get_default_config()
    trainer.global_step = 10
    trainer.max_staleness_steps = 2
    trainer.mini_batch_size = 2
    trainer.all_metrics = {}
    trainer._training_metrics_enabled = True
    groups = [_generated_group("a", 10), _generated_group("b", 8)]
    events = []

    class BeforePostprocess(Exception):
        pass

    def stop_after_telemetry(*args):
        raise BeforePostprocess

    trainer.postprocess_trajectory_batch = stop_after_telemetry
    monkeypatch.setattr(
        "skyrl_train.fully_async_trainer.record_event", lambda name, body, **kw: events.append((name, body))
    )
    with pytest.raises(BeforePostprocess):
        trainer.convert_generation_group_mini_batch_to_training_input(groups)
    assert events == [
        ("consumed_age", {"age": 0, "groups": 1, "sequences": 2, "response_tokens": 2}),
        ("consumed_age", {"age": 2, "groups": 1, "sequences": 2, "response_tokens": 2}),
    ]
    for _, body in events:
        assert event_fields(EventBody(body), budget=100_000) == body
