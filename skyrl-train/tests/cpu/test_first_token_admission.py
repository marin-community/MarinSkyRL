"""Exercise the actual producer's admission stamp after an in-flight weight sync."""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from omegaconf import OmegaConf

from skyrl_train.config.utils import get_default_config
from skyrl_train.fully_async_trainer import FullyAsyncRayPPOTrainer, _GroupFreshness


@pytest.mark.asyncio
@pytest.mark.parametrize("enabled,versions,expected", [(True, [2, 3], 3), (False, [2, 3], 1), (True, [None, 3], None)])
async def test_actual_producer_uses_earliest_sampled_version_and_logs_both(monkeypatch, enabled, versions, expected):
    trainer = object.__new__(FullyAsyncRayPPOTrainer)
    trainer.cfg = get_default_config()
    OmegaConf.update(trainer.cfg, "trainer.fully_async.first_token_admission", enabled)
    trainer.cfg.generator.n_samples_per_prompt = 2
    trainer.global_step = 4
    trainer._published_policy_version = 0
    trainer._async_observations_enabled = False
    trainer._staleness_manager = SimpleNamespace(
        acquire_submission_slot=AsyncMock(), on_rollout_accepted=AsyncMock(), cancel_submission_slot=AsyncMock()
    )
    row = {"uid": "sample", "prompt": [{"role": "user", "content": "2+2?"}], "env_class": "gsm8k", "env_extras": {}}
    trainer._next_generation_prompts = AsyncMock(side_effect=[[row], asyncio.CancelledError()])
    batch = {"response_ids": [[10], [11]], "policy_versions_at_first_token": versions}

    async def generate(*args, **kwargs):
        trainer._published_policy_version = 3
        return batch

    trainer.trajectory_runner = SimpleNamespace(run=generate)
    groups = []

    async def enqueue(queues, group):
        groups.append(group)
        return _GroupFreshness.FRESH

    trainer._enqueue_if_fresh = enqueue
    queues = SimpleNamespace(completed=asyncio.Queue(), mark_producer_finished=AsyncMock(), producer_failure=None)
    events = []
    monkeypatch.setattr(
        "skyrl_train.fully_async_trainer.record_event", lambda name, fields, **kw: events.append((name, fields))
    )
    if expected is None:
        with pytest.raises(RuntimeError, match="requires native version evidence"):
            await trainer._run_generate_for_a_group_loop(queues)
        assert not groups
        trainer._staleness_manager.cancel_submission_slot.assert_awaited_once()
    else:
        await trainer._run_generate_for_a_group_loop(queues)
        assert len(groups) == 1
        assert groups[0].earliest_model_step == expected
        stamp = next(fields for name, fields in events if name == "rollout_admission_stamp")
        assert stamp["submission_model_step"] == 1
        assert stamp["first_token_model_step"] == 3
        assert stamp["admission_model_step"] == expected
        assert batch["submission_model_step"] == 1 and batch["first_token_model_step"] == 3
    queues.mark_producer_finished.assert_awaited_once()
