"""Actual-loop regression coverage for synchronous Megatron checkpoint continuation.

The model work is stubbed, but this deliberately runs ``RayPPOTrainer._train_loop``
with a real ``StatefulDataLoader``.  In particular, it must not replace the loop
with a hand-written approximation: reaching the iterator's terminal
``StopIteration`` is the behavior under test.
"""

from __future__ import annotations

import asyncio
from collections import deque
from dataclasses import dataclass
import json
from pathlib import Path
import random
from types import SimpleNamespace

import numpy as np
from omegaconf import OmegaConf
import torch
from torch.utils.data import Dataset
from torchdata.stateful_dataloader import StatefulDataLoader

from skyrl_train.callbacks.base import CallbackHandler, TrainerCallback, TrainerControl
from skyrl_train.checkpoint_generation import new_attempt_path, resolve_checkpoint_payload
from skyrl_train.hf_export_schema import TRAINER_STATE_FILENAME
import skyrl_train.trainer as trainer_module
from skyrl_train.trainer import RayPPOTrainer
from skyrl_train.utils.trainer_utils import ResumeMode


class _Rows(Dataset):
    def __len__(self) -> int:
        return 4

    def __getitem__(self, index: int) -> int:
        return index


class _SaveAtStepTwo(TrainerCallback):
    error_behavior = "raise"

    def on_step_end(self, state, control, **_kwargs):
        if state.global_step == 2:
            control.should_save = True
        return control


class _SaveAtStepTwoWithEpochEndRng(_SaveAtStepTwo):
    def __init__(self):
        self.epoch_end_rng_draws: list[_RngDraw] = []

    def on_epoch_end(self, state, control, **_kwargs):
        if state.epoch == 0:
            self.epoch_end_rng_draws.append(_draw_from_driver_rng())
        return control


_RngDraw = tuple[float, float, float]


def _driver_rng_state():
    return random.getstate(), np.random.get_state(), torch.get_rng_state()


def _restore_rng_state(state) -> None:
    python_state, numpy_state, torch_state = state
    random.setstate(python_state)
    np.random.set_state(numpy_state)
    torch.set_rng_state(torch_state)


def _checkpoint_rng_state(state) -> dict[str, object]:
    python_state, numpy_state, torch_state = state
    return {"python": python_state, "numpy": numpy_state, "torch_cpu": torch_state}


def _draw_from_driver_rng() -> _RngDraw:
    return random.random(), float(np.random.random()), float(torch.rand(1).item())


class _SaveAtStepOneWithLateRng(TrainerCallback):
    error_behavior = "raise"

    def __init__(self, early_state: list | None = None):
        self.early_state = early_state
        self.events: list[str] = []
        self.expected_next_draw: _RngDraw | None = None

    def on_step_end(self, state, control, **_kwargs):
        if state.global_step == 1:
            control.should_save = True
            if self.early_state is not None:
                self.early_state.append(_driver_rng_state())
        return control

    def on_log(self, state, control, logs, **_kwargs):
        if state.global_step == 1:
            self.events.append("on_log")
            _draw_from_driver_rng()
            late_state = _driver_rng_state()
            self.expected_next_draw = _draw_from_driver_rng()
            _restore_rng_state(late_state)
        return control

    def on_save(self, state, control, **_kwargs):
        if state.global_step == 1:
            self.events.append("on_save")
            _draw_from_driver_rng()
        return control


class _NoopTracker:
    def log(self, *_args, **_kwargs) -> None:
        return None


class _NoopPolicyActors:
    def async_run_ray_method(self, *_args, **_kwargs):
        return []


class _Tokenizer:
    def decode(self, _tokens) -> str:
        return "response"


class _EpochBoundaryLoopHarness(RayPPOTrainer):
    """Use the production loop while replacing GPU/model work with CPU no-ops."""

    def __init__(
        self,
        checkpoint_root: Path,
        *,
        epochs: int,
        resume: bool,
        resume_step: int = 2,
        callbacks: list[TrainerCallback] | None = None,
        record_rng: bool = False,
        skipped_admission_batches: int = 0,
        enable_http_endpoint: bool = False,
    ):
        self.cfg = OmegaConf.create(
            {
                "trainer": {
                    "strategy": "megatron",
                    "epochs": epochs,
                    "train_batch_size": 2,
                    "step_wise_training": False,
                    "dump_data_batch": False,
                    "resume_mode": "from_path" if resume else "none",
                    "resume_path": str(checkpoint_root / f"global_step_{resume_step}"),
                    "restore_dataloader_state": True,
                    "reset_distillation_token_count_on_resume": False,
                    "reset_global_step_on_resume": False,
                    "ckpt_interval": 1,
                    "ckpt_path": str(checkpoint_root),
                    "max_ckpts_to_keep": -1,
                    "algorithm": {
                        "use_kl_in_reward": False,
                        "use_tis": False,
                        "dynamic_sampling": {"type": None},
                    },
                },
                "generator": {
                    "n_samples_per_prompt": 1,
                    "backend": "vllm",
                    "sampling_params": {},
                    "enable_http_endpoint": enable_http_endpoint,
                },
                "environment": {"env_class": "test"},
            }
        )
        self.train_dataloader = StatefulDataLoader(
            _Rows(),
            batch_size=2,
            shuffle=False,
            num_workers=0,
            drop_last=True,
            generator=torch.Generator().manual_seed(0),
        )
        self.total_training_steps = len(self.train_dataloader) * epochs
        self.resume_mode = ResumeMode.FROM_PATH if resume else ResumeMode.NONE
        self.global_step = 0
        self.colocate_all = False
        self.eval_dataset = None
        self.policy_model = _NoopPolicyActors()
        self.critic_model = None
        self.ref_model = None
        self.tokenizer = _Tokenizer()
        self.tracker = _NoopTracker()
        self.callback_handler = CallbackHandler([_SaveAtStepTwo()] if callbacks is None else callbacks)
        self._control = TrainerControl()
        self.all_metrics = {}
        self.all_timings = {}
        self.all_startup_timings = {}
        self._checkpoint_save_failures = 0.0
        self._last_saved_step = None
        self._active_checkpoint_payload_path = None
        self._checkpoint_required_files = None
        self._last_optimizer_step_finished_at = None
        self._pending_checkpoint_upload = None
        self._pending_sync_prompts = []
        self._domain_balancer = None
        self.distillation_scored_tokens_total = 0
        self.group_admission_state = None
        self._sync_distillation_runtime = None
        self._step_time_history = deque(maxlen=5)
        self.processed_batches: list[tuple[int, ...]] = []
        self.record_rng = record_rng
        self.rng_draws: list[_RngDraw] = []
        self.skipped_admission_batches = skipped_admission_batches
        self.published_iterator_finished: list[bool] = []

    def init_weight_sync_state(self) -> None:
        return None

    async def _start_draft_trainer(self) -> None:
        return None

    async def _sync_policy_for_rollouts(self, *, reason: str) -> None:
        return None

    async def _begin_speculator_capture(self) -> None:
        return None

    async def _seal_speculator_capture(self) -> None:
        return None

    async def _start_speculator_update(self) -> None:
        return None

    async def _poll_speculator_lifecycle(self) -> None:
        return None

    def _log_startup_timings(self) -> None:
        return None

    def _select_sync_generation_prompts(self, entries):
        return entries

    def _remove_tail_data(self, entries):
        return entries

    async def generate(self, request):
        return {"response_ids": [[1]], "rewards": [0.0], "batch_ids": request["batch_ids"]}

    def _update_curriculum_sampler(self, *_args) -> None:
        return None

    def handle_group_admission(self, trajectory_batch, uids):
        keep_sampling = self.skipped_admission_batches > 0
        self.skipped_admission_batches -= int(keep_sampling)
        return SimpleNamespace(trajectory_batch=trajectory_batch, uids=uids, keep_sampling=keep_sampling)

    def postprocess_trajectory_batch(self, trajectory_batch, _uids):
        return trajectory_batch

    def select_trajectories(self, trajectory_batch, uids):
        return trajectory_batch, uids

    def convert_to_training_input(self, trajectory_batch, _uids):
        return {"sequences": trajectory_batch["batch_ids"], "batch_ids": trajectory_batch["batch_ids"]}

    async def _forward_with_optional_distillation(self, _trajectory_batch, training_input):
        return training_input

    def compute_advantages_and_returns(self, training_input):
        return training_input

    def finalize_advantages_for_training(self, training_input):
        return training_input

    def train_critic_and_policy(self, training_input):
        self.processed_batches.append(tuple(int(item) for item in training_input["batch_ids"]))
        if self.record_rng:
            self.rng_draws.append(_draw_from_driver_rng())
        return {}

    def _log_optimizer_step_completed(self, **_kwargs) -> None:
        return None

    def _get_ref_update_callback(self):
        return None

    def _log_metrics_stdout(self, *_args, **_kwargs) -> None:
        return None

    def _log_training_step_completed(self, **_kwargs) -> None:
        return None

    async def _finalize_training(self, *, completed_step: int, epoch: int) -> None:
        return None

    def _offload_policy_optimizer(self, *_args, **_kwargs) -> None:
        return None

    def _save_checkpoint_payloads(self, *, step: int | None = None, include_continuation_state: bool = True) -> None:
        """Create a tiny rank payload while preserving production seal timing."""
        step = self.global_step if step is None else step
        step_path = Path(self.cfg.trainer.ckpt_path, f"global_step_{step}")
        attempt_path = Path(new_attempt_path(str(step_path)))
        attempt_path.mkdir(parents=True)
        self._active_checkpoint_payload_path = str(attempt_path)
        required_files = {
            "data.pt",
            TRAINER_STATE_FILENAME,
            "policy/__0_0.distcp",
            "policy/worker_receipts.json",
        }
        if self._can_replay_driver_rng():
            required_files.add(trainer_module.DRIVER_RNG_STATE_FILENAME)
        self._checkpoint_required_files = required_files

        policy_path = attempt_path / "policy"
        policy_path.mkdir()
        (policy_path / "__0_0.distcp").write_bytes(b"rank payload")
        (policy_path / "worker_receipts.json").write_text(json.dumps({"component": "policy", "ranks": [0]}))
        if include_continuation_state:
            self._write_checkpoint_continuation_state(step)

    def _publish_checkpoint(self, step: int | None = None) -> None:
        self.published_iterator_finished.append(bool(self.train_dataloader.state_dict()["_iterator_finished"]))
        super()._publish_checkpoint(step)


@dataclass(frozen=True)
class _RngRoundTrip:
    events: tuple[str, ...]
    expected_next_draw: _RngDraw
    uninterrupted_next_draw: _RngDraw
    resumed_next_draw: _RngDraw


def _mid_epoch_rng_roundtrip(tmp_path, monkeypatch, *, mutate_to_early_capture: bool) -> _RngRoundTrip:
    _patch_loop_dependencies(monkeypatch)
    original_state = _driver_rng_state()
    try:
        random.seed(37)
        np.random.seed(37)
        torch.manual_seed(37)

        early_state: list = []
        callback = _SaveAtStepOneWithLateRng(early_state)
        if mutate_to_early_capture:
            real_capture = trainer_module._capture_driver_rng_state
            monkeypatch.setattr(
                trainer_module,
                "_capture_driver_rng_state",
                lambda: _checkpoint_rng_state(early_state[0]) if early_state else real_capture(),
            )

        first = _EpochBoundaryLoopHarness(
            tmp_path,
            epochs=1,
            resume=False,
            callbacks=[callback],
            record_rng=True,
        )
        asyncio.run(first._train_loop())
        assert first.processed_batches == [(0, 1), (2, 3)]
        assert callback.expected_next_draw is not None

        random.seed(100)
        np.random.seed(100)
        torch.manual_seed(100)
        resumed = _EpochBoundaryLoopHarness(
            tmp_path,
            epochs=1,
            resume=True,
            resume_step=1,
            callbacks=[_SaveAtStepOneWithLateRng()],
            record_rng=True,
        )
        asyncio.run(resumed._train_loop())
        assert resumed.processed_batches == [(2, 3)]

        return _RngRoundTrip(
            events=tuple(callback.events),
            expected_next_draw=callback.expected_next_draw,
            uninterrupted_next_draw=first.rng_draws[1],
            resumed_next_draw=resumed.rng_draws[0],
        )
    finally:
        _restore_rng_state(original_state)


def _patch_loop_dependencies(monkeypatch) -> None:
    monkeypatch.setattr(
        trainer_module,
        "prepare_trajectory_request",
        lambda entries, *_args, **_kwargs: (
            {"prompts": ["prompt"], "batch_ids": [int(item) for item in entries]},
            [str(int(item)) for item in entries],
        ),
    )
    monkeypatch.setattr(trainer_module, "get_sampling_params_for_backend", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(trainer_module, "log_example", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(trainer_module, "publish_step_timings", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(trainer_module, "record_policy_step", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(trainer_module.ray, "get", lambda refs: refs)


def test_megatron_epoch_checkpoint_resume_does_not_skip_next_epoch(tmp_path, monkeypatch):
    """The next epoch must run after resuming an epoch-boundary checkpoint."""

    _patch_loop_dependencies(monkeypatch)

    first = _EpochBoundaryLoopHarness(tmp_path, epochs=1, resume=False)
    asyncio.run(first._train_loop())
    assert first.processed_batches == [(0, 1), (2, 3)]

    resumed = _EpochBoundaryLoopHarness(tmp_path, epochs=2, resume=True)
    asyncio.run(resumed._train_loop())
    assert resumed.processed_batches == [(0, 1), (2, 3)]


def test_megatron_http_checkpoint_late_seals_without_driver_rng_replay(tmp_path, monkeypatch):
    """HTTP mode keeps exact data continuation but excludes process-global RNG."""

    _patch_loop_dependencies(monkeypatch)
    restore_calls: list[dict[str, object]] = []
    monkeypatch.setattr(trainer_module, "_restore_driver_rng_state", restore_calls.append)

    first = _EpochBoundaryLoopHarness(
        tmp_path,
        epochs=1,
        resume=False,
        enable_http_endpoint=True,
    )
    asyncio.run(first._train_loop())
    assert first.processed_batches == [(0, 1), (2, 3)]
    assert first.published_iterator_finished == [True]

    payload = Path(resolve_checkpoint_payload(str(tmp_path / "global_step_2"), verify_files=True))
    assert not (payload / trainer_module.DRIVER_RNG_STATE_FILENAME).exists()

    resumed = _EpochBoundaryLoopHarness(
        tmp_path,
        epochs=2,
        resume=True,
        enable_http_endpoint=True,
        callbacks=[],
    )
    asyncio.run(resumed._train_loop())
    assert resumed.processed_batches == [(0, 1), (2, 3)]
    assert restore_calls == []


def test_megatron_epoch_boundary_checkpoint_replays_rng_before_next_iterator(tmp_path, monkeypatch):
    """A fresh resume must match the first step of the uninterrupted next epoch."""

    _patch_loop_dependencies(monkeypatch)
    original_state = _driver_rng_state()
    try:
        random.seed(53)
        np.random.seed(53)
        torch.manual_seed(53)
        callback = _SaveAtStepTwoWithEpochEndRng()
        first = _EpochBoundaryLoopHarness(
            tmp_path,
            epochs=2,
            resume=False,
            callbacks=[callback],
            record_rng=True,
        )
        asyncio.run(first._train_loop())
        assert first.processed_batches == [(0, 1), (2, 3), (0, 1), (2, 3)]
        assert len(callback.epoch_end_rng_draws) == 1
        uninterrupted_next_epoch_draw = first.rng_draws[2]

        random.seed(101)
        np.random.seed(101)
        torch.manual_seed(101)
        resumed = _EpochBoundaryLoopHarness(
            tmp_path,
            epochs=2,
            resume=True,
            resume_step=2,
            callbacks=[],
            record_rng=True,
        )
        asyncio.run(resumed._train_loop())
        assert resumed.processed_batches == [(0, 1), (2, 3)]
        assert resumed.rng_draws[0] == uninterrupted_next_epoch_draw
    finally:
        _restore_rng_state(original_state)


def test_megatron_mid_epoch_checkpoint_replays_rng_after_late_callbacks(tmp_path, monkeypatch):
    result = _mid_epoch_rng_roundtrip(tmp_path, monkeypatch, mutate_to_early_capture=False)

    assert result.events == ("on_save", "on_log")
    assert result.uninterrupted_next_draw == result.expected_next_draw
    assert result.resumed_next_draw == result.expected_next_draw


def test_megatron_mid_epoch_checkpoint_rng_oracle_detects_early_capture(tmp_path, monkeypatch):
    result = _mid_epoch_rng_roundtrip(tmp_path, monkeypatch, mutate_to_early_capture=True)

    assert result.events == ("on_save", "on_log")
    assert result.uninterrupted_next_draw != result.expected_next_draw
    assert result.resumed_next_draw != result.expected_next_draw


def test_megatron_skipped_batch_last_physical_batch_checkpoint_seals_epoch(tmp_path, monkeypatch):
    """A skipped batch must not make the last physical batch publish before exhaustion."""

    _patch_loop_dependencies(monkeypatch)
    callback = _SaveAtStepOneWithLateRng()
    first = _EpochBoundaryLoopHarness(
        tmp_path,
        epochs=1,
        resume=False,
        callbacks=[callback],
        skipped_admission_batches=1,
    )
    asyncio.run(first._train_loop())

    assert first.processed_batches == [(2, 3)]
    assert first.published_iterator_finished == [True]
    payload = Path(resolve_checkpoint_payload(str(tmp_path / "global_step_1"), verify_files=True))
    trainer_state = torch.load(payload / TRAINER_STATE_FILENAME, map_location="cpu", weights_only=False)
    assert trainer_state["global_step"] == 1
    assert trainer_state["resume_epoch"] == 1
    assert trainer_state["at_epoch_boundary"] is True

    resumed = _EpochBoundaryLoopHarness(
        tmp_path,
        epochs=1,
        resume=True,
        resume_step=1,
        callbacks=[],
    )
    asyncio.run(resumed._train_loop())
    assert resumed.processed_batches == []
