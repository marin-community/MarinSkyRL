"""Completion receipts follow real lifecycle, save, and evaluation behavior."""

import asyncio
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from omegaconf import OmegaConf
from skyrl_train.callbacks import CallbackHandler, TrainerControl
from skyrl_train.callbacks.builtin import (
    BufferCheckpointCallback,
    CheckpointCallback,
    EvaluationCallback,
    HFModelSaveCallback,
)
from skyrl_train.config.utils import get_default_config
from skyrl_train.entrypoints.main_base import BasePPOExp
from skyrl_train.fully_async_trainer import FullyAsyncRayPPOTrainer, _GenerationQueues
from skyrl_train.trainer import RayPPOTrainer
from skyrl_train.training_completion import write_training_receipt
from skyrl_train.trajectory_runners.base import TrajectoryRunner

from marinskyrl.training_completion import TrainingReceipt, validate_completion_config


def completion_config(tmp_path, mode="metrics"):
    cfg = get_default_config()
    cfg.trainer.completion = {
        "mode": mode,
        "run_id": "training-run",
        "attempt_id": "attempt-one",
        "request_fingerprint": "a" * 64,
        "receipt_uri": str(tmp_path / "receipt.json"),
    }
    cfg.trainer.ckpt_path = str(tmp_path / "checkpoints")
    cfg.trainer.ckpt_interval = -1
    cfg.trainer.hf_save_interval = -1
    return cfg


@pytest.mark.parametrize("failure", [None, "setup", "train", "shutdown", "tracker"])
def test_receipt_requires_successful_experiment_lifecycle(tmp_path, failure):
    cfg = completion_config(tmp_path)
    receipt_path = Path(cfg.trainer.completion.receipt_uri)
    finished = []

    class Tracker:
        def finish(self, exit_code):
            assert not receipt_path.exists()
            finished.append(exit_code)
            if failure == "tracker":
                raise RuntimeError("tracker failed")

    class Trainer:
        global_step = 0

        async def train(self):
            assert not receipt_path.exists()
            self.global_step = 3
            if failure == "train":
                raise RuntimeError("train failed")

        async def shutdown(self):
            assert not receipt_path.exists()
            if failure == "shutdown":
                raise RuntimeError("shutdown failed")

    class Experiment(BasePPOExp):
        def _setup_trainer(self):
            self.tracker = Tracker()
            if failure == "setup":
                raise RuntimeError("setup failed")
            return Trainer()

    experiment = object.__new__(Experiment)
    experiment.cfg = cfg
    if failure:
        with pytest.raises(RuntimeError, match="failed"):
            experiment._run()
        assert not receipt_path.exists()
    else:
        experiment._run()
        value = json.loads(receipt_path.read_text())
        assert value["global_step"] == 3
        assert value["attempt_id"] == "attempt-one"
        assert "checkpoint" not in value
        assert TrainingReceipt.from_dict(value).to_dict() == value
    assert finished == [1 if failure in {"setup", "train", "shutdown"} else 0]
    assert not Path(cfg.trainer.ckpt_path).exists()


@pytest.mark.parametrize(
    "override",
    [
        {"ckpt_interval": 5},
        {"hf_save_interval": 5},
        {"callbacks": [{"type": "checkpoint"}]},
        {"callbacks": [{"type": "hf_model_save", "save_steps": 2}]},
    ],
)
def test_metrics_rejects_legacy_and_explicit_saving(tmp_path, override):
    cfg = completion_config(tmp_path)
    for key, value in override.items():
        OmegaConf.update(cfg, "trainer." + key, value, force_add=True)
    with pytest.raises(ValueError, match="forbids"):
        validate_completion_config(cfg)
    assert not list(tmp_path.iterdir())


class LocalPolicyWorker:
    def async_run_ray_method(self, _dispatch, _method, *, ckpt_dir, tokenizer):
        path = Path(ckpt_dir)
        path.mkdir(parents=True)
        (path / "weights.distcp").write_bytes(b"native-policy-shard")
        return []


class Dataloader(list):
    def state_dict(self):
        return {"position": len(self)}


def bare_trainer(cfg, cls=RayPPOTrainer):
    trainer = object.__new__(cls)
    trainer.cfg = cfg
    trainer.global_step = 7
    trainer.total_training_steps = 7
    trainer.num_steps_per_epoch = 7
    trainer.train_dataloader = Dataloader([None] * 7)
    trainer.all_metrics = {}
    trainer.all_timings = {}
    trainer._control = TrainerControl()
    trainer._last_saved_step = None
    trainer.colocate_all = False
    trainer.eval_dataset = None
    trainer.policy_model = LocalPolicyWorker()
    trainer.critic_model = None
    trainer.tokenizer = SimpleNamespace(decode=lambda tokens: "answer")
    trainer.tracker = SimpleNamespace(log=lambda *args, **kwargs: None)
    return trainer


@pytest.mark.parametrize("cls", [RayPPOTrainer, FullyAsyncRayPPOTrainer])
@pytest.mark.parametrize("mode", ["metrics", "checkpoint"])
def test_completion_saving_controls_and_final_native_receipt(tmp_path, monkeypatch, cls, mode):
    cfg = completion_config(tmp_path, mode)
    trainer = bare_trainer(cfg, cls)
    # Ray is the external boundary; save_checkpoints itself writes real native metadata.
    monkeypatch.setattr("skyrl_train.trainer.ray.get", lambda refs: refs)
    trainer.callback_handler = CallbackHandler(
        [CheckpointCallback(save_steps=10, save_on_train_end=False), HFModelSaveCallback(save_steps=1)]
    )
    state = trainer._create_trainer_state(epoch=0)
    if mode == "metrics":
        asyncio.run(trainer._save_intermediate_checkpoint(state))
    trainer.handle_hf_export()
    asyncio.run(trainer._finalize_training(completed_step=7, epoch=0))
    receipt = write_training_receipt(cfg, trainer)
    value = json.loads(Path(cfg.trainer.completion.receipt_uri).read_text())
    if mode == "metrics":
        assert "checkpoint" not in value
        assert not Path(cfg.trainer.ckpt_path).exists()
    else:
        checkpoint = value["checkpoint"]
        state_path = Path(checkpoint["checkpoint_path"]) / "trainer_state.pt"
        assert checkpoint["global_step"] == 7
        assert checkpoint["trainer_state_sha256"] == hashlib.sha256(state_path.read_bytes()).hexdigest()
        assert {file["path"] for file in checkpoint["files"]} == {
            "data.pt",
            "trainer_state.pt",
            "policy/weights.distcp",
        }
        assert not list(tmp_path.rglob("hf_export_request.json"))
    assert TrainingReceipt.from_dict(value) == receipt


@pytest.mark.parametrize("corruption", ["missing", "marker", "state", "data", "policy"])
def test_invalid_native_checkpoint_cannot_publish_receipt(tmp_path, monkeypatch, corruption):
    cfg = completion_config(tmp_path, "checkpoint")
    trainer = bare_trainer(cfg)
    monkeypatch.setattr("skyrl_train.trainer.ray.get", lambda refs: refs)
    if corruption != "missing":
        trainer.save_checkpoints()
        root = Path(cfg.trainer.ckpt_path)
        step_dir = root / "global_step_7"
        if corruption == "marker":
            (root / "latest_ckpt_global_step.txt").write_text("6")
        elif corruption == "state":
            torch.save({"global_step": 6}, step_dir / "trainer_state.pt")
        elif corruption == "data":
            (step_dir / "data.pt").unlink()
        elif corruption == "policy":
            (step_dir / "policy/weights.distcp").unlink()
    with pytest.raises((ValueError, FileNotFoundError)):
        write_training_receipt(cfg, trainer)
    assert not Path(cfg.trainer.completion.receipt_uri).exists()


class EvaluationRunner(TrajectoryRunner):
    def __init__(self, fail_once):
        self.steps = []
        self.fail_once = fail_once

    async def _run(self, input_batch, disable_tqdm=False):
        self.steps.append(input_batch["batch_metadata"].global_step)
        if self.fail_once:
            self.fail_once = False
            raise RuntimeError("evaluation failed")
        return {
            "prompt_token_ids": [[1]],
            "response_ids": [[2]],
            "rewards": [1.0],
            "loss_masks": [[1]],
            "stop_reasons": ["stop"],
            "rollout_logprobs": None,
        }


@pytest.mark.parametrize("cls", [RayPPOTrainer, FullyAsyncRayPPOTrainer])
@pytest.mark.parametrize("fail_once", [False, True])
def test_final_evaluation_deduplicates_only_successful_same_step(tmp_path, cls, fail_once):
    cfg = completion_config(tmp_path)
    cfg.trainer.dump_eval_results = False
    cfg.generator.eval_n_samples_per_prompt = 1
    trainer = bare_trainer(cfg, cls)
    trainer.eval_dataset = object()
    trainer.eval_dataloader = [
        [
            {
                "prompt": [{"role": "user", "content": "question"}],
                "env_class": None,
                "env_extras": {"data_source": "heldout"},
                "uid": "question-one",
            }
        ]
    ]
    runner = EvaluationRunner(fail_once)
    trainer.trajectory_runner = runner
    trainer.trajectory_sink = None
    trainer.callback_handler = CallbackHandler([EvaluationCallback(eval_steps=7)])
    if fail_once:
        with pytest.raises(RuntimeError, match="evaluation failed"):
            asyncio.run(trainer.eval())
    else:
        metrics = asyncio.run(trainer.eval())
        assert metrics["eval/all/avg_score"] == 1.0
    asyncio.run(trainer._finalize_training(completed_step=7, epoch=0))
    asyncio.run(trainer._finalize_training(completed_step=7, epoch=0))
    assert runner.steps == ([7, 7] if fail_once else [7])
    asyncio.run(trainer._finalize_training(completed_step=8, epoch=0))
    assert runner.steps[-1] == 8


def test_checkpoint_mode_preserves_periodic_save_and_avoids_duplicate_final_write(tmp_path, monkeypatch):
    cfg = completion_config(tmp_path, "checkpoint")
    trainer = bare_trainer(cfg)
    monkeypatch.setattr("skyrl_train.trainer.ray.get", lambda refs: refs)
    trainer.callback_handler = CallbackHandler([CheckpointCallback(save_steps=7)])
    state = trainer._create_trainer_state(epoch=0)
    control = asyncio.run(trainer.callback_handler.call_event_async("on_step_end", state, trainer._control))
    if control.should_save:
        asyncio.run(trainer._save_intermediate_checkpoint(state))
    state_path = Path(cfg.trainer.ckpt_path) / "global_step_7/trainer_state.pt"
    original = state_path.read_bytes()
    # The worker refuses to overwrite an existing policy directory, so a second
    # final save would fail instead of hiding a duplicate expensive write.
    asyncio.run(trainer._finalize_training(completed_step=7, epoch=0))
    write_training_receipt(cfg, trainer)
    assert state_path.read_bytes() == original
    assert Path(cfg.trainer.completion.receipt_uri).exists()


@pytest.mark.parametrize("mode", ["metrics", "checkpoint"])
def test_metrics_shutdown_does_not_modify_resumed_checkpoint(tmp_path, mode):
    cfg = completion_config(tmp_path, mode)
    root = Path(cfg.trainer.ckpt_path)
    step_dir = root / "global_step_6"
    step_dir.mkdir(parents=True)
    (root / "latest_ckpt_global_step.txt").write_text("6")
    queues = _GenerationQueues(
        completed=asyncio.Queue(maxsize=1), retries=asyncio.Queue(), condition=asyncio.Condition()
    )
    queues.retries.put_nowait([{"uid": "untrained-prompt"}])
    callback = BufferCheckpointCallback()
    callback.bind_queues(queues)
    trainer = bare_trainer(cfg, FullyAsyncRayPPOTrainer)
    trainer._buffer_checkpoint_callback = callback
    asyncio.run(trainer._flush_generation_buffer_on_shutdown())
    artifact_path = step_dir / "generation_buffer_state.pt"
    if mode == "metrics":
        assert not artifact_path.exists()
    else:
        state = torch.load(artifact_path, weights_only=False)
        assert state["retry_prompts"] == [[{"uid": "untrained-prompt"}]]
    assert (root / "latest_ckpt_global_step.txt").read_text() == "6"


@pytest.mark.parametrize(
    "corruption",
    ["boolean_version", "negative_step", "boolean_step", "duplicate", "zero", "traversal", "policy_metadata"],
)
def test_receipt_parser_rejects_invalid_completion_evidence(tmp_path, monkeypatch, corruption):
    cfg = completion_config(tmp_path, "checkpoint")
    trainer = bare_trainer(cfg)
    monkeypatch.setattr("skyrl_train.trainer.ray.get", lambda refs: refs)
    trainer.save_checkpoints()
    receipt = write_training_receipt(cfg, trainer).to_dict()
    if corruption == "boolean_version":
        receipt["schema_version"] = True
    elif corruption == "negative_step":
        receipt["global_step"] = -1
    elif corruption == "boolean_step":
        receipt["checkpoint"]["global_step"] = True
    elif corruption == "duplicate":
        receipt["checkpoint"]["files"].append(receipt["checkpoint"]["files"][0])
    elif corruption == "zero":
        receipt["checkpoint"]["files"][0]["size"] = 0
    elif corruption == "traversal":
        receipt["checkpoint"]["files"][0]["path"] = "../data.pt"
    else:
        for file in receipt["checkpoint"]["files"]:
            if file["path"].startswith("policy/"):
                file["path"] = "policy/config.json"
    with pytest.raises(ValueError):
        TrainingReceipt.from_dict(receipt)
