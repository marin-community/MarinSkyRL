from contextlib import nullcontext

import pytest
from omegaconf import OmegaConf

from skyrl_train.config.rollout_validation import EntrypointOperation
from skyrl_train.entrypoints import main_generate
from skyrl_train.entrypoints.main_generate import EvalOnlyEntrypoint


class LifecycleRunner:
    def __init__(self) -> None:
        self.events = []
        self.sink = None

    def set_trajectory_sink(self, sink) -> None:
        if self.events and sink is not self.sink:
            raise RuntimeError("cannot replace a started worker's sink")
        self.sink = sink

    async def startup(self) -> None:
        self.events.append("startup")

    async def shutdown(self) -> None:
        self.events.append("shutdown")


class RecordingTracker:
    def __init__(self) -> None:
        self.calls = []

    def log(self, *args, **kwargs) -> None:
        self.calls.append((args, kwargs))


@pytest.mark.asyncio
async def test_eval_only_uses_generation_engine_without_initial_wake(monkeypatch):
    experiment = object.__new__(EvalOnlyEntrypoint)
    experiment.cfg = OmegaConf.create(
        {
            "trainer": {"evaluation_runner": "training", "policy": {"model": {"lora": {"adapter_path": None}}}},
            "generator": {},
        }
    )
    experiment.eval_dataset = ["prompt"]
    experiment.tokenizer = object()
    inference_client = object()
    trajectory_runner = LifecycleRunner()
    tracker = RecordingTracker()

    def create_inference_engine_client(*, operation: EntrypointOperation):
        assert operation is EntrypointOperation.GENERATE
        return inference_client

    async def evaluate(**kwargs):
        assert trajectory_runner.sink is not None
        trajectory_runner.set_trajectory_sink(kwargs["trajectory_sink"])
        assert kwargs["eval_dataloader"] == "dataloader"
        assert kwargs["trajectory_runner"] is trajectory_runner
        trajectory_runner.events.append("evaluate")
        return {"reward": 1.0}

    experiment.create_inference_engine_client = create_inference_engine_client
    experiment.get_trajectory_runner = lambda *_args: trajectory_runner
    experiment.get_tracker = lambda: nullcontext(tracker)
    monkeypatch.setattr(main_generate, "build_eval_dataloader", lambda *_args, **_kwargs: "dataloader")
    monkeypatch.setattr(main_generate, "evaluate", evaluate)

    result = await experiment._evaluate()

    assert result == {"reward": 1.0}
    assert trajectory_runner.events == ["startup", "evaluate", "shutdown"]
    assert tracker.calls == [(({"reward": 1.0},), {"step": 0, "commit": True})]
