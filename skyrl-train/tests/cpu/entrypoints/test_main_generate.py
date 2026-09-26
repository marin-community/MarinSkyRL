import pytest
from omegaconf import OmegaConf

from skyrl_train.config.trajectory_runner_capabilities import EntrypointOperation
from skyrl_train.entrypoints import main_generate
from skyrl_train.entrypoints.main_generate import EvalOnlyEntrypoint


class RecordingTracker:
    def __init__(self) -> None:
        self.calls = []

    def log(self, *args, **kwargs) -> None:
        self.calls.append((args, kwargs))


@pytest.mark.asyncio
async def test_eval_only_uses_generation_engine_without_initial_wake(monkeypatch):
    experiment = object.__new__(EvalOnlyEntrypoint)
    experiment.cfg = OmegaConf.create({"trainer": {"policy": {"model": {"lora": {"adapter_path": None}}}}})
    experiment.eval_dataset = ["prompt"]
    experiment.tokenizer = object()
    inference_client = object()
    trajectory_runner = object()
    tracker = RecordingTracker()

    def create_inference_engine_client(*, operation: EntrypointOperation):
        assert operation is EntrypointOperation.GENERATE
        return inference_client

    async def evaluate(**kwargs):
        assert kwargs["eval_dataloader"] == "dataloader"
        assert kwargs["trajectory_runner"] is trajectory_runner
        return {"reward": 1.0}

    experiment.create_inference_engine_client = create_inference_engine_client
    experiment.get_trajectory_runner = lambda *_args: trajectory_runner
    experiment.get_tracker = lambda: tracker
    monkeypatch.setattr(main_generate, "build_dataloader", lambda *_args, **_kwargs: "dataloader")
    monkeypatch.setattr(main_generate, "evaluate", evaluate)

    result = await experiment._evaluate()

    assert result == {"reward": 1.0}
    assert tracker.calls == [(({"reward": 1.0},), {"step": 0, "commit": True})]
