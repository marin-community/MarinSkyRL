import json

import pytest
from omegaconf import OmegaConf

from skyrl_train.config.trajectory_runner_capabilities import EntrypointOperation
from skyrl_train.entrypoints import main_generate
from skyrl_train.entrypoints.main_generate import EvalOnlyEntrypoint


def test_pilot_loads_validation_when_default_evaluation_interval_is_disabled(tmp_path):
    path = tmp_path / "validation.jsonl"
    rows = [
        {"prompt": [{"role": "user", "content": "Validation prompt"}], "extra_info": {"source_id": str(index)}}
        for index in range(2)
    ]
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    experiment = object.__new__(EvalOnlyEntrypoint)
    experiment.tokenizer = None
    experiment.cfg = OmegaConf.create(
        {
            "trainer": {"eval_interval": -1, "max_prompt_length": 1, "pivot_pilot": {"arm": "rl_tool_name"}},
            "data": {"val_data": [str(path)], "prompt_length_policy": "keep"},
        }
    )

    dataset = experiment.get_eval_dataset()

    assert len(dataset) == 2
    assert dataset[1][0] == rows[1]["prompt"]
    assert dataset[1][2]["extra_info"]["source_id"] == "1"


class LifecycleRunner:
    def __init__(self) -> None:
        self.events = []

    async def startup(self) -> None:
        self.events.append("startup")

    def set_trajectory_sink(self, sink) -> None:
        self.events.append("attach_sink")

    async def shutdown(self) -> None:
        self.events.append("shutdown")


class RecordingTracker:
    def __init__(self) -> None:
        self.calls = []
        self.finished = False

    def log(self, *args, **kwargs) -> None:
        self.calls.append((args, kwargs))

    def finish(self, exit_code=0) -> None:
        self.finished = True


@pytest.mark.asyncio
async def test_eval_only_uses_generation_engine_without_initial_wake(monkeypatch):
    experiment = object.__new__(EvalOnlyEntrypoint)
    experiment.cfg = OmegaConf.create(
        {"trainer": {"policy": {"model": {"lora": {"adapter_path": None}}}}, "generator": {"pivot_profiling": False}}
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
        assert kwargs["eval_dataloader"] == "dataloader"
        assert kwargs["trajectory_runner"] is trajectory_runner
        trajectory_runner.events.append("evaluate")
        return {"reward": 1.0}

    experiment.create_inference_engine_client = create_inference_engine_client
    experiment.get_trajectory_runner = lambda *_args: trajectory_runner
    experiment.get_tracker = lambda: tracker
    monkeypatch.setattr(main_generate, "build_eval_dataloader", lambda *_args, **_kwargs: "dataloader")
    monkeypatch.setattr(main_generate, "evaluate", evaluate)

    class Sink:
        def close(self):
            trajectory_runner.events.append("close_sink")

    monkeypatch.setattr(main_generate, "make_trajectory_sink", lambda *_args: Sink())

    result = await experiment._evaluate()

    assert result == {"reward": 1.0}
    assert tracker.finished
    assert trajectory_runner.events == ["attach_sink", "startup", "evaluate", "shutdown", "close_sink"]
    assert tracker.calls == [(({"reward": 1.0},), {"step": 0, "commit": True})]
