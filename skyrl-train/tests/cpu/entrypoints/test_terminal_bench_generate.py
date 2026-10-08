from pathlib import Path

import pytest
from omegaconf import OmegaConf

from skyrl_train.config.trajectory_runner_capabilities import EntrypointOperation
from skyrl_train.entrypoints.terminal_bench_generate import TerminalBenchGenerateExp
from skyrl_train.trajectory_runners.types import BatchMetadata, TrajectoryRequestBatch
from skyrl_train.trajectory_runners.trajectory_retention import RetentionSink
from skyrl_train.trajectory_runners.harbor.dataset import TerminalBenchTaskDataset


class RecordingTrajectoryRunner:
    def __init__(self) -> None:
        self.requests: list[TrajectoryRequestBatch] = []
        self.events: list[str] = []

    def set_trajectory_sink(self, sink: RetentionSink) -> None:
        sink.bind_runner(type(self).__name__)

    async def startup(self) -> None:
        self.events.append("startup")

    async def start_eval_session(self, *, run_name: str, eval_step: int) -> None:
        self.events.append(f"start_eval {run_name} {eval_step}")

    async def run(self, request: TrajectoryRequestBatch) -> None:
        self.events.append("run")
        self.requests.append(request)

    async def stop_eval_session(self) -> None:
        self.events.append("stop_eval")

    async def shutdown(self) -> None:
        self.events.append("shutdown")


@pytest.mark.parametrize("shuffle", [False, True])
def test_terminal_bench_generate_batches_all_tasks_with_global_indices(tmp_path: Path, shuffle: bool):
    runner = RecordingTrajectoryRunner()
    experiment = object.__new__(TerminalBenchGenerateExp)
    experiment.cfg = OmegaConf.create(
        {
            "data": {"shuffle": shuffle},
            "generator": {
                "backend": "vllm",
                "n_samples_per_prompt": 8,
                "sampling_params": {
                    "max_generate_length": 4096,
                    "temperature": 0.7,
                    "top_p": 0.95,
                    "top_k": 20,
                    "min_p": 0.0,
                    "logprobs": None,
                },
            },
            "environment": {"env_class": "terminal_bench"},
            # run() configures progress and enters the trainer telemetry lifecycle before _run().
            "trainer": {
                "seed": 7,
                "run_name": "generate",
                "eval_batch_size": 2,
                "eval_num_prompts": 1,
                "progress": {
                    "mode": "tqdm",
                    "min_interval_seconds": 0.5,
                    "heartbeat_seconds": 15,
                    "percent_step": 5,
                    "count_step": 1000,
                },
            },
        }
    )
    task_paths = [tmp_path / f"task-{task}" for task in ("a", "b", "c", "d", "e")]
    for path in task_paths:
        path.mkdir()
        (path / "instruction.md").write_text("Use the available tools.")
    experiment.train_dataset = TerminalBenchTaskDataset([str(tmp_path)])
    experiment.tokenizer = object()
    inference_client = object()

    def create_inference_engine_client(*, operation: EntrypointOperation):
        assert operation is EntrypointOperation.GENERATE
        return inference_client

    experiment.create_inference_engine_client = create_inference_engine_client
    experiment.get_trajectory_runner = lambda cfg, tokenizer, client: runner

    experiment.run()

    assert runner.events == ["startup", "start_eval generate 0", "run", "run", "run", "stop_eval", "shutdown"]
    assert [len(request["prompts"]) for request in runner.requests] == [16, 16, 8]
    indices = [extras["task_index"] for request in runner.requests for extras in request["env_extras"]]
    order = indices[::8]
    assert sorted(order) == list(range(5))
    assert (order != list(range(5))) is shuffle
    assert [prompt for request in runner.requests for prompt in request["prompts"]] == [
        str(task_paths[index]) for index in order for _ in range(8)
    ]
    assert indices == [index for index in order for _ in range(8)]
    assert [
        trajectory_id.to_string() for request in runner.requests for trajectory_id in request["trajectory_ids"]
    ] == [f"{task_paths[index].name}_{repetition_id}" for index in order for repetition_id in range(8)]
    for request in runner.requests:
        assert request["env_classes"] == ["terminal_bench"] * len(request["prompts"])
        assert request["batch_metadata"] == BatchMetadata(global_step=0, training_phase="eval")

    runner.requests.clear()
    experiment.run()
    assert [extras["task_index"] for request in runner.requests for extras in request["env_extras"]] == indices

    experiment.cfg.trainer.seed = 8
    runner.requests.clear()
    experiment.run()
    new_indices = [extras["task_index"] for request in runner.requests for extras in request["env_extras"]]
    assert sorted(new_indices) == sorted(indices)
    assert (new_indices != indices) is shuffle
