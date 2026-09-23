from pathlib import Path

import pytest
from omegaconf import OmegaConf

from skyrl_train.config.trajectory_runner_capabilities import EntrypointOperation
from skyrl_train.entrypoints.main_generate import run_evaluation_only
from skyrl_train.entrypoints.terminal_bench_generate import TerminalBenchGenerateExp
from skyrl_train.inference_engines.vllm.stats import InferenceStatsSnapshot
from skyrl_train.trajectory_runners.base import TrajectoryBatch, TrajectoryRunner
from skyrl_train.trajectory_runners.harbor.dataset import TerminalBenchTaskDataset
from tests.cpu.util import example_dummy_config


class _InferenceClient:
    def __init__(self) -> None:
        self.stopped = False

    async def get_stats(self, *, read_mode) -> InferenceStatsSnapshot:
        del read_mode
        return InferenceStatsSnapshot(engines=())

    def shutdown_http_endpoint(self) -> None:
        pass

    async def teardown(self) -> None:
        self.stopped = True


class _StartupRequiredRunner(TrajectoryRunner):
    def __init__(self) -> None:
        self.started = False
        self.stopped = False

    async def startup(self) -> None:
        self.started = True

    async def _run(self, input_batch, disable_tqdm: bool = False) -> TrajectoryBatch:
        del disable_tqdm
        if not self.started:
            raise RuntimeError("trajectory runner was not started")
        size = len(input_batch["prompts"])
        return {
            "prompt_token_ids": [[1]] * size,
            "response_ids": [[2]] * size,
            "rewards": [1.0] * size,
            "loss_masks": [[1]] * size,
            "stop_reasons": ["stop"] * size,
            "rollout_logprobs": None,
        }

    async def shutdown(self) -> None:
        self.stopped = True


class _Tracker:
    def __init__(self) -> None:
        self.metrics = None

    def log(self, metrics, *, step: int, commit: bool) -> None:
        assert step == 0
        assert commit
        self.metrics = metrics


def test_terminal_bench_generate_reads_validation_tasks(tmp_path: Path) -> None:
    task_dir = tmp_path / "tasks" / "task-a"
    task_dir.mkdir(parents=True)
    (task_dir / "instruction.md").write_text("Do the task")

    experiment = object.__new__(TerminalBenchGenerateExp)
    experiment.cfg = OmegaConf.create(
        {
            "trainer": {"eval_interval": 1},
            "data": {"train_data": [], "val_data": [str(tmp_path / "tasks")]},
        }
    )

    assert experiment.get_train_dataset() is None
    assert list(experiment.get_eval_dataset()) == [
        {
            "prompt": str(task_dir),
            "env_class": None,
            "env_extras": {"data_source": str(task_dir)},
            "uid": "task-a",
        }
    ]


@pytest.mark.asyncio
async def test_terminal_bench_generate_starts_runner_before_rollout(tmp_path: Path) -> None:
    task_dir = tmp_path / "tasks" / "task-a"
    task_dir.mkdir(parents=True)
    (task_dir / "instruction.md").write_text("Do the task")

    cfg = example_dummy_config()
    cfg.generator.backend = "vllm"
    cfg.generator.enable_http_endpoint = True
    cfg.generator.eval_n_samples_per_prompt = 1
    cfg.generator.eval_sampling_params.max_generate_length = 20
    cfg.trainer.eval_batch_size = 1
    cfg.trainer.dump_eval_results = False
    cfg.trainer.export_path = str(tmp_path)
    cfg.environment.env_class = "terminal_bench"

    inference_client = _InferenceClient()
    runner = _StartupRequiredRunner()
    tracker = _Tracker()
    experiment = object.__new__(TerminalBenchGenerateExp)
    experiment.cfg = cfg
    experiment.eval_dataset = TerminalBenchTaskDataset(data_files=[str(tmp_path / "tasks")])
    experiment.tokenizer = type("Tokenizer", (), {"decode": lambda _self, _tokens: "done"})()

    def create_inference_engine_client(*, operation: EntrypointOperation):
        assert operation is EntrypointOperation.GENERATE
        return inference_client

    experiment.create_inference_engine_client = create_inference_engine_client
    experiment.get_trajectory_runner = lambda *_args: runner
    experiment.get_tracker = lambda: tracker

    metrics = await run_evaluation_only(experiment)

    assert metrics["eval/all/avg_score"] == 1.0
    assert tracker.metrics == metrics
    assert runner.stopped
    assert inference_client.stopped
