from omegaconf import OmegaConf

from skyrl_train.entrypoints.terminal_bench_generate import TerminalBenchGenerateExp
from skyrl_train.trajectory_runners.types import BatchMetadata, TrajectoryRequestBatch


class RecordingTrajectoryRunner:
    def __init__(self) -> None:
        self.request: TrajectoryRequestBatch | None = None
        self.events: list[str] = []

    async def startup(self) -> None:
        self.events.append("startup")

    async def run(self, request: TrajectoryRequestBatch) -> None:
        self.events.append("run")
        self.request = request

    async def shutdown(self) -> None:
        self.events.append("shutdown")


class RecordingInferenceClient:
    def __init__(self, events: list[str]) -> None:
        self.events = events

    async def wake_up(self) -> None:
        self.events.append("wake_up")


def test_terminal_bench_generate_builds_complete_evaluation_request():
    runner = RecordingTrajectoryRunner()
    experiment = object.__new__(TerminalBenchGenerateExp)
    experiment.cfg = OmegaConf.create(
        {
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
        }
    )
    experiment.train_dataset = [
        {"uid": "task-a", "prompt": "task-a-path", "env_class": None, "env_extras": {"split": "train"}},
        {"uid": "task-b", "prompt": "task-b-path", "env_class": None, "env_extras": {"split": "train"}},
    ]
    experiment.tokenizer = object()
    inference_client = RecordingInferenceClient(runner.events)
    experiment.create_inference_engine_client = lambda: inference_client
    experiment.get_trajectory_runner = lambda cfg, tokenizer, client: runner

    experiment.run()

    assert runner.events == ["wake_up", "startup", "run", "shutdown"]
    assert runner.request is not None
    assert runner.request["prompts"] == ["task-a-path"] * 8 + ["task-b-path"] * 8
    trajectory_ids = runner.request["trajectory_ids"]
    assert trajectory_ids is not None
    assert [trajectory_id.to_string() for trajectory_id in trajectory_ids] == [
        f"task-{task}_{repetition_id}" for task in ("a", "b") for repetition_id in range(8)
    ]
    assert runner.request["env_classes"] == ["terminal_bench"] * 16
    assert runner.request["batch_metadata"] == BatchMetadata(global_step=0, training_phase="eval")
