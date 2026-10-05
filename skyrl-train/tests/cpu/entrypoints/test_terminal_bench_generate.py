from omegaconf import OmegaConf
from taskcompendium.models import TaskSpec

from skyrl_train.config.rollout_validation import EntrypointOperation
from skyrl_train.dataset.harbor import HarborTaskDataset
from skyrl_train.entrypoints.terminal_bench_generate import TerminalBenchGenerateExp
from skyrl_train.trajectory_runners.types import BatchMetadata, TrajectoryRequestBatch


class RecordingTrajectoryRunner:
    def __init__(self) -> None:
        self.request: TrajectoryRequestBatch | None = None
        self.events: list[str] = []

    async def startup(self) -> None:
        self.events.append("startup")

    async def start_eval_session(self, *, run_name: str, eval_step: int) -> None:
        self.events.append(f"start_eval {run_name} {eval_step}")

    async def run(self, request: TrajectoryRequestBatch) -> None:
        self.events.append("run")
        self.request = request

    async def stop_eval_session(self) -> None:
        self.events.append("stop_eval")

    async def shutdown(self) -> None:
        self.events.append("shutdown")


def test_terminal_bench_generate_builds_complete_evaluation_request(tmp_path):
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
            # run() configures progress and enters the trainer telemetry lifecycle before _run().
            "trainer": {
                "run_name": "generate",
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
    sources = tmp_path / "sources"
    for name in ("task-a", "task-b"):
        source = sources / name
        (source / "tests").mkdir(parents=True)
        (source / "instruction.md").write_text(name)
        (source / "task.toml").write_text('[environment]\ndocker_image = "fixture"\n')
        (source / "tests/test.sh").write_text("echo 1 > /logs/verifier/reward.txt\n")

    class Tokenizer:
        def apply_chat_template(self, messages, add_generation_prompt):
            return [1, 2]

    experiment.tokenizer = Tokenizer()
    experiment.train_dataset = HarborTaskDataset(
        [str(sources)], experiment.tokenizer, 100, cache_dir=tmp_path / "cache", num_workers=1
    )
    inference_client = object()

    def create_inference_engine_client(*, operation: EntrypointOperation):
        assert operation is EntrypointOperation.GENERATE
        return inference_client

    experiment.create_inference_engine_client = create_inference_engine_client
    experiment.get_trajectory_runner = lambda cfg, tokenizer, client: runner

    experiment.run()

    assert runner.events == ["startup", "start_eval generate 0", "run", "stop_eval", "shutdown"]
    assert runner.request is not None
    assert [prompt[0]["content"] for prompt in runner.request["prompts"]] == ["task-a"] * 8 + ["task-b"] * 8
    trajectory_ids = runner.request["trajectory_ids"]
    assert trajectory_ids is not None
    assert [trajectory_id.to_string() for trajectory_id in trajectory_ids] == [
        f"task-{task}_{repetition_id}" for task in ("a", "b") for repetition_id in range(8)
    ]
    assert runner.request["env_classes"] == ["taskcompendium"] * 16
    assert [TaskSpec.model_validate_json(extra["task_spec"]).id for extra in runner.request["env_extras"]] == (
        ["task-a"] * 8 + ["task-b"] * 8
    )
    assert runner.request["batch_metadata"] == BatchMetadata(global_step=0, training_phase="eval")
