import json

from omegaconf import OmegaConf
from taskcompendium.models import (
    AnswerType,
    ConversationInput,
    EnvironmentRequirements,
    NoGrader,
    PlainText,
    Source,
    TaskSpec,
    TextMessage,
)

from skyrl_train.config.rollout_validation import EntrypointOperation
from skyrl_train.config.utils import get_default_config
from marinskyrl.task_format import TASK_SPEC_COLUMN
from skyrl_train.dataset.prepared_tasks import TaskSpecDataset
from skyrl_train.entrypoints.task_generate import TaskGenerateExp
from skyrl_train.rollouts.task_lowering import source_execution_settings
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


def test_task_generate_builds_complete_evaluation_request(tmp_path):
    runner = RecordingTrajectoryRunner()
    experiment = object.__new__(TaskGenerateExp)
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
            "environment": {"env_class": "taskcompendium"},
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
    source = tmp_path / "tasks.jsonl"
    source.write_text(
        "\n".join(
            json.dumps(
                {
                    TASK_SPEC_COLUMN: TaskSpec(
                        id=name,
                        context=ConversationInput(events=(TextMessage(role="user", content=name),)),
                        environment_requirements=EnvironmentRequirements(),
                        answer_type=AnswerType.TEXT,
                        answer_format=PlainText(),
                        grader=NoGrader(reason="Generation fixture"),
                        source=Source(dataset="fixture", revision="1", row=name, importer_revision="1"),
                    ).model_dump_json()
                }
            )
            for name in ("task-a", "task-b")
        )
    )

    class Tokenizer:
        def apply_chat_template(self, messages, add_generation_prompt):
            return [1, 2]

    experiment.tokenizer = Tokenizer()
    execution, source_executions = source_execution_settings(
        OmegaConf.to_container(get_default_config().environment.task_sessions, resolve=True)
    )
    experiment.train_dataset = TaskSpecDataset(
        [str(source)],
        experiment.tokenizer,
        100,
        execution=execution,
        source_executions=source_executions,
        num_workers=1,
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
    assert [prompt[0]["content"] for prompt in runner.request.public_prompts] == ["task-a"] * 8 + ["task-b"] * 8
    trajectory_ids = runner.request.trajectory_ids
    assert trajectory_ids is not None
    assert [trajectory_id.to_string() for trajectory_id in trajectory_ids] == [
        f"task-{task}_{repetition_id}" for task in ("a", "b") for repetition_id in range(8)
    ]
    assert [row.metadata.metric_class for row in runner.request.rows] == ["taskcompendium"] * 16
    assert [row.lowered.task.id for row in runner.request.rows] == ["task-a"] * 8 + ["task-b"] * 8
    assert runner.request.batch_metadata == BatchMetadata(global_step=0, training_phase="eval")
