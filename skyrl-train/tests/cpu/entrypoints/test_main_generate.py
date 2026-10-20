from contextlib import nullcontext

import pytest
from omegaconf import OmegaConf
from taskcompendium.models import ScriptGrader, TextMessage, TokenizedInput
from tokenizers import Tokenizer, models
from transformers import PreTrainedTokenizerFast

from skyrl_train.config.rollout_validation import EntrypointOperation
from skyrl_train.config.utils import get_default_config
from skyrl_train.dataset.tasks import write_tasks
from skyrl_train.entrypoints import main_generate
from skyrl_train.entrypoints.main_generate import EvalOnlyEntrypoint
from tests.cpu.task_specs import NATIVE_ENVIRONMENT, lowered_task, text_task


def test_eval_only_task_dataset_preserves_private_grader_and_native_tokens(tmp_path):
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=Tokenizer(models.WordLevel({"[UNK]": 0}, unk_token="[UNK]")),
        unk_token="[UNK]",
        chat_template="{% for message in messages %}{{ message.content }} {% endfor %}",
    )
    grader = ScriptGrader(argv=("sh", "/tests/private-grader.sh"), environment=NATIVE_ENVIRONMENT)
    shell_task = text_task("shell-private").model_copy(update={"grader": grader})
    native_task = text_task("native-private").model_copy(
        update={
            "grader": grader,
            "context": TokenizedInput(
                events=(TextMessage(role="user", content="Display only " * 100),),
                token_ids=(1, 2, 3, 4),
                tokenizer_model="fixture/tokenizer",
                tokenizer_revision="a" * 40,
            ),
        }
    )
    expected = [
        lowered_task(shell_task, verifier_backend="daytona"),
        lowered_task(native_task, "token_prefix", max_turns=1, verifier_backend="daytona"),
    ]
    path = tmp_path / "validation.parquet"
    write_tasks(path, expected)
    experiment = object.__new__(EvalOnlyEntrypoint)
    experiment.cfg = get_default_config()
    experiment.cfg.trainer.evaluation_runner = "task"
    experiment.cfg.trainer.max_prompt_length = 8
    experiment.cfg.data.val_data = [str(path)]
    experiment.tokenizer = tokenizer

    dataset = experiment.get_eval_dataset()
    selected = dataset.collate_fn(list(dataset))

    assert [row["input"].spec for row in selected] == expected
    assert [row["uid"] for row in selected] == ["0", "1"]


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
