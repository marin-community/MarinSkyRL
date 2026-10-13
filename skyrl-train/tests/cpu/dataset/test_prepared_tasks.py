import json
import subprocess
import sys

import pytest
from datasets import Dataset
from rolloutengine.spec import TaskRuntimeSpec
from taskcompendium.models import TaskSpec

from marinskyrl.task_format import TASK_SPEC_COLUMN
from marinskyrl.task_preparation import source_row_task
from skyrl_train.config.utils import get_default_config
from skyrl_train.dataset.prepared_tasks import TaskSpecDataset
from skyrl_train.rollouts.task_lowering import source_execution_settings
from skyrl_train.rollouts.task_projections import WholeTaskProjection
from skyrl_train.rollouts.task_worker import TaskRolloutWorker
from skyrl_train.trajectory_runners.trajectory_processing import prepare_trajectory_request
from tests.cpu.rollouts.engine_fakes import ConversationClient, Tokenizer
from tests.cpu.task_specs import machine_runtime, session_spec, task_dataset, text_task


class PromptTokenizer:
    def apply_chat_template(self, messages, add_generation_prompt):
        return "".join(message["content"] for message in messages)


def task_row(prompt: str, index: int) -> dict:
    return {TASK_SPEC_COLUMN: text_task(f"fixture-{index}", prompt=prompt, source="fixture").model_dump_json()}


def test_prepared_hf_selector_reads_the_requested_revision_and_subdirectory(monkeypatch):
    revision = "a" * 40

    def read_hub_dataset(path, *, revision=None, data_dir=None, keep_in_memory=False):
        prompt = (
            "Pinned question" if (path, revision, data_dir) == ("org/tasks", "a" * 40, "prepared") else "Wrong source"
        )
        return {"train": Dataset.from_list([task_row(prompt, 0)])}

    monkeypatch.setattr("datasets.load_dataset", read_hub_dataset)
    dataset = task_dataset([f"org/tasks@{revision}::prepared"], PromptTokenizer(), num_workers=1)
    assert dataset[0].lowered.task.context.events[0].content == "Pinned question"


@pytest.mark.parametrize("source_max_turns", [None, 3])
def test_task_specs_preserve_definitions_and_use_run_session_settings(tmp_path, source_max_turns):
    source = tmp_path / "tasks.jsonl"
    input_row = task_row("Prepared task", 0)
    source.write_text(json.dumps(input_row))
    settings = {
        "runtime": TaskRuntimeSpec(
            task_machine=machine_runtime("docker"), verifier_machine=machine_runtime("docker")
        ).model_dump(),
        "session": session_spec(max_turns=2).model_dump(),
        "sources": {} if source_max_turns is None else {"fixture": {"session": {"max_turns": source_max_turns}}},
    }
    execution, source_executions = source_execution_settings(settings)
    dataset = TaskSpecDataset(
        [str(source)],
        PromptTokenizer(),
        100,
        execution=execution,
        source_executions=source_executions,
        num_workers=1,
    )
    lowered = dataset[0].lowered

    assert lowered.task == TaskSpec.model_validate_json(input_row[TASK_SPEC_COLUMN])
    assert lowered.session.max_turns == (source_max_turns or 2)
    assert lowered.runtime.task_machine is None
    assert lowered.runtime.verifier_machine is None
    assert dataset.dataframe.cache_files == []


def test_prepared_loader_rejects_raw_source_rows(tmp_path):
    source = tmp_path / "raw.jsonl"
    source.write_text(json.dumps({"prompt": [{"role": "user", "content": "Raw question"}], "env_class": "gsm8k"}))
    with pytest.raises(ValueError, match="prepared TaskSpec"):
        task_dataset([str(source)], PromptTokenizer())


@pytest.mark.parametrize("num_workers", [1, 2])
def test_prepared_tasks_preserve_global_row_indices_after_prompt_filtering(tmp_path, num_workers):
    rows = [
        {
            "prompt": [{"role": "user", "content": content}],
            "env_class": "gsm8k",
            "reward_spec": {"ground_truth": "PRIVATE_ANSWER"},
            "data_source": "fixture",
        }
        for content in ("First question", "x" * 200, "Last question")
    ]
    source = tmp_path / "source.jsonl"
    source.write_text(
        "\n".join(
            json.dumps(
                {TASK_SPEC_COLUMN: source_row_task(row, index, source_name="fixture", config={}).model_dump_json()}
            )
            for index, row in enumerate(rows)
        )
    )
    dataset = task_dataset(
        [str(source)], PromptTokenizer(), 100, session=session_spec("gsm8k"), num_workers=num_workers
    )
    prepared = dataset.collate_fn([dataset[index] for index in range(len(dataset))])
    tasks = [row["input"].lowered.task for row in prepared]
    assert [task.context.events[0].content for task in tasks] == ["First question", "Last question"]
    assert [task.source.row for task in tasks] == ["0", "2"]
    assert dataset.dataframe.cache_files == []


@pytest.mark.asyncio
async def test_preparation_cli_preserves_private_grading_and_labels_after_source_removal(tmp_path):
    source = tmp_path / "source.jsonl"
    source.write_text(
        json.dumps(
            {
                "prompt": [{"role": "user", "content": "What is six plus six?"}],
                "reward_spec": {"ground_truth": "12"},
                "data_source": "arithmetic",
                "teacher_route": "math",
                "metric_tags": ["fixture/math"],
                "extra_info": {"data_source": "easy", "grade": 3},
            }
        )
    )
    prepared_path = tmp_path / "tasks.parquet"
    subprocess.run(
        [
            sys.executable,
            "-m",
            "marinskyrl.task_preparation",
            "--input",
            str(source),
            "--output",
            str(prepared_path),
            "--source",
            "fixture",
        ],
        check=True,
        timeout=60,
    )
    source.unlink()
    dataset = task_dataset([str(prepared_path)], Tokenizer(), session=session_spec("gsm8k"))
    prompts = dataset.collate_fn([dataset[0]])
    request, _ = prepare_trajectory_request(prompts, 1, {"logprobs": True}, "train", 1)
    config = get_default_config().generator
    client = ConversationClient(["#### 12"])
    worker = TaskRolloutWorker(config, WholeTaskProjection(config, Tokenizer()), client, {}, shutdown_timeout=30)
    try:
        batch = await worker.run(request)
    finally:
        await worker.shutdown()

    assert batch["unshaped_rewards"] == [1.0]
    assert batch["teacher_route_keys"] == ["math"]
    assert batch["data_sources"] == ["arithmetic"]
    assert batch["loss_masks"] == [[1, 1]]
    assert dataset[0].metadata.metric_tags == ("fixture/math",)
    assert dataset.dataframe[0]["extra_info"] == {"data_source": "easy", "grade": 3}
    assert "ground_truth" not in json.dumps(client.requests[0]["prompts"])


def test_preparation_cli_rejects_different_sessions_with_the_same_source_before_output(tmp_path):
    source = tmp_path / "mixed.jsonl"
    source.write_text(
        "\n".join(
            json.dumps(
                {
                    "prompt": [{"role": "user", "content": "Question"}],
                    "env_class": session_name,
                    "data_source": "same-source",
                    "reward_spec": {"ground_truth": "12"},
                }
            )
            for session_name in ("gsm8k", "aime")
        )
    )
    output = tmp_path / "tasks.parquet"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "marinskyrl.task_preparation",
            "--input",
            str(source),
            "--output",
            str(output),
            "--source",
            "fixture",
        ],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode != 0, result.stderr
    assert not output.exists()
