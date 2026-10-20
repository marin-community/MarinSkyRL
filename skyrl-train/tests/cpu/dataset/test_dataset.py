import json

import pytest
from unittest.mock import patch
from datasets import Dataset
from transformers import BatchEncoding, PreTrainedTokenizerFast
from tokenizers import Tokenizer, models, pre_tokenizers
from taskcompendium.models import AnswerType, FinalAction, FunctionDefinition, NoGrader, Source
from omegaconf import OmegaConf
from skyrl_train.config.utils import get_default_config
from skyrl_train.dataset import PromptDataset
from skyrl_train.dataset.harbor_import import HARBOR_TRIAL_SESSION, harbor_trial_task
from skyrl_train.dataset.tasks import SourceTaskDataset, TaskDataset, source_row_task, write_tasks
from skyrl_train.entrypoints.main_base import BasePPOExp
from skyrl_train.task_inputs import task_messages
from skyrl_gym.rollout_specs import rollout_spec
from tests.cpu.task_specs import lowered_task


@pytest.mark.parametrize("num_workers", [1, 2])
def test_native_task_filter_counts_tool_definitions(tmp_path, num_workers):
    backend = Tokenizer(models.WordLevel({"[UNK]": 0}, unk_token="[UNK]"))
    backend.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend,
        unk_token="[UNK]",
        chat_template="{% for m in messages %}{{ m.content }} {% endfor %}"
        "{% for tool in tools or [] %}{{ tool.function.description }} {% endfor %}",
    )
    configs = OmegaConf.to_container(get_default_config().environment.task_sessions, resolve=True)
    base = source_row_task(
        {
            "prompt": [{"role": "user", "content": "Choose a tool"}],
            "env_class": "gsm8k",
            "reward_spec": {"ground_truth": "1"},
        },
        0,
        source_name="tool-budget",
        environment_configs=configs,
    )
    tasks = []
    for name, description in (("short", "Find a value"), ("long", "word " * 100)):
        task = base.task.model_copy(
            update={
                "id": name,
                "answer_type": AnswerType.NATIVE_ACTION,
                "answer_format": FinalAction(require_call=True, max_calls=1),
                "grader": NoGrader(reason="Loader-only test"),
                "final_tools": (
                    FunctionDefinition(name="lookup", description=description, parameters={"type": "object"}),
                ),
            }
        )
        tasks.append(rollout_spec(task, {**base.limits.model_dump(), "task_session": "shellbox", "command_timeout": 1}))
    path = tmp_path / "tools.parquet"
    write_tasks(path, tasks)
    dataset = TaskDataset([str(path)], tokenizer, 30, num_workers=num_workers)
    admitted = [row["input"].spec.task.id for row in dataset.collate_fn(list(dataset))]
    assert admitted == ["short"]


class _StubTokenizer:
    """Picklable tokenizer stub for DataLoader worker tests.

    MagicMock cannot survive pickling across multiprocessing boundaries
    (newer dill/datasets versions enforce this), so tests that spin up
    DataLoader workers need a real picklable object.
    """

    def apply_chat_template(self, messages, add_generation_prompt):
        return messages if isinstance(messages, str) else "".join(message["content"] for message in messages)


@pytest.fixture
def mock_tokenizer():
    return _StubTokenizer()


@pytest.fixture
def sample_dataset():
    # 3 samples: one too long, two valid
    data = {
        "prompt": [
            "short prompt",  # length 13
            "a" * 120,  # length 120
            "b" * 200,  # length 200 (to be filtered out if max len < 200)
        ],
        "answer": ["a1", "a2", "a3"],
    }
    return Dataset.from_dict(data)


@pytest.mark.parametrize("probe_enabled", [False, True], ids=["evaluation", "probe_without_evaluation"])
def test_eval_dataset_filtering(mock_tokenizer, sample_dataset, tmp_path, probe_enabled):
    path = tmp_path / "validation.parquet"
    sample_dataset.map(
        lambda row: {
            "prompt": [{"role": "user", "content": row["prompt"]}],
            "env_class": "gsm8k",
            "reward_spec": {"ground_truth": row["answer"]},
        }
    ).to_parquet(path)
    experiment = object.__new__(BasePPOExp)
    experiment.cfg = get_default_config()
    experiment.cfg.data.val_data = [str(path)]
    experiment.cfg.trainer.max_prompt_length = 150
    experiment.cfg.trainer.eval_interval = -1 if probe_enabled else 1
    experiment.cfg.trainer.mismatch_probe.enabled = probe_enabled
    experiment.tokenizer = mock_tokenizer

    dataset = experiment.get_eval_dataset()

    assert dataset is not None
    assert len(dataset) == 2
    rows = dataset.collate_fn([dataset[0], dataset[1]])
    assert [task_messages(row["input"].spec) for row in rows] == [
        [{"role": "user", "content": "short prompt"}],
        [{"role": "user", "content": "a" * 120}],
    ]
    assert [row["uid"] for row in rows] == ["0", "1"]
    tasks = [row["input"].spec.task for row in rows]
    assert [task.source.row for task in tasks] == ["0", "1"]
    assert all(row["input"].metadata.metric_class == "gsm8k" for row in rows)


@pytest.mark.parametrize("num_workers", [1, 2])
@pytest.mark.parametrize(("nested_source", "expected_source"), [(None, "fixture"), ("nested-source", "nested-source")])
def test_source_tasks_preserve_row_indices_and_source_labels_after_filtering(
    tmp_path, num_workers, nested_source, expected_source
):
    rows = [
        {
            "prompt": [{"role": "user", "content": content}],
            "env_class": "gsm8k",
            "reward_spec": {"ground_truth": "PRIVATE_ANSWER"},
            "teacher_route": "math",
            "data_source": "fixture",
            "extra_info": {"grade": index, "data_source": nested_source},
        }
        for index, content in enumerate(("First question", "x" * 200, "Last question"))
    ]
    source = tmp_path / "source.jsonl"
    source.write_text("\n".join(json.dumps(row) for row in rows))

    dataset = SourceTaskDataset(
        [str(source)],
        _StubTokenizer(),
        100,
        environment_configs=OmegaConf.to_container(get_default_config().environment.task_sessions, resolve=True),
        num_workers=num_workers,
    )
    prepared = dataset.collate_fn([dataset[index] for index in range(len(dataset))])
    assert [task_messages(row["input"].spec) for row in prepared] == [rows[0]["prompt"], rows[2]["prompt"]]
    tasks = [row["input"].spec.task for row in prepared]
    assert [task.source.row for task in tasks] == ["0", "2"]
    assert [row["input"].metadata.data_source for row in prepared] == [expected_source, expected_source]
    assert dataset.dataframe.cache_files == []


@patch("datasets.load_dataset")
def test_prompt_dataset_filtering_counts_batch_encoding_input_ids(mock_load_dataset, sample_dataset):
    mock_load_dataset.return_value = {"train": sample_dataset}

    class BatchTokenizer:
        def apply_chat_template(self, messages, add_generation_prompt):
            del add_generation_prompt
            return BatchEncoding({"input_ids": list(messages), "attention_mask": [1] * len(messages)})

    dataset = PromptDataset(
        datasets=["dummy.parquet"],
        tokenizer=BatchTokenizer(),
        max_prompt_length=150,
        num_workers=1,
    )

    assert len(dataset) == 2


@pytest.mark.parametrize("num_workers", [1, 2])
@pytest.mark.parametrize("reverse", [False, True])
def test_native_tasks_with_different_extras_preserve_each_row(tmp_path, num_workers, reverse):
    rows = [
        {
            "prompt": [
                {"role": "user", "content": "First question"},
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {"id": "call-1", "type": "function", "function": {"name": "lookup", "arguments": "{}"}}
                    ],
                },
                {"role": "tool", "content": "Result", "tool_call_id": "call-1"},
            ],
            "env_class": "gsm8k",
            "reward_spec": {"ground_truth": "1"},
            "prompt_token_ids": [17, 23],
        },
        {
            "prompt": [{"role": "user", "content": "Second question"}],
            "env_class": "gsm8k",
            "reward_spec": {"ground_truth": "2"},
            "teacher_route": "math",
        },
    ]
    if reverse:
        rows.reverse()
    configs = OmegaConf.to_container(get_default_config().environment.task_sessions, resolve=True)
    tasks = [
        source_row_task(row, index, source_name="mixed", environment_configs=configs) for index, row in enumerate(rows)
    ]
    path = tmp_path / "mixed.parquet"
    write_tasks(path, tasks)

    dataset = TaskDataset([str(path)], _StubTokenizer(), 100, num_workers=num_workers)

    inputs = dataset.collate_fn([dataset[index] for index in range(len(dataset))])
    for index, source in enumerate(rows):
        task = inputs[index]["input"]
        assert task.metadata.metric_class == source["env_class"]
        assert inputs[index]["uid"] == str(index)
        assert task.spec == tasks[index]
        assert task.metadata.teacher_route == source.get("teacher_route")


@pytest.mark.parametrize("reverse", [False, True])
def test_mixed_original_harbor_and_source_tasks_collate_without_source_metadata_assumption(tmp_path, reverse):
    package = tmp_path / "harbor-task"
    package.mkdir()
    (package / "instruction.md").write_text("Repair the program.")
    (package / "task.toml").write_text('[verifier]\nenvironment_mode = "shared"\n')
    harbor = lowered_task(
        harbor_trial_task(package, source=Source(dataset="harbor", revision="1", row="0", importer_revision="1")),
        HARBOR_TRIAL_SESSION,
    )
    configs = OmegaConf.to_container(get_default_config().environment.task_sessions, resolve=True)
    source = source_row_task(
        {
            "prompt": [{"role": "user", "content": "What is one plus one?"}],
            "env_class": "gsm8k",
            "reward_spec": {"ground_truth": "2"},
            "teacher_route": "math",
        },
        0,
        source_name="math",
        environment_configs=configs,
    )
    tasks = [harbor, source]
    if reverse:
        tasks.reverse()
    path = tmp_path / "mixed-harbor.parquet"
    write_tasks(path, tasks)
    dataset = TaskDataset([str(path)], _StubTokenizer(), 100, num_workers=1)

    inputs = dataset.collate_fn(list(dataset))

    assert [row["input"].spec for row in inputs] == tasks
    by_metric_class = {row["input"].metadata.metric_class: row["input"] for row in inputs}
    assert by_metric_class["gsm8k"].metadata.teacher_route == "math"
    assert by_metric_class[HARBOR_TRIAL_SESSION].metadata.teacher_route is None
    assert by_metric_class[HARBOR_TRIAL_SESSION].group_grader is None


@pytest.mark.parametrize(
    ("dataset_spec", "available_split", "error"),
    [
        pytest.param("my_hf_dataset", "train", None, id="name-defaults-to-train"),
        pytest.param("my_hf_dataset:validation", "validation", None, id="explicit-split"),
        pytest.param("my_hf_dataset", "validation", r"Split `train` not found", id="missing-default-train"),
        pytest.param("my_hf_dataset:bogus", "train", r"Split `bogus` not found", id="missing-explicit-split"),
    ],
)
@patch("datasets.load_dataset")
def test_prompt_dataset_hf_split_selection(
    mock_load_dataset, mock_tokenizer, sample_dataset, dataset_spec, available_split, error
):
    mock_load_dataset.return_value = {available_split: sample_dataset}

    def build():
        return PromptDataset(datasets=[dataset_spec], tokenizer=mock_tokenizer, max_prompt_length=150, num_workers=1)

    if error is not None:
        with pytest.raises(ValueError, match=error):
            build()
        return
    dataset = build()
    assert len(dataset) == 2
    messages, _env, extra, _uid = dataset[1]
    assert messages == "a" * 120
    assert extra == {"answer": "a2"}


@patch("datasets.load_dataset")
def test_prompt_dataset_uids(mock_load_dataset, sample_dataset, mock_tokenizer):
    # When only a dataset name is provided, we default to the 'train' split
    mock_load_dataset.return_value = {"train": sample_dataset}

    ds = PromptDataset(
        datasets=["my_hf_dataset"],
        tokenizer=mock_tokenizer,
        max_prompt_length=1024,
        num_workers=1,
        prompt_key="prompt",
        env_class_key="env_class",
    )
    rows = [ds[i] for i in range(len(ds))]
    uids = [row[3] for row in rows]

    # UID is simply the row index
    assert uids[0] == "0"
    # UIDs must be unique
    assert len(set(uids)) == len(uids)

    rows_again = [ds[i] for i in range(len(ds))]
    uids_again = [row[3] for row in rows_again]
    # When sampled the second time, UIDs should not change
    assert uids_again == uids
