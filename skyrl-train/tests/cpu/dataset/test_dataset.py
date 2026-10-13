import json

import pytest
from unittest.mock import patch
from datasets import Dataset
from transformers import BatchEncoding
from omegaconf import OmegaConf
from taskcompendium.models import (
    Source,
)
from skyrl_train.config.utils import get_default_config
from skyrl_train.dataset import PromptDataset
from marinskyrl.task_format import TASK_SPEC_COLUMN
from skyrl_train.entrypoints.main_base import BasePPOExp
from tests.cpu.task_specs import text_task
from tests.cpu.tiny_training.tiny_model import build_tiny_policy


def task_row(prompt: str, index: int) -> dict:
    return {
        TASK_SPEC_COLUMN: text_task(f"fixture-{index}", prompt=prompt, source="fixture")
        .model_copy(update={"source": Source(dataset="fixture", revision="1", row=str(index), importer_revision="1")})
        .model_dump_json()
    }


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
def test_eval_dataset_filtering(mock_tokenizer, tmp_path, probe_enabled):
    path = tmp_path / "validation.parquet"
    Dataset.from_list(
        [task_row(content, index) for index, content in enumerate(("short prompt", "a" * 40, "b" * 200))]
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
    assert [row["input"].lowered.task.context.events[0].content for row in rows] == ["short prompt", "a" * 40]
    assert [row["uid"] for row in rows] == ["fixture-0", "fixture-1"]
    tasks = [row["input"].lowered.task for row in rows]
    assert [task.source.row for task in tasks] == ["0", "1"]
    assert all(row["input"].metadata.metric_class == "taskcompendium" for row in rows)


def test_generated_rl_rejects_unprepared_source_rows(tmp_path):
    source = tmp_path / "raw.jsonl"
    source.write_text(json.dumps({"prompt": [{"role": "user", "content": "Raw question"}], "env_class": "gsm8k"}))
    experiment = object.__new__(BasePPOExp)
    experiment.cfg = get_default_config()
    experiment.tokenizer = _StubTokenizer()

    with pytest.raises(ValueError, match="prepared TaskSpec"):
        experiment.task_dataset([str(source)])


@pytest.mark.parametrize("override_source", ["train-source", "val-source", "misspelled-source"])
def test_driver_matches_source_overrides_against_train_and_eval_tasks(tmp_path, override_source):
    train_path = tmp_path / "train.jsonl"
    eval_path = tmp_path / "eval.jsonl"
    for path, source in ((train_path, "train-source"), (eval_path, "val-source")):
        path.write_text(json.dumps({TASK_SPEC_COLUMN: text_task(source, source=source).model_dump_json()}))
    cfg = get_default_config()
    cfg.trainer.policy.model.path = str(build_tiny_policy(tmp_path / "model"))
    cfg.trainer.train_batch_size = 1
    cfg.trainer.eval_interval = 1
    cfg.trainer.placement.colocate_all = False
    cfg.data.train_data = [str(train_path)]
    cfg.data.val_data = [str(eval_path)]
    OmegaConf.update(
        cfg,
        f"environment.task_sessions.sources.{override_source}",
        {"session": {"max_turns": 3}},
        force_add=True,
    )
    if override_source == "misspelled-source":
        with pytest.raises(ValueError, match="Unknown task execution sources"):
            BasePPOExp(cfg)
        return
    experiment = BasePPOExp(cfg)
    selected = experiment.train_dataset if override_source == "train-source" else experiment.eval_dataset
    assert selected[0].lowered.session.max_turns == 3


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
    messages, extra, _uid = dataset[1]
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
    )
    rows = [ds[i] for i in range(len(ds))]
    uids = [row[2] for row in rows]

    # UID is simply the row index
    assert uids[0] == "0"
    # UIDs must be unique
    assert len(set(uids)) == len(uids)

    rows_again = [ds[i] for i in range(len(ds))]
    uids_again = [row[2] for row in rows_again]
    # When sampled the second time, UIDs should not change
    assert uids_again == uids
