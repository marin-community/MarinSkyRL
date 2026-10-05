import pytest
from unittest.mock import patch
from datasets import Dataset
from transformers import BatchEncoding
from taskcompendium.models import TaskSpec
from skyrl_train.config.utils import get_default_config
from skyrl_train.dataset import PromptDataset
from skyrl_train.entrypoints.main_base import BasePPOExp


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
    experiment.cfg.data.task_cache_dir = str(tmp_path / "tasks")
    experiment.cfg.trainer.max_prompt_length = 150
    experiment.cfg.trainer.eval_interval = -1 if probe_enabled else 1
    experiment.cfg.trainer.mismatch_probe.enabled = probe_enabled
    experiment.tokenizer = mock_tokenizer

    dataset = experiment.get_eval_dataset()

    assert dataset is not None
    assert len(dataset) == 2
    rows = dataset.collate_fn([dataset[0], dataset[1]])
    assert [row["prompt"] for row in rows] == [
        [{"role": "user", "content": "short prompt"}],
        [{"role": "user", "content": "a" * 120}],
    ]
    assert [row["uid"] for row in rows] == ["0", "1"]
    tasks = [TaskSpec.model_validate_json(row["env_extras"]["task_spec"]) for row in rows]
    assert [task.source.row for task in tasks] == ["0", "1"]
    assert all(row["env_class"] == "gsm8k" for row in rows)


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
