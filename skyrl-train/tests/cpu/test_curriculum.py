"""Tests for adaptive curriculum sampling (skyrl_train.curriculum)."""

import numpy as np
import pytest
from datasets import Dataset
from torch.utils.data import SequentialSampler

from skyrl_train.curriculum import CurriculumConfig, CurriculumSampler, SamplingKind, dataset_bins
from skyrl_train.dataset import PromptDataset
from skyrl_train.utils.trainer_utils import build_dataloader
from tests.cpu.util import example_dummy_config


class _StubCurriculumDataset:
    """Minimal PromptDataset stand-in: a dataframe with `extra_info` plus map-style access.

    Module scope so DataLoader pickling would work if needed; bins maps
    data_source name -> (grade, row_count), rows laid out in insertion order.
    """

    def __init__(self, bins: dict[str, tuple[int, int]]):
        prompts, infos = [], []
        for name, (grade, rows) in bins.items():
            for _ in range(rows):
                prompts.append(f"prompt-{len(prompts)}")
                infos.append({"data_source": name, "grade": grade})
        self.dataframe = Dataset.from_dict({"prompt": prompts, "extra_info": infos})

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        return idx

    def collate_fn(self, batch):
        return batch


def _sampler(bins, kind, seed=0, **overrides):
    dataset = _StubCurriculumDataset(bins)
    config = CurriculumConfig(kind=SamplingKind(kind), **overrides)
    return CurriculumSampler(dataset, config, seed)


TWO_BINS = {"g0-easy": (0, 1), "g1-hard": (1, 1)}


def test_one_pass_yields_len_dataset_indices():
    sampler = _sampler({"g0-easy": (0, 3), "g1-hard": (1, 7)}, "thompson")
    indices = list(iter(sampler))
    assert len(sampler) == 10
    assert len(indices) == 10
    assert all(0 <= i < 10 for i in indices)


def test_draws_deterministic_given_seed():
    bins = {"g0-easy": (0, 5), "g1-hard": (1, 5)}
    draws_a = list(iter(_sampler(bins, "thompson", seed=7)))
    draws_b = list(iter(_sampler(bins, "thompson", seed=7)))
    draws_c = list(iter(_sampler(bins, "thompson", seed=8)))
    assert draws_a == draws_b
    assert draws_a != draws_c


def test_state_dict_roundtrip_resumes_identical_draws():
    bins = {"g0-easy": (0, 15), "g1-hard": (1, 15)}
    sampler = _sampler(bins, "thompson", seed=3)
    it = iter(sampler)
    [next(it) for _ in range(5)]
    sampler.update(["0", "0", "20", "20"], [0.0, 1.0, 1.0, 1.0], 2)
    [next(it) for _ in range(2)]

    snapshot = sampler.state_dict()
    expected = [next(it) for _ in range(10)]

    restored = _sampler(bins, "thompson", seed=99)
    restored.load_state_dict(snapshot)
    restored_it = iter(restored)
    assert [next(restored_it) for _ in range(10)] == expected
    np.testing.assert_allclose(restored.informative, sampler.informative)
    np.testing.assert_allclose(restored.total, sampler.total)


def test_update_informative_accounting_with_decay():
    sampler = _sampler(TWO_BINS, "naive", decay=0.5)
    # Row 0 is the g0 bin, row 1 the g1 bin. One informative group each for bin 0, none for bin 1.
    sampler.update(["0", "0", "1", "1"], [0.0, 1.0, 1.0, 1.0], 2)
    np.testing.assert_allclose(sampler.informative, [1.0, 0.0])
    np.testing.assert_allclose(sampler.total, [1.0, 1.0])

    sampler.update(["0", "0"], [1.0, 1.0], 2)
    np.testing.assert_allclose(sampler.informative, [0.5, 0.0])
    np.testing.assert_allclose(sampler.total, [1.5, 0.5])
    metrics = sampler.metrics()
    assert metrics["curriculum/g0-easy/informative_frac"] == pytest.approx(0.5 / 1.5)
    assert metrics["curriculum/g0-easy/groups"] == 1.0
    assert metrics["curriculum/g1-hard/groups"] == 0.0


def test_update_rejects_partial_groups():
    sampler = _sampler(TWO_BINS, "naive")
    with pytest.raises(ValueError, match="not a multiple"):
        sampler.update(["0", "0", "0"], [0.0, 1.0, 0.0], 2)


def test_naive_weights_match_row_counts():
    sampler = _sampler({"g0-easy": (0, 2), "g1-hard": (1, 18)}, "naive")
    np.testing.assert_allclose(sampler.weights, [0.1, 0.9])


def test_grade_uniform_equalizes_grades():
    sampler = _sampler({"g0-a": (0, 2), "g0-b": (0, 2), "g1-c": (1, 12)}, "grade-uniform")
    np.testing.assert_allclose(sampler.weights, [0.25, 0.25, 0.5])


def test_thompson_concentrates_on_informative_bins():
    sampler = _sampler(TWO_BINS, "thompson", seed=0)
    for _ in range(30):
        # Bin 0 (row 0) yields informative groups; bin 1 (row 1) is always all-pass.
        sampler.update(["0", "0", "1", "1"], [0.0, 1.0, 1.0, 1.0], 2)
    assert sampler.weights[0] > 0.7
    assert sampler.weights[1] >= 0.05 / 2  # epsilon floor


def test_grade_prior_prefers_easy_then_follows_evidence():
    bins = {"g0-easy": (0, 1), "g2-hard": (2, 1)}
    sampler = _sampler(bins, "grade-prior", seed=1)
    weight_sum = np.zeros(2)
    for _ in range(300):
        sampler.update([], [], 1)  # no evidence; redraws weights from the grade-seeded prior
        weight_sum += sampler.weights
    assert weight_sum[0] > weight_sum[1]

    for _ in range(30):
        # Evidence flips: the hard bin (row 1) is informative, the easy bin saturated.
        sampler.update(["0", "0", "1", "1"], [1.0, 1.0, 0.0, 1.0], 2)
    assert sampler.weights[1] > sampler.weights[0]


def test_grade_adaptive_advances_after_low_signal_window():
    sampler = _sampler(
        TWO_BINS, "grade-adaptive", adaptive_window=3, adaptive_min_informative=0.1, adaptive_exploration=0.2
    )
    np.testing.assert_allclose(sampler.weights, [0.8, 0.2])
    for _ in range(3):
        sampler.update(["0", "0"], [1.0, 1.0], 2)  # level-0 bin saturated: all-pass groups
    assert sampler.metrics()["curriculum/level"] == 1.0
    np.testing.assert_allclose(sampler.weights, [0.2, 0.8])

    # At the max grade the level stays put even under sustained low signal.
    for _ in range(5):
        sampler.update(["1", "1"], [0.0, 0.0], 2)
    assert sampler.metrics()["curriculum/level"] == 1.0


def test_grade_adaptive_informative_signal_resets_window():
    sampler = _sampler(TWO_BINS, "grade-adaptive", adaptive_window=3, adaptive_min_informative=0.1)
    for _ in range(2):
        sampler.update(["0", "0"], [1.0, 1.0], 2)
    sampler.update(["0", "0"], [0.0, 1.0], 2)  # informative group resets the counter
    for _ in range(2):
        sampler.update(["0", "0"], [1.0, 1.0], 2)
    assert sampler.metrics()["curriculum/level"] == 0.0


def test_dataset_bins_requires_consistent_metadata():
    missing_column = _StubCurriculumDataset({"bin": (0, 1)})
    missing_column.dataframe = Dataset.from_dict({"prompt": ["p"]})
    with pytest.raises(ValueError, match="extra_info"):
        dataset_bins(missing_column)

    inconsistent = _StubCurriculumDataset({"bin": (0, 1)})
    inconsistent.dataframe = Dataset.from_dict(
        {"extra_info": [{"data_source": "bin", "grade": 0}, {"data_source": "bin", "grade": 1}]}
    )
    with pytest.raises(ValueError, match="inconsistent grades"):
        dataset_bins(inconsistent)


class _StubTokenizer:
    """Picklable tokenizer stub; length filtering just measures the raw prompt."""

    def apply_chat_template(self, messages, add_generation_prompt):
        return messages


def _parquet_prompt_dataset(tmp_path):
    """Two bins: g0-easy with 2 rows (rows 0-1), g1-hard with 18 rows (rows 2-19)."""
    rows = [("g0-easy", 0)] * 2 + [("g1-hard", 1)] * 18
    data = {
        "prompt": [f"prompt {i}" for i in range(len(rows))],
        "env_class": ["gsm8k"] * len(rows),
        "extra_info": [{"data_source": name, "grade": grade} for name, grade in rows],
    }
    parquet_path = str(tmp_path / "train.parquet")
    Dataset.from_dict(data).to_parquet(parquet_path)
    return PromptDataset(
        datasets=[parquet_path], tokenizer=_StubTokenizer(), max_prompt_length=100, num_workers=1
    )


def _curriculum_config(kind):
    config = example_dummy_config()
    config.data.sampling.kind = kind
    config.trainer.train_batch_size = 5
    return config


def test_build_dataloader_grade_uniform_draw_frequencies(tmp_path):
    dataset = _parquet_prompt_dataset(tmp_path)
    config = _curriculum_config("grade-uniform")
    dataloader = build_dataloader(config, dataset, is_train=True)

    assert isinstance(dataloader.sampler, CurriculumSampler)
    assert dataloader.num_workers == 0  # draws must not be prefetched ahead of weight updates

    easy_draws = total_draws = 0
    for _ in range(30):
        for batch in dataloader:
            for item in batch:
                easy_draws += int(item["uid"]) < 2
                total_draws += 1
    # grade-uniform puts half the mass on the 2-row g0 bin vs its 10% row share.
    assert total_draws == 30 * 20
    assert 0.4 < easy_draws / total_draws < 0.6


def test_build_dataloader_eval_loader_ignores_curriculum(tmp_path):
    dataset = _parquet_prompt_dataset(tmp_path)
    dataloader = build_dataloader(_curriculum_config("thompson"), dataset, is_train=False)
    assert isinstance(dataloader.sampler, SequentialSampler)


def test_build_dataloader_rejects_unsupported_modes(tmp_path):
    dataset = _parquet_prompt_dataset(tmp_path)
    config = _curriculum_config("thompson")
    with pytest.raises(ValueError, match="fully async"):
        build_dataloader(config, dataset, is_train=True, is_fully_async=True)
    config.trainer.step_wise_training = True
    with pytest.raises(ValueError, match="step_wise_training"):
        build_dataloader(config, dataset, is_train=True)


def test_stateful_dataloader_checkpoint_resumes_draws(tmp_path):
    dataset = _parquet_prompt_dataset(tmp_path)
    config = _curriculum_config("thompson")

    dataloader = build_dataloader(config, dataset, is_train=True)
    it = iter(dataloader)
    [next(it) for _ in range(2)]
    dataloader.sampler.update(["0", "0", "5", "5"], [0.0, 1.0, 1.0, 1.0], 2)
    snapshot = dataloader.state_dict()
    remaining = list(it)

    resumed = build_dataloader(config, dataset, is_train=True)
    resumed.load_state_dict(snapshot)
    assert list(iter(resumed)) == remaining
    np.testing.assert_allclose(resumed.sampler.total, dataloader.sampler.total)
