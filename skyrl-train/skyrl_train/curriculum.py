"""Adaptive curriculum sampling over prompt-dataset bins.

Training rows carry a bin assignment in their ``extra_info`` column: ``data_source`` names the
bin (e.g. ``"g0-gsm8k"``) and ``grade`` orders bins by difficulty (0 easiest). CurriculumSampler
draws dataset rows with replacement according to per-bin weights, updated each training step from
per-sample rollout rewards. A prompt group (one uid's rollouts) is "informative" when its rewards
are not all equal — an all-pass or all-fail group contributes no GRPO gradient signal. Alongside
the group-level informative counts, the sampler tracks sample-level solved counts (reward > 0),
whose decayed pass rate drives the directional kinds: ``learnability`` and ``grade-prior`` weight
bins by a Thompson draw on p·(1−p), which peaks at a 50% pass rate and so distinguishes bins that
are too easy from bins that are too hard.
"""

from collections import defaultdict
from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Dict, Iterator, List

import numpy as np
from omegaconf import DictConfig
from torch.utils.data import Sampler

from skyrl_train.dataset import PromptDataset

EXTRA_INFO_KEY = "extra_info"
DATA_SOURCE_KEY = "data_source"
GRADE_KEY = "grade"


class SamplingKind(StrEnum):
    NAIVE = "naive"
    THOMPSON = "thompson"
    LEARNABILITY = "learnability"
    GRADE_UNIFORM = "grade-uniform"
    GRADE_ADAPTIVE = "grade-adaptive"
    GRADE_PRIOR = "grade-prior"


@dataclass(frozen=True)
class CurriculumConfig:
    """Weight-policy parameters, mirroring the ``data.sampling`` config subtree."""

    kind: SamplingKind
    decay: float = 0.95
    epsilon: float = 0.05
    prior_alpha: float = 1.0
    prior_beta: float = 1.0
    grade_prior_strength: float = 8.0
    grade_prior_high: float = 0.85
    grade_prior_low: float = 0.05
    adaptive_exploration: float = 0.2
    adaptive_window: int = 10
    adaptive_min_informative: float = 0.1

    @classmethod
    def from_dict_config(cls, cfg: DictConfig) -> "CurriculumConfig":
        return cls(
            kind=SamplingKind(cfg.kind),
            decay=cfg.decay,
            epsilon=cfg.epsilon,
            prior_alpha=cfg.prior_alpha,
            prior_beta=cfg.prior_beta,
            grade_prior_strength=cfg.grade_prior_strength,
            grade_prior_high=cfg.grade_prior_high,
            grade_prior_low=cfg.grade_prior_low,
            adaptive_exploration=cfg.adaptive_exploration,
            adaptive_window=cfg.adaptive_window,
            adaptive_min_informative=cfg.adaptive_min_informative,
        )


@dataclass(frozen=True)
class DatasetBins:
    """Row -> bin structure extracted from a post-filter PromptDataset."""

    names: List[str]  # bin index -> data_source name, sorted by (grade, name)
    grades: np.ndarray  # bin index -> grade
    row_counts: np.ndarray  # bin index -> number of rows
    bin_rows: List[np.ndarray]  # bin index -> row indices
    row_to_bin: np.ndarray  # row index -> bin index


def dataset_bins(dataset: PromptDataset) -> DatasetBins:
    """Build per-row bin assignments from the dataset's ``extra_info`` column."""
    if EXTRA_INFO_KEY not in dataset.dataframe.column_names:
        raise ValueError(f"Curriculum sampling requires an `{EXTRA_INFO_KEY}` column in the training dataset")

    grade_by_name: Dict[str, int] = {}
    name_by_row: List[str] = []
    for row, info in enumerate(dataset.dataframe[EXTRA_INFO_KEY]):
        if not info or info.get(DATA_SOURCE_KEY) is None or info.get(GRADE_KEY) is None:
            raise ValueError(
                f"Row {row}: curriculum sampling requires `{DATA_SOURCE_KEY}` and `{GRADE_KEY}` in `{EXTRA_INFO_KEY}`"
            )
        name = info[DATA_SOURCE_KEY]
        grade = int(info[GRADE_KEY])
        if grade_by_name.setdefault(name, grade) != grade:
            raise ValueError(f"Bin `{name}` has inconsistent grades: {grade_by_name[name]} and {grade}")
        name_by_row.append(name)

    names = sorted(grade_by_name, key=lambda name: (grade_by_name[name], name))
    bin_index = {name: b for b, name in enumerate(names)}
    row_to_bin = np.array([bin_index[name] for name in name_by_row], dtype=np.int64)
    bin_rows = [np.flatnonzero(row_to_bin == b) for b in range(len(names))]
    return DatasetBins(
        names=names,
        grades=np.array([grade_by_name[name] for name in names], dtype=np.int64),
        row_counts=np.array([len(rows) for rows in bin_rows], dtype=np.int64),
        bin_rows=bin_rows,
        row_to_bin=row_to_bin,
    )


class _CurriculumSamplerIterator(Iterator[int]):
    """One epoch pass over the sampler; Stateful for torchdata's snapshot protocol.

    Only the within-pass position is stored here — the draw stream itself is owned by the
    sampler's persistent RNG, whose state torchdata saves separately via the sampler-level
    ``state_dict``, so restore does not replay draws.
    """

    def __init__(self, sampler: "CurriculumSampler"):
        self.sampler = sampler
        self.yielded = 0
        self._batch_rows: set[int] = set()

    def __iter__(self) -> "_CurriculumSamplerIterator":
        return self

    def __next__(self) -> int:
        if self.yielded >= len(self.sampler):
            raise StopIteration
        if self.yielded % self.sampler.batch_size == 0:
            self._batch_rows.clear()
        self.yielded += 1
        row = self.sampler._draw(exclude=self._batch_rows)
        self._batch_rows.add(row)
        return row

    def state_dict(self) -> Dict[str, Any]:
        return {"yielded": self.yielded, "batch_rows": sorted(self._batch_rows)}

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        self.yielded = state_dict["yielded"]
        self._batch_rows = set(state_dict["batch_rows"])


class CurriculumSampler(Sampler[int]):
    """Samples dataset rows with replacement from per-bin curriculum weights.

    Each epoch pass yields exactly ``len(dataset)`` indices so the trainer's len()-based
    epoch/step arithmetic is unchanged. Draws are uniform within a bin and deterministic
    given the seed and draw history. The sampler and its iterator implement the torchdata
    Stateful protocol, so StatefulDataLoader checkpoints resume the exact draw stream and
    the decayed per-bin sufficient statistics.
    """

    def __init__(self, dataset: PromptDataset, config: CurriculumConfig, seed: int, batch_size: int):
        if batch_size < 1:
            raise ValueError(f"batch_size must be positive, got {batch_size}")
        self.config = config
        self.bins = dataset_bins(dataset)
        self.num_rows = len(dataset)
        self.batch_size = batch_size
        self.grade_values = np.unique(self.bins.grades)  # sorted ascending
        self.rng = np.random.default_rng(seed)

        num_bins = len(self.bins.names)
        self.informative = np.zeros(num_bins)
        self.total = np.zeros(num_bins)
        self.solved = np.zeros(num_bins)
        self.samples = np.zeros(num_bins)
        self.step_groups = np.zeros(num_bins)
        self.level_pos = 0  # index into grade_values; grade-adaptive only
        self.low_signal_steps = 0
        self.draw_count = 0
        self.weights = self._compute_weights()

    def __len__(self) -> int:
        return self.num_rows

    def __iter__(self) -> _CurriculumSamplerIterator:
        return _CurriculumSamplerIterator(self)

    def _draw(self, exclude: set[int] = frozenset()) -> int:
        """Draw one dataset row, rejecting rows already used in the current batch.

        Duplicate rows within one train batch would merge into a single uid group and
        violate the trainer's exact physical-group-size invariant. Rejection is cheap:
        every bin is far larger than a train batch.
        """
        self.draw_count += 1
        for _ in range(1000):
            bin_idx = int(self.rng.choice(len(self.weights), p=self.weights))
            rows = self.bins.bin_rows[bin_idx]
            row = int(rows[self.rng.integers(len(rows))])
            if row not in exclude:
                return row
        raise RuntimeError(f"Could not draw a batch-unique row after 1000 attempts (batch_size={self.batch_size})")

    def update(self, uids: List[str], rewards: List[float], n_samples_per_prompt: int) -> None:
        """Fold one training step's per-sample rewards into the bin statistics.

        ``uids`` are dataset row indices as strings, one per rollout sample; a group is one
        uid's rollouts. All bin stats decay by ``decay`` before this step's counts are added.
        """
        if len(uids) != len(rewards):
            raise ValueError(f"Got {len(uids)} uids but {len(rewards)} rewards")

        groups: Dict[str, List[float]] = defaultdict(list)
        for uid, reward in zip(uids, rewards):
            groups[uid].append(reward)

        num_bins = len(self.bins.names)
        step_informative = np.zeros(num_bins)
        step_total = np.zeros(num_bins)
        step_solved = np.zeros(num_bins)
        step_samples = np.zeros(num_bins)
        for uid, group_rewards in groups.items():
            if len(group_rewards) % n_samples_per_prompt != 0:
                raise ValueError(
                    f"uid {uid} has {len(group_rewards)} samples, not a multiple of "
                    f"n_samples_per_prompt={n_samples_per_prompt}"
                )
            bin_idx = int(self.bins.row_to_bin[int(uid)])
            step_total[bin_idx] += 1
            if any(reward != group_rewards[0] for reward in group_rewards):
                step_informative[bin_idx] += 1
            step_samples[bin_idx] += len(group_rewards)
            step_solved[bin_idx] += sum(reward > 0 for reward in group_rewards)

        self.informative = self.config.decay * self.informative + step_informative
        self.total = self.config.decay * self.total + step_total
        self.solved = self.config.decay * self.solved + step_solved
        self.samples = self.config.decay * self.samples + step_samples
        self.step_groups = step_total
        if self.config.kind == SamplingKind.GRADE_ADAPTIVE:
            self._maybe_advance_level()
        self.weights = self._compute_weights()

    def _maybe_advance_level(self) -> None:
        """Advance the level once its decayed informative fraction stays low for a full window."""
        level_mask = self.bins.grades == self.grade_values[self.level_pos]
        level_total = self.total[level_mask].sum()
        if level_total <= 0:
            return
        informative_frac = self.informative[level_mask].sum() / level_total
        if informative_frac < self.config.adaptive_min_informative:
            self.low_signal_steps += 1
        else:
            self.low_signal_steps = 0
        if self.low_signal_steps >= self.config.adaptive_window and self.level_pos < len(self.grade_values) - 1:
            self.level_pos += 1
            self.low_signal_steps = 0

    def _compute_weights(self) -> np.ndarray:
        kind = self.config.kind
        counts = self.bins.row_counts.astype(np.float64)
        num_bins = len(counts)

        if kind == SamplingKind.NAIVE:
            # Uniform over rows: weight each bin by its row count.
            weights = counts
        elif kind == SamplingKind.GRADE_UNIFORM:
            weights = np.zeros(num_bins)
            for grade in self.grade_values:
                mask = self.bins.grades == grade
                weights[mask] = counts[mask] / counts[mask].sum() / len(self.grade_values)
        elif kind == SamplingKind.THOMPSON:
            return self._thompson_weights(
                np.full(num_bins, self.config.prior_alpha), np.full(num_bins, self.config.prior_beta)
            )
        elif kind == SamplingKind.LEARNABILITY:
            return self._learnability_weights(
                np.full(num_bins, self.config.prior_alpha), np.full(num_bins, self.config.prior_beta)
            )
        elif kind == SamplingKind.GRADE_PRIOR:
            # Learnability with a grade-seeded prior: the expected pass rate falls linearly
            # from grade_prior_high at the lowest grade to grade_prior_low at the highest.
            grades = self.bins.grades.astype(np.float64)
            grade_span = grades.max() - grades.min()
            fraction = (grades - grades.min()) / grade_span if grade_span > 0 else np.zeros(num_bins)
            mean = self.config.grade_prior_high + fraction * (
                self.config.grade_prior_low - self.config.grade_prior_high
            )
            strength = self.config.grade_prior_strength
            return self._learnability_weights(1.0 + strength * mean, 1.0 + strength * (1.0 - mean))
        elif kind == SamplingKind.GRADE_ADAPTIVE:
            level_mask = self.bins.grades == self.grade_values[self.level_pos]
            if level_mask.all():
                weights = counts
            else:
                weights = np.zeros(num_bins)
                weights[level_mask] = (1.0 - self.config.adaptive_exploration) * counts[level_mask]
                weights[level_mask] /= counts[level_mask].sum()
                weights[~level_mask] = self.config.adaptive_exploration * counts[~level_mask]
                weights[~level_mask] /= counts[~level_mask].sum()
        else:
            raise ValueError(f"Unknown sampling kind: {kind}")

        return weights / weights.sum()

    def _thompson_weights(self, prior_alpha: np.ndarray, prior_beta: np.ndarray) -> np.ndarray:
        samples = self.rng.beta(prior_alpha + self.informative, prior_beta + (self.total - self.informative))
        return self._floor_and_normalize(samples)

    def _learnability_weights(self, prior_alpha: np.ndarray, prior_beta: np.ndarray) -> np.ndarray:
        """Thompson draw on the pass rate, weighted by p·(1−p) so mid-difficulty bins dominate."""
        pass_rate = self.rng.beta(prior_alpha + self.solved, prior_beta + (self.samples - self.solved))
        return self._floor_and_normalize(pass_rate * (1.0 - pass_rate))

    def _floor_and_normalize(self, weights: np.ndarray) -> np.ndarray:
        weights = weights / weights.sum()
        weights = np.maximum(weights, self.config.epsilon / len(weights))
        return weights / weights.sum()

    def metrics(self) -> Dict[str, float]:
        """Per-bin curriculum metrics for the current step."""
        out = {}
        for bin_idx, name in enumerate(self.bins.names):
            total = self.total[bin_idx]
            out[f"curriculum/{name}/weight"] = float(self.weights[bin_idx])
            out[f"curriculum/{name}/informative_frac"] = float(self.informative[bin_idx] / total) if total > 0 else 0.0
            samples = self.samples[bin_idx]
            out[f"curriculum/{name}/pass_rate"] = float(self.solved[bin_idx] / samples) if samples > 0 else 0.0
            out[f"curriculum/{name}/groups"] = float(self.step_groups[bin_idx])
        if self.config.kind == SamplingKind.GRADE_ADAPTIVE:
            out["curriculum/level"] = float(self.grade_values[self.level_pos])
        return out

    def state_dict(self) -> Dict[str, Any]:
        return {
            "informative": self.informative.copy(),
            "total": self.total.copy(),
            "solved": self.solved.copy(),
            "samples": self.samples.copy(),
            "step_groups": self.step_groups.copy(),
            "weights": self.weights.copy(),
            "level_pos": self.level_pos,
            "low_signal_steps": self.low_signal_steps,
            "draw_count": self.draw_count,
            "rng_state": self.rng.bit_generator.state,
        }

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        self.informative = state_dict["informative"].copy()
        self.total = state_dict["total"].copy()
        self.solved = state_dict["solved"].copy()
        self.samples = state_dict["samples"].copy()
        self.step_groups = state_dict["step_groups"].copy()
        self.weights = state_dict["weights"].copy()
        self.level_pos = state_dict["level_pos"]
        self.low_signal_steps = state_dict["low_signal_steps"]
        self.draw_count = state_dict["draw_count"]
        self.rng.bit_generator.state = state_dict["rng_state"]
