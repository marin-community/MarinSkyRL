"""Tests for adaptive curriculum sampling (skyrl_train.curriculum)."""

import numpy as np
import pytest
from datasets import Dataset

from skyrl_train.curriculum import CurriculumConfig, CurriculumOrder, SamplingKind, WeightingKind, dataset_bins
from skyrl_train.rollouts.loader import JudgedGroup


class _StubCurriculumDataset:
    """Minimal PromptDataset stand-in: a dataframe with `extra_info` plus map-style access.

    bins maps data_source name -> (grade, row_count), rows laid out in insertion order.
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


def _order(bins, kind, seed=0, window_size=2, **overrides):
    dataset = _StubCurriculumDataset(bins)
    config = CurriculumConfig(kind=SamplingKind(kind), **overrides)
    return CurriculumOrder(dataset, config, seed=seed, window_size=window_size)


def _observe(order, uids, rewards):
    """Report one step's groups, given one uid and one reward per sample."""
    groups: dict[str, list[float]] = {}
    for uid, reward in zip(uids, rewards, strict=True):
        groups.setdefault(uid, []).append(reward)
    return order.observe([JudgedGroup(uid, tuple(group)) for uid, group in groups.items()])


def _draws(order, count):
    return [order.next_index() for _ in range(count)]


TWO_BINS = {"g0-easy": (0, 1), "g1-hard": (1, 1)}


def test_draws_deterministic_given_seed():
    bins = {"g0-easy": (0, 5), "g1-hard": (1, 5)}
    draws_a = _draws(_order(bins, "thompson", seed=7), 10)
    draws_b = _draws(_order(bins, "thompson", seed=7), 10)
    draws_c = _draws(_order(bins, "thompson", seed=8), 10)
    assert draws_a == draws_b
    assert draws_a != draws_c


def test_state_dict_roundtrip_resumes_identical_draws():
    bins = {"g0-easy": (0, 15), "g1-hard": (1, 15)}
    order = _order(bins, "thompson", seed=3)
    _draws(order, 5)
    _observe(order, ["0", "0", "20", "20"], [0.0, 1.0, 1.0, 1.0])
    _draws(order, 2)

    snapshot = order.state_dict()
    expected = _draws(order, 10)

    restored = _order(bins, "thompson", seed=99)
    restored.load_state_dict(snapshot)
    assert _draws(restored, 10) == expected
    np.testing.assert_allclose(restored.stats.informative, order.stats.informative)
    np.testing.assert_allclose(restored.stats.total, order.stats.total)
    np.testing.assert_allclose(restored.stats.solved, order.stats.solved)
    np.testing.assert_allclose(restored.stats.samples, order.stats.samples)


def test_update_informative_accounting_with_decay():
    order = _order(TWO_BINS, "naive", decay=0.5)
    # Row 0 is the g0 bin, row 1 the g1 bin. One informative group each for bin 0, none for bin 1.
    _observe(order, ["0", "0", "1", "1"], [0.0, 1.0, 1.0, 1.0])
    np.testing.assert_allclose(order.stats.informative, [1.0, 0.0])
    np.testing.assert_allclose(order.stats.total, [1.0, 1.0])

    _observe(order, ["0", "0"], [1.0, 1.0])
    np.testing.assert_allclose(order.stats.informative, [0.5, 0.0])
    np.testing.assert_allclose(order.stats.total, [1.5, 0.5])
    metrics = order.metrics()
    assert metrics["curriculum/g0-easy/informative_frac"] == pytest.approx(0.5 / 1.5)
    assert metrics["curriculum/g0-easy/groups"] == 1.0
    assert metrics["curriculum/g1-hard/groups"] == 0.0


def test_row_stats_track_per_visit_decay_and_recency():
    order = _order(TWO_BINS, "naive", instance_decay=0.5)
    _observe(order, ["0", "0"], [1.0, 0.0])
    _observe(order, ["1", "1"], [1.0, 1.0])
    # Row 0's second visit decays its earlier counts by 0.5 before adding the new group.
    _observe(order, ["0", "0"], [1.0, 1.0])
    rows = order.rows
    np.testing.assert_array_equal(rows.visits, [2, 1])
    np.testing.assert_array_equal(rows.last_step, [3, 2])
    np.testing.assert_allclose(rows.samples, [0.5 * 2 + 2, 2.0])
    np.testing.assert_allclose(rows.solved, [0.5 * 1 + 2, 2.0])


def test_row_stats_summary_metrics():
    order = _order({"g0-easy": (0, 4)}, "naive")
    for _ in range(2):
        # Row 0 always passes (mastered), row 1 always fails (dead).
        _observe(order, ["0", "0", "1", "1"], [1.0, 1.0, 0.0, 0.0])
    _observe(order, ["2", "2"], [1.0, 0.0])  # row 2: one mixed visit
    metrics = order.metrics()
    assert metrics["curriculum/instances/visited_frac"] == pytest.approx(3 / 4)
    assert metrics["curriculum/instances/mean_pass"] == pytest.approx((1.0 + 0.0 + 0.5) / 3)
    assert metrics["curriculum/instances/mastered_frac"] == pytest.approx(1 / 3)
    assert metrics["curriculum/instances/dead_frac"] == pytest.approx(1 / 3)


def test_row_stats_checkpoint_roundtrip():
    order = _order(TWO_BINS, "naive")
    _observe(order, ["0", "0"], [1.0, 0.0])
    restored = _order(TWO_BINS, "naive")
    restored.load_state_dict(order.state_dict())
    assert restored.update_count == 1
    np.testing.assert_array_equal(restored.rows.visits, order.rows.visits)
    np.testing.assert_array_equal(restored.rows.last_step, order.rows.last_step)
    np.testing.assert_allclose(restored.rows.samples, order.rows.samples)
    np.testing.assert_allclose(restored.rows.solved, order.rows.solved)


def test_naive_weights_match_row_counts():
    order = _order({"g0-easy": (0, 2), "g1-hard": (1, 18)}, "naive")
    np.testing.assert_allclose(order.weights, [0.1, 0.9])


def test_grade_uniform_equalizes_grades():
    order = _order({"g0-a": (0, 2), "g0-b": (0, 2), "g1-c": (1, 12)}, "grade-uniform")
    np.testing.assert_allclose(order.weights, [0.25, 0.25, 0.5])


def test_thompson_concentrates_on_informative_bins():
    order = _order(TWO_BINS, "thompson", seed=0)
    for _ in range(30):
        # Bin 0 (row 0) yields informative groups; bin 1 (row 1) is always all-pass.
        _observe(order, ["0", "0", "1", "1"], [0.0, 1.0, 1.0, 1.0])
    assert order.weights[0] > 0.7
    assert order.weights[1] >= 0.05 / 2  # epsilon floor


def test_learnability_concentrates_on_mid_pass_rate_bin():
    bins = {"g0-easy": (0, 1), "g1-mid": (1, 1), "g2-hard": (2, 1)}
    order = _order(bins, "learnability", seed=0)
    for _ in range(30):
        # Bin 0 (row 0) all-pass, bin 1 (row 1) 50% pass, bin 2 (row 2) all-fail.
        _observe(order, ["0", "0", "1", "1", "2", "2"], [1.0, 1.0, 0.0, 1.0, 0.0, 0.0])
    assert order.weights[1] > 0.5
    assert order.weights[1] > order.weights[0]
    assert order.weights[1] > order.weights[2]


def _starved_bin_weights(reversion_mass):
    """Drive bin 0 at a 50% pass rate and bin 1 with one all-fail group of 8 rollouts per
    step — the epsilon-floor trickle of a starved bin. Returns per-bin weights averaged
    over the last 10 of 20 steps."""
    order = _order({"g0-mid": (0, 4), "g1-dead": (1, 4)}, "learnability", seed=2, reversion_mass=reversion_mass)
    mid, dead = [1.0] * 4 + [0.0] * 4, [0.0] * 8
    weight_sum = np.zeros(2)
    for step in range(20):
        _observe(order, ["0"] * 8 + ["4"] * 8, mid + dead)
        if step >= 10:
            weight_sum += order.weights
    return weight_sum / 10


def test_reversion_mass_recovers_starved_bin():
    # Without reversion the floor trickle re-confirms pass 0 forever, so the bin stays
    # pinned near the epsilon floor (0.05 / 2 per bin).
    starved = _starved_bin_weights(reversion_mass=0.0)
    assert starved[1] <= 2 * 0.05 / 2
    # With reversion the bin's pass estimate settles near 2.0 * 0.5 / (8 + 2.0) = 0.1,
    # lifting its weight back into re-probeable range.
    recovered = _starved_bin_weights(reversion_mass=2.0)
    assert recovered[1] >= 0.3 * recovered.max()


def test_reversion_mass_barely_moves_actively_sampled_bin():
    def run(reversion_mass):
        order = _order({"g0-mid": (0, 5), "g1-low": (1, 5)}, "learnability", seed=0, reversion_mass=reversion_mass)
        # 50 rollouts/step per bin: bin 0 (rows 0-4) at pass 0.5, bin 1 (rows 5-9) at 0.2.
        uids = [str(row) for row in range(10) for _ in range(10)]
        rewards = ([1.0] * 5 + [0.0] * 5) * 5 + ([1.0] * 2 + [0.0] * 8) * 5
        weight_sum = np.zeros(2)
        for step in range(30):
            _observe(order, uids, rewards)
            if step >= 20:
                weight_sum += order.weights
        return weight_sum / 10

    # Real counts dominate: 2.0 pseudo-rollouts against 50 real ones move the actively
    # sampled bin's normalized weight by well under 5%.
    without, with_reversion = run(0.0), run(2.0)
    assert abs(with_reversion[0] - without[0]) / without[0] < 0.05


def test_pass_rate_metric_tracks_solved_samples():
    order = _order(TWO_BINS, "naive")
    _observe(order, ["0", "0", "0", "0"], [1.0, 0.0, 0.0, 0.0])
    metrics = order.metrics()
    assert metrics["curriculum/g0-easy/pass_rate"] == pytest.approx(0.25)
    assert metrics["curriculum/g1-hard/pass_rate"] == 0.0  # no samples yet


def test_grade_prior_prefers_easy_then_follows_evidence():
    bins = {"g0-easy": (0, 1), "g2-hard": (2, 1)}
    order = _order(bins, "grade-prior", seed=1)
    weight_sum = np.zeros(2)
    for _ in range(300):
        _observe(order, [], [])  # no evidence; redraws weights from the grade-seeded prior
        weight_sum += order.weights
    # Prior mean pass rate 0.85 at the low grade vs 0.05 at the high grade: the easy
    # bin's p*(1-p) learnability is larger in expectation until evidence says otherwise.
    assert weight_sum[0] > weight_sum[1]

    for _ in range(30):
        # Evidence flips: the easy bin (row 0) saturates at 100% pass, the hard bin
        # (row 1) sits at the 50% learnability peak.
        _observe(order, ["0", "0", "1", "1"], [1.0, 1.0, 0.0, 1.0])
    assert order.weights[1] > order.weights[0]


def test_grade_adaptive_advances_after_low_signal_window():
    order = _order(
        TWO_BINS, "grade-adaptive", adaptive_window=3, adaptive_min_informative=0.1, adaptive_exploration=0.2
    )
    np.testing.assert_allclose(order.weights, [0.8, 0.2])
    for _ in range(3):
        _observe(order, ["0", "0"], [1.0, 1.0])  # level-0 bin saturated: all-pass groups
    assert order.metrics()["curriculum/level"] == 1.0
    np.testing.assert_allclose(order.weights, [0.2, 0.8])

    # At the max grade the level stays put even under sustained low signal.
    for _ in range(5):
        _observe(order, ["1", "1"], [0.0, 0.0])
    assert order.metrics()["curriculum/level"] == 1.0


def test_grade_adaptive_informative_signal_resets_window():
    order = _order(TWO_BINS, "grade-adaptive", adaptive_window=3, adaptive_min_informative=0.1)
    for _ in range(2):
        _observe(order, ["0", "0"], [1.0, 1.0])
    _observe(order, ["0", "0"], [0.0, 1.0])  # informative group resets the counter
    for _ in range(2):
        _observe(order, ["0", "0"], [1.0, 1.0])
    assert order.metrics()["curriculum/level"] == 0.0


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


def test_draws_are_unique_within_each_window():
    bins = {"g0-easy": (0, 8), "g1-hard": (1, 8)}
    order = _order(bins, "thompson", window_size=4)
    indices = _draws(order, 16)
    for start in range(0, 16, 4):
        batch = indices[start : start + 4]
        assert len(set(batch)) == len(batch)


def test_group_informative_weights_low_pass_bin_near_mid_bin():
    """At n=16, a 1-in-16 bin is nearly as informative as a 50% bin; both dwarf the extremes.

    pass-variance would give the low bin ~23% of the mid bin's weight; the
    group-informative curve keeps it above 60%.
    """
    bins = {"g0-easy": (0, 1), "g1-low": (1, 1), "g2-mid": (2, 1), "g3-dead": (3, 1)}
    order = _order(bins, "learnability", seed=0, weighting=WeightingKind.GROUP_INFORMATIVE, group_size=16)
    low = [1.0] + [0.0] * 15
    mid = [1.0] * 8 + [0.0] * 8
    for _ in range(60):
        _observe(
            order,
            ["0"] * 16 + ["1"] * 16 + ["2"] * 16 + ["3"] * 16,
            [1.0] * 16 + low + mid + [0.0] * 16,
        )
    assert order.weights[1] > 0.6 * order.weights[2]
    assert order.weights[1] > 3 * order.weights[0]
    assert order.weights[1] > 3 * order.weights[3]


def test_group_informative_requires_group_size():
    # The guard lives in the curve builder, so a order with group-informative
    # weighting still fails fast at construction when group_size is missing.
    with pytest.raises(ValueError, match="group_size"):
        _order(TWO_BINS, "learnability", weighting=WeightingKind.GROUP_INFORMATIVE)
