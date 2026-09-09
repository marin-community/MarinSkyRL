"""Verify admission permutations independently of version stamping and driver clocks."""

from dataclasses import dataclass
import random

from skyrl_train.group_admission import AdmissionOrder, injected_delay_steps, order_admission_candidates


@dataclass
class Group:
    uid: str
    earliest_model_step: int
    response_lengths: tuple[int, ...]


def groups():
    return [Group(uid, 10 - age, (3, 50, 7, 100)) for uid, age in zip("abcde", (3, 0, 2, 0, 1), strict=True)]


def test_permutations_preserve_group_identity_and_freshness_ties():
    source = groups()
    for policy, expected in [
        (AdmissionOrder.FIFO, "abcde"),
        (AdmissionOrder.LIFO, "edcba"),
        (AdmissionOrder.FRESHEST_FIRST, "bdeca"),
    ]:
        result = order_admission_candidates(source, global_step=10, policy=policy, rng=random.Random(17))
        assert "".join(group.uid for group in result) == expected
        assert {id(group) for group in result} == {id(group) for group in source}
        assert all(len(group.response_lengths) == 4 for group in result)
    assert "".join(group.uid for group in source) == "abcde"
    assert [group.earliest_model_step for group in source] == [7, 10, 8, 10, 9]


def test_weighted_order_matches_power_keys_and_favors_young_groups():
    source = groups()
    rng = random.Random(17)
    uniforms = [rng.random() for _ in source]
    oracle = sorted(
        zip(source, uniforms, strict=True), key=lambda pair: pair[1] ** (11 - pair[0].earliest_model_step), reverse=True
    )
    actual = order_admission_candidates(
        source, global_step=10, policy=AdmissionOrder.AGE_WEIGHTED, rng=random.Random(17)
    )
    assert actual == [group for group, _ in oracle]
    draws = random.Random(29)
    first = [
        order_admission_candidates(source, global_step=10, policy=AdmissionOrder.AGE_WEIGHTED, rng=draws)[0].uid
        for _ in range(1000)
    ]
    assert sum(uid in "bd" for uid in first) > 600


def test_zero_delay_does_not_consume_global_randomness():
    before = random.getstate()
    assert [injected_delay_steps(str(uid), seed=17, maximum=0) for uid in range(10)] == [0] * 10
    assert random.getstate() == before


def test_injected_delays_repeat_for_same_uid_and_seed_with_uniform_support():
    first = [injected_delay_steps(str(uid), seed=17, maximum=3) for uid in range(2000)]
    second = [injected_delay_steps(str(uid), seed=17, maximum=3) for uid in range(2000)]
    other_seed = [injected_delay_steps(str(uid), seed=29, maximum=3) for uid in range(2000)]
    assert first == second and first != other_seed
    counts = [first.count(delay) for delay in range(4)]
    chi_square = sum((count - 500) ** 2 / 500 for count in counts)
    assert chi_square < 11.344866730144373  # chi-square(3) 99th percentile
