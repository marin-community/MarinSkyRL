from collections import Counter

import datasets
import pytest

from skyrl_train.domain_sampling import DomainWeightedSampler, weighted_quotas


def test_domain_weighted_sampler_emits_exact_quota_per_batch_and_resumes():
    dataset = datasets.Dataset.from_dict({"teacher_route": ["math"] * 8 + ["code"] * 8 + ["if"] * 8})
    sampler = DomainWeightedSampler(dataset, weights={"math": 2, "code": 2, "if": 1}, seed=7, batch_size=6)
    iterator = iter(sampler)
    first_batch = [next(iterator) for _ in range(6)]

    assert Counter(dataset[index]["teacher_route"] for index in first_batch) == {
        "math": 3,
        "code": 2,
        "if": 1,
    }
    assert len(set(first_batch)) == 6

    iterator_state = iterator.state_dict()
    expected_next = [next(iterator) for _ in range(6)]
    restored_sampler = DomainWeightedSampler(dataset, weights={"math": 2, "code": 2, "if": 1}, seed=7, batch_size=6)
    restored = iter(restored_sampler)
    restored.load_state_dict(iterator_state)

    assert [next(restored) for _ in range(6)] == expected_next
    assert Counter(dataset[index]["teacher_route"] for index in expected_next) == {
        "math": 2,
        "code": 3,
        "if": 1,
    }
    assert weighted_quotas(1024, {"math": 2, "code": 2, "if": 1}, 0) == {"math": 410, "code": 409, "if": 205}
    assert weighted_quotas(1024, {"math": 2, "code": 2, "if": 1}, 1) == {"math": 409, "code": 410, "if": 205}


def test_domain_weighted_sampler_rejects_empty_or_underfilled_route_pools():
    dataset = datasets.Dataset.from_dict({"teacher_route": ["math"] * 8 + ["code"] * 8 + ["if"]})

    with pytest.raises(ValueError, match="if"):
        DomainWeightedSampler(dataset, weights={"math": 2, "code": 2, "if": 1}, seed=7, batch_size=11)
