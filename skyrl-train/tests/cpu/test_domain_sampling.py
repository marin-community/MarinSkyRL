from collections import Counter

import datasets
import pytest

from skyrl_train.domain_sampling import DomainWeightedOrder, weighted_quotas

WEIGHTS = {"math": 2, "code": 2, "if": 1}


def _routes() -> datasets.Dataset:
    return datasets.Dataset.from_dict({"teacher_route": ["math"] * 8 + ["code"] * 8 + ["if"] * 8})


def test_domain_weighted_order_emits_exact_quota_per_window_and_resumes():
    dataset = _routes()
    order = DomainWeightedOrder(dataset, WEIGHTS, seed=7, window_size=6)
    first_window = [order.next_index() for _ in range(6)]

    assert Counter(dataset[index]["teacher_route"] for index in first_window) == {"math": 3, "code": 2, "if": 1}
    assert len(set(first_window)) == 6

    state = order.state_dict()
    expected_next = [order.next_index() for _ in range(6)]
    restored = DomainWeightedOrder(dataset, WEIGHTS, seed=7, window_size=6)
    restored.load_state_dict(state)

    assert [restored.next_index() for _ in range(6)] == expected_next
    assert Counter(dataset[index]["teacher_route"] for index in expected_next) == {"math": 2, "code": 3, "if": 1}
    assert weighted_quotas(1024, WEIGHTS, 0) == {"math": 410, "code": 409, "if": 205}
    assert weighted_quotas(1024, WEIGHTS, 1) == {"math": 409, "code": 410, "if": 205}


def test_domain_weighted_order_continues_past_the_dataset_size():
    dataset = _routes()
    # Five rows split 2:2:1 exactly, so every window has the same mixture.
    order = DomainWeightedOrder(dataset, WEIGHTS, seed=7, window_size=5)
    draws = [order.next_index() for _ in range(10 * len(dataset))]

    routes = Counter(dataset[index]["teacher_route"] for index in draws)
    assert routes == {"math": 96, "code": 96, "if": 48}


def test_domain_weighted_order_rejects_empty_or_underfilled_route_pools():
    dataset = datasets.Dataset.from_dict({"teacher_route": ["math"] * 8 + ["code"] * 8 + ["if"]})

    with pytest.raises(ValueError, match="if"):
        DomainWeightedOrder(dataset, WEIGHTS, seed=7, window_size=11)
