from collections import Counter

import datasets
from omegaconf import OmegaConf
import pytest

from skyrl_train.domain_sampling import DomainWeightedSampler, weighted_quotas
from skyrl_train.utils.trainer_utils import build_dataloader


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


def test_training_dataloader_uses_exact_domain_weights():
    class RouteDataset:
        dataframe = datasets.Dataset.from_dict({"teacher_route": ["math"] * 8 + ["code"] * 8 + ["if"] * 8})

        def __len__(self):
            return len(self.dataframe)

        def __getitem__(self, index):
            return self.dataframe[index]

        def collate_fn(self, rows):
            return rows

    config = OmegaConf.create(
        {
            "data": {
                "sampling": {"kind": "domain-weighted", "seed": 7, "domain_weights": {"math": 2, "code": 2, "if": 1}},
                "shuffle": True,
            },
            "trainer": {"train_batch_size": 6, "seed": 7, "step_wise_training": False, "epochs": 1},
            "generator": {"enable_http_endpoint": False},
        }
    )

    dataloader = build_dataloader(config, RouteDataset(), is_train=True)
    iterator = iter(dataloader)
    batch = next(iterator)
    checkpoint = dataloader.state_dict()
    expected_next = next(iterator)
    resumed = build_dataloader(config, RouteDataset(), is_train=True)
    resumed.load_state_dict(checkpoint)

    assert Counter(row["teacher_route"] for row in batch) == {"math": 3, "code": 2, "if": 1}
    assert next(iter(resumed)) == expected_next
