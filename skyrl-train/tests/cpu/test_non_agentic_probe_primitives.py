"""The probe's advantage primitive tracks the estimator exactly and the analytic value within tolerance."""

import sys

import pytest
import torch

from tests.cpu.test_non_agentic_native_probe_engine import native_probe  # noqa: F401

ANALYTIC = [[-1.0, 1.0], [-1.0, -1.0]]


@pytest.fixture
def primitives(native_probe):  # noqa: F811
    return sys.modules["skyrl_train.entrypoints.non_agentic_probe_primitives"]


def test_final_advantage_checks_match_the_estimator_and_the_analytic_value_on_cpu(primitives):
    evidence = primitives.final_advantage_checks(torch.device("cpu"))
    assert evidence["after_finalization"] == ANALYTIC == evidence["analytic_expectation"]
    assert evidence["max_abs_deviation_from_analytic"] == 0.0
    assert [record["post_override"] for record in evidence["response_evidence"]] == [-1.0, -1.0]


def test_one_ulp_rsqrt_like_cuda_is_tracked_rather_than_hard_coded(primitives, monkeypatch):
    exact = torch.Tensor.rsqrt
    monkeypatch.setattr(torch.Tensor, "rsqrt", lambda self: torch.nextafter(exact(self), torch.zeros_like(self)))
    evidence = primitives.final_advantage_checks(torch.device("cpu"))
    assert evidence["after_finalization"][0][1] != 1.0
    assert 0.0 < evidence["max_abs_deviation_from_analytic"] < 1e-6


def test_changed_estimator_semantics_still_fail(primitives, monkeypatch):
    import skyrl_train.trainer as trainer_module

    def doubled(data):
        data = primitives_normalize(data)
        data["advantages"] = data["advantages"] * 2
        return data

    primitives_normalize = primitives.normalize_advantages_dict
    monkeypatch.setattr(trainer_module, "normalize_advantages_dict", doubled)
    monkeypatch.setattr(primitives, "normalize_advantages_dict", doubled)
    with pytest.raises(AssertionError):
        primitives.final_advantage_checks(torch.device("cpu"))
