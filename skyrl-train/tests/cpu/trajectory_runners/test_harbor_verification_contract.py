from types import SimpleNamespace

import pytest
from skyrl_gym.verification import VerificationStatus
from skyrl_train.evaluate import _calculate_eval_metrics
from skyrl_train.trajectory_runners.harbor.contracts import verification_from_harbor_result


def test_zero_harbor_score_is_a_verified_result():
    result = SimpleNamespace(
        verifier_result=SimpleNamespace(rewards={"reward": 0.0}, stdout="failed"),
        exception_info=None,
    )

    verification = verification_from_harbor_result(result)

    assert verification.status is VerificationStatus.VERIFIED
    assert verification.score == 0.0
    assert verification.passed is False


def test_missing_harbor_verifier_is_explicitly_unavailable():
    verification = verification_from_harbor_result(SimpleNamespace(verifier_result=None, exception_info=None))

    assert verification.status is VerificationStatus.UNAVAILABLE
    assert verification.score is None


def test_harbor_exception_without_verdict_is_a_verification_error():
    verification = verification_from_harbor_result(
        SimpleNamespace(
            verifier_result=None,
            exception_info=SimpleNamespace(exception_type="VerifierTimeoutError"),
        )
    )

    assert verification.status is VerificationStatus.ERROR
    assert verification.diagnostics["exception_type"] == "VerifierTimeoutError"


def test_malformed_harbor_verifier_result_is_a_verification_error():
    verification = verification_from_harbor_result(
        SimpleNamespace(verifier_result=SimpleNamespace(rewards={}), exception_info=None)
    )

    assert verification.status is VerificationStatus.ERROR
    assert verification.score is None
    assert verification.diagnostics["missing_field"] == "rewards.reward"


@pytest.mark.parametrize("exception_type", ["EnvironmentStartTimeoutError", "VerifierTimeoutError"])
def test_harbor_eval_coverage_excludes_infrastructure_but_keeps_verified_timeout_zero(exception_type):
    outcomes = [
        SimpleNamespace(verifier_result=SimpleNamespace(rewards={"reward": 1.0}, stdout="passed"), exception_info=None),
        SimpleNamespace(
            verifier_result=SimpleNamespace(rewards={"reward": 0.0}, stdout="failed"),
            exception_info=SimpleNamespace(exception_type="AgentTimeoutError"),
        ),
        SimpleNamespace(verifier_result=None, exception_info=SimpleNamespace(exception_type=exception_type)),
    ]
    batch = {
        "response_ids": [[1], [2], [3]],
        "rewards": [1.0, 0.0, 0.0],
        "verification_results": [verification_from_harbor_result(result) for result in outcomes],
    }

    metrics = _calculate_eval_metrics(batch, ["correct", "policy-timeout", "infra-error"], ["bfcl"] * 3, 1)

    assert metrics["eval/all/num_attempted"] == 3
    assert metrics["eval/all/num_scored"] == 2
    assert metrics["eval/all/avg_verifier_score"] == 0.5
    assert metrics["eval/all/verifier_score_coverage"] == pytest.approx(2 / 3)
