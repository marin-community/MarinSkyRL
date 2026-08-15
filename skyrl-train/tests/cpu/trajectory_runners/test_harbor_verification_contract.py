from types import SimpleNamespace

from skyrl_gym.verification import VerificationStatus
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
