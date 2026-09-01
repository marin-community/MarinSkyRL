from types import SimpleNamespace

import pytest
from skyrl_gym.verification import RewardResult
from omegaconf import OmegaConf

try:
    import skyrl_train.trajectory_runners.harbor.runner as harbor_runner_module
    from skyrl_train.trajectory_runners.harbor.runner import HarborTrajectoryRunner
except ImportError:
    pytest.skip("harbor deps unavailable (agentic RL extra not installed)", allow_module_level=True)

from harbor.verifier.verifier import VerifierOutputParseError
from skyrl_train.metric_names import IDENTITY_AWARE_REWARD_METRIC_PREFIX
from skyrl_train.trajectory_runners.types import TrajectoryID
from skyrl_train.utils.harbor_errors import ErrorHandlingConfig


def _runner_output(
    trial: int,
    outcomes: dict[str, str],
    aggregate_reward: float,
    *,
    verifier_test_collection_factory,
    truncation_penalized: bool = False,
):
    return SimpleNamespace(
        trajectory_id=TrajectoryID(instance_id="task", repetition_id=trial),
        verifier_tests=verifier_test_collection_factory(trial, outcomes),
        disposition=SimpleNamespace(baseline_eligible=True),
        reward_result=RewardResult(unshaped_reward=aggregate_reward, optimization_reward=aggregate_reward),
        truncation_penalized=truncation_penalized,
    )


def _runner(shaper: str | None = None):
    runner = object.__new__(HarborTrajectoryRunner)
    runner._reward_shaping_enabled = True
    runner._reward_shaping_config = {"shaper_kwargs": {}}
    if shaper is not None:
        runner._reward_shaping_config["reward_shaper"] = shaper
    runner._truncation_penalty = 0.25
    return runner


class _Tokenizer:
    def apply_chat_template(self, *_args, **_kwargs):
        return [1]


def _trial_runner() -> HarborTrajectoryRunner:
    runner = object.__new__(HarborTrajectoryRunner)
    runner._error_handling_config = ErrorHandlingConfig(
        enable_error_classification=True,
        passthrough_exceptions=frozenset({"TurnCapExhaustedError"}),
    )
    runner._rollout_logprobs_required = False
    runner._reward_shaping_enabled = False
    runner._collect_rollout_details = False
    runner._moe_router_replay = False
    runner._tito_full = None
    runner._tis_splice = True
    runner._truncation_penalty = 0.0
    runner._enable_token_reward_channel = False
    runner._chat_template_kwargs = {}
    runner.custom_chat_template_content = None
    runner.tokenizer = _Tokenizer()
    runner.trajectory_runner_cfg = OmegaConf.create(
        {"sampling_params": {"max_generate_length": 16}, "max_input_length": 16}
    )
    return runner


@pytest.mark.parametrize(
    ("exception_info", "agent_stop_reason", "expected_exception", "expected_treatment"),
    [
        (
            SimpleNamespace(exception_type="TurnCapExhaustedError"),
            "turn_cap_exhausted",
            "TurnCapExhaustedError",
            "passthrough",
        ),
        (None, "complete", None, None),
    ],
)
def test_verified_harbor_result_preserves_terminal_disposition(
    monkeypatch, exception_info, agent_stop_reason, expected_exception, expected_treatment
):
    monkeypatch.setattr(
        harbor_runner_module,
        "get_response_ids_and_loss_mask_from_messages",
        lambda *_args, **_kwargs: ([10, 11], [1, 1], None),
    )
    result = SimpleNamespace(
        verifier_result=SimpleNamespace(rewards={"reward": 1.0}, stdout="passed"),
        exception_info=exception_info,
        agent_result=SimpleNamespace(
            metadata={
                "all_messages": [
                    {"role": "user", "content": "solve it"},
                    {"role": "assistant", "content": "done"},
                ],
                "summarization_count": 0,
                "stop_reason": agent_stop_reason,
            },
            rollout_details=None,
        ),
    )

    output = _trial_runner()._process_trial_result(
        result,
        TrajectoryID(instance_id="task", repetition_id=0),
    )

    assert output.verification.score == 1.0
    assert output.reward_result.unshaped_reward == 1.0
    assert output.reward_result.optimization_reward == 1.0
    assert output.evidence.stop_reason == agent_stop_reason
    assert output.disposition.loss_eligible
    assert output.disposition.baseline_eligible
    assert output.disposition.exception_type == expected_exception
    assert output.error_treatment == expected_treatment


def test_harbor_runner_applies_identity_aware_shaping_as_the_default(verifier_test_collection_factory):
    outputs = [
        _runner_output(
            0,
            {"uniform": "passed", "mixed": "passed"},
            1.0,
            verifier_test_collection_factory=verifier_test_collection_factory,
        ),
        _runner_output(
            1,
            {"uniform": "passed", "mixed": "failed"},
            0.5,
            verifier_test_collection_factory=verifier_test_collection_factory,
        ),
    ]

    metrics = _runner()._apply_identity_aware_reward_shaping(outputs)

    assert [output.reward_result.optimization_reward for output in outputs] == [1.0, 0.0]
    assert metrics[f"{IDENTITY_AWARE_REWARD_METRIC_PREFIX}/groups"] == 1


def test_legacy_pass_ratio_is_an_explicit_backup_mode(verifier_test_collection_factory):
    outputs = [
        _runner_output(
            0,
            {"uniform": "passed", "mixed": "passed"},
            1.0,
            verifier_test_collection_factory=verifier_test_collection_factory,
        ),
        _runner_output(
            1,
            {"uniform": "passed", "mixed": "failed"},
            0.5,
            verifier_test_collection_factory=verifier_test_collection_factory,
        ),
    ]

    metrics = _runner("pass_ratio")._apply_identity_aware_reward_shaping(outputs)

    assert [output.reward_result.optimization_reward for output in outputs] == [1.0, 0.5]
    assert metrics == {}


def test_identity_aware_shaping_preserves_the_downstream_truncation_penalty(verifier_test_collection_factory):
    outputs = [
        _runner_output(
            0,
            {"mixed": "passed"},
            0.75,
            verifier_test_collection_factory=verifier_test_collection_factory,
            truncation_penalized=True,
        ),
        _runner_output(
            1,
            {"mixed": "failed"},
            0.0,
            verifier_test_collection_factory=verifier_test_collection_factory,
        ),
    ]

    _runner()._apply_identity_aware_reward_shaping(outputs)

    assert [output.reward_result.optimization_reward for output in outputs] == [0.75, 0.0]


def test_unrecognized_verifier_output_is_binned_as_a_verifier_failure():
    runner = object.__new__(HarborTrajectoryRunner)
    runner._error_handling_config = ErrorHandlingConfig(enable_error_classification=True)
    runner._reward_shaping_enabled = True
    runner._reward_shaping_config = {
        "enable_reward_shaping": True,
        "reward_shaper": "threshold",
        "reward_shaping_fallback": False,
    }
    result = SimpleNamespace(
        verifier_result=SimpleNamespace(rewards={"reward": 0.0}, stdout="unrecognized verifier output"),
        exception_info=None,
        agent_result=SimpleNamespace(
            metadata={
                "all_messages": [
                    {"role": "user", "content": "solve it"},
                    {"role": "assistant", "content": "done"},
                ],
                "summarization_count": 0,
            }
        ),
    )

    with pytest.raises(VerifierOutputParseError):
        runner._process_trial_result(result, TrajectoryID(instance_id="task", repetition_id=0))

    treatment, exception_type = runner._classify_exception(VerifierOutputParseError("unrecognized output"))
    assert treatment == "mask"
    assert exception_type == "VerifierOutputParseError"
