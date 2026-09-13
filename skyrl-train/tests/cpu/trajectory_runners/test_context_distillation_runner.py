"""trainer.algorithm.context_distillation in the Harbor runner: the training prompt loses the guidance span, the
served prompt rides along, an absent marker trains as is, a failed edit masks or raises, evaluation never strips."""

from types import SimpleNamespace

import pytest
from omegaconf import OmegaConf

try:
    import skyrl_train.trajectory_runners.harbor.runner as harbor_runner_module
    from skyrl_train.trajectory_runners.harbor.runner import HarborTrajectoryRunner
except ImportError:
    pytest.skip("harbor deps unavailable (agentic RL extra not installed)", allow_module_level=True)

from skyrl_train.trajectory_runners.context_distillation import (
    DEFAULT_END_MARKER,
    DEFAULT_START_MARKER,
    ContextDistillationConfig,
    ContextEditError,
    ContextEditStatus,
)
from skyrl_train.trajectory_runners.types import TrajectoryID
from skyrl_train.utils.harbor_errors import ErrorHandlingConfig

BLOCK = "Make every source change with a short Python script."
PROMPTED = f"Task Description:\nFix foo().{DEFAULT_START_MARKER}{BLOCK}{DEFAULT_END_MARKER}\nroot@host:/app#"
BARE = f"Task Description:\nFix foo().{DEFAULT_END_MARKER}\nroot@host:/app#"
NO_END = f"Task Description:\nFix foo().{DEFAULT_START_MARKER}{BLOCK}"
ENABLED = ContextDistillationConfig(enabled=True)


class _CharTokenizer:
    """One token per character of the rendered messages: exact, so prompts compare by text."""

    def apply_chat_template(self, messages, **_kwargs):
        return [ord(ch) for message in messages for ch in message["content"]]


def _ids(text):
    return [ord(ch) for ch in text]


def _runner(config):
    runner = object.__new__(HarborTrajectoryRunner)
    runner._error_handling_config = ErrorHandlingConfig(enable_error_classification=True)
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
    runner.tokenizer = _CharTokenizer()
    runner.trajectory_runner_cfg = OmegaConf.create(
        {"sampling_params": {"max_generate_length": 16}, "max_input_length": 4096}
    )
    runner._context_distillation = config
    return runner


def _result(first_user_content):
    return SimpleNamespace(
        verifier_result=SimpleNamespace(rewards={"reward": 1.0}, stdout="passed"),
        exception_info=None,
        agent_result=SimpleNamespace(
            metadata={
                "all_messages": [
                    {"role": "user", "content": first_user_content},
                    {"role": "assistant", "content": "done"},
                ],
                "summarization_count": 0,
                "stop_reason": "complete",
            },
            rollout_details=None,
        ),
    )


def _process(runner, content, **kwargs):
    return runner._process_trial_result(_result(content), TrajectoryID(instance_id="task", repetition_id=0), **kwargs)


@pytest.fixture(autouse=True)
def _short_responses(monkeypatch):
    monkeypatch.setattr(
        harbor_runner_module,
        "get_response_ids_and_loss_mask_from_messages",
        lambda *_args, **_kwargs: ([10, 11], [1, 1], None),
    )


def test_disabled_keeps_the_served_prompt_and_adds_nothing():
    output = _process(_runner(ContextDistillationConfig.disabled()), PROMPTED)
    assert list(output.evidence.prompt_token_ids) == _ids(PROMPTED)
    assert output.rollout_prompt_token_ids is None
    assert output.context_edit is None
    assert output.disposition.loss_eligible


def test_enabled_trains_on_the_bare_prompt_and_keeps_the_served_one():
    output = _process(_runner(ENABLED), PROMPTED)
    assert list(output.evidence.prompt_token_ids) == _ids(BARE)
    assert output.rollout_prompt_token_ids == _ids(PROMPTED)
    assert output.context_edit.status is ContextEditStatus.STRIPPED
    assert output.disposition.loss_eligible and output.disposition.baseline_eligible
    # the response stream is untouched by the edit
    assert list(output.evidence.response_token_ids) == [10, 11]
    assert output.loss_mask == [1, 1]


def test_enabled_leaves_an_unprompted_rollout_alone():
    output = _process(_runner(ENABLED), BARE)
    assert list(output.evidence.prompt_token_ids) == _ids(BARE)
    assert output.rollout_prompt_token_ids is None
    assert output.context_edit.status is ContextEditStatus.ABSENT
    assert output.disposition.loss_eligible


def test_failed_edit_masks_the_sample_but_keeps_its_reward_in_the_baseline():
    output = _process(_runner(ContextDistillationConfig(enabled=True, on_failure="mask")), NO_END)
    assert output.context_edit.status is ContextEditStatus.END_MISSING
    assert not output.disposition.loss_eligible
    assert output.disposition.baseline_eligible
    assert "context distillation" in output.disposition.reason
    assert list(output.evidence.prompt_token_ids) == _ids(NO_END)
    assert output.rollout_prompt_token_ids is None
    assert output.reward_result.optimization_reward == 1.0


def test_failed_edit_raises_by_default():
    with pytest.raises(ContextEditError, match="end_missing"):
        _process(_runner(ENABLED), NO_END)


def test_an_overlong_span_is_a_failure_too():
    with pytest.raises(ContextEditError, match="span_too_long"):
        _process(_runner(ContextDistillationConfig(enabled=True, max_removed_chars=8)), PROMPTED)


def test_evaluation_never_strips():
    output = _process(_runner(ENABLED), PROMPTED, is_eval=True)
    assert list(output.evidence.prompt_token_ids) == _ids(PROMPTED)
    assert output.rollout_prompt_token_ids is None
    assert output.context_edit is None


def test_custom_markers_are_honoured():
    config = ContextDistillationConfig(enabled=True, start_marker="<<hint>>", end_marker="<<end>>")
    output = _process(_runner(config), "task<<hint>>be careful<<end>>state")
    assert list(output.evidence.prompt_token_ids) == _ids("task<<end>>state")
    assert output.rollout_prompt_token_ids == _ids("task<<hint>>be careful<<end>>state")
