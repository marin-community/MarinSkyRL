"""Loop-behavior reward shaping (``composite_loop``): terminate and antithrash components.

Invariants:
  * With components disabled, ``composite_loop`` equals the standalone ``pass_ratio`` reward (G1).
  * The summed shaped delta is clamped to ``+/- total_shaping_cap`` and a failing trajectory never
    ends net-positive (G2).
  * Antithrash counts consecutive, content-keyed repeats only: re-running a command after a
    different edit is the good loop and is never penalized.
"""

import pytest

from skyrl_train.trajectory_runners.harbor.runner import detect_termination_signals
from skyrl_train.utils.reward_shaping import (
    detect_repeated_actions,
    shape_reward_from_output,
    shape_reward_with_components,
)

GREEN_STDOUT = "===== 5 passed in 0.12s ====="
RED_STDOUT = "===== 5 failed in 0.10s ====="

# Spec magnitudes applied when a component is enabled without explicit magnitudes.
GREEN_BONUS = 0.3
RED_PENALTY = 0.3
NOTERM_PENALTY = 0.2
PER_REPEAT = 0.02
ANTITHRASH_CAP = 0.1

TERMINATE_ON = {"loop_shaping": {"terminate": {"enabled": True}}}
ANTITHRASH_ON = {"loop_shaping": {"antithrash": {"enabled": True}}}
DISABLED_WITH_MAGNITUDES = {
    "loop_shaping": {
        "terminate": {"enabled": False, "green_bonus": 0.3, "red_penalty": 0.3, "noterm_penalty": 0.2},
        "antithrash": {"enabled": False, "per_repeat_penalty": 0.02, "cap": 0.1},
    }
}


def _assistant(content: str) -> dict:
    return {"role": "assistant", "content": content}


def _user(content: str) -> dict:
    return {"role": "user", "content": content}


def _cat_heredoc(body: str) -> str:
    return f"cat > file.py <<'EOF'\n{body}\nEOF"


def _repeated(action: str, count: int) -> list[dict]:
    chat = []
    for _ in range(count):
        chat.extend([_assistant(action), _user("ok")])
    return chat


def _ctx(mark_complete: bool, verifier_reward: float, premature_stop: bool) -> dict:
    return {"mark_complete": mark_complete, "verifier_reward": verifier_reward, "premature_stop": premature_stop}


def _shape(stdout, original_reward, shaper_kwargs, chat_history=None, trajectory_context=None):
    return shape_reward_with_components(
        stdout=stdout,
        original_reward=original_reward,
        shaper_kwargs=shaper_kwargs,
        chat_history=chat_history,
        shaper_name="composite_loop",
        trajectory_context=trajectory_context,
    )


# A chat that would trigger antithrash if it were enabled.
_THRASH_CHAT = [_user("fix the bug"), *_repeated("pytest -q", 3)]

VERIFIER_OUTPUTS = [
    ("pytest_all_pass", "===== 5 passed in 0.12s =====", 1.0, _THRASH_CHAT),
    ("pytest_partial", "===== 1 failed, 62 passed, 2 xfailed in 2.39s =====", 0.0, _THRASH_CHAT),
    ("pytest_all_fail", "===== 5 failed in 0.10s =====", 0.0, _THRASH_CHAT),
    ("pytest_with_skip", "===== 3 passed, 2 skipped in 0.30s =====", 0.0, _THRASH_CHAT),
    ("pytest_errors", "===== 2 passed, 1 error in 0.50s =====", 0.0, _THRASH_CHAT),
    ("unittest_ok", "Ran 5 tests in 0.003s\nOK", 1.0, _THRASH_CHAT),
    ("unittest_failed", "Ran 5 tests in 0.003s\nFAILED (failures=2, errors=1)", 0.0, _THRASH_CHAT),
    ("collection_error", "ERROR collecting tests\nerror during collection", 0.0, _THRASH_CHAT),
    ("empty_stdout", "", 0.0, _THRASH_CHAT),
    ("none_stdout", None, 0.0, _THRASH_CHAT),
    ("unparseable_reward_one", "some random log line with no test markers", 1.0, _THRASH_CHAT),
    ("unparseable_reward_zero", "garbage", 0.0, _THRASH_CHAT),
    ("no_chat", "===== 4 passed, 1 failed in 1.0s =====", 0.0, None),
]


@pytest.mark.parametrize("shaper_kwargs", [{}, DISABLED_WITH_MAGNITUDES], ids=["defaults", "disabled_with_magnitudes"])
@pytest.mark.parametrize(
    "stdout,original_reward,chat_history", [v[1:] for v in VERIFIER_OUTPUTS], ids=[v[0] for v in VERIFIER_OUTPUTS]
)
def test_disabled_components_match_pass_ratio(shaper_kwargs, stdout, original_reward, chat_history):
    baseline = shape_reward_from_output(
        stdout=stdout,
        original_reward=original_reward,
        parser_name=None,
        shaper_name="pass_ratio",
        shaper_kwargs={},
        fallback_to_original=True,
        chat_history=chat_history,
    )
    reward, components = _shape(
        stdout, original_reward, shaper_kwargs, chat_history, _ctx(True, original_reward, False)
    )

    assert reward == baseline
    assert components["outcome"] == baseline
    assert components["terminate"] == 0.0
    assert components["antithrash"] == 0.0
    assert components["shaping_total"] == 0.0


@pytest.mark.parametrize(
    "stdout,base,ctx,expected_terminate",
    [
        pytest.param(GREEN_STDOUT, 1.0, _ctx(True, 1.0, False), GREEN_BONUS, id="green_complete"),
        pytest.param(RED_STDOUT, 0.0, _ctx(True, 0.0, False), -RED_PENALTY, id="red_complete"),
        pytest.param(RED_STDOUT, 0.0, _ctx(False, 0.0, True), -NOTERM_PENALTY, id="red_no_terminate"),
        pytest.param(GREEN_STDOUT, 1.0, _ctx(False, 1.0, True), -NOTERM_PENALTY, id="green_no_terminate"),
    ],
)
def test_terminate_component_by_outcome(stdout, base, ctx, expected_terminate):
    reward, comps = _shape(stdout, base, TERMINATE_ON, trajectory_context=ctx)
    assert comps["outcome"] == base
    assert comps["terminate"] == pytest.approx(expected_terminate)
    assert reward == pytest.approx(base + expected_terminate)
    if base == 0.0:
        assert reward <= 0.0


def test_terminate_custom_magnitudes_override_fallback():
    cfg = {
        "loop_shaping": {
            "terminate": {"enabled": True, "green_bonus": 0.1, "red_penalty": 0.25, "noterm_penalty": 0.05}
        }
    }
    _, green = _shape(GREEN_STDOUT, 1.0, cfg, trajectory_context=_ctx(True, 1.0, False))
    _, red = _shape(RED_STDOUT, 0.0, cfg, trajectory_context=_ctx(True, 0.0, False))
    _, noterm = _shape(RED_STDOUT, 0.0, cfg, trajectory_context=_ctx(False, 0.0, True))
    assert green["terminate"] == pytest.approx(0.1)
    assert red["terminate"] == pytest.approx(-0.25)
    assert noterm["terminate"] == pytest.approx(-0.05)


def test_total_shaping_cap_clamps_oversized_green_bonus():
    cfg = {"loop_shaping": {"terminate": {"enabled": True, "green_bonus": 5.0}, "total_shaping_cap": 0.3}}
    reward, comps = _shape(GREEN_STDOUT, 1.0, cfg, trajectory_context=_ctx(True, 1.0, False))
    assert comps["terminate"] == pytest.approx(5.0)
    assert comps["shaping_total"] == pytest.approx(0.3)
    assert reward == pytest.approx(1.3)


def test_total_shaping_cap_clamps_summed_penalties_on_failing_trajectory():
    """Antithrash and terminate both fire negative; the sum is clamped to -total_shaping_cap."""
    cfg = {
        "loop_shaping": {
            "antithrash": {"enabled": True},
            "terminate": {"enabled": True},
            "total_shaping_cap": 0.15,
        }
    }
    chat = _repeated(_cat_heredoc("x = 1"), 10)
    reward, comps = _shape(RED_STDOUT, 0.0, cfg, chat, _ctx(True, 0.0, False))
    assert comps["antithrash"] == pytest.approx(-ANTITHRASH_CAP)
    assert comps["terminate"] == pytest.approx(-RED_PENALTY)
    assert comps["shaping_total"] == pytest.approx(-0.15)
    assert reward <= 0.0


@pytest.mark.parametrize(
    "chat,mark_complete",
    [
        pytest.param([_user("do it"), _assistant("ok <task_complete>true</task_complete>")], True, id="xml_marker"),
        pytest.param([_assistant('{"plan": "x", "task_complete": true}')], True, id="json_marker"),
        pytest.param(
            [{"role": "assistant", "content": "", "tool_calls": [{"function_name": "mark_task_complete"}]}],
            True,
            id="tool_call_marker",
        ),
        pytest.param([_assistant("<task_complete>false</task_complete>")], False, id="false_marker"),
        pytest.param(
            [_user("do it"), _assistant("still working <think>hmm</think>"), _user("output")], False, id="no_marker"
        ),
        pytest.param([], False, id="empty_history"),
    ],
)
def test_detect_termination_signals(chat, mark_complete):
    ctx = detect_termination_signals(chat, 1.0)
    assert ctx == _ctx(mark_complete, 1.0, not mark_complete)


def test_detected_green_completion_earns_bonus():
    chat = [_user("fix"), _assistant("done <task_complete>true</task_complete>")]
    ctx = detect_termination_signals(chat, 1.0)
    reward, comps = _shape(GREEN_STDOUT, 1.0, TERMINATE_ON, trajectory_context=ctx)
    assert comps["terminate"] == pytest.approx(GREEN_BONUS)
    assert reward == pytest.approx(1.0 + GREEN_BONUS)


def test_antithrash_penalizes_identical_heredoc_writes():
    chat = [_user("write file.py"), *_repeated(_cat_heredoc("def f():\n    return 1"), 4)]
    assert detect_repeated_actions(chat) == 3

    reward, comps = _shape(GREEN_STDOUT, 1.0, ANTITHRASH_ON, chat)
    assert comps["antithrash"] == pytest.approx(-3 * PER_REPEAT)
    assert comps["shaping_total"] == pytest.approx(-3 * PER_REPEAT)
    assert reward == pytest.approx(1.0 - 3 * PER_REPEAT)


def _edit_tool_call(body: str) -> dict:
    return {
        "role": "assistant",
        "content": "",
        "tool_calls": [{"function_name": "write_file", "arguments": {"path": "f.py", "content": body}}],
    }


_RUN_TOOL_CALL = {
    "role": "assistant",
    "content": "",
    "tool_calls": [{"function_name": "run", "arguments": {"cmd": "pytest -q"}}],
}


@pytest.mark.parametrize(
    "chat",
    [
        pytest.param(
            [
                _user("fix f"),
                _assistant(_cat_heredoc("def f():\n    return 1")),
                _user("wrote file"),
                _assistant("pytest -q test_f.py"),
                _user("1 failed"),
                _assistant(_cat_heredoc("def f():\n    return 2")),
                _user("wrote file"),
                _assistant("pytest -q test_f.py"),
                _user("1 passed"),
            ],
            id="text_actions",
        ),
        pytest.param(
            [
                _user("fix"),
                _edit_tool_call("return 1"),
                _user("out"),
                _RUN_TOOL_CALL,
                _user("fail"),
                _edit_tool_call("return 2"),
                _user("out"),
                _RUN_TOOL_CALL,
                _user("pass"),
            ],
            id="tool_calls",
        ),
    ],
)
def test_antithrash_rerun_after_different_edit_not_penalized(chat):
    """The same test command after a different edit is the good loop, not a thrash."""
    assert detect_repeated_actions(chat) == 0
    reward, comps = _shape(GREEN_STDOUT, 1.0, ANTITHRASH_ON, chat)
    assert comps["antithrash"] == 0.0
    assert reward == pytest.approx(1.0)


def test_antithrash_penalty_clamped_at_component_cap():
    chat = _repeated(_cat_heredoc("x = 1"), 10)
    assert detect_repeated_actions(chat) == 9
    _, comps = _shape(GREEN_STDOUT, 1.0, ANTITHRASH_ON, chat)
    assert comps["antithrash"] == pytest.approx(-ANTITHRASH_CAP)


def test_antithrash_custom_per_repeat_penalty():
    cfg = {"loop_shaping": {"antithrash": {"enabled": True, "per_repeat_penalty": 0.07}}}
    _, comps = _shape(GREEN_STDOUT, 1.0, cfg, _repeated(_cat_heredoc("x = 1"), 2))
    assert comps["antithrash"] == pytest.approx(-0.07)


@pytest.mark.parametrize(
    "first,second,expected_repeats",
    [
        pytest.param(
            "cat > f.py <<'EOF'\ndef g():\n    return 1\nEOF",
            "cat  > f.py <<'EOF'  \n\ndef g():\n\treturn 1\nEOF ",
            1,
            id="whitespace_only_difference",
        ),
        pytest.param(
            "cat > f.py <<'EOF'\ndef g():\n    return 1\nEOF",
            "cat > f.py <<'EOF'\ndef g():\n    return 2\nEOF",
            0,
            id="content_difference",
        ),
        pytest.param(
            "<think>let me try this</think>\nrm -rf build",
            "<think>trying again, different reasoning</think>\nrm -rf build",
            1,
            id="think_block_ignored",
        ),
    ],
)
def test_detect_repeated_actions_payload_normalization(first, second, expected_repeats):
    chat = [_assistant(first), _user("x"), _assistant(second), _user("x")]
    assert detect_repeated_actions(chat) == expected_repeats


def test_detect_repeated_actions_counts_consecutive_runs_only():
    chat = [_assistant("cmd A"), _user("x"), _assistant("cmd B"), _user("x"), _assistant("cmd A"), _user("x")]
    assert detect_repeated_actions(chat) == 0


def test_detect_repeated_actions_skips_empty_turns():
    chat = [_assistant("make test"), _user("x"), _assistant(""), _user("x"), _assistant("make test"), _user("x")]
    assert detect_repeated_actions(chat) == 1
