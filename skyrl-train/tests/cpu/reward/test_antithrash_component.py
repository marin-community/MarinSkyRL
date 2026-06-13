"""Stage-2 gate for the loop-behavior reward shaper's ``antithrash`` component.

Pure-CPU tests of ``skyrl_train.utils.reward_shaping`` (the ``composite_loop``
container's M3 anti-thrash component + the ``detect_repeated_actions`` util).
Touches no GPU / DTensor / Ray.

Asserts the Stage-2 invariants from
``notes/RL/skyrl/loop_behavior_reward_stages/stage2_antithrash_scope.md``:

  1. 4x byte-identical ``cat > file.py`` (same bytes) -> penalty = min(3*per, cap).
  2. **pytest run twice after DIFFERENT edits -> NO penalty** (the critical
     false-positive guard: re-running a command after real new work is the GOOD
     loop and must never be penalized; dedup is content-keyed, not type-keyed).
  3. edit->test->edit->test (different edits) -> no penalty.
  4. ``antithrash.enabled=false`` -> byte-identical to Stage-0/1 no-op (G1).
  5. Penalty respects the component cap AND the G2 total clamp.
"""

from __future__ import annotations

import pytest

from skyrl_train.utils.reward_shaping import (
    _ANTITHRASH_DEFAULTS,
    CompositeLoopShaper,
    detect_repeated_actions,
    shape_reward_with_components,
)

# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------

PER_REPEAT = _ANTITHRASH_DEFAULTS["per_repeat_penalty"]  # 0.02
CAP = _ANTITHRASH_DEFAULTS["cap"]  # 0.1

# Green / red verifier stdout (pass_ratio -> 1.0 / 0.0), as in the Stage-1 gate.
GREEN_STDOUT = "===== 5 passed in 0.12s ====="
RED_STDOUT = "===== 5 failed in 0.10s ====="
GREEN_BASE = 1.0
RED_BASE = 0.0

# Turn ON antithrash with the spec default magnitudes (per 0.02 / cap 0.1 via fallback).
ANTITHRASH_ON = {"loop_shaping": {"antithrash": {"enabled": True}}}


def _assistant(content: str) -> dict:
    return {"role": "assistant", "content": content}


def _user(content: str) -> dict:
    return {"role": "user", "content": content}


# A heredoc file-write payload (command + body). The exact-bytes write.
def _cat_heredoc(body: str, fname: str = "file.py") -> str:
    return f"cat > {fname} <<'EOF'\n{body}\nEOF"


def _shape(stdout, original_reward, shaper_kwargs, chat_history):
    return shape_reward_with_components(
        stdout=stdout,
        original_reward=original_reward,
        shaper_kwargs=shaper_kwargs,
        chat_history=chat_history,
        shaper_name="composite_loop",
        trajectory_context=None,
    )


# ---------------------------------------------------------------------------
# Invariant 1 — 4x byte-identical cat heredoc -> penalty = min(3*per, cap)
# ---------------------------------------------------------------------------


def test_four_identical_cat_heredocs_penalized():
    same = _cat_heredoc("def f():\n    return 1")
    chat = [
        _user("write file.py"),
        _assistant(same),
        _user("ok"),
        _assistant(same),
        _user("ok"),
        _assistant(same),
        _user("ok"),
        _assistant(same),
        _user("ok"),
    ]
    # 4 identical consecutive actions -> 3 repeats beyond the first.
    assert detect_repeated_actions(chat) == 3

    expected_penalty = min(3 * PER_REPEAT, CAP)  # min(0.06, 0.1) = 0.06
    reward, comps = _shape(GREEN_STDOUT, GREEN_BASE, ANTITHRASH_ON, chat)
    assert comps["antithrash"] == pytest.approx(-expected_penalty)
    assert comps["shaping_total"] == pytest.approx(-expected_penalty)
    assert reward == pytest.approx(GREEN_BASE - expected_penalty)


# ---------------------------------------------------------------------------
# Invariant 2 — pytest twice after DIFFERENT edits -> NO penalty (false-positive guard)
# ---------------------------------------------------------------------------


def test_pytest_rerun_after_different_edits_is_not_penalized():
    """THE critical guard. editA -> pytest -> editB -> pytest. The two pytest
    commands are byte-identical to each other, but a *different* edit sits
    between them, so this is the GOOD diagnose->edit->test->iterate loop, NOT a
    thrash. Content-keyed + consecutive-only counting -> 0 repeats -> 0 penalty."""
    edit_a = _cat_heredoc("def f():\n    return 1")  # first attempt
    edit_b = _cat_heredoc("def f():\n    return 2")  # DIFFERENT bytes
    pytest_cmd = "pytest -q test_f.py"
    chat = [
        _user("fix f"),
        _assistant(edit_a),
        _user("wrote file"),
        _assistant(pytest_cmd),
        _user("1 failed"),
        _assistant(edit_b),
        _user("wrote file"),
        _assistant(pytest_cmd),  # same command string, but after different work
        _user("1 passed"),
    ]
    assert detect_repeated_actions(chat) == 0, "re-run after a different edit must NOT count"

    reward, comps = _shape(GREEN_STDOUT, GREEN_BASE, ANTITHRASH_ON, chat)
    assert comps["antithrash"] == 0.0
    assert comps["shaping_total"] == 0.0
    assert reward == pytest.approx(GREEN_BASE)


def test_pytest_rerun_tool_call_form_after_different_edits_not_penalized():
    """Same guard, but actions are structured tool_calls (name + arguments).
    Keying on the full payload (name AND arguments) means identical edits with
    different bodies are distinct, and the re-run loop is not penalized."""

    def edit_tc(body: str) -> dict:
        return {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {"function_name": "write_file", "arguments": {"path": "f.py", "content": body}}
            ],
        }

    def run_tc() -> dict:
        return {
            "role": "assistant",
            "content": "",
            "tool_calls": [{"function_name": "run", "arguments": {"cmd": "pytest -q"}}],
        }

    chat = [
        _user("fix"),
        edit_tc("return 1"),
        _user("out"),
        run_tc(),
        _user("fail"),
        edit_tc("return 2"),  # different arguments -> distinct payload
        _user("out"),
        run_tc(),  # identical to the earlier run, but not consecutive
        _user("pass"),
    ]
    assert detect_repeated_actions(chat) == 0
    _, comps = _shape(GREEN_STDOUT, GREEN_BASE, ANTITHRASH_ON, chat)
    assert comps["antithrash"] == 0.0


# ---------------------------------------------------------------------------
# Invariant 3 — edit->test->edit->test (different edits) -> no penalty
# ---------------------------------------------------------------------------


def test_edit_test_edit_test_loop_not_penalized():
    chat = [
        _user("fix"),
        _assistant(_cat_heredoc("v = 1")),
        _user("out"),
        _assistant("pytest -q"),
        _user("fail"),
        _assistant(_cat_heredoc("v = 2")),  # different edit
        _user("out"),
        _assistant("pytest -q"),
        _user("pass"),
    ]
    assert detect_repeated_actions(chat) == 0
    reward, comps = _shape(GREEN_STDOUT, GREEN_BASE, ANTITHRASH_ON, chat)
    assert comps["antithrash"] == 0.0
    assert reward == pytest.approx(GREEN_BASE)


# ---------------------------------------------------------------------------
# Invariant 4 — antithrash.enabled=false -> byte-identical to Stage-0/1 (G1)
# ---------------------------------------------------------------------------

# A trajectory that WOULD be penalized if antithrash were on.
_THRASH_CHAT = [
    _user("write"),
    _assistant(_cat_heredoc("x = 1")),
    _user("ok"),
    _assistant(_cat_heredoc("x = 1")),
    _user("ok"),
    _assistant(_cat_heredoc("x = 1")),
    _user("ok"),
]

_DISABLED_VARIANTS = [
    ("default_no_kwargs", {}),
    ("explicit_disabled", {"loop_shaping": {"antithrash": {"enabled": False}}}),
    (
        "disabled_with_magnitudes",
        {"loop_shaping": {"antithrash": {"enabled": False, "per_repeat_penalty": 0.02, "cap": 0.1}}},
    ),
]


@pytest.mark.parametrize("label,kwargs", _DISABLED_VARIANTS)
@pytest.mark.parametrize(
    "stdout,base",
    [(GREEN_STDOUT, GREEN_BASE), (RED_STDOUT, RED_BASE)],
)
def test_disabled_is_byte_identical_to_stage0(label, kwargs, stdout, base):
    """With antithrash disabled, the reward equals the Stage-0/1 no-op exactly,
    even on a trajectory full of identical repeated actions (G1)."""
    # Stage-0/1 reference: same config, no antithrash component live (a chat with
    # no repeats would also give base, but we use the thrash chat to prove the
    # disabled component never fires).
    ref_reward, ref_comps = _shape(stdout, base, kwargs, _THRASH_CHAT)
    assert ref_comps["antithrash"] == 0.0
    assert ref_comps["shaping_total"] == 0.0
    assert ref_comps["outcome"] == pytest.approx(base)
    assert ref_reward == pytest.approx(base), f"[{label}] drifted with antithrash off"


def test_disabled_matches_terminate_only_path():
    """antithrash off + terminate off == pure Stage-0 outcome, on the thrash chat."""
    reward, comps = _shape(GREEN_STDOUT, GREEN_BASE, {}, _THRASH_CHAT)
    assert comps["antithrash"] == 0.0
    assert comps["terminate"] == 0.0
    assert reward == pytest.approx(GREEN_BASE)


# ---------------------------------------------------------------------------
# Invariant 5 — penalty respects the component cap AND the G2 total clamp
# ---------------------------------------------------------------------------


def test_penalty_clamped_at_component_cap():
    """Many identical repeats -> raw penalty exceeds cap -> clamped to -cap."""
    same = _cat_heredoc("x = 1")
    # 10 identical consecutive actions -> 9 repeats; 9 * 0.02 = 0.18 > cap 0.1.
    chat = []
    for _ in range(10):
        chat.append(_assistant(same))
        chat.append(_user("ok"))
    assert detect_repeated_actions(chat) == 9

    _, comps = _shape(GREEN_STDOUT, GREEN_BASE, ANTITHRASH_ON, chat)
    # Component-level clamp at -cap (0.1); 0.18 would exceed it.
    assert comps["antithrash"] == pytest.approx(-CAP)


def test_custom_cap_respected():
    same = _cat_heredoc("x = 1")
    chat = []
    for _ in range(6):  # 5 repeats; 5 * 0.02 = 0.10 > custom cap 0.05
        chat.append(_assistant(same))
        chat.append(_user("ok"))
    cfg = {"loop_shaping": {"antithrash": {"enabled": True, "cap": 0.05}}}
    _, comps = _shape(GREEN_STDOUT, GREEN_BASE, cfg, chat)
    assert comps["antithrash"] == pytest.approx(-0.05)


def test_custom_per_repeat_penalty_respected():
    same = _cat_heredoc("x = 1")
    chat = [_assistant(same), _user("ok"), _assistant(same), _user("ok")]  # 1 repeat
    cfg = {"loop_shaping": {"antithrash": {"enabled": True, "per_repeat_penalty": 0.07}}}
    _, comps = _shape(GREEN_STDOUT, GREEN_BASE, cfg, chat)
    assert comps["antithrash"] == pytest.approx(-0.07)


def test_g2_total_clamp_with_antithrash_and_terminate():
    """When antithrash AND terminate both fire negative, the SUMMED shaped delta
    is clamped to -total_shaping_cap (G2), not just each component's own cap."""
    same = _cat_heredoc("x = 1")
    chat = []
    for _ in range(10):  # antithrash -> -0.1 (its own cap)
        chat.append(_assistant(same))
        chat.append(_user("ok"))
    cfg = {
        "loop_shaping": {
            "antithrash": {"enabled": True},
            "terminate": {"enabled": True},
            "total_shaping_cap": 0.15,
        }
    }
    # red-complete -> terminate = -0.3; antithrash = -0.1; sum = -0.4 -> clamp -0.15.
    reward, comps = shape_reward_with_components(
        stdout=RED_STDOUT,
        original_reward=RED_BASE,
        shaper_kwargs=cfg,
        chat_history=chat,
        shaper_name="composite_loop",
        trajectory_context={"mark_complete": True, "verifier_reward": 0.0, "premature_stop": False},
    )
    assert comps["antithrash"] == pytest.approx(-0.1)
    assert comps["terminate"] == pytest.approx(-0.3)
    assert comps["shaping_total"] == pytest.approx(-0.15)  # G2 total clamp
    # G2 anchor: a failing trajectory stays net-non-positive.
    assert reward <= 0.0


def test_g2_failing_trajectory_stays_nonpositive_with_antithrash():
    same = _cat_heredoc("x = 1")
    chat = [_assistant(same), _user("ok"), _assistant(same), _user("ok")]
    reward, comps = _shape(RED_STDOUT, RED_BASE, ANTITHRASH_ON, chat)
    assert comps["outcome"] == 0.0
    assert comps["antithrash"] < 0.0
    assert reward <= 0.0


# ---------------------------------------------------------------------------
# detect_repeated_actions util — edge cases
# ---------------------------------------------------------------------------


def test_detect_empty_and_none():
    assert detect_repeated_actions(None) == 0
    assert detect_repeated_actions([]) == 0
    assert detect_repeated_actions([_user("only user")]) == 0


def test_detect_single_action_no_repeat():
    assert detect_repeated_actions([_assistant("ls -la")]) == 0


def test_detect_whitespace_normalized_only():
    """Trivial-whitespace-only differences collapse to the same payload, but a
    real content difference does NOT."""
    a = _assistant("cat > f.py <<'EOF'\ndef g():\n    return 1\nEOF")
    # Same bytes, but with trailing spaces + extra blank line + tab-vs-space indent.
    b = _assistant("cat  > f.py <<'EOF'  \n\ndef g():\n\treturn 1\nEOF ")
    chat = [a, _user("x"), b, _user("x")]
    assert detect_repeated_actions(chat) == 1, "whitespace-only diff must be treated as identical"

    # A genuine content difference (return 1 -> return 2) is NOT a repeat.
    c = _assistant("cat > f.py <<'EOF'\ndef g():\n    return 2\nEOF")
    chat2 = [a, _user("x"), c, _user("x")]
    assert detect_repeated_actions(chat2) == 0, "different body must NOT be a repeat"


def test_detect_think_block_ignored():
    """Differing <think> reasoning around an identical action still counts as a
    repeat (the action is the same; the prose is not part of the payload)."""
    a = _assistant("<think>let me try this</think>\nrm -rf build")
    b = _assistant("<think>trying again, different reasoning</think>\nrm -rf build")
    chat = [a, _user("x"), b, _user("x")]
    assert detect_repeated_actions(chat) == 1


def test_detect_consecutive_only_not_global():
    """A->B->A is two distinct consecutive payloads with no run -> 0 repeats."""
    chat = [
        _assistant("cmd A"),
        _user("x"),
        _assistant("cmd B"),
        _user("x"),
        _assistant("cmd A"),  # same as first, but not consecutive
        _user("x"),
    ]
    assert detect_repeated_actions(chat) == 0


def test_detect_empty_turns_do_not_break_run():
    """A no-payload turn between two identical actions does not break the run."""
    same = "make test"
    chat = [
        _assistant(same),
        _user("x"),
        _assistant(""),  # empty / no actionable payload -> skipped
        _user("x"),
        _assistant(same),
        _user("x"),
    ]
    assert detect_repeated_actions(chat) == 1


# ---------------------------------------------------------------------------
# Direct CompositeLoopShaper API
# ---------------------------------------------------------------------------


def test_shaper_direct_api_antithrash():
    same = _cat_heredoc("x = 1")
    chat = [_assistant(same), _user("x"), _assistant(same), _user("x"), _assistant(same), _user("x")]
    shaper = CompositeLoopShaper(loop_shaping={"antithrash": {"enabled": True}})
    reward = shaper.shape(GREEN_STDOUT, GREEN_BASE, chat, None)
    # 3 identical -> 2 repeats -> 2 * 0.02 = 0.04 penalty.
    assert reward == pytest.approx(GREEN_BASE - 2 * PER_REPEAT)
