"""Stage-1 gate for the loop-behavior reward shaper's ``terminate`` component.

Pure-CPU tests of ``skyrl_train.utils.reward_shaping`` (the ``composite_loop``
container's M2 "only-then-complete" termination component) plus the generator's
``detect_termination_signals`` helper that threads the finished-trajectory
signals into the shaper. Touches no GPU / DTensor / Ray.

Asserts the Stage-1 invariants from
``notes/RL/skyrl/loop_behavior_reward_stages/stage1_termination_scope.md``:

  1. green-complete (mark_complete + verifier>0) -> base + green_bonus (G2 cap).
  2. red-complete (mark_complete + verifier==0)  -> base - red_penalty AND the
     total stays <= 0 (no net-positive on a fail; G2).
  3. no-terminate (premature_stop / wall)        -> base - noterm_penalty.
  4. ``terminate.enabled=false``                  -> byte-identical to Stage 0 (G1).
  5. green_bonus never fires on verifier==0 (it can never rescue a fail).
  6. asymmetry guard: red-complete and no-terminate are *comparably* penalized so
     the policy is not pushed to NEVER terminate.
"""

from __future__ import annotations

import pytest

from skyrl_train.utils.reward_shaping import (
    _TERMINATE_DEFAULTS,
    CompositeLoopShaper,
    shape_reward_with_components,
)

# The generator helper that derives the trajectory_context. Imported lazily-safe:
# the module pulls in harbor-compat shims, so guard the import and skip the
# helper-specific tests if the heavy generator deps are unavailable (the shaper
# tests below never need it).
try:
    from examples.terminal_bench.terminal_bench_generator import (
        detect_termination_signals,
    )

    _HAVE_GENERATOR = True
except Exception:  # pragma: no cover - exercised only when harbor deps missing
    detect_termination_signals = None  # type: ignore[assignment]
    _HAVE_GENERATOR = False


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# A green verifier stdout (pass_ratio -> 1.0) and a red one (pass_ratio -> 0.0).
GREEN_STDOUT = "===== 5 passed in 0.12s ====="
RED_STDOUT = "===== 5 failed in 0.10s ====="

# Outcome (pass_ratio) for the above, used as the "base" reward in assertions.
GREEN_BASE = 1.0
RED_BASE = 0.0

# loop_shaping config that turns the terminate component ON with the spec
# default magnitudes (green 0.3 / red 0.3 / noterm 0.2 resolved via fallback).
TERMINATE_ON = {"loop_shaping": {"terminate": {"enabled": True}}}

GREEN_BONUS = _TERMINATE_DEFAULTS["green_bonus"]
RED_PENALTY = _TERMINATE_DEFAULTS["red_penalty"]
NOTERM_PENALTY = _TERMINATE_DEFAULTS["noterm_penalty"]


def _ctx(mark_complete: bool, verifier_reward: float, premature_stop: bool):
    return {
        "mark_complete": mark_complete,
        "verifier_reward": verifier_reward,
        "premature_stop": premature_stop,
    }


def _shape(stdout, original_reward, shaper_kwargs, trajectory_context):
    return shape_reward_with_components(
        stdout=stdout,
        original_reward=original_reward,
        shaper_kwargs=shaper_kwargs,
        chat_history=None,
        shaper_name="composite_loop",
        trajectory_context=trajectory_context,
    )


# ---------------------------------------------------------------------------
# Invariant 1 — green-complete -> base + green_bonus (within G2 cap)
# ---------------------------------------------------------------------------


def test_green_complete_adds_green_bonus():
    reward, comps = _shape(
        GREEN_STDOUT, GREEN_BASE, TERMINATE_ON, _ctx(True, GREEN_BASE, False)
    )
    assert comps["outcome"] == GREEN_BASE
    assert comps["terminate"] == pytest.approx(GREEN_BONUS)
    # shaping_total is the clamped sum; green_bonus (0.3) == cap (0.3) -> survives.
    assert comps["shaping_total"] == pytest.approx(GREEN_BONUS)
    assert reward == pytest.approx(GREEN_BASE + GREEN_BONUS)


# ---------------------------------------------------------------------------
# Invariant 2 — red-complete -> base - red_penalty AND total stays <= 0 (G2)
# ---------------------------------------------------------------------------


def test_red_complete_subtracts_red_penalty_and_stays_nonpositive():
    reward, comps = _shape(
        RED_STDOUT, RED_BASE, TERMINATE_ON, _ctx(True, RED_BASE, False)
    )
    assert comps["outcome"] == RED_BASE  # 0.0
    assert comps["terminate"] == pytest.approx(-RED_PENALTY)
    assert reward == pytest.approx(RED_BASE - RED_PENALTY)
    # G2: a failing trajectory can never end net-positive.
    assert reward <= 0.0, f"red-complete went net-positive: {reward}"


# ---------------------------------------------------------------------------
# Invariant 3 — no-terminate (wall) -> base - noterm_penalty
# ---------------------------------------------------------------------------


def test_no_terminate_subtracts_noterm_penalty():
    # never marked complete, ran to the wall (premature_stop True), verifier red.
    reward, comps = _shape(
        RED_STDOUT, RED_BASE, TERMINATE_ON, _ctx(False, RED_BASE, True)
    )
    assert comps["terminate"] == pytest.approx(-NOTERM_PENALTY)
    assert reward == pytest.approx(RED_BASE - NOTERM_PENALTY)
    assert reward <= 0.0


def test_no_terminate_on_green_verifier_still_penalized():
    # Pathological-but-possible: tests pass yet the agent never confirmed done.
    # noterm penalty still applies (we want decisive termination), but it can
    # never push a green trajectory below its own merit beyond the cap.
    reward, comps = _shape(
        GREEN_STDOUT, GREEN_BASE, TERMINATE_ON, _ctx(False, GREEN_BASE, True)
    )
    assert comps["terminate"] == pytest.approx(-NOTERM_PENALTY)
    assert reward == pytest.approx(GREEN_BASE - NOTERM_PENALTY)


# ---------------------------------------------------------------------------
# Invariant 4 — terminate.enabled=false -> byte-identical to Stage-0 no-op (G1)
# ---------------------------------------------------------------------------

_DISABLED_VARIANTS = [
    ("default_no_kwargs", {}),
    ("explicit_disabled", {"loop_shaping": {"terminate": {"enabled": False}}}),
    (
        "disabled_with_magnitudes",
        {
            "loop_shaping": {
                "terminate": {
                    "enabled": False,
                    "green_bonus": 0.3,
                    "red_penalty": 0.3,
                    "noterm_penalty": 0.2,
                }
            }
        },
    ),
]


@pytest.mark.parametrize("label,kwargs", _DISABLED_VARIANTS)
@pytest.mark.parametrize(
    "stdout,base,ctx",
    [
        (GREEN_STDOUT, GREEN_BASE, _ctx(True, GREEN_BASE, False)),
        (RED_STDOUT, RED_BASE, _ctx(True, RED_BASE, False)),
        (RED_STDOUT, RED_BASE, _ctx(False, RED_BASE, True)),
    ],
)
def test_disabled_is_byte_identical_to_stage0(label, kwargs, stdout, base, ctx):
    """With terminate disabled, the reward equals the Stage-0 no-op exactly,
    REGARDLESS of the trajectory_context (G1)."""
    # Stage-0 reference: same config, no trajectory_context threaded.
    ref_reward, ref_comps = shape_reward_with_components(
        stdout=stdout,
        original_reward=base,
        shaper_kwargs=kwargs,
        chat_history=None,
        shaper_name="composite_loop",
        trajectory_context=None,
    )
    # Stage-1 path: same config, full trajectory_context present.
    got_reward, got_comps = _shape(stdout, base, kwargs, ctx)

    assert got_reward == ref_reward, f"[{label}] reward drifted with terminate off"
    assert got_comps["terminate"] == 0.0
    assert got_comps["shaping_total"] == 0.0
    assert got_comps["outcome"] == ref_comps["outcome"]
    # And the disabled reward is exactly the outcome term (== pass_ratio).
    assert got_reward == pytest.approx(base)


# ---------------------------------------------------------------------------
# Invariant 5 — green_bonus only fires when verifier > 0 (never rescues a fail)
# ---------------------------------------------------------------------------


def test_green_bonus_never_fires_on_failing_verifier():
    # mark_complete True but verifier red: must be a PENALTY, not a bonus.
    reward, comps = _shape(
        RED_STDOUT, RED_BASE, TERMINATE_ON, _ctx(True, RED_BASE, False)
    )
    assert comps["terminate"] < 0.0, "green_bonus must not fire on a red verifier"
    assert comps["terminate"] != pytest.approx(GREEN_BONUS)
    assert reward <= 0.0


def test_green_bonus_requires_strictly_positive_verifier():
    # Boundary: verifier exactly 0.0 in the context -> no bonus, red penalty.
    _, comps = _shape(RED_STDOUT, 0.0, TERMINATE_ON, _ctx(True, 0.0, False))
    assert comps["terminate"] == pytest.approx(-RED_PENALTY)


# ---------------------------------------------------------------------------
# Invariant 6 — asymmetry guard: red-complete vs no-terminate comparably penalized
# ---------------------------------------------------------------------------


def test_red_complete_and_noterm_comparably_penalized():
    """Risk mitigation: if red_penalty >> noterm_penalty the policy learns
    "never finish" is safer than "finish red". Assert the two penalties are of
    the same order so neither strategy dominates."""
    _, red_comps = _shape(
        RED_STDOUT, RED_BASE, TERMINATE_ON, _ctx(True, RED_BASE, False)
    )
    _, noterm_comps = _shape(
        RED_STDOUT, RED_BASE, TERMINATE_ON, _ctx(False, RED_BASE, True)
    )
    red_pen = -red_comps["terminate"]
    noterm_pen = -noterm_comps["terminate"]
    assert red_pen > 0.0 and noterm_pen > 0.0
    # Comparable: same order of magnitude (within 2x), and not so lopsided that
    # never-terminating becomes a strictly cheaper escape than finishing red.
    ratio = red_pen / noterm_pen
    assert 0.5 <= ratio <= 2.0, (
        f"penalties not comparable: red={red_pen}, noterm={noterm_pen}, ratio={ratio}"
    )
    # Spec defaults: red 0.3, noterm 0.2 -> finishing-red is penalized slightly
    # MORE than never-finishing, so the policy is never pushed to avoid
    # terminating; but the gap is small enough not to encourage false "done".
    assert red_pen >= noterm_pen


# ---------------------------------------------------------------------------
# Custom-magnitude config is honoured verbatim
# ---------------------------------------------------------------------------


def test_custom_magnitudes_override_fallback():
    cfg = {
        "loop_shaping": {
            "terminate": {
                "enabled": True,
                "green_bonus": 0.1,
                "red_penalty": 0.25,
                "noterm_penalty": 0.05,
            }
        }
    }
    _, green = _shape(GREEN_STDOUT, GREEN_BASE, cfg, _ctx(True, GREEN_BASE, False))
    assert green["terminate"] == pytest.approx(0.1)
    _, red = _shape(RED_STDOUT, RED_BASE, cfg, _ctx(True, RED_BASE, False))
    assert red["terminate"] == pytest.approx(-0.25)
    _, noterm = _shape(RED_STDOUT, RED_BASE, cfg, _ctx(False, RED_BASE, True))
    assert noterm["terminate"] == pytest.approx(-0.05)


def test_g2_cap_clamps_oversized_green_bonus():
    """Even a large custom green_bonus is clamped to the G2 total_shaping_cap."""
    cfg = {
        "loop_shaping": {
            "terminate": {"enabled": True, "green_bonus": 5.0},
            "total_shaping_cap": 0.3,
        }
    }
    reward, comps = _shape(GREEN_STDOUT, GREEN_BASE, cfg, _ctx(True, GREEN_BASE, False))
    assert comps["terminate"] == pytest.approx(5.0)  # raw signed delta, pre-clamp
    assert comps["shaping_total"] == pytest.approx(0.3)  # clamped
    assert reward == pytest.approx(GREEN_BASE + 0.3)


# ---------------------------------------------------------------------------
# Enabled-but-no-context contributes nothing (defensive; preserves identity)
# ---------------------------------------------------------------------------


def test_enabled_without_context_is_noop():
    reward, comps = _shape(GREEN_STDOUT, GREEN_BASE, TERMINATE_ON, None)
    assert comps["terminate"] == 0.0
    assert reward == pytest.approx(GREEN_BASE)


# ---------------------------------------------------------------------------
# Direct CompositeLoopShaper API (mirrors the container call site)
# ---------------------------------------------------------------------------


def test_shaper_direct_api_green_and_red():
    shaper = CompositeLoopShaper(loop_shaping={"terminate": {"enabled": True}})
    r_green = shaper.shape(GREEN_STDOUT, GREEN_BASE, None, _ctx(True, GREEN_BASE, False))
    assert r_green == pytest.approx(GREEN_BASE + GREEN_BONUS)
    r_red = shaper.shape(RED_STDOUT, RED_BASE, None, _ctx(True, RED_BASE, False))
    assert r_red == pytest.approx(RED_BASE - RED_PENALTY)
    assert r_red <= 0.0


# ---------------------------------------------------------------------------
# Generator helper: detect_termination_signals (skipped if harbor deps absent)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _HAVE_GENERATOR, reason="generator harbor deps unavailable")
class TestDetectTerminationSignals:
    def test_xml_marker_detected(self):
        chat = [
            {"role": "user", "content": "do it"},
            {"role": "assistant", "content": "ok <task_complete>true</task_complete>"},
        ]
        ctx = detect_termination_signals(chat, 1.0)
        assert ctx["mark_complete"] is True
        assert ctx["premature_stop"] is False
        assert ctx["verifier_reward"] == 1.0

    def test_json_marker_detected(self):
        chat = [
            {"role": "assistant", "content": '{"plan": "x", "task_complete": true}'},
        ]
        ctx = detect_termination_signals(chat, 0.0)
        assert ctx["mark_complete"] is True

    def test_tool_call_marker_detected(self):
        chat = [
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [{"function_name": "mark_task_complete"}],
            },
        ]
        ctx = detect_termination_signals(chat, 1.0)
        assert ctx["mark_complete"] is True

    def test_no_marker_is_premature_stop(self):
        chat = [
            {"role": "user", "content": "do it"},
            {"role": "assistant", "content": "still working <think>hmm</think>"},
            {"role": "user", "content": "output"},
        ]
        ctx = detect_termination_signals(chat, 0.0)
        assert ctx["mark_complete"] is False
        assert ctx["premature_stop"] is True

    def test_false_marker_not_complete(self):
        chat = [
            {"role": "assistant", "content": "<task_complete>false</task_complete>"},
        ]
        ctx = detect_termination_signals(chat, 0.0)
        assert ctx["mark_complete"] is False

    def test_empty_or_none_history(self):
        for hist in (None, [], [{"role": "user", "content": "x"}]):
            ctx = detect_termination_signals(hist, 0.0)
            assert ctx["mark_complete"] is False
            assert ctx["premature_stop"] is True

    def test_end_to_end_green_complete(self):
        """detect -> shape: a green completed trajectory earns the bonus."""
        chat = [
            {"role": "user", "content": "fix"},
            {"role": "assistant", "content": "done <task_complete>true</task_complete>"},
        ]
        ctx = detect_termination_signals(chat, 1.0)
        reward, comps = _shape(GREEN_STDOUT, 1.0, TERMINATE_ON, ctx)
        assert comps["terminate"] == pytest.approx(GREEN_BONUS)
        assert reward == pytest.approx(1.0 + GREEN_BONUS)
