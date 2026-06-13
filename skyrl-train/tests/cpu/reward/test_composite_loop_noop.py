"""Stage-0 gate for the loop-behavior reward shaper (`composite_loop`).

These are pure-CPU tests of ``skyrl_train.utils.reward_shaping``; they touch no
GPU / DTensor / Ray. They assert the four Stage-0 invariants from
``notes/RL/skyrl/loop_behavior_reward_stages/stage0_config_scaffold_scope.md``:

  1. Defaults parse, no-op: ``composite_loop`` with all components disabled is
     **bit-identical** to the standalone ``pass_ratio`` reward on a battery of
     synthetic ``(stdout, original_reward, chat_history)`` fixtures (G1).
  2. G1 config no-op: the default ``reward_shaper`` stays ``pass_ratio``
     (``composite_loop`` is opt-in); the ``loop_shaping`` block is additive.
  3. G2 clamp: a synthetic shaped component is clamped to ``+/- total_shaping_cap``
     and can never move a failing (``original_reward == 0``) trajectory net-positive.
  4. Components reported: ``reward_components`` carries every component key, all 0.0
     at the default.
"""

from __future__ import annotations

import pytest

from skyrl_train.utils.reward_shaping import (
    DEFAULT_LOOP_SHAPING_CONFIG,
    CompositeLoopShaper,
    shape_reward_from_output,
    shape_reward_with_components,
)


# ---------------------------------------------------------------------------
# Synthetic trajectory / verifier fixtures
# ---------------------------------------------------------------------------
#
# Each fixture is (stdout, original_reward, chat_history). They span the live
# code paths of the pass_ratio shaper:
#   - parseable pytest summaries (all-pass, partial, all-fail, with xfail/skip)
#   - unittest output
#   - unparseable / empty stdout (fallback-to-original path)
#   - collection-error stdout (parse returns None -> fallback)

_CHAT = [
    {"role": "user", "content": "fix the bug"},
    {"role": "assistant", "content": "<think>plan</think> ok"},
    {"role": "user", "content": "1 failed, 2 passed"},
]

SYNTHETIC_FIXTURES = [
    # (label, stdout, original_reward, chat_history)
    ("pytest_all_pass", "===== 5 passed in 0.12s =====", 1.0, _CHAT),
    ("pytest_partial", "===== 1 failed, 62 passed, 2 xfailed in 2.39s =====", 0.0, _CHAT),
    ("pytest_all_fail", "===== 5 failed in 0.10s =====", 0.0, _CHAT),
    ("pytest_with_skip", "===== 3 passed, 2 skipped in 0.30s =====", 0.0, _CHAT),
    ("pytest_errors", "===== 2 passed, 1 error in 0.50s =====", 0.0, _CHAT),
    ("unittest_ok", "Ran 5 tests in 0.003s\nOK", 1.0, _CHAT),
    ("unittest_failed", "Ran 5 tests in 0.003s\nFAILED (failures=2, errors=1)", 0.0, _CHAT),
    ("collection_error", "ERROR collecting tests\nerror during collection", 0.0, _CHAT),
    ("empty_stdout", "", 0.0, _CHAT),
    ("none_stdout", None, 0.0, _CHAT),
    ("unparseable", "some random log line with no test markers", 1.0, _CHAT),
    ("no_chat", "===== 4 passed, 1 failed in 1.0s =====", 0.0, None),
    ("orig_reward_one_unparse", "garbage", 1.0, _CHAT),
    ("orig_reward_zero_unparse", "garbage", 0.0, _CHAT),
]


def _ids(fixtures):
    return [f[0] for f in fixtures]


# ---------------------------------------------------------------------------
# Invariant 1 — composite_loop (all components off) == pass_ratio, bit-for-bit
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "label,stdout,original_reward,chat_history",
    SYNTHETIC_FIXTURES,
    ids=_ids(SYNTHETIC_FIXTURES),
)
def test_composite_loop_noop_bit_identical_to_pass_ratio(label, stdout, original_reward, chat_history):
    """G1: with defaults, composite_loop reward == standalone pass_ratio reward."""
    # Baseline: exactly what the non-composite generator path computes today.
    baseline = shape_reward_from_output(
        stdout=stdout,
        original_reward=original_reward,
        parser_name=None,
        shaper_name="pass_ratio",
        shaper_kwargs={},
        fallback_to_original=True,
        chat_history=chat_history,
    )

    # composite_loop with default (no-op) loop_shaping config.
    loop_reward, components = shape_reward_with_components(
        stdout=stdout,
        original_reward=original_reward,
        parser_name=None,
        shaper_kwargs={},  # -> CompositeLoopShaper() defaults
        chat_history=chat_history,
        shaper_name="composite_loop",
    )

    # Bit-for-bit identity (use repr to catch any float drift, not just ==).
    assert loop_reward == baseline, f"[{label}] composite_loop {loop_reward!r} != pass_ratio {baseline!r}"
    # The outcome term must itself equal the baseline (no clamp applied at default).
    assert (
        components["outcome"] == baseline
    ), f"[{label}] outcome term {components['outcome']!r} != baseline {baseline!r}"
    # No shaping applied at default.
    assert components["shaping_total"] == 0.0


def test_composite_loop_explicit_outcome_weight_one_noop():
    """An explicit (rather than defaulted) no-op config is still bit-identical."""
    cfg = {
        "loop_shaping": {
            "outcome_weight": 1.0,
            "terminate": {"enabled": False, "green_bonus": 0.0, "red_penalty": 0.0, "noterm_penalty": 0.0},
            "antithrash": {"enabled": False, "per_repeat_penalty": 0.0, "cap": 0.0},
            "total_shaping_cap": 0.3,
        }
    }
    for label, stdout, orig, chat in SYNTHETIC_FIXTURES:
        baseline = shape_reward_from_output(
            stdout=stdout, original_reward=orig, shaper_name="pass_ratio", chat_history=chat
        )
        reward, _ = shape_reward_with_components(
            stdout=stdout,
            original_reward=orig,
            shaper_kwargs=cfg,
            chat_history=chat,
            shaper_name="composite_loop",
        )
        assert reward == baseline, f"[{label}] {reward!r} != {baseline!r}"


# ---------------------------------------------------------------------------
# Invariant 2 — opt-in only; default shaper stays pass_ratio; block is additive
# ---------------------------------------------------------------------------


def test_shape_reward_with_components_defaults_to_composite_not_loop():
    """Default shaper_name is the legacy weighted composite, not composite_loop.

    composite_loop must be opt-in: nothing routes to it unless explicitly named.
    """
    import inspect

    sig = inspect.signature(shape_reward_with_components)
    assert sig.parameters["shaper_name"].default == "composite"


def test_default_loop_config_is_full_noop():
    """The shipped default loop_shaping config is the no-op (all disabled, 0 weights)."""
    cfg = DEFAULT_LOOP_SHAPING_CONFIG
    assert cfg["outcome_weight"] == 1.0
    assert cfg["total_shaping_cap"] == 0.3
    for comp in ("terminate", "antithrash"):
        block = cfg[comp]
        assert block["enabled"] is False, f"{comp} must default disabled"
        for k, v in block.items():
            if k == "enabled":
                continue
            assert v == 0.0, f"{comp}.{k} must default to 0.0, got {v}"


def test_partial_override_keeps_other_defaults():
    """A partial loop_shaping override deep-merges onto defaults."""
    shaper = CompositeLoopShaper(loop_shaping={"terminate": {"enabled": True}})
    # The overridden field takes effect...
    assert shaper.config["terminate"]["enabled"] is True
    # ...but the other terminate fields keep their defaults...
    assert shaper.config["terminate"]["green_bonus"] == 0.0
    # ...and the untouched component block + top-level keys are intact.
    assert shaper.config["antithrash"]["enabled"] is False
    assert shaper.config["outcome_weight"] == 1.0
    assert shaper.config["total_shaping_cap"] == 0.3


# ---------------------------------------------------------------------------
# Invariant 3 — G2 clamp behavior (with a synthetic component injected)
# ---------------------------------------------------------------------------


class _ProbeLoopShaper(CompositeLoopShaper):
    """Test double: inject a fixed shaped delta to exercise the G2 clamp.

    Stage 0 ships no live components, so to test the clamp we subclass and emit a
    synthetic component value. This validates the clamp + ground-truth-anchor
    logic that Stages 1-2 will rely on, without enabling any real shaping.
    """

    def __init__(self, *args, probe_delta: float = 0.0, **kwargs):
        super().__init__(*args, **kwargs)
        self._probe_delta = probe_delta

    def _compute_components(self, stdout, original_reward, chat_history, *args, **kwargs):
        # Forward any extra args (Stage 1+ added an optional ``trajectory_context``
        # param) so the probe stays compatible across stages.
        comps = super()._compute_components(stdout, original_reward, chat_history, *args, **kwargs)
        comps["terminate"] = self._probe_delta
        return comps


def test_g2_clamp_caps_positive_shaping():
    """A large positive shaped delta is clamped to +total_shaping_cap."""
    shaper = _ProbeLoopShaper(probe_delta=5.0)  # cap defaults to 0.3
    reward, comps = shaper.shape_with_components(
        stdout="===== 5 passed in 0.1s =====", original_reward=1.0, chat_history=_CHAT
    )
    assert comps["shaping_total"] == 0.3  # clamped
    # outcome (pass_ratio=1.0) + clamped shaping (0.3) == 1.3
    assert reward == pytest.approx(1.3)


def test_g2_clamp_caps_negative_shaping():
    """A large negative shaped delta is clamped to -total_shaping_cap."""
    shaper = _ProbeLoopShaper(probe_delta=-5.0)
    reward, comps = shaper.shape_with_components(
        stdout="===== 5 passed in 0.1s =====", original_reward=1.0, chat_history=_CHAT
    )
    assert comps["shaping_total"] == -0.3
    assert reward == pytest.approx(0.7)


def test_g2_failing_trajectory_cannot_go_net_positive():
    """Ground-truth anchor: a failing (outcome==0) trajectory stays <= 0.

    Even a maximal positive shaped delta cannot push a zero-outcome trajectory
    net-positive.
    """
    shaper = _ProbeLoopShaper(probe_delta=5.0)
    reward, comps = shaper.shape_with_components(
        stdout="===== 5 failed in 0.1s =====", original_reward=0.0, chat_history=_CHAT
    )
    assert comps["outcome"] == 0.0
    assert reward <= 0.0, f"failing trajectory went net-positive: {reward}"


def test_g2_custom_cap_respected():
    """A non-default total_shaping_cap is honored."""
    shaper = _ProbeLoopShaper(loop_shaping={"total_shaping_cap": 0.1}, probe_delta=5.0)
    reward, comps = shaper.shape_with_components(
        stdout="===== 5 passed in 0.1s =====", original_reward=1.0, chat_history=_CHAT
    )
    assert comps["shaping_total"] == 0.1
    assert reward == pytest.approx(1.1)


# ---------------------------------------------------------------------------
# Invariant 4 — reward_components carries every component key at 0.0
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "label,stdout,original_reward,chat_history",
    SYNTHETIC_FIXTURES,
    ids=_ids(SYNTHETIC_FIXTURES),
)
def test_reward_components_all_keys_present_and_zero(label, stdout, original_reward, chat_history):
    """Every behavioral component key is present and 0.0 at the default config."""
    _, components = shape_reward_with_components(
        stdout=stdout,
        original_reward=original_reward,
        shaper_kwargs={},
        chat_history=chat_history,
        shaper_name="composite_loop",
    )
    # Behavioral components always present...
    for key in ("terminate", "antithrash"):
        assert key in components, f"[{label}] missing component key {key}"
        assert components[key] == 0.0, f"[{label}] {key} != 0.0 at default"
    # ...plus the bookkeeping keys.
    assert "outcome" in components
    assert "shaping_total" in components
    assert components["shaping_total"] == 0.0
