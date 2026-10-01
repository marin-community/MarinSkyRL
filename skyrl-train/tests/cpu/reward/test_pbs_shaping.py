"""Stage C (F6) — PBS shaping vector + rloo_n_pbs advantage estimator.

Validates:
  * compute_pbs_token_shaping builds a true potential-difference vector on the
    EDIT-token span, policy-invariant (telescopes to Φ_final − Φ_0) and bounded.
  * No test signal / no edit turn ⇒ all-zero vector (outcome-only).
  * rloo_n_pbs estimator == pure RLOO-N when token_level_shaping is None / zeros
    (flag-off byte-identical), and adds the channel additively/masked otherwise.
"""

from types import SimpleNamespace

import numpy as np
import torch

from skyrl_train.utils.span_tagger import SPAN_OTHER, SPAN_THINK, SPAN_ACTION, SPAN_EDIT
from skyrl_train.utils.pbs_shaping import compute_pbs_token_shaping
from skyrl_train.utils.test_delta_parser import TestRunResult
from skyrl_train.utils.advantage_estimators import compute_rloo_n_outcome_advantage, compute_rloo_n_pbs_advantage
from skyrl_train.group_admission import GroupAdvantageInvariant


def _cfg():
    return SimpleNamespace(rloo_n_filter_zero_reward_groups=False)


def _invariant():
    return GroupAdvantageInvariant.minimum_baseline_eligible(physical_group_size=2, minimum_group_size=2)


def _run(idx, passed, failed):
    return TestRunResult(
        message_index=idx, passed=passed, failed=failed, total_runnable=passed + failed, framework="pytest"
    )


# ---------------------------------------------------------------------------
# PBS vector: shape, location, telescoping (policy invariance), bound
# ---------------------------------------------------------------------------


def test_pbs_credits_edit_tokens_only():
    # Layout: [think, edit, edit, OTHER(obs), think, edit] — 2 edit turns.
    tags = [SPAN_THINK, SPAN_EDIT, SPAN_EDIT, SPAN_OTHER, SPAN_THINK, SPAN_EDIT]
    # Two test runs: 0% -> 50% -> 100%.
    runs = [_run(99, 0, 2), _run(99, 1, 1), _run(99, 2, 0)]
    vec = compute_pbs_token_shaping(None, tags, gamma=1.0, max_total_shaping=10.0, test_runs=runs)
    assert len(vec) == len(tags)
    # Non-zero ONLY on EDIT positions (1,2,5); zero elsewhere.
    for j, t in enumerate(tags):
        if t != SPAN_EDIT:
            assert vec[j] == 0.0, f"non-edit token {j} got shaping"
    assert any(vec[j] != 0.0 for j in (1, 2, 5))


def test_pbs_telescopes_to_potential_difference():
    # Policy invariance check: with gamma=1, sum of shaping == Φ_final − Φ_0.
    tags = [SPAN_EDIT, SPAN_OTHER, SPAN_EDIT]
    runs = [_run(99, 1, 3), _run(99, 3, 1)]  # 0.25 -> 0.75
    vec = compute_pbs_token_shaping(None, tags, gamma=1.0, max_total_shaping=10.0, test_runs=runs)
    total = sum(vec)
    # Φ(s0)=0, Φ(final)=0.75 ; telescopes to 0.75.
    assert abs(total - 0.75) < 1e-9


def test_pbs_bounded():
    tags = [SPAN_EDIT, SPAN_EDIT]
    runs = [_run(99, 10, 0)]  # frac 1.0 -> potential 1.0 > bound 0.3
    vec = compute_pbs_token_shaping(None, tags, gamma=1.0, max_total_shaping=0.3, test_runs=runs)
    assert abs(sum(vec)) <= 0.3 + 1e-9
    assert abs(sum(vec) - 0.3) < 1e-9  # scaled to the ceiling


def test_pbs_no_test_signal_is_zeros():
    tags = [SPAN_EDIT, SPAN_THINK, SPAN_EDIT]
    vec = compute_pbs_token_shaping(None, tags, test_runs=[])
    assert vec == [0.0, 0.0, 0.0]


def test_pbs_no_edit_turn_is_zeros():
    # Tests moved but the turn is ACTION (not EDIT) -> nothing to credit.
    tags = [SPAN_THINK, SPAN_ACTION, SPAN_OTHER]
    runs = [_run(99, 1, 0)]
    vec = compute_pbs_token_shaping(None, tags, test_runs=runs)
    assert all(v == 0.0 for v in vec)


def test_pbs_scatter_is_uniform_within_turn():
    tags = [SPAN_EDIT, SPAN_EDIT, SPAN_EDIT]  # one 3-token edit turn
    runs = [_run(99, 2, 2)]  # 0 -> 0.5
    vec = compute_pbs_token_shaping(None, tags, gamma=1.0, max_total_shaping=10.0, test_runs=runs)
    assert abs(sum(vec) - 0.5) < 1e-9
    assert abs(vec[0] - vec[1]) < 1e-12 and abs(vec[1] - vec[2]) < 1e-12


# ---------------------------------------------------------------------------
# rloo_n_pbs estimator: byte-identical when off, additive when on
# ---------------------------------------------------------------------------


def _setup_batch():
    bsz, seqlen = 4, 6
    token_level_rewards = torch.zeros(bsz, seqlen)
    token_level_rewards[:, -1] = torch.tensor([1.0, 0.0, 1.0, 0.0])
    response_mask = torch.ones(bsz, seqlen)
    index = np.array(["g0", "g0", "g1", "g1"])
    return token_level_rewards, response_mask, index


def test_estimator_none_shaping_is_pure_rloo_n():
    tlr, rm, idx = _setup_batch()
    base, _ = compute_rloo_n_outcome_advantage(
        token_level_rewards=tlr, response_mask=rm, index=idx, config=_cfg(), group_advantage_invariant=_invariant()
    )
    adv, ret = compute_rloo_n_pbs_advantage(
        token_level_rewards=tlr,
        response_mask=rm,
        index=idx,
        config=_cfg(),
        group_advantage_invariant=_invariant(),
        token_level_shaping=None,
    )
    assert torch.equal(adv, base)
    assert torch.equal(ret, base)


def test_estimator_adds_shaping_at_exact_tokens():
    tlr, rm, idx = _setup_batch()
    base, _ = compute_rloo_n_outcome_advantage(
        token_level_rewards=tlr, response_mask=rm, index=idx, config=_cfg(), group_advantage_invariant=_invariant()
    )
    shaping = torch.zeros_like(rm)
    shaping[0, 2] = 0.3
    shaping[2, 4] = -0.1
    adv, _ = compute_rloo_n_pbs_advantage(
        token_level_rewards=tlr,
        response_mask=rm,
        index=idx,
        config=_cfg(),
        group_advantage_invariant=_invariant(),
        token_level_shaping=shaping,
    )
    delta = adv - base
    assert torch.isclose(delta[0, 2], torch.tensor(0.3))
    assert torch.isclose(delta[2, 4], torch.tensor(-0.1))
    mask = torch.ones_like(delta, dtype=torch.bool)
    mask[0, 2] = False
    mask[2, 4] = False
    assert torch.allclose(delta[mask], torch.zeros_like(delta[mask]))


def test_shaping_masked_outside_response():
    # Shaping that lands on a masked (response_mask==0) token must NOT enter.
    tlr, rm, idx = _setup_batch()
    rm[3, 5] = 0.0  # mask a token
    base, _ = compute_rloo_n_outcome_advantage(
        token_level_rewards=tlr, response_mask=rm, index=idx, config=_cfg(), group_advantage_invariant=_invariant()
    )
    shaping = torch.zeros_like(rm)
    shaping[3, 5] = 0.5  # on the masked token
    adv, _ = compute_rloo_n_pbs_advantage(
        token_level_rewards=tlr,
        response_mask=rm,
        index=idx,
        config=_cfg(),
        group_advantage_invariant=_invariant(),
        token_level_shaping=shaping,
    )
    assert torch.equal(adv, base)  # masked out
