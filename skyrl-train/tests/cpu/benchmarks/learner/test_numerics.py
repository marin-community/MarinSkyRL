import math

import numpy as np
from omegaconf import OmegaConf
import pytest
import torch

from skyrl_train.benchmarks.learner import numerics
from skyrl_train.benchmarks.learner.numerics import (
    CalibrationPolicy,
    NumericsProvenance,
    NumericsRecord,
    Signal,
    SignalKind,
    calibrate_reference,
    compare_to_reference,
    logprob_signals,
    numerics_config_fingerprint,
    parameter_signals,
    read_reference,
    write_reference,
)
from skyrl_train.dataset.routed_expert_batch import RoutedExpertRows
from skyrl_train.distributed.megatron.parameter_fingerprint import parameter_delta_statistics
from skyrl_train.training_batch import TrainingInputBatch

POLICY = CalibrationPolicy(
    multiplier=4.0,
    floors={SignalKind.RELATIVE: 1e-3, SignalKind.ABSOLUTE: 1e-4, SignalKind.RELATIVE_L2: 1e-3},
    weight_resolution=numerics.BF16_RELATIVE_SPACING,
)


def _provenance(layout: str = "tp1-ep1") -> NumericsProvenance:
    return NumericsProvenance(
        checkpoint="hero-k4",
        batch_sha256="abc",
        config_fingerprint="cfg",
        num_updates=1,
        layout=layout,
        source_revision="rev",
    )


def _bf16_update(shape, seed: int, scale: float = 1e-2) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    before = torch.randn(shape, generator=generator).to(torch.bfloat16)
    after = (before.float() + scale * torch.randn(shape, generator=generator)).to(torch.bfloat16)
    return before, after


def _record(statistics, layout: str = "tp1-ep1", extra: dict[str, Signal] | None = None) -> NumericsRecord:
    signals = parameter_signals(statistics)
    signals["train_status/update0/policy_loss"] = Signal(SignalKind.ABSOLUTE, (0.25,))
    signals.update(extra or {})
    return NumericsRecord(provenance=_provenance(layout), signals=signals)


def _groups(updates):
    return {group: parameter_delta_statistics(before, after) for group, (before, after) in updates.items()}


def test_single_layer_change_and_misplaced_update_fail_only_their_group():
    # Updates of ~25 bf16 steps per element, well above the weight-resolution floor.
    updates = {
        f"decoder.layers.{layer}.mlp.linear_fc1.weight": _bf16_update((32, 32), seed=layer, scale=0.1)
        for layer in range(3)
    }
    reference = calibrate_reference([_record(_groups(updates)), _record(_groups(updates), layout="tp2-ep1")], POLICY)
    assert compare_to_reference(reference, _record(_groups(updates))).passed

    target = "decoder.layers.1.mlp.linear_fc1.weight"
    before, after = updates[target]
    # Zeroed update for one layer.
    zeroed = dict(updates, **{target: (before, before.clone())})
    failures = {check.name for check in compare_to_reference(reference, _record(_groups(zeroed))).failures}
    assert {f"params/{target}/delta_norm", f"params/{target}/sketch", "params/total_delta_norm"} <= failures
    assert all(target in name or name == "params/total_delta_norm" for name in failures)

    # The right update applied to the wrong rows keeps every norm but breaks the sketch.
    delta = after.float() - before.float()
    rolled = (before.float() + delta.roll(1, dims=0)).to(torch.bfloat16)
    misplaced = dict(updates, **{target: (before, rolled)})
    failures = {check.name for check in compare_to_reference(reference, _record(_groups(misplaced))).failures}
    assert f"params/{target}/sketch" in failures


def test_calibration_scales_largest_pairwise_spread_with_floor():
    def run(loss: float, layout: str) -> NumericsRecord:
        return NumericsRecord(
            provenance=_provenance(layout),
            signals={
                "loss": Signal(SignalKind.ABSOLUTE, (loss,)),
                "grad_norm": Signal(SignalKind.RELATIVE, (2.0,)),
            },
        )

    reference = calibrate_reference([run(1.0, "a"), run(1.01, "b"), run(0.995, "c")], POLICY)
    assert reference.spreads["loss"] == pytest.approx(0.015)
    assert reference.tolerances["loss"] == pytest.approx(0.06)
    # Identical runs leave only the floor.
    assert reference.tolerances["grad_norm"] == POLICY.floors[SignalKind.RELATIVE]

    assert compare_to_reference(reference, run(1.05, "d")).passed
    failures = compare_to_reference(reference, run(1.07, "d")).failures
    assert [check.name for check in failures] == ["loss"]


def test_comparison_fails_missing_extra_and_nonfinite_signals():
    base = {"loss": Signal(SignalKind.ABSOLUTE, (1.0,)), "norm": Signal(SignalKind.RELATIVE, (3.0,))}
    reference = calibrate_reference(
        [NumericsRecord(_provenance(), base), NumericsRecord(_provenance("b"), base)], POLICY
    )
    missing = NumericsRecord(_provenance(), {"loss": base["loss"]})
    extra = NumericsRecord(_provenance(), {**base, "new": Signal(SignalKind.ABSOLUTE, (0.0,))})
    nonfinite = NumericsRecord(_provenance(), {**base, "loss": Signal(SignalKind.ABSOLUTE, (math.nan,))})
    assert [check.name for check in compare_to_reference(reference, missing).failures] == ["norm"]
    assert [check.name for check in compare_to_reference(reference, extra).failures] == ["new"]
    assert [check.name for check in compare_to_reference(reference, nonfinite).failures] == ["loss"]

    other_batch = NumericsRecord(NumericsProvenance(**{**_provenance().__dict__, "batch_sha256": "other"}), base)
    with pytest.raises(ValueError, match="provenance"):
        compare_to_reference(reference, other_batch)


def test_reference_json_round_trip_preserves_verdicts(tmp_path):
    updates = {"decoder.layers.0.self_attention.linear_qkv.weight": _bf16_update((16, 8), seed=3)}
    logprobs = torch.log(torch.rand(4, 10, generator=torch.Generator().manual_seed(4)))
    mask = torch.ones(4, 10)
    extra = logprob_signals("logprobs/pre_update", logprobs, mask)
    reference = calibrate_reference(
        [_record(_groups(updates), extra=extra), _record(_groups(updates), layout="b", extra=extra)], POLICY
    )
    path = str(tmp_path / "reference.json")
    write_reference(path, reference)
    loaded = read_reference(path)
    assert loaded == reference

    shifted = logprob_signals("logprobs/pre_update", logprobs + 1e-2, mask)
    candidate = _record(_groups(updates), extra=shifted)
    assert compare_to_reference(loaded, candidate).failures == compare_to_reference(reference, candidate).failures
    assert not compare_to_reference(loaded, candidate).passed


def test_logprob_signals_follow_masked_tokens_only(monkeypatch):
    monkeypatch.setattr(numerics, "LOGPROB_SAMPLE_SIZE", 16)
    generator = torch.Generator().manual_seed(5)
    logprobs = -torch.rand(6, 20, generator=generator)
    mask = torch.zeros(6, 20)
    mask[:, :12] = 1
    reference = logprob_signals("lp", logprobs, mask)
    sampled = reference["lp/sampled_tokens"].values
    assert len(sampled) == 16
    assert set(sampled) <= set(logprobs[:, :12].reshape(-1).tolist())

    padding_changed = logprobs.clone()
    padding_changed[:, 12:] = 0.0
    assert logprob_signals("lp", padding_changed, mask) == reference

    token_changed = logprobs.clone()
    token_changed[2, 3] -= 0.5
    changed = logprob_signals("lp", token_changed, mask)
    assert changed["lp/sequence_mean"] != reference["lp/sequence_mean"]
    assert changed["lp/token_mean"] != reference["lp/token_mean"]


def test_config_fingerprint_ignores_layout_but_not_optimizer():
    cfg = OmegaConf.create(
        {
            "trainer": {
                "seed": 42,
                "algorithm": {"policy_loss_type": "regular"},
                "policy": {"optimizer_config": {"lr": 1e-6}, "megatron_config": {"tensor_model_parallel_size": 1}},
                "policy_mini_batch_size": 64,
                "update_epochs_per_batch": 1,
                "micro_train_batch_size_per_gpu": 1,
            },
            "generator": {"sampling_params": {"temperature": 1.0}},
        }
    )
    fingerprint = numerics_config_fingerprint(cfg)
    relayout = cfg.copy()
    relayout.trainer.policy.megatron_config.tensor_model_parallel_size = 4
    relayout.trainer.micro_train_batch_size_per_gpu = 2
    assert numerics_config_fingerprint(relayout) == fingerprint
    retuned = cfg.copy()
    retuned.trainer.policy.optimizer_config.lr = 2e-6
    assert numerics_config_fingerprint(retuned) != fingerprint


def test_batch_identity_distinguishes_replay_routes_from_native_routing():
    batch = TrainingInputBatch({"sequences": torch.zeros(1, 4, dtype=torch.long)})
    native = numerics.batch_identity("digest", batch)
    rows = np.zeros((3, 2, 8), dtype=np.int16)
    batch.routed_expert_rows = RoutedExpertRows((rows,), 3, 384)
    replay = numerics.batch_identity("digest", batch)
    rows[0, 1, 0] = 5
    rerouted = numerics.batch_identity("digest", batch)

    assert len({native, replay, rerouted}) == 3
    assert numerics.batch_identity("digest", batch) == rerouted


def test_weight_resolution_floor_covers_sub_step_updates_and_not_resolved_ones():
    def run(layout: str, small_delta: float, large_delta: float) -> NumericsRecord:
        statistics = {
            # A norm-like group: weights near 1, update far below one bf16 step (~4e-3 per element).
            "norm": numerics.ParameterGroupStatistics(
                numel=4.0, delta_square_sum=small_delta**2, before_square_sum=4.0, sketches=(small_delta,) * 4
            ),
            # A matrix group whose update is ~100 bf16 steps.
            "matrix": numerics.ParameterGroupStatistics(
                numel=4.0, delta_square_sum=large_delta**2, before_square_sum=1e-4, sketches=(large_delta,) * 4
            ),
        }
        return NumericsRecord(provenance=_provenance(layout), signals=parameter_signals(statistics))

    reference = calibrate_reference([run("a", 1e-4, 4e-3), run("b", 1e-4, 4e-3)], POLICY)
    # Sub-step noise: the stored-dtype rounding moves the norm group's delta by 40%; within its floor.
    assert compare_to_reference(reference, run("c", 1.4e-4, 4e-3)).passed
    # The resolved group still has a tight tolerance: a 10% change fails.
    failures = {check.name for check in compare_to_reference(reference, run("c", 1e-4, 4.4e-3)).failures}
    assert "params/matrix/delta_norm" in failures and "params/norm/delta_norm" not in failures
