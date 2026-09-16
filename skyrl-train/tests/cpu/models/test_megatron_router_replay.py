"""CPU tests for the megatron-free half of Megatron router replay (R3).

Covers the layer mapping, the sequence-major flatten, the TP sequence-parallel
slice, the per-layer controller state machine (forward / recompute FIFO), the
geometry validation, and the dense-target builder shared with the FSDP path.
"""

import numpy as np
import pytest
import torch

from skyrl_train.models.megatron_router_replay import (
    LayerReplayHandle,
    MegatronRouterReplay,
    capture_layer_indices,
    expand_moe_layer_freq,
    num_moe_layers,
    sequence_major_flatten,
    slice_sequence_parallel,
    validate_replay_geometry,
)
from skyrl_train.models.router_replay import SENTINEL_EXPERT_ID, dense_replay_targets


def _fake_compute_topk(scores, topk, num_groups=None, group_topk=None):
    """Deterministic stand-in for mcore's _compute_topk: last-k columns reversed."""
    indices = torch.arange(scores.shape[1] - topk, scores.shape[1], device=scores.device).expand(
        scores.shape[0], -1
    ).flip(-1)
    return scores.gather(1, indices), indices


class TestCaptureLayerIndices:
    def test_maps_one_indexed_layer_numbers_to_capture_indices(self):
        assert capture_layer_indices([0, 1, 1, 0, 1]) == {2: 0, 3: 1, 5: 2}
        assert num_moe_layers([0, 1, 1, 0, 1]) == 3

    def test_all_moe_pattern_is_identity_shift(self):
        assert capture_layer_indices([1, 1, 1, 1]) == {1: 0, 2: 1, 3: 2, 4: 3}

    def test_dense_only_pattern_is_empty(self):
        assert capture_layer_indices([0, 0, 0]) == {}
        assert num_moe_layers([0, 0, 0]) == 0


class TestSequenceMajorFlatten:
    def test_round_trip_indexing_matches_s_times_b_plus_b(self):
        torch.manual_seed(0)
        b, s, response_len, l, k = 3, 5, 2, 4, 2
        rollout = torch.randint(0, 8, (b, response_len, l, k))
        dense, _ = dense_replay_targets(rollout, b, s, response_len)
        flat = sequence_major_flatten(dense)
        assert flat.shape == (s * b, l, k)
        for _ in range(20):
            bi = int(torch.randint(0, b, ()))
            si = int(torch.randint(0, s, ()))
            assert torch.equal(flat[si * b + bi], dense[bi, si])


class TestSliceSequenceParallel:
    def test_concatenated_rank_slices_reproduce_full_tensor(self):
        torch.manual_seed(1)
        seq_len, b, tp = 4, 3, 2
        flat = torch.randn(seq_len * b, 7)
        r0 = slice_sequence_parallel(flat, seq_len=seq_len, batch_size=b, tp_rank=0, tp_size=tp)
        r1 = slice_sequence_parallel(flat, seq_len=seq_len, batch_size=b, tp_rank=1, tp_size=tp)
        assert r0.shape == (seq_len // tp * b, 7)
        assert torch.equal(torch.cat([r0, r1], dim=0), flat)

    def test_non_divisible_seq_len_raises(self):
        flat = torch.zeros(3 * 2, 1)
        with pytest.raises(ValueError, match="divis"):
            slice_sequence_parallel(flat, seq_len=3, batch_size=2, tp_rank=0, tp_size=2)


def _masked_target_rows(n, topk, num_experts, n_masked, device="cpu"):
    # Deterministic non-zero targets so the all-K-sentinel convention stays unambiguous.
    targets = torch.arange(1, topk + 1, device=device).expand(n, topk).contiguous() % num_experts
    mask = torch.zeros(n, dtype=torch.bool, device=device)
    mask[:n_masked] = True
    return targets, mask


class TestControllerSingleForward:
    def test_masked_rows_replay_targets_unmasked_rows_stay_native(self):
        torch.manual_seed(2)
        n, num_experts, topk = 6, 8, 2
        scores = torch.randn(n, num_experts)
        targets, mask = _masked_target_rows(n, topk, num_experts, n_masked=3)
        native_values, native_idx = _fake_compute_topk(scores, topk)

        controller = MegatronRouterReplay(local_layer_indices=[0], recompute_enabled=False)
        handle = LayerReplayHandle(controller, layer_idx=0)
        controller.begin_forward({0: targets}, mask)
        probs, idx = handle.get_replay_topk(scores, topk, None, None, _fake_compute_topk)
        controller.end_forward()

        assert torch.equal(idx[mask], targets[mask])
        assert torch.equal(idx[~mask], native_idx[~mask])
        # probs are the live scores gathered at the returned indices (the R3 crux)
        assert torch.allclose(probs, scores.gather(1, idx))
        controller.assert_drained()

    def test_probs_carry_gradient_through_the_gate(self):
        scores = torch.randn(4, 8, requires_grad=True)
        targets = torch.tensor([[1, 2], [3, 4], [5, 6], [0, 1]])
        mask = torch.tensor([True, True, True, True])
        controller = MegatronRouterReplay(local_layer_indices=[0], recompute_enabled=False)
        handle = LayerReplayHandle(controller, layer_idx=0)
        controller.begin_forward({0: targets}, mask)
        probs, idx = handle.get_replay_topk(scores, 2, None, None, _fake_compute_topk)
        controller.end_forward()
        probs.sum().backward()
        assert scores.grad is not None and torch.isfinite(scores.grad).all()

    def test_layer_consumed_twice_in_one_forward_raises(self):
        scores = torch.randn(4, 8)
        targets, mask = _masked_target_rows(4, 2, 8, 4)
        controller = MegatronRouterReplay(local_layer_indices=[0], recompute_enabled=False)
        handle = LayerReplayHandle(controller, layer_idx=0)
        controller.begin_forward({0: targets}, mask)
        handle.get_replay_topk(scores, 2, None, None, _fake_compute_topk)
        with pytest.raises(ValueError, match="layer 0"):
            handle.get_replay_topk(scores, 2, None, None, _fake_compute_topk)
        controller.end_forward()

    def test_begin_forward_while_in_forward_raises(self):
        scores = torch.randn(4, 8)
        targets, mask = _masked_target_rows(4, 2, 8, 4)
        controller = MegatronRouterReplay(local_layer_indices=[0], recompute_enabled=False)
        handle = LayerReplayHandle(controller, layer_idx=0)
        controller.begin_forward({0: targets}, mask)
        handle.get_replay_topk(scores, 2, None, None, _fake_compute_topk)
        with pytest.raises(RuntimeError, match="IDLE"):
            controller.begin_forward({0: targets}, mask)
        controller.end_forward()
        assert controller.pop_metrics()["hit_fraction"] == 1.0


class TestControllerRecomputeFifo:
    def test_pp_interleave_serves_each_recompute_with_its_own_micro_batch(self):
        scores = torch.randn(4, 8)
        layer = 0
        controller = MegatronRouterReplay(local_layer_indices=[layer], recompute_enabled=True)
        handle = LayerReplayHandle(controller, layer_idx=layer)

        mb = []
        for value in range(3):
            targets = torch.full((4, 2), value, dtype=torch.long)
            mask = torch.ones(4, dtype=torch.bool)
            mb.append((targets, mask))

        def forward(i):
            controller.begin_forward({layer: mb[i][0]}, mb[i][1])
            probs, idx = handle.get_replay_topk(scores, 2, None, None, _fake_compute_topk)
            controller.end_forward()
            return idx

        def recompute(i):
            _, idx = handle.get_replay_topk(scores, 2, None, None, _fake_compute_topk)
            return idx

        f1 = forward(0)
        assert torch.equal(f1, mb[0][0])
        f2 = forward(1)
        assert torch.equal(f2, mb[1][0])
        r1 = recompute(0)
        assert torch.equal(r1, mb[0][0]), "first recompute must replay the FIRST forward"
        f3 = forward(2)
        assert torch.equal(f3, mb[2][0])
        r2 = recompute(1)
        assert torch.equal(r2, mb[1][0])
        r3 = recompute(2)
        assert torch.equal(r3, mb[2][0])
        controller.assert_drained()

    def test_recompute_with_empty_fifo_raises(self):
        scores = torch.randn(4, 8)
        controller = MegatronRouterReplay(local_layer_indices=[0], recompute_enabled=True)
        handle = LayerReplayHandle(controller, layer_idx=0)
        with pytest.raises(RuntimeError, match="recompute without a recorded forward"):
            handle.get_replay_topk(scores, 2, None, None, _fake_compute_topk)

    def test_assert_drained_with_outstanding_recompute_names_the_layer(self):
        scores = torch.randn(4, 8)
        controller = MegatronRouterReplay(local_layer_indices=[0], recompute_enabled=True)
        handle = LayerReplayHandle(controller, layer_idx=0)
        targets, mask = _masked_target_rows(4, 2, 8, 4)
        controller.begin_forward({0: targets}, mask)
        handle.get_replay_topk(scores, 2, None, None, _fake_compute_topk)
        controller.end_forward()
        with pytest.raises(RuntimeError, match="layer 0 has 1 outstanding"):
            controller.assert_drained()

    def test_without_recompute_the_fifo_never_grows(self):
        scores = torch.randn(4, 8)
        controller = MegatronRouterReplay(local_layer_indices=[0], recompute_enabled=False)
        handle = LayerReplayHandle(controller, layer_idx=0)
        targets, mask = _masked_target_rows(4, 2, 8, 4)
        controller.begin_forward({0: targets}, mask)
        handle.get_replay_topk(scores, 2, None, None, _fake_compute_topk)
        controller.end_forward()
        controller.assert_drained()
        with pytest.raises(RuntimeError, match="recompute without a recorded forward"):
            handle.get_replay_topk(scores, 2, None, None, _fake_compute_topk)


class TestEndForward:
    def test_unconsumed_bracket_layer_raises(self):
        scores = torch.randn(4, 8)
        t0, mask = _masked_target_rows(4, 2, 8, 4)
        t1, _ = _masked_target_rows(4, 2, 8, 4)
        controller = MegatronRouterReplay(local_layer_indices=[0, 1], recompute_enabled=False)
        handle0 = LayerReplayHandle(controller, layer_idx=0)
        handle1 = LayerReplayHandle(controller, layer_idx=1)
        controller.begin_forward({0: t0, 1: t1}, mask)
        handle0.get_replay_topk(scores, 2, None, None, _fake_compute_topk)
        with pytest.raises(ValueError, match=r"layer\(s\) \[1\] never fired"):
            controller.end_forward()


class TestMetrics:
    def test_pop_metrics_reports_hit_and_sentinel_fractions_then_resets(self):
        torch.manual_seed(3)
        scores = torch.randn(8, 8)
        targets, mask = _masked_target_rows(8, 2, 8, n_masked=4)
        targets[2] = SENTINEL_EXPERT_ID  # masked row whose target row is sentinel: a layout bug
        response_mask = torch.zeros(8, dtype=torch.bool)
        response_mask[:6] = True  # two response rows fell through to native (capture loss)
        controller = MegatronRouterReplay(local_layer_indices=[0], recompute_enabled=False)
        handle = LayerReplayHandle(controller, layer_idx=0)
        controller.begin_forward({0: targets}, mask, response_mask)
        handle.get_replay_topk(scores, 2, None, None, _fake_compute_topk)
        controller.end_forward()
        metrics = controller.pop_metrics()
        assert metrics["hit_fraction"] == pytest.approx(3 / 4)
        assert metrics["sentinel_fraction"] == pytest.approx(2 / 6)
        assert controller.pop_metrics() == {"hit_fraction": 1.0, "sentinel_fraction": 0.0}


class TestValidateReplayGeometry:
    def _kwargs(self, **overrides):
        num_experts = 8
        targets = torch.randint(0, num_experts, (2, 5, 3, 2))
        kwargs = dict(
            num_layers_captured=3,
            expected_moe_layers=3,
            topk_captured=2,
            expected_topk=2,
            num_experts=num_experts,
            targets=targets,
            response_len=5,
            num_actions=5,
        )
        kwargs.update(overrides)
        return kwargs

    def test_valid_geometry_passes(self):
        validate_replay_geometry(**self._kwargs())

    def test_layer_count_mismatch_names_quantities(self):
        with pytest.raises(ValueError, match="expected_moe_layers"):
            validate_replay_geometry(**self._kwargs(num_layers_captured=2))

    def test_topk_mismatch_names_quantities(self):
        with pytest.raises(ValueError, match="expected_topk"):
            validate_replay_geometry(**self._kwargs(topk_captured=4))

    def test_topk_below_two_is_rejected(self):
        with pytest.raises(ValueError, match="expected_topk"):
            validate_replay_geometry(**self._kwargs(expected_topk=1, topk_captured=1))

    def test_out_of_range_expert_id_names_quantities(self):
        targets = torch.randint(0, 8, (2, 5, 3, 2))
        targets[0, 0, 0, 0] = 8
        with pytest.raises(ValueError, match="num_experts"):
            validate_replay_geometry(**self._kwargs(targets=targets))

    def test_response_len_mismatch_names_quantities(self):
        with pytest.raises(ValueError, match="response_len"):
            validate_replay_geometry(**self._kwargs(num_actions=4))

    def test_list_num_actions_raises_not_implemented(self):
        with pytest.raises(NotImplementedError, match="scalar num_actions"):
            validate_replay_geometry(**self._kwargs(num_actions=[5, 5]))


class TestExpandMoeLayerFreq:
    def test_integer_freq_matches_mcore_every_n_layers(self):
        # Mirrors gpt_layer_specs.py: 1 if i % freq == 0 for 0-based i.
        assert expand_moe_layer_freq(2, 6) == [1, 0, 1, 0, 1, 0]
        assert expand_moe_layer_freq(1, 3) == [1, 1, 1]

    def test_list_freq_is_passed_through(self):
        pattern = [0, 1, 1, 0, 1]
        assert expand_moe_layer_freq(pattern, 5) == pattern

    def test_list_freq_length_mismatch_raises(self):
        with pytest.raises(ValueError, match="moe_layer_freq"):
            expand_moe_layer_freq([0, 1], 5)

    def test_unsupported_freq_type_raises(self):
        with pytest.raises(ValueError, match="moe_layer_freq"):
            expand_moe_layer_freq("every other", 4)


class TestDenseReplayTargets:
    def _reference_dense(self, rollout, batch_size, seq_len, num_actions):
        """Inline copy of the pre-refactor builder in HFModelWrapper."""
        device = rollout.device
        re = rollout.to(device=device, dtype=torch.long)
        _, response_len, _, _ = re.shape
        full = torch.full((batch_size, seq_len, re.shape[2], re.shape[3]), SENTINEL_EXPERT_ID, dtype=torch.long, device=device)
        full[:, seq_len - response_len : seq_len, :, :] = re
        response_pos = torch.zeros(batch_size, seq_len, dtype=torch.bool, device=device)
        response_pos[:, seq_len - response_len : seq_len] = True
        non_sentinel = (full != SENTINEL_EXPERT_ID).any(dim=-1).all(dim=-1)
        return full, response_pos & non_sentinel

    def test_matches_pre_refactor_builder_on_random_inputs(self):
        torch.manual_seed(4)
        for batch_size, seq_len, response_len, num_layers, topk, num_experts in [
            (3, 12, 5, 4, 2, 8),
            (1, 8, 8, 1, 4, 6),
            (2, 10, 3, 2, 2, 4),
        ]:
            rollout = torch.randint(0, num_experts, (batch_size, response_len, num_layers, topk))
            if batch_size >= 2:
                rollout[1] = SENTINEL_EXPERT_ID  # fully-lost capture for one sample
            got_full, got_mask = dense_replay_targets(rollout, batch_size, seq_len, response_len)
            want_full, want_mask = self._reference_dense(rollout, batch_size, seq_len, response_len)
            assert torch.equal(got_full, want_full)
            assert torch.equal(got_mask, want_mask)

    def test_list_num_actions_raises_not_implemented(self):
        rollout = torch.zeros(2, 5, 3, 2, dtype=torch.long)
        with pytest.raises(NotImplementedError, match="scalar num_actions"):
            dense_replay_targets(rollout, 2, 8, num_actions=[5, 5])

    def test_batch_size_mismatch_raises(self):
        rollout = torch.zeros(2, 5, 3, 2, dtype=torch.long)
        with pytest.raises((ValueError, AssertionError)):
            dense_replay_targets(rollout, 3, 8, 5)


def test_module_has_no_megatron_imports():
    import ast

    import skyrl_train.models.megatron_router_replay as module

    tree = ast.parse(open(module.__file__).read())
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module)
    assert not [name for name in imported if name.split(".")[0] == "megatron"]
