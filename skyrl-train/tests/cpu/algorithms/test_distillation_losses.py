"""Tests for distillation utility functions."""

import pytest
import torch
import torch.nn.functional as F

from skyrl_train.utils.distillation_utils import (
    best_of_n_select,
    reconstruct_sparse_log_probs,
    forward_kl_from_sparse,
    reverse_kl_from_sparse,
    jsd_from_sparse,
    kl_from_sparse_teacher_logprobs,
)


class TestBestOfNSelect:
    def test_basic_selection(self):
        rewards = [0.1, 0.5, 0.3, 0.2, 0.8, 0.4]
        indices = best_of_n_select(rewards, n_samples_per_prompt=3)
        assert indices == [1, 4]  # 0.5 from group 0, 0.8 from group 1

    def test_single_sample_per_prompt(self):
        rewards = [0.1, 0.5, 0.3]
        indices = best_of_n_select(rewards, n_samples_per_prompt=1)
        assert indices == [0, 1, 2]

    def test_all_same_rewards(self):
        rewards = [0.5, 0.5, 0.5, 0.5]
        indices = best_of_n_select(rewards, n_samples_per_prompt=2)
        # Should pick first in each group (max returns first occurrence)
        assert len(indices) == 2
        assert indices[0] in [0, 1]
        assert indices[1] in [2, 3]

    def test_negative_rewards(self):
        rewards = [-0.5, -0.1, -0.3, -0.8]
        indices = best_of_n_select(rewards, n_samples_per_prompt=2)
        assert indices == [1, 2]  # -0.1 and -0.3

    def test_n_equals_total(self):
        rewards = [0.1, 0.5, 0.3, 0.8]
        indices = best_of_n_select(rewards, n_samples_per_prompt=4)
        assert indices == [3]

    def test_invalid_length(self):
        with pytest.raises(AssertionError):
            best_of_n_select([0.1, 0.2, 0.3], n_samples_per_prompt=2)


class TestReconstructSparseLogProbs:
    def test_basic_reconstruction(self):
        B, S, K, V = 2, 3, 2, 5
        top_k_logprobs = torch.tensor([
            [[-0.5, -1.0], [-0.3, -0.7], [-0.1, -0.9]],
            [[-0.2, -0.8], [-0.4, -0.6], [-0.5, -1.5]],
        ])
        top_k_indices = torch.tensor([
            [[0, 2], [1, 3], [4, 0]],
            [[3, 1], [2, 4], [0, 3]],
        ])
        full = reconstruct_sparse_log_probs(top_k_logprobs, top_k_indices, V)
        assert full.shape == (B, S, V)
        # Check that specified positions have correct values
        assert full[0, 0, 0].item() == pytest.approx(-0.5)
        assert full[0, 0, 2].item() == pytest.approx(-1.0)
        # Check that unspecified positions are -inf
        assert full[0, 0, 1].item() == float("-inf")
        assert full[0, 0, 3].item() == float("-inf")

    def test_shape(self):
        B, S, K, V = 4, 8, 16, 1000
        top_k_logprobs = torch.randn(B, S, K)
        top_k_indices = torch.randint(0, V, (B, S, K))
        full = reconstruct_sparse_log_probs(top_k_logprobs, top_k_indices, V)
        assert full.shape == (B, S, V)


class TestKLFromSparse:
    """Test KL divergence computation from sparse teacher logprobs."""

    @pytest.fixture
    def setup_distributions(self):
        """Create known teacher and student distributions for testing."""
        B, S, V = 2, 3, 10
        K = 5  # keep top-5

        # Create teacher logits and compute full distribution
        teacher_logits = torch.randn(B, S, V)
        teacher_log_probs_full = F.log_softmax(teacher_logits, dim=-1)

        # Get top-K from teacher
        top_k_values, top_k_indices = teacher_log_probs_full.topk(K, dim=-1)

        # Create student distribution
        student_logits = torch.randn(B, S, V)
        student_log_probs_full = F.log_softmax(student_logits, dim=-1)

        return {
            "teacher_top_k_logprobs": top_k_values,
            "teacher_top_k_indices": top_k_indices,
            "teacher_log_probs_full": teacher_log_probs_full,
            "student_log_probs_full": student_log_probs_full,
            "B": B, "S": S, "V": V, "K": K,
        }

    def test_forward_kl_shape(self, setup_distributions):
        d = setup_distributions
        kl = forward_kl_from_sparse(
            d["teacher_top_k_logprobs"], d["teacher_top_k_indices"],
            d["student_log_probs_full"],
        )
        assert kl.shape == (d["B"], d["S"])

    def test_forward_kl_non_negative(self, setup_distributions):
        d = setup_distributions
        kl = forward_kl_from_sparse(
            d["teacher_top_k_logprobs"], d["teacher_top_k_indices"],
            d["student_log_probs_full"],
        )
        # Forward KL should be non-negative (up to floating point error)
        assert (kl >= -1e-6).all()

    def test_forward_kl_zero_when_same(self):
        """Forward KL should be ~0 when teacher == student."""
        B, S, V, K = 1, 2, 8, 8  # K == V, so no approximation
        logits = torch.randn(B, S, V)
        log_probs = F.log_softmax(logits, dim=-1)
        top_k_values, top_k_indices = log_probs.topk(K, dim=-1)

        kl = forward_kl_from_sparse(top_k_values, top_k_indices, log_probs)
        assert torch.allclose(kl, torch.zeros_like(kl), atol=1e-5)

    def test_forward_kl_matches_dense_when_k_equals_v(self):
        """When K == V, sparse forward KL should match dense computation."""
        B, S, V = 1, 2, 8
        teacher_logits = torch.randn(B, S, V)
        student_logits = torch.randn(B, S, V)
        teacher_log_probs = F.log_softmax(teacher_logits, dim=-1)
        student_log_probs = F.log_softmax(student_logits, dim=-1)
        teacher_probs = teacher_log_probs.exp()

        # Dense KL
        dense_kl = (teacher_probs * (teacher_log_probs - student_log_probs)).sum(dim=-1)

        # Sparse KL with K == V
        top_k_values, top_k_indices = teacher_log_probs.topk(V, dim=-1)
        sparse_kl = forward_kl_from_sparse(top_k_values, top_k_indices, student_log_probs)

        assert torch.allclose(dense_kl, sparse_kl, atol=1e-5)

    def test_reverse_kl_shape(self, setup_distributions):
        d = setup_distributions
        kl = reverse_kl_from_sparse(
            d["teacher_top_k_logprobs"], d["teacher_top_k_indices"],
            d["student_log_probs_full"],
        )
        assert kl.shape == (d["B"], d["S"])

    def test_reverse_kl_non_negative_when_k_equals_v(self):
        """Reverse KL is non-negative when K == V (exact, not approximated)."""
        B, S, V, K = 1, 2, 8, 8
        teacher_logits = torch.randn(B, S, V)
        student_logits = torch.randn(B, S, V)
        teacher_log_probs = F.log_softmax(teacher_logits, dim=-1)
        student_log_probs = F.log_softmax(student_logits, dim=-1)
        top_k_values, top_k_indices = teacher_log_probs.topk(K, dim=-1)

        kl = reverse_kl_from_sparse(top_k_values, top_k_indices, student_log_probs)
        assert (kl >= -1e-5).all()

    def test_jsd_shape(self, setup_distributions):
        d = setup_distributions
        jsd = jsd_from_sparse(
            d["teacher_top_k_logprobs"], d["teacher_top_k_indices"],
            d["student_log_probs_full"],
        )
        assert jsd.shape == (d["B"], d["S"])

    def test_jsd_non_negative(self, setup_distributions):
        d = setup_distributions
        jsd = jsd_from_sparse(
            d["teacher_top_k_logprobs"], d["teacher_top_k_indices"],
            d["student_log_probs_full"],
        )
        assert (jsd >= -1e-6).all()

    def test_jsd_bounded_by_log2(self, setup_distributions):
        """JSD is bounded by log(2) for alpha=0.5."""
        d = setup_distributions
        jsd = jsd_from_sparse(
            d["teacher_top_k_logprobs"], d["teacher_top_k_indices"],
            d["student_log_probs_full"],
        )
        assert (jsd <= torch.log(torch.tensor(2.0)) + 1e-5).all()

    def test_jsd_zero_when_same(self):
        """JSD should be ~0 when teacher == student."""
        B, S, V, K = 1, 2, 8, 8
        logits = torch.randn(B, S, V)
        log_probs = F.log_softmax(logits, dim=-1)
        top_k_values, top_k_indices = log_probs.topk(K, dim=-1)

        jsd = jsd_from_sparse(top_k_values, top_k_indices, log_probs)
        assert torch.allclose(jsd, torch.zeros_like(jsd), atol=1e-5)

    def test_loss_mask(self, setup_distributions):
        d = setup_distributions
        mask = torch.zeros(d["B"], d["S"])
        mask[:, 0] = 1.0  # only first position

        kl = forward_kl_from_sparse(
            d["teacher_top_k_logprobs"], d["teacher_top_k_indices"],
            d["student_log_probs_full"], loss_mask=mask,
        )
        # Masked positions should be zero
        assert (kl[:, 1:] == 0).all()

    def test_unified_interface(self, setup_distributions):
        d = setup_distributions
        args = (d["teacher_top_k_logprobs"], d["teacher_top_k_indices"], d["student_log_probs_full"])

        fwd = kl_from_sparse_teacher_logprobs(*args, kl_type="forward")
        rev = kl_from_sparse_teacher_logprobs(*args, kl_type="reverse")
        jsd = kl_from_sparse_teacher_logprobs(*args, kl_type="jsd")

        assert fwd.shape == rev.shape == jsd.shape == (d["B"], d["S"])

        with pytest.raises(ValueError):
            kl_from_sparse_teacher_logprobs(*args, kl_type="invalid")
