import pytest
import torch

from skyrl_train.config.objective_spec import LossReduction
from skyrl_train.objective.reduction import policy_data_weights, reduce_to_step, step_counts


@pytest.mark.parametrize("mode", list(LossReduction))
@pytest.mark.parametrize("row", ["policy", "mask", "teacher"])
@pytest.mark.parametrize(("micro_batch", "dp_size"), [(1, 1), (2, 1), (4, 1), (1, 2), (2, 2)])
def test_step_reduction_matches_full_batch_for_any_split(mode, row, micro_batch, dp_size):
    mask = torch.tensor([[1, 0, 0, 0], [1, 1, 1, 0], [0, 0, 0, 0], [1, 1, 1, 1]], dtype=torch.float64)
    tags = torch.tensor([[1, 0, 0, 0], [1, 0, 1, 0], [0, 0, 0, 0], [1, 1, 0, 0]])
    policy = policy_data_weights(mask, tags, 0.25)
    teacher = mask * torch.tensor([[0, 0, 0, 0], [1, 0, 1, 0], [0, 0, 0, 0], [0, 1, 1, 0]])
    advantages = torch.tensor([[2, 99, 99, 99], [0, 0, 0, 99], [99, 99, 99, 99], [1, -1, 2, 2]])
    data = {"policy": policy, "mask": mask, "teacher": teacher}[row]
    numerator_weights = torch.tensor([[0, 9, 9, 9], [2, 0.5, 0.3, 9], [9, 9, 9, 9], [1, 0, 3, 0.1]])
    reference_values = torch.arange(1, 17, dtype=torch.float64).reshape(4, 4).requires_grad_()
    reference = reference_values.new_zeros(())
    token_count = sum(float(data[i, j]) for i in range(4) for j in range(4))
    row_count = sum(any(data[i, j] > 0 for j in range(4)) for i in range(4))
    nonzero_rows = sum(any(policy[i, j] > 0 and advantages[i, j] != 0 for j in range(4)) for i in range(4))
    for i in range(4):
        row_weight = sum(float(data[i, j]) for j in range(4))
        for j in range(4):
            if data[i, j] == 0:
                continue
            term = reference_values[i, j] * data[i, j] * numerator_weights[i, j]
            if mode is LossReduction.SEQUENCE_MEAN:
                term = term / row_weight
            reference = reference + term
    denominator = {
        LossReduction.TOKEN_MEAN: max(token_count, 1),
        LossReduction.SEQUENCE_MEAN: max(row_count, 1),
        LossReduction.SEQ_MEAN_TOKEN_SUM_NORM: max(row_count, 1) * 4,
        LossReduction.SEQ_MEAN_TOKEN_SUM_NORM_GLOBAL: max(nonzero_rows, 1) * 4,
    }[mode]
    reference = reference / denominator
    reference.backward()

    values = reference_values.detach().clone().requires_grad_()
    rank_rows = 4 // dp_size
    num_microbatches = rank_rows // micro_batch
    total = values.new_zeros(())
    for rank in range(dp_size):
        rank_start, rank_end = rank * rank_rows, (rank + 1) * rank_rows
        remote_counts = torch.zeros(7, dtype=torch.float64)
        for i in range(4):
            if rank_start <= i < rank_end:
                continue
            for offset, weights in ((0, policy), (2, mask), (4, teacher)):
                remote_counts[offset] += sum(float(weights[i, j]) for j in range(4))
                remote_counts[offset + 1] += int(any(weights[i, j] > 0 for j in range(4)))
            remote_counts[6] += int(any(policy[i, j] > 0 and advantages[i, j] != 0 for j in range(4)))
        chunks = [slice(i, i + micro_batch) for i in range(rank_start, rank_end, micro_batch)]
        counts = step_counts(
            [policy[c] for c in chunks],
            [mask[c] for c in chunks],
            [teacher[c] for c in chunks],
            [advantages[c] for c in chunks],
            max_seq_len=4,
            all_reduce_sum=lambda local: local + remote_counts,
        )
        row_counts = getattr(counts, row)
        for chunk in chunks:
            contribution = reduce_to_step(
                values[chunk],
                data[chunk],
                row_counts,
                mode,
                max_seq_len=counts.max_seq_len,
                nonzero_advantage_rows=counts.nonzero_advantage_rows,
                numerator_weights=numerator_weights[chunk],
            )
            loss_scale = num_microbatches * dp_size
            cp_size = 1
            total = total + contribution * loss_scale * cp_size / num_microbatches / (dp_size * cp_size)
    total.backward()
    torch.testing.assert_close(total, reference, rtol=1e-7, atol=1e-7)
    torch.testing.assert_close(values.grad, reference_values.grad, rtol=1e-7, atol=1e-7)


@pytest.mark.parametrize("mode", list(LossReduction))
def test_empty_and_masked_positions_have_zero_value_and_gradient(mode):
    weights = torch.tensor([[0.0, 0.0], [0.25, 0.0]])
    values = torch.tensor([[torch.nan, -1e9], [2.0, torch.nan]], requires_grad=True)
    advantages = torch.tensor([[99.0, 99.0], [0.0, torch.nan]])
    counts = step_counts([weights], [weights > 0], [], [advantages], 2, lambda value: value)
    result = reduce_to_step(
        values,
        weights,
        counts.policy,
        mode,
        max_seq_len=2,
        nonzero_advantage_rows=counts.nonzero_advantage_rows,
        numerator_weights=torch.tensor([[torch.nan, torch.nan], [3.0, torch.nan]]),
    )
    result.backward()
    expected = {
        LossReduction.TOKEN_MEAN: 1.5,
        LossReduction.SEQUENCE_MEAN: 6.0,
        LossReduction.SEQ_MEAN_TOKEN_SUM_NORM: 0.75,
        LossReduction.SEQ_MEAN_TOKEN_SUM_NORM_GLOBAL: 0.75,
    }[mode]
    assert result.item() == pytest.approx(expected)
    torch.testing.assert_close(values.grad, torch.tensor([[0.0, 0.0], [expected / 2, 0.0]]))
    assert counts.nonzero_advantage_rows == 0

    empty = torch.zeros_like(weights)
    counts = step_counts([empty], [empty], [], [advantages], 2, lambda value: value)
    empty_values = torch.full_like(weights, torch.nan, requires_grad=True)
    result = reduce_to_step(
        empty_values, empty, counts.policy, mode, max_seq_len=2, nonzero_advantage_rows=counts.nonzero_advantage_rows
    )
    result.backward()
    assert result.item() == 0
    torch.testing.assert_close(empty_values.grad, empty)
