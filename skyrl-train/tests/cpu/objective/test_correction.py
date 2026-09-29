import pytest
import torch
from omegaconf import OmegaConf

from skyrl_train.config.objective_spec import OffPolicyCorrection, load_correction
from skyrl_train.dataset.replay_buffer import NaiveReplayBuffer
from skyrl_train.objective.correction import compute_correction
from skyrl_train.training_batch import TrainingBatchIterator, TrainingInputBatch


@pytest.mark.parametrize(
    ("preset", "ratios", "mask", "expected", "truncated", "masked"),
    [
        ("tis", [[0.25, 1.5, 4.0]], [[1, 1, 1]], [[0.25, 1.5, 2.0]], 1 / 3, 0),
        ("icepop", [[0.25, 0.5, 1.5, 5.0, 8.0]], [[1] * 5], [[0, 0.5, 1.5, 5.0, 0]], 1 / 5, 2 / 5),
        (
            "seq_mask_tis",
            [[4.0, 0.25, float("nan")], [1.01, 1.02, 1.0]],
            [[1, 1, 0], [1, 1, 0]],
            [[2.0, 0.25, 0], [0, 0, 0]],
            1 / 4,
            1 / 2,
        ),
        (
            "outlier_mask",
            [[1e-4, 100, float("nan")], [1.0, 101, 1.0]],
            [[1, 1, 0], [1, 1, 1]],
            [[1, 1, 0], [0, 0, 0]],
            0,
            3 / 5,
        ),
    ],
)
def test_correction_presets_match_importance_weight_oracles(preset, ratios, mask, expected, truncated, masked):
    ratios = torch.tensor(ratios)
    mask = torch.tensor(mask)
    rollout = torch.full_like(ratios, -3, requires_grad=True)
    old = (ratios.log() + rollout).detach().requires_grad_()

    result = compute_correction(old, rollout, mask, load_correction(preset))

    rows, response_length = mask.shape
    batch = TrainingInputBatch(
        {
            "sequences": torch.ones(rows, response_length + 1, dtype=torch.long),
            "attention_mask": torch.ones(rows, response_length + 1, dtype=torch.long),
            "action_log_probs": old.detach(),
            "rollout_logprobs": rollout.detach(),
            "base_action_log_probs": None,
            "values": None,
            "returns": None,
            "advantages": torch.ones_like(old),
            "loss_mask": mask,
            "response_mask": torch.ones_like(mask),
            "correction_weights": result.weights,
        }
    )
    batch.metadata = {"response_length": response_length}
    replay = NaiveReplayBuffer(sample_batch_size=rows)
    for experience in TrainingBatchIterator(batch, sample_batch_size=1):
        replay.append(experience)
    restored = replay.collate_fn([replay[index] for index in range(len(replay))])

    expected = torch.tensor(expected, dtype=torch.float32)
    torch.testing.assert_close(torch.stack(restored.correction_weights), expected)
    assert not result.weights.requires_grad
    assert result.metrics["policy/correction/weight_mean"] == pytest.approx(expected.sum().item() / mask.sum().item())
    assert result.metrics["policy/correction/truncated_fraction"] == pytest.approx(truncated)
    assert result.metrics["policy/correction/masked_fraction"] == pytest.approx(masked)


@pytest.mark.parametrize("preset", ["none", "tis", "icepop", "seq_mask_tis", "outlier_mask"])
def test_correction_on_policy_is_identity_on_eligible_tokens(preset):
    logprobs = torch.tensor([[-1.0, float("nan"), -3.0], [float("nan")] * 3])
    mask = torch.tensor([[1, 0, 1], [0, 0, 0]])

    result = compute_correction(logprobs, logprobs, mask, load_correction(preset))

    torch.testing.assert_close(result.weights, mask.float())
    assert all(torch.isfinite(torch.tensor(value)) for value in result.metrics.values())


@pytest.mark.parametrize(
    ("aggregate", "expected"),
    [("geometric", [[2.0, 2.0, 0], [0, 0, 0]]), ("product", [[3.0, 3.0, 0], [0, 0, 0]])],
)
def test_sequence_truncation_broadcasts_only_over_trainable_tokens(aggregate, expected):
    correction = OffPolicyCorrection.from_config(
        OmegaConf.create(
            {"name": "custom", "rules": [{"kind": "sequence", "aggregate": aggregate, "action": "truncate", "high": 3}]}
        )
    )
    old = torch.tensor([[4.0, 1.0, float("nan")], [float("nan")] * 3]).log()
    mask = torch.tensor([[1, 1, 0], [0, 0, 0]])

    result = compute_correction(old, torch.zeros_like(old), mask, correction)

    torch.testing.assert_close(result.weights, torch.tensor(expected))
    assert result.metrics["policy/correction/truncated_fraction"] == (0 if aggregate == "geometric" else 1)


def test_sequence_product_clamps_log_sum_before_exponentiating():
    correction = OffPolicyCorrection.from_config(
        OmegaConf.create(
            {
                "name": "custom",
                "rules": [{"kind": "sequence", "aggregate": "product", "action": "truncate", "high": 1e20}],
            }
        )
    )
    old = torch.tensor([[100.0, 100.0], [-100.0, -100.0]])

    result = compute_correction(old, torch.zeros_like(old), torch.ones_like(old), correction)

    torch.testing.assert_close(result.weights, torch.tensor([[20.0, 20.0], [-20.0, -20.0]]).exp())
