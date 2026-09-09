import pytest
import torch

from skyrl_train.training_batch import TrainingInputBatch
from skyrl_train.trainer import RayPPOTrainer
from omegaconf import OmegaConf
from skyrl_train.utils.non_agentic_advantages import cap_truncated_advantages


def batch():
    data = TrainingInputBatch(
        {
            "advantages": torch.tensor([[2.0, 2.0, 0.0], [-2.0, -2.0, 0.0], [0.5, 0.5, 0.0]]),
            "loss_mask": torch.tensor([[1, 1, 0], [1, 1, 0], [1, 0, 0]]),
            "non_agentic_truncated": torch.tensor([True, True, False]),
        }
    )
    data.metadata = {}
    return data


def test_cap_after_normalization_preserves_negative_and_complete_actions():
    data = batch()
    result = cap_truncated_advantages(data, -1.0)
    assert torch.equal(result["advantages"], torch.tensor([[-1.0, -1.0, 0.0], [-2.0, -2.0, 0.0], [0.5, 0.5, 0.0]]))
    records = result.metadata["non_agentic_advantage_override"]
    assert [(r["pre_override"], r["post_override"]) for r in records] == [(2.0, -1.0), (-2.0, -2.0), (0.5, 0.5)]
    assert [r["policy_action_count"] for r in records] == [2, 2, 1]


def test_nonconstant_token_advantage_is_not_misreported_as_grpo():
    data = batch()
    data["advantages"][0, 1] = 3
    with pytest.raises(ValueError, match="constant"):
        cap_truncated_advantages(data, -1)


@pytest.mark.parametrize("cap", [0, 1, float("nan"), float("inf")])
def test_invalid_cap_rejected(cap):
    with pytest.raises(ValueError, match="finite and negative"):
        cap_truncated_advantages(batch(), cap)


@pytest.mark.parametrize("cap", [None, -1.0])
def test_trainer_finalization_applies_optional_cap_after_normalization(cap):
    trainer = RayPPOTrainer.__new__(RayPPOTrainer)
    trainer.cfg = OmegaConf.create(
        {
            "trainer": {
                "algorithm": {
                    "advantage_batch_normalize": True,
                    "non_agentic_truncated_advantage_cap": cap,
                }
            }
        }
    )
    data = TrainingInputBatch(
        {
            "advantages": torch.tensor([[3.0, 3.0], [-3.0, -3.0]]),
            "response_mask": torch.ones(2, 2),
            "loss_mask": torch.tensor([[1.0, 0.0], [1.0, 1.0]]),
            "rewards": torch.tensor([[0.0, 1.0], [0.0, 0.0]]),
            "non_agentic_truncated": torch.tensor([True, False]),
        }
    )
    data.metadata = {"uids": ["first", "second"]}
    finalized = trainer.finalize_advantages_for_training(data)
    # Mean zero and RMS three give exact normalized values +1/-1.
    # The forced non-action retains +1; only the first sampled action is capped.
    expected = torch.tensor([[1.0, 1.0], [-1.0, -1.0]])
    if cap is not None:
        expected[0, 0] = -1
        first = finalized.metadata["non_agentic_advantage_override"][0]
        assert first["pre_override"] == 1 and first["post_override"] == -1
    assert torch.equal(finalized["advantages"], expected)
    assert "rewards" not in finalized and "uids" not in finalized.metadata
