import pytest
import torch

from skyrl_train.utils import torch_utils
from skyrl_train.utils.torch_utils import chunked_cross_entropy_from_log_probs, chunked_entropy_from_logits

ATTENTION_MASK = torch.tensor([[1.0, 1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 0.0, 0.0, 0.0]])


@pytest.fixture
def logits():
    # Five positions with a chunk size of two exercises a partial final chunk.
    return torch.randn(2, 5, 7, generator=torch.Generator().manual_seed(0), dtype=torch.float64)


@pytest.fixture(autouse=True)
def small_chunks(monkeypatch):
    monkeypatch.setattr(torch_utils, "CHUNK_SIZE", 2)


def _reference_entropy(logits):
    return torch.distributions.Categorical(logits=logits).entropy()


def test_chunked_cross_entropy_from_log_probs_matches_categorical_entropy(logits):
    result = chunked_cross_entropy_from_log_probs(torch.log_softmax(logits, dim=-1))
    torch.testing.assert_close(result, _reference_entropy(logits))


@pytest.mark.parametrize("requires_grad", [False, True])
@pytest.mark.parametrize("attention_mask", [None, ATTENTION_MASK], ids=["unmasked", "masked"])
def test_chunked_entropy_from_logits_matches_categorical_entropy(logits, requires_grad, attention_mask):
    result = chunked_entropy_from_logits(logits, requires_grad=requires_grad, attention_mask=attention_mask)

    expected = _reference_entropy(logits)
    if attention_mask is not None:
        expected = expected * attention_mask
    torch.testing.assert_close(result.detach(), expected)
