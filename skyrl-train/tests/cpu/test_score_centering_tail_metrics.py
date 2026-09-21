import math

import torch

from skyrl_train.utils.score_centering import masked_topk_tail_mass


def test_tail_mass_ignores_masked_nonfinite_logprobs():
    logprobs = torch.tensor([[[math.log(0.6), math.log(0.2)], [float("nan"), float("inf")]]])
    mask = torch.tensor([[True, False]])

    tail = masked_topk_tail_mass(logprobs, mask)

    torch.testing.assert_close(tail, torch.tensor([[0.2, 0.0]]))
