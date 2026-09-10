"""Scalar oracles independent of the production vectorized mask implementation.

Band semantics match verl d040717b's rollout-correction integration anchor;
the row veto is the separately specified INTELLECT-3 threshold. The M2 mask
enumerates every harmful prefix and does not reuse production tensor logic.
"""

import math

import torch


def offpolicy_keep_reference(delta, selected, low=0.5, high=5.0, veto=1e-5):
    keep = []
    for row, valid in zip(delta.detach().cpu().tolist(), selected.cpu().tolist(), strict=True):
        ratios = [math.exp(max(-20.0, min(20.0, value))) for value in row]
        vetoed = any(flag and ratio < veto for ratio, flag in zip(ratios, valid, strict=True))
        keep.append(
            [bool(flag and not vetoed and low <= ratio <= high) for ratio, flag in zip(ratios, valid, strict=True)]
        )
    return torch.tensor(keep, dtype=torch.bool, device=delta.device)
