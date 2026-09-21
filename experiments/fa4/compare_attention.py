"""Compare pointwise attention outputs and gradients from two probe arms."""

from __future__ import annotations

import argparse
import json

import torch


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference")
    parser.add_argument("candidate")
    args = parser.parse_args()
    reference = torch.load(args.reference, map_location="cpu", weights_only=True)
    candidate = torch.load(args.candidate, map_location="cpu", weights_only=True)
    assert reference["result"]["shape"] == candidate["result"]["shape"], "shape/configuration mismatch"
    comparisons = {}
    for name in ("output", "dq", "dk", "dv"):
        actual = candidate[name]
        expected = reference[name]
        assert actual.shape == expected.shape, (name, actual.shape, expected.shape)
        difference = (actual - expected).abs()
        comparisons[name] = {
            "max_abs": difference.max().item(),
            "mean_abs": difference.mean().item(),
            "rms": difference.square().mean().sqrt().item(),
            "finite": bool(torch.isfinite(actual).all().item()),
        }
    print(json.dumps(comparisons, indent=2))


if __name__ == "__main__":
    main()
