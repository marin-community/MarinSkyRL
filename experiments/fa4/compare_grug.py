"""Compare matched Grug policy outputs and optimizer-induced weight changes."""

from __future__ import annotations

import argparse
import json

import torch


def difference(actual: torch.Tensor, expected: torch.Tensor) -> dict[str, float | bool]:
    assert actual.shape == expected.shape, (actual.shape, expected.shape)
    delta = (actual.float() - expected.float()).abs()
    return {
        "max_abs": delta.max().item(),
        "mean_abs": delta.mean().item(),
        "rms": delta.square().mean().sqrt().item(),
        "finite": bool(torch.isfinite(actual).all().item()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference")
    parser.add_argument("candidate")
    args = parser.parse_args()
    reference = torch.load(args.reference, map_location="cpu", weights_only=True)
    candidate = torch.load(args.candidate, map_location="cpu", weights_only=True)
    config_keys = (
        "world_size",
        "context_parallel_size",
        "sample_packing",
        "cp_comm_type",
        "shape",
        "prompt_length",
        "response_length",
        "steps",
    )
    assert all(reference["result"][key] == candidate["result"][key] for key in config_keys)
    assert torch.equal(reference["response_mask"], candidate["response_mask"])
    assert torch.equal(reference["reference_logprobs"], candidate["reference_logprobs"])
    valid = reference["response_mask"].bool()
    comparisons = {
        "before_logprobs": difference(candidate["before_logprobs"][valid], reference["before_logprobs"][valid]),
        "after_logprobs": difference(candidate["after_logprobs"][valid], reference["after_logprobs"][valid]),
    }
    for name in reference["before_weights"]:
        assert name in candidate["before_weights"]
        comparisons[f"initial/{name}"] = difference(
            candidate["before_weights"][name], reference["before_weights"][name]
        )
        candidate_change = candidate["after_weights"][name] - candidate["before_weights"][name]
        reference_change = reference["after_weights"][name] - reference["before_weights"][name]
        comparisons[f"update/{name}"] = difference(candidate_change, reference_change)
    print(json.dumps(comparisons, indent=2))


if __name__ == "__main__":
    main()
