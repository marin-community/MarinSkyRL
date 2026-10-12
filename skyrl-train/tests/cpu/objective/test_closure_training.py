import pytest
import torch

from rolloutengine.closure_repair import ClosureTargetKind
from skyrl_gym.verification import VerificationResult
from skyrl_train.closure_training import ClosureTrainingTargets, closure_target_metrics, closure_training_targets
from skyrl_train.ftpo import VerifiedFTPOBoundary


@pytest.mark.parametrize("terminator", ["<|eot_id|>", "<|end_of_text|>"])
@pytest.mark.parametrize("continuation,repair", [("", "]}"), ("]", "}"), ("]}", None)])
def test_stopped_turn_repairs_only_missing_closure_tokens(terminator, continuation, repair):
    terminal_id = 1_000_000

    def decode(tokens):
        return "".join(terminator if token == terminal_id else chr(token) for token in tokens)

    def encode(text):
        return list(map(ord, text))

    prefix = '{"a":[1'
    prompt = encode(prefix)
    response = encode(continuation) + [terminal_id]
    verification = VerificationResult.verified(
        float(repair is None),
        diagnostics={
            "stop_reason": "stop",
            "closure": {
                "prefix": prefix,
                "expected": '{"a":[1]}',
                "format": "json",
                "family": "json_nested",
                "control": "close_required",
            },
        },
    )
    targets = closure_training_targets(
        [prompt],
        [response],
        [[1] * len(response)],
        torch.full((1, len(response), 1), -1),
        [verification],
        eligible=[True],
        encode=encode,
        decode=decode,
        eos_token_ids=frozenset(),
    )
    assert targets.boundaries == [None]
    if repair is None:
        assert targets.repairs == [None]
    else:
        target = targets.repairs[0]
        assert target.kind == ClosureTargetKind.MISSING_SUFFIX
        assert target.response_position == len(continuation)
        assert decode(target.repair_token_ids) == repair


def test_target_metrics_separate_candidates_filtered_ftpo_and_repairs():
    verification = VerificationResult.verified(
        0.0, diagnostics={"closure": {"family": "json_nested", "control": "close_required"}}
    )
    continuation = VerificationResult.verified(
        0.0, diagnostics={"closure": {"family": "code_fence", "control": "continue_required"}}
    )
    targets = ClosureTrainingTargets(
        boundaries=[
            VerifiedFTPOBoundary(0, (7,)),
            VerifiedFTPOBoundary(0, (7,)),
            VerifiedFTPOBoundary(0, ()),
            None,
            None,
        ],
        repairs=[None] * 5,
    )
    chosen = torch.zeros((5, 2, 1), dtype=torch.bool)
    chosen[0, 0, 0] = True
    chosen[1, 0, 0] = True
    loss_mask = torch.tensor([[1, 0], [0, 0], [0, 0], [1, 1], [0, 0]])
    repair_mask = torch.zeros((5, 2), dtype=torch.bool)
    repair_mask[3] = True
    metrics = closure_target_metrics(
        [verification, verification, continuation, verification, None], targets, chosen, loss_mask, repair_mask
    )
    prefix = "closure/targets/json_nested/close_required"
    assert metrics == {
        f"{prefix}/rows": 3,
        f"{prefix}/verified_boundaries": 2,
        f"{prefix}/boundaries_with_candidates": 2,
        f"{prefix}/selected_ftpo_tokens": 1,
        f"{prefix}/repair_tokens": 2,
        "closure/targets/code_fence/continue_required/rows": 1,
        "closure/targets/code_fence/continue_required/verified_boundaries": 1,
        "closure/targets/code_fence/continue_required/boundaries_with_candidates": 0,
        "closure/targets/code_fence/continue_required/selected_ftpo_tokens": 0,
        "closure/targets/code_fence/continue_required/repair_tokens": 0,
    }
