import pytest
import torch

from rolloutengine.closure_repair import ClosureTargetKind
from skyrl_gym.verification import VerificationResult
from skyrl_train.closure_training import closure_training_targets


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
