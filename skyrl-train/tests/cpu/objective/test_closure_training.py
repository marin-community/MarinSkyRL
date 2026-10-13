import pytest
import torch

from skyrl_train.closure_repair import ClosureTargetKind
from skyrl_gym.verification import VerificationResult
from skyrl_train.closure_training import ClosureTrainingTargets, closure_target_metrics, closure_training_targets
from skyrl_train.ftpo import VerifiedFTPOBoundary, select_ftpo_candidates
from skyrl_train.config.ftpo import FTPOBoundary, FTPOConfig
from skyrl_train.trajectory_runners.trajectory_reward_shaping import LoopCreditConfig


@pytest.mark.parametrize("boundary", [FTPOBoundary.VERIFIED_CLOSURE, FTPOBoundary.VERIFIED_CLOSURE_AND_CONTINUATION])
@pytest.mark.parametrize("has_continuation", [False, True])
def test_early_closure_selects_only_recorded_continuations_at_one_token(boundary, has_continuation):
    pieces = {1_000_000: '"]', 1_000_001: "ultiple", 1_000_002: "ult", 1_000_003: "wrong"}

    def decode(tokens):
        return "".join(pieces[token] if token in pieces else chr(token) for token in tokens)

    prefix = '["m'
    prompt = list(map(ord, "assistant " + prefix))
    response = [1_000_000, ord(" ")]
    ids = [1_000_000, 1_000_001, 1_000_002, 1_000_003] if has_continuation else [1_000_000, 1_000_003, 120, 121]
    candidates = torch.tensor([[ids, ids]])
    verification = VerificationResult.verified(
        0,
        diagnostics={
            "closure": {
                "prefix": prefix,
                "expected": '["multiple"]',
                "format": "json",
                "family": "json_array",
                "control": "continue_required",
            }
        },
    )
    targets = closure_training_targets(
        [prompt],
        [response],
        [[1, 1]],
        candidates,
        [verification],
        boundary=boundary,
        eligible=[True],
        encode=lambda text: list(map(ord, text)),
        decode=decode,
        eos_token_ids=frozenset(),
    )
    chosen, weights = select_ftpo_candidates(
        [response],
        [[1, 1]],
        candidates,
        torch.tensor([[[-0.1, -1.0, -10.0, -0.2]] * 2]),
        prompt_ids=[prompt],
        decode=decode,
        loop=LoopCreditConfig(),
        config=FTPOConfig(boundary=boundary, min_p=0.01, chosen_balance_strength=0),
        seed=42,
        eligible=[True],
        verified_boundaries=targets.boundaries,
    )
    assert targets.repairs == [None]
    if boundary == FTPOBoundary.VERIFIED_CLOSURE_AND_CONTINUATION and has_continuation:
        assert targets.boundaries == [VerifiedFTPOBoundary(0, (1_000_001, 1_000_002))]
        assert chosen.nonzero().tolist() == [[0, 0, 1]]
        assert weights.tolist() == [[1.0, 0.0]]
        assert decode(prompt + [int(candidates[chosen][0])]) == 'assistant ["multiple'
    else:
        assert not chosen.any()
        assert not weights.any()


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
        boundary=FTPOBoundary.VERIFIED_CLOSURE,
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


@pytest.mark.parametrize("boundary", [FTPOBoundary.VERIFIED_CLOSURE, FTPOBoundary.VERIFIED_CLOSURE_AND_CONTINUATION])
@pytest.mark.parametrize(
    "body,suffix,expected_kind",
    [
        ('{"name":"write_result","arguments":{"result":[]}}', [], "repair"),
        ('{"name":"wrong","arguments":{"result":[]}}', [], "none"),
        ('{"name":"write_result","arguments":{"result":', [], "none"),
        ('{"name":"write_result","arguments":{"result":[]}}', [1_000_001], "none"),
        ("", [1_000_000], "opener"),
    ],
)
def test_retention_targets_repair_closers_without_changing_opener_selection(boundary, body, suffix, expected_kind):
    pieces = {
        1_000_000: "<tool_call>",
        1_000_001: "</tool_call>",
        1_000_002: "<|eot_id|>",
        1_000_003: "<|start_header_id|>",
        1_000_004: "<|end_header_id|>",
    }

    def decode(tokens):
        return "".join(pieces[token] if token in pieces else chr(token) for token in tokens)

    def encode(text):
        if text == "</tool_call>":
            return [1_000_001]
        return list(map(ord, text))

    prompt = [1_000_003, *encode("assistant"), 1_000_004, 1_000_000]
    response = [*encode(body), *suffix, 1_000_002]
    expected = '<tool_call>{"name":"write_result","arguments":{"result":[]}}</tool_call>'
    verification = VerificationResult.verified(
        0,
        diagnostics={
            "stop_reason": "stop",
            "closure": {
                "prefix": "<tool_call>",
                "expected": expected,
                "format": "tool",
                "family": "repeated_opener_retention",
                "control": "opener_retention",
            },
        },
    )
    targets = closure_training_targets(
        [prompt],
        [response],
        [[1] * len(response)],
        torch.full((1, len(response), 1), ord("{")),
        [verification],
        boundary=boundary,
        eligible=[True],
        encode=encode,
        decode=decode,
        eos_token_ids=frozenset(),
    )
    if expected_kind == "opener":
        assert targets.boundaries == [VerifiedFTPOBoundary(0, (ord("{"),))]
        assert targets.repairs == [None]
    elif expected_kind == "repair":
        assert targets.boundaries == [None]
        repair = targets.repairs[0]
        assert repair.kind == ClosureTargetKind.MISSING_SUFFIX
        assert repair.response_position == len(body)
        assert repair.repair_token_ids == (1_000_001,)
        assert (
            decode(response[: repair.response_position] + list(repair.repair_token_ids))
            == expected[len("<tool_call>") :]
        )
    else:
        assert targets.boundaries == [None]
        assert targets.repairs == [None]


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
