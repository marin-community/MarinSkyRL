"""Versioned post-thinking answer verification for non-agentic reasoning tasks."""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

from skyrl_gym.envs.aime.verifier import AIMEVerifier
from skyrl_gym.envs.gsm8k.utils import compute_score as gsm8k_score
from skyrl_gym.envs.reasoning_gym.scoring import score_response as reasoning_gym_score
from skyrl_gym.verification import RolloutEvidence

THINKING_CONTRACT_VERSION = "post-thinking-native-v1"
ACCEPTED_STOPS = frozenset({"complete", "end_turn", "eos", "stop"})
EOS_MARKERS = frozenset({"<|im_end|>", "<|eot_id|>", "<|end_of_text|>", "<|endoftext|>"})
THINKING_MARKERS = frozenset({"<think>", "</think>", "<|start_think|>", "<|end_think|>"})
FORBIDDEN_MARKERS = EOS_MARKERS | THINKING_MARKERS | {"<|start_header_id|>", "<|end_header_id|>", "<|im_start|>"}


class TokenDecoder(Protocol):
    def token_to_id(self, token: str) -> int | None: ...

    def id_to_token(self, token: int) -> str | None: ...

    def decode(self, ids: Sequence[int], skip_special_tokens: bool = True) -> str: ...


@dataclass(frozen=True)
class ThinkingContractResult:
    parser_protocol: str
    boundary_status: str
    verifier_reward: float
    contract_correct: int
    score_contract_completed: int
    legacy_full_text_reward: float
    thinking_closed: bool


def post_thinking_segment(
    decoder: TokenDecoder, prompt_tokens: Sequence[int], response_tokens: Sequence[int]
) -> tuple[str | None, str]:
    """Resolve one inherited thinking span without accepting role continuations."""
    start = decoder.token_to_id("<|start_think|>")
    end = decoder.token_to_id("<|end_think|>")
    if start is None or end is None:
        raise ValueError("Post-thinking verification requires declared thinking boundary tokens")
    prompt = list(prompt_tokens)
    tokens = list(response_tokens)
    if start not in prompt:
        return None, "missing_thinking_prompt"
    last_start = len(prompt) - 1 - prompt[::-1].index(start)
    if decoder.decode(prompt[last_start + 1 :], skip_special_tokens=False).strip():
        return None, "invalid_thinking_prompt"
    if start in tokens:
        return None, "unexpected_thinking_start"
    count = tokens.count(end)
    if count != 1:
        return None, "missing_thinking_end" if count == 0 else "multiple_thinking_ends"
    tokens = tokens[tokens.index(end) + 1 :]
    while tokens and (
        decoder.id_to_token(tokens[-1]) in EOS_MARKERS
        or decoder.decode(tokens[-1:], skip_special_tokens=False).isspace()
    ):
        tokens.pop()
    if any(decoder.id_to_token(token) in FORBIDDEN_MARKERS for token in tokens):
        return None, "role_or_thinking_continuation"
    return decoder.decode(tokens, skip_special_tokens=False), "resolved"


def native_answer_reward(env_class: str, ground_truth: str, response: str) -> float:
    """Use the task's existing unshaped reward implementation, including its scale."""
    if env_class == "gsm8k":
        return float(gsm8k_score(response, ground_truth))
    if env_class == "aime":
        score = AIMEVerifier(ground_truth).verify(RolloutEvidence(response=response)).score
        assert score is not None
        return float(score)
    if env_class == "reasoning_gym":
        return float(reasoning_gym_score(response, ground_truth))
    raise ValueError(f"Post-thinking verification is not defined for {env_class!r}")


def score_thinking_contract(
    *,
    env_class: str,
    ground_truth: str,
    native_response: str,
    prompt_tokens: Sequence[int],
    response_tokens: Sequence[int],
    stop_reason: str,
    decoder: TokenDecoder,
) -> ThinkingContractResult:
    """Keep corrected native reward separate from binary correctness and stopping.

    Unresolved thinking supplies no answer to the native verifier. In particular,
    AIME retains its native negative reward instead of an invented zero reward.
    An accepted stop is required only for the completed metric, not raw reward.
    """
    segment, status = post_thinking_segment(decoder, prompt_tokens, response_tokens)
    corrected = native_answer_reward(env_class, ground_truth, "" if segment is None else segment)
    legacy = native_answer_reward(env_class, ground_truth, native_response)
    correct = int(corrected >= 1.0) if env_class == "reasoning_gym" else int(corrected == 1.0)
    if status != "resolved":
        assert correct == 0
    end = decoder.token_to_id("<|end_think|>")
    return ThinkingContractResult(
        parser_protocol=THINKING_CONTRACT_VERSION,
        boundary_status=status,
        verifier_reward=corrected,
        contract_correct=correct,
        score_contract_completed=correct * int(stop_reason in ACCEPTED_STOPS),
        legacy_full_text_reward=legacy,
        thinking_closed=end is not None and list(response_tokens).count(end) == 1,
    )
