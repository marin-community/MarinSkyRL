"""Deterministic sampled/forced-token accounting for opt-in K16 interventions."""

from dataclasses import dataclass
from collections import Counter, deque
from typing import Sequence


INTERVENTION_VERSION = "non-agentic-token-intervention-v1"


@dataclass(frozen=True)
class TokenIntervention:
    protocol: str
    kind: str
    thinking_end_id: int
    eos_id: int
    force_close_after: int = 3072
    repetition_window: int = 256
    repetition_ngram: int = 16
    repetition_fraction: float = 0.5

    def __post_init__(self):
        if self.protocol != INTERVENTION_VERSION or self.kind not in ("force_close", "repetition_stop"):
            raise ValueError("Unknown token intervention protocol or treatment")
        if min(self.thinking_end_id, self.eos_id) < 0 or self.thinking_end_id == self.eos_id:
            raise ValueError("Interventions require distinct, verified tokenizer IDs")
        if self.force_close_after <= 0 or not 0 < self.repetition_ngram <= self.repetition_window:
            raise ValueError("Invalid intervention token bounds")
        if not 0 < self.repetition_fraction <= 1:
            raise ValueError("Invalid repeated n-gram fraction")


def repeated_ngram_fraction(tokens: Sequence[int], n: int) -> float:
    """Stage-1 duplicate fraction over overlapping n-grams, not token coverage."""
    if n <= 0:
        raise ValueError("n must be positive")
    grams = [tuple(tokens[i : i + n]) for i in range(max(0, len(tokens) - n + 1))]
    return 1 - len(set(grams)) / len(grams) if grams else 0.0


class InterventionState:
    """Incrementally replay a native sequence without replacing any sampled token.

    The vLLM adapter receives a live list of output IDs. The runner and auditor
    replay that same list to validate forced positions and loss masks.
    """

    def __init__(self, config: TokenIntervention):
        self.config = config
        self.seen: list[int] = []
        self.sampled: list[int] = []
        self.forced_positions: list[int] = []
        self.closed = False
        self.repetition_stopped = False
        self.recent_grams: deque[tuple[int, ...]] = deque()
        self.gram_counts: Counter[tuple[int, ...]] = Counter()

    def next_forced_token(self) -> int | None:
        config = self.config
        if config.kind == "force_close":
            if not self.closed and len(self.sampled) >= config.force_close_after:
                return config.thinking_end_id
        elif not self.repetition_stopped and len(self.sampled) >= config.repetition_window:
            fraction = 1 - len(self.gram_counts) / len(self.recent_grams)
            if fraction >= config.repetition_fraction:
                return config.eos_id
        return None

    def advance(self, output_ids: Sequence[int]) -> int | None:
        if len(output_ids) < len(self.seen) or list(output_ids[: len(self.seen)]) != self.seen:
            raise ValueError("Native output IDs changed during intervention replay")
        for token in output_ids[len(self.seen) :]:
            if self.repetition_stopped:
                raise ValueError("Native generation continued after forced repetition EOS")
            forced = self.next_forced_token()
            if forced is not None:
                if token != forced:
                    raise ValueError("Native output does not contain the prescribed forced token")
                self.forced_positions.append(len(self.seen))
                self.repetition_stopped = self.config.kind == "repetition_stop"
            else:
                self.sampled.append(token)
                n = self.config.repetition_ngram
                if len(self.sampled) >= n:
                    gram = tuple(self.sampled[-n:])
                    self.recent_grams.append(gram)
                    self.gram_counts[gram] += 1
                    if len(self.recent_grams) > self.config.repetition_window - n + 1:
                        oldest = self.recent_grams.popleft()
                        self.gram_counts[oldest] -= 1
                        if not self.gram_counts[oldest]:
                            del self.gram_counts[oldest]
            self.closed |= token == self.config.thinking_end_id
            self.seen.append(token)
        return self.next_forced_token()


def intervention_trace(output_ids: Sequence[int], config: TokenIntervention) -> dict:
    state = InterventionState(config)
    state.advance(output_ids)
    forced = set(state.forced_positions)
    return {
        "protocol": config.protocol,
        "kind": config.kind,
        "forced_positions": state.forced_positions,
        "sampled_positions": [i for i in range(len(output_ids)) if i not in forced],
        "sampled_token_count": len(state.sampled),
        "repetition_stopped": state.repetition_stopped,
        "thinking_closed": state.closed,
    }
