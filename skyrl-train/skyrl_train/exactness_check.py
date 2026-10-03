"""An online check, under exact numerics, that the trainer computes the rollout engines' log-probabilities bit for bit.

After the first weight sync and every ``every_weight_syncs``-th one, each live engine generates one fixed short response
(``ExactnessCheck.generate``). The trainer's next policy forward, while the policy still holds the weights the engines
received, scores those responses (``ExactnessCheck.batch``), and every response token's log-probability must equal the
engine's (``ExactnessCheck.compare``).
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np
import torch
from loguru import logger
from omegaconf import DictConfig
from transformers import PreTrainedTokenizerBase

from skyrl_train.config.numerics import ExactnessFailure
from skyrl_train.dataset.preprocess import convert_prompts_responses_to_batch_tensors
from skyrl_train.dataset.routed_expert_batch import RoutedExpertRows
from skyrl_train.inference_engines.base import InferenceEngineOutput
from skyrl_train.inference_engines.utils import get_sampling_params_for_backend
from skyrl_train.training_batch import ENGINE_DP_RANKS_KEY, TrainingInputBatch

# The fixed prompt and the response length of every check: a few decode steps on each engine.
CHECK_TEXT = "Every token's log-probability is the same in the trainer as in the inference engine that sampled it."
CHECK_RESPONSE_TOKENS = 16
CHECK_SEED = 0
METRIC_PREFIX = "exactness_check"
# Weight syncs after which the check always runs: the first after startup and after a checkpoint restore.
ALWAYS_CHECKED_SYNCS = frozenset({"initial", "checkpoint_restore"})


class EngineGenerator(Protocol):
    """The inference engine client calls the check makes."""

    def live_engine_indices(self) -> list[int]: ...

    async def generate_on_engine(
        self, engine_idx: int, prompt_token_ids: list[int], sampling_params: dict[str, Any]
    ) -> InferenceEngineOutput: ...


@dataclass(frozen=True)
class EngineSample:
    """One engine's response to the check prompt, its log-probabilities, the rank of the engine that served it and,
    under router replay, its expert routes."""

    engine_dp_rank: int
    response_token_ids: list[int]
    response_logprobs: list[float]
    routed_experts: np.ndarray | None


class ExactnessCheck:
    """Engine samples taken after a weight sync, and their comparison with the trainer's scores."""

    def __init__(self, cfg: DictConfig, tokenizer: PreTrainedTokenizerBase):
        check = cfg.trainer.algorithm.exactness_check
        self.every_weight_syncs = int(check.every_weight_syncs)
        self.on_failure = ExactnessFailure(check.on_failure)
        self.tokenizer = tokenizer
        self.prompt_token_ids = tokenizer.encode(CHECK_TEXT, add_special_tokens=False)
        self.sampling_params = {
            **get_sampling_params_for_backend(cfg.generator.backend, cfg.generator.sampling_params),
            "max_tokens": CHECK_RESPONSE_TOKENS,
            "logprobs": 0,
            "seed": CHECK_SEED,
            "stop": None,
        }
        self.pending: list[EngineSample] | None = None
        self._training_syncs = 0
        self._generate_seconds = 0.0

    def count_sync(self, sync_reason: str) -> bool:
        """Count a weight sync made for ``sync_reason``; True when the check follows it."""
        if sync_reason in ALWAYS_CHECKED_SYNCS:
            return True
        self._training_syncs += 1
        return self.every_weight_syncs > 0 and self._training_syncs % self.every_weight_syncs == 0

    async def generate(self, client: EngineGenerator) -> None:
        """Every live engine's response to the check prompt, kept until the trainer scores it."""
        started = time.monotonic()
        outputs = await asyncio.gather(
            *(
                client.generate_on_engine(index, self.prompt_token_ids, self.sampling_params)
                for index in client.live_engine_indices()
            )
        )
        self.pending = [
            EngineSample(
                engine_dp_rank=output["engine_dp_ranks"][0],
                response_token_ids=list(output["response_ids"][0]),
                response_logprobs=list(output["response_logprobs"][0]),
                routed_experts=output["routed_experts"][0] if "routed_experts" in output else None,
            )
            for output in outputs
        ]
        self._generate_seconds = time.monotonic() - started

    def batch(self, dp_size: int, num_experts: int | None) -> TrainingInputBatch:
        """The pending responses as a policy forward batch, repeated to a multiple of ``dp_size`` rows, with their
        expert routes when the engines returned them."""
        if self.pending is None:
            raise RuntimeError("the exactness check has no engine samples to score")
        rows = [self.pending[index % len(self.pending)] for index in range(-(-len(self.pending) // dp_size) * dp_size)]
        responses = [row.response_token_ids for row in rows]
        sequences, attention_mask, action_mask, *_ = convert_prompts_responses_to_batch_tensors(
            self.tokenizer,
            [self.prompt_token_ids] * len(rows),
            responses,
            rewards=[[0.0] * len(response) for response in responses],
            loss_masks=[[1] * len(response) for response in responses],
        )
        routes = [row.routed_experts for row in rows]
        if any(route is None for route in routes) and any(route is not None for route in routes):
            raise ValueError("only some engines returned expert routes for the exactness check")
        batch = TrainingInputBatch(
            {
                "sequences": sequences,
                "attention_mask": attention_mask,
                ENGINE_DP_RANKS_KEY: torch.tensor([row.engine_dp_rank for row in rows], dtype=torch.long),
            },
            routed_expert_rows=(
                None if routes[0] is None else RoutedExpertRows(tuple(routes), action_mask.shape[1], num_experts)
            ),
        )
        batch.metadata = {"response_length": action_mask.shape[1]}
        return batch

    def compare(self, logprobs: torch.Tensor, score_seconds: float) -> dict[str, float]:
        """The check's metrics from the trainer's ``[rows, response_length]`` scores of ``batch()``'s first rows; under
        ``on_failure=stop``, raises unless every score has the engine's float32 bits."""
        samples, self.pending = self.pending, None
        if samples is None:
            raise RuntimeError("the exactness check has no engine samples to compare")
        if logprobs.shape[0] < len(samples):
            raise ValueError(f"{logprobs.shape[0]} trainer score rows for {len(samples)} engine samples")
        mismatched = tokens = 0
        largest = 0.0
        for sample, scores in zip(samples, logprobs.float().cpu(), strict=False):
            engine = np.asarray(sample.response_logprobs, dtype=np.float32)
            trainer = scores[: engine.size].numpy()
            mismatched += int((trainer.view(np.int32) != engine.view(np.int32)).sum())
            tokens += engine.size
            largest = max(largest, float(np.abs(trainer.astype(np.float64) - engine.astype(np.float64)).max(initial=0)))
        metrics = {
            f"{METRIC_PREFIX}/engines": float(len(samples)),
            f"{METRIC_PREFIX}/tokens": float(tokens),
            f"{METRIC_PREFIX}/mismatched_tokens": float(mismatched),
            f"{METRIC_PREFIX}/max_abs_difference": largest,
            f"{METRIC_PREFIX}/generate_seconds": self._generate_seconds,
            f"{METRIC_PREFIX}/score_seconds": score_seconds,
        }
        if mismatched:
            message = (
                f"Exact numerics: {mismatched} of {tokens} log-probabilities the trainer computed for {len(samples)} "
                f"engines' check responses differ from the engines' (largest difference {largest:.3g})"
            )
            if self.on_failure is ExactnessFailure.STOP:
                raise RuntimeError(message)
            logger.error(message)
        return metrics
