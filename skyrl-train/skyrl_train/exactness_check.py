"""An online check that the trainer computes the rollout engines' log-probabilities bit for bit.

After a qualifying weight sync each live engine generates one fixed short response (``ExactnessCheck.start_generation``,
collected by ``finish_generation``). The trainer's next policy forward, while the policy still holds the weights the
engines received, scores those responses (``ExactnessCheck.batch``), and ``ExactnessCheck.compare`` counts the response
tokens whose float32 log-probability differs from the engine's, in total and per engine rank. The check runs under every
numerics mode; native and batch-invariant runs report their real mismatch.
"""

import asyncio
import time
from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np
import torch
from loguru import logger
from omegaconf import DictConfig
from transformers import PreTrainedTokenizerBase

from skyrl_train.config.numerics import CheckFailure, ExactnessCheckConfig, NumericsMode
from skyrl_train.dataset.preprocess import convert_prompts_responses_to_batch_tensors
from skyrl_train.dataset.routed_expert_batch import RoutedExpertRows
from skyrl_train.inference_engines.base import InferenceEngineOutput
from skyrl_train.inference_engines.utils import get_sampling_params_for_backend
from skyrl_train.training_batch import TrainingInputBatch
from skyrl_train.trajectory_runners.types import ENGINE_DP_RANKS_KEY, ROLLOUT_ENGINE_DP_RANKS_KEY

# The fixed prompt and response length of every check: a few decode steps on each engine.
CHECK_TEXT = "Every token's log-probability is the same in the trainer as in the inference engine that sampled it."
CHECK_RESPONSE_TOKENS = 16
CHECK_SEED = 0
METRIC_PREFIX = "exactness_check"
# Weight syncs the check always follows: the first after startup, and the re-sync after a colocated checkpoint save.
INITIAL_SYNC = "initial"
CHECKPOINT_SAVE_RESYNC = "checkpoint_restore"
ALWAYS_CHECKED_SYNCS = frozenset({INITIAL_SYNC, CHECKPOINT_SAVE_RESYNC})
# The weight syncs every_weight_syncs counts.
TRAINING_SYNC = "training_step"


class EngineSampler(Protocol):
    """The inference engine client call the check makes."""

    async def generate_on_live_engines(
        self, prompt_token_ids: list[int], sampling_params: dict[str, Any]
    ) -> list[InferenceEngineOutput]: ...


@dataclass(frozen=True)
class EngineSample:
    """One engine's response to the check prompt, its log-probabilities, the rank of the engine that served it and,
    when the engine returned them, its expert routes aligned with the response tokens."""

    engine_dp_rank: int
    response_token_ids: list[int]
    response_logprobs: list[float]
    routed_experts: np.ndarray | None


class ExactnessCheck:
    """Engine samples taken after a weight sync, and their comparison with the trainer's scores."""

    def __init__(
        self,
        config: ExactnessCheckConfig,
        mode: NumericsMode,
        generator: DictConfig,
        tokenizer: PreTrainedTokenizerBase,
    ):
        self.every_weight_syncs = config.every_weight_syncs
        self.on_failure = config.on_failure
        self.mode = mode
        self.tokenizer = tokenizer
        self.prompt_token_ids = tokenizer.encode(CHECK_TEXT, add_special_tokens=False)
        self.sampling_params = {
            **get_sampling_params_for_backend(generator.backend, generator.sampling_params),
            "max_tokens": CHECK_RESPONSE_TOKENS,
            "logprobs": 0,
            "seed": CHECK_SEED,
            "stop": None,
        }
        self.pending: list[EngineSample] | None = None
        self.generate_failures = 0
        self._generation: asyncio.Task[list[EngineSample]] | None = None
        self._training_syncs = 0
        self._generate_seconds = 0.0

    def count_sync(self, reason: str) -> bool:
        """Count a weight sync made for ``reason``; True when the check follows it."""
        if reason in ALWAYS_CHECKED_SYNCS:
            return True
        if reason != TRAINING_SYNC:
            return False
        self._training_syncs += 1
        return self._training_syncs % self.every_weight_syncs == 0

    def start_generation(self, client: EngineSampler) -> None:
        """Start every live engine's response to the check prompt; ``finish_generation`` collects them."""
        if self._generation is not None:
            self._generation.cancel()
        self._generation = asyncio.create_task(self._generate(client))

    async def finish_generation(self) -> dict[str, float]:
        """Wait for the started generation and keep its samples for ``batch``; a failed generation under
        ``on_failure=log`` is skipped and counted in ``exactness_check/generate_failures``."""
        if self._generation is None:
            return {}
        generation, self._generation = self._generation, None
        try:
            self.pending = await generation
        except Exception as error:
            if self.on_failure is CheckFailure.STOP:
                raise
            self.generate_failures += 1
            self.pending = None
            logger.warning("Exactness check skipped: the engines did not generate the check responses: {}", error)
            return {f"{METRIC_PREFIX}/generate_failures": float(self.generate_failures)}
        return {}

    async def _generate(self, client: EngineSampler) -> list[EngineSample]:
        started = time.monotonic()
        outputs = await client.generate_on_live_engines(self.prompt_token_ids, self.sampling_params)
        samples = []
        for output in outputs:
            if output["response_logprobs"] is None:
                raise ValueError("the exactness check needs the engines' response log-probabilities")
            samples.append(
                EngineSample(
                    engine_dp_rank=output[ENGINE_DP_RANKS_KEY][0],
                    response_token_ids=list(output["response_ids"][0]),
                    response_logprobs=list(output["response_logprobs"][0]),
                    routed_experts=(output.get("routed_experts") or [None])[0],
                )
            )
        self._generate_seconds = time.monotonic() - started
        return samples

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
                ROLLOUT_ENGINE_DP_RANKS_KEY: torch.tensor([row.engine_dp_rank for row in rows], dtype=torch.long),
            },
            routed_expert_rows=(
                None if routes[0] is None else RoutedExpertRows(tuple(routes), action_mask.shape[1], num_experts)
            ),
        )
        batch.metadata = {"response_length": action_mask.shape[1]}
        return batch

    def compare(self, logprobs: torch.Tensor, score_seconds: float) -> dict[str, float]:
        """The check's metrics from the trainer's ``[rows, response_length]`` scores of ``batch()``, in total and per
        engine rank; under ``on_failure=stop``, raises unless every score has the engine's float32 bits."""
        samples, self.pending = self.pending, None
        if samples is None:
            raise RuntimeError("the exactness check has no engine samples to compare")
        if logprobs.shape[0] < len(samples):
            raise ValueError(f"{logprobs.shape[0]} trainer score rows for {len(samples)} engine samples")
        tokens_by_rank: dict[int, int] = {}
        mismatched_by_rank: dict[int, int] = {}
        largest = 0.0
        for row, scores in enumerate(logprobs.float().cpu()):
            sample = samples[row % len(samples)]
            engine = np.asarray(sample.response_logprobs, dtype=np.float32)
            trainer = scores[: engine.size].numpy()
            rank = sample.engine_dp_rank
            tokens_by_rank[rank] = tokens_by_rank.get(rank, 0) + engine.size
            mismatched_by_rank[rank] = mismatched_by_rank.get(rank, 0) + int(
                (trainer.view(np.int32) != engine.view(np.int32)).sum()
            )
            largest = max(largest, float(np.abs(trainer.astype(np.float64) - engine.astype(np.float64)).max(initial=0)))
        tokens, mismatched = sum(tokens_by_rank.values()), sum(mismatched_by_rank.values())
        metrics = {
            f"{METRIC_PREFIX}/engines": float(len(samples)),
            f"{METRIC_PREFIX}/tokens": float(tokens),
            f"{METRIC_PREFIX}/mismatched_tokens": float(mismatched),
            f"{METRIC_PREFIX}/max_abs_difference": largest,
            f"{METRIC_PREFIX}/generate_seconds": self._generate_seconds,
            f"{METRIC_PREFIX}/score_seconds": score_seconds,
        }
        for rank, count in tokens_by_rank.items():
            metrics[f"{METRIC_PREFIX}/engine_{rank}/tokens"] = float(count)
            metrics[f"{METRIC_PREFIX}/engine_{rank}/mismatched_tokens"] = float(mismatched_by_rank[rank])
        if mismatched:
            message = (
                f"Exactness check: {mismatched} of {tokens} log-probabilities the trainer computed for {len(samples)} "
                f"engines' check responses differ from the engines' (largest difference {largest:.3g})"
            )
            if self.on_failure is CheckFailure.STOP:
                raise RuntimeError(message)
            # A mode that promises bit-identical results has a defect; the others measure their expected mismatch.
            if self.mode.bit_identical:
                logger.error(message)
            else:
                logger.info(message)
        return metrics
