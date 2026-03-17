"""
Teacher inference engine client for distillation.

Wraps vLLM inference engines to provide teacher model scoring and generation
for on-policy and off-policy distillation workflows.
"""

import asyncio
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple

import torch
from loguru import logger

from skyrl_train.inference_engines.base import (
    InferenceEngineInterface,
    InferenceEngineInput,
    InferenceEngineOutput,
)


@dataclass
class TeacherScoringOutput:
    """Output from teacher scoring of student-generated sequences.

    Attributes:
        top_k_logprobs: [B, S, K] top-K log-probability values from teacher at each
            response token position. These are the raw logprobs from vLLM, not normalized.
        top_k_indices: [B, S, K] corresponding vocabulary token IDs.
        chosen_token_logprobs: [B, S] teacher's log-probability of the actual
            (student-generated) token at each position.
    """

    top_k_logprobs: torch.Tensor  # [B, S, K]
    top_k_indices: torch.Tensor  # [B, S, K] (long)
    chosen_token_logprobs: torch.Tensor  # [B, S]


class TeacherInferenceEngineClient:
    """Client for teacher model inference via vLLM.

    Supports two modes:
    - **Scoring** (on-policy distillation): Score student-generated sequences by
      feeding prompt+response as a "prompt" to teacher vLLM with `prompt_logprobs`.
    - **Generation** (off-policy distillation): Generate completions from the teacher.

    The teacher engines are separate vLLM instances that do NOT participate in
    weight sync (teacher weights are static). They run on their own GPUs.
    """

    def __init__(
        self,
        inference_engines: List[InferenceEngineInterface],
        top_k_logprobs: int = 256,
    ):
        self.inference_engines = inference_engines
        self.top_k_logprobs = top_k_logprobs
        self._engine_idx = 0  # simple round-robin

    def _next_engine(self) -> InferenceEngineInterface:
        """Round-robin engine selection."""
        engine = self.inference_engines[self._engine_idx % len(self.inference_engines)]
        self._engine_idx += 1
        return engine

    async def score_sequences(
        self,
        prompt_token_ids: List[List[int]],
        response_token_ids: List[List[int]],
        top_k_logprobs: Optional[int] = None,
    ) -> TeacherScoringOutput:
        """Score student-generated sequences with the teacher model.

        Concatenates each (prompt, response) pair into a single sequence and feeds
        it to the teacher vLLM engine with `prompt_logprobs` to get the teacher's
        per-token log-probabilities over the response region.

        Args:
            prompt_token_ids: List of prompt token ID sequences, length B.
            response_token_ids: List of response token ID sequences, length B.
            top_k_logprobs: Override for top-K (default: self.top_k_logprobs).

        Returns:
            TeacherScoringOutput with per-token top-K logprobs for the response region.
        """
        k = top_k_logprobs or self.top_k_logprobs
        B = len(prompt_token_ids)
        assert len(response_token_ids) == B

        # Concatenate prompt + response as a single "prompt" to teacher
        full_sequences = []
        prompt_lengths = []
        for prompt, response in zip(prompt_token_ids, response_token_ids):
            full_sequences.append(list(prompt) + list(response))
            prompt_lengths.append(len(prompt))

        # Use vLLM with prompt_logprobs to score (generate max_tokens=1 to get prompt logprobs)
        engine = self._next_engine()
        input_batch: InferenceEngineInput = {
            "prompts": None,
            "prompt_token_ids": full_sequences,
            "sampling_params": {
                "max_tokens": 1,  # We only want prompt_logprobs, not generation
                "prompt_logprobs": k,
                "temperature": 1.0,
            },
            "session_ids": None,
        }

        output = await engine.generate(input_batch)

        # Extract prompt_logprobs for the response region from vLLM output
        # NOTE: The actual extraction depends on vLLM's output format for prompt_logprobs.
        # This is a placeholder that needs to be connected to the vLLM engine's
        # prompt_logprobs output. See Phase 1.3 of the plan for the vLLM engine changes.
        return self._extract_response_logprobs(output, prompt_lengths, response_token_ids, k)

    def _extract_response_logprobs(
        self,
        output: InferenceEngineOutput,
        prompt_lengths: List[int],
        response_token_ids: List[List[int]],
        k: int,
    ) -> TeacherScoringOutput:
        """Extract top-K logprobs for the response region from vLLM output.

        Processes the InferenceEngineOutput.prompt_logprobs field (populated by
        vLLM when SamplingParams(prompt_logprobs=K) is set) into padded tensors.

        vLLM prompt_logprobs format per sample:
            List[Optional[Dict[int, float]]]
        where each dict maps token_id → logprob value. The first position is
        always None (no logprob for the first token). Each dict contains the
        actual prompt token plus up to K-1 top alternatives.

        Args:
            output: InferenceEngineOutput with prompt_logprobs populated.
            prompt_lengths: Length of each prompt (to locate response region).
            response_token_ids: The actual response tokens (for chosen-token logprobs).
            k: Number of top-K logprobs requested.

        Returns:
            TeacherScoringOutput with padded tensors.
        """
        B = len(prompt_lengths)
        max_response_len = max(len(r) for r in response_token_ids)

        # Initialize output tensors
        all_top_k_logprobs = torch.full((B, max_response_len, k), float("-inf"))
        all_top_k_indices = torch.zeros((B, max_response_len, k), dtype=torch.long)
        all_chosen_logprobs = torch.zeros((B, max_response_len))

        # Extract from vLLM prompt_logprobs
        # Format: List[List[Optional[Dict[int, float]]]] — [batch][position]
        prompt_logprobs_batch = output.get("prompt_logprobs")
        if prompt_logprobs_batch is not None:
            for i in range(B):
                prompt_len = prompt_lengths[i]
                response_len = len(response_token_ids[i])
                sample_prompt_logprobs = prompt_logprobs_batch[i]

                for t in range(response_len):
                    # Position in the full sequence (prompt + response)
                    pos = prompt_len + t
                    if pos >= len(sample_prompt_logprobs):
                        continue
                    pos_logprobs = sample_prompt_logprobs[pos]
                    if pos_logprobs is None:
                        continue

                    # Sort by logprob descending, take top-K
                    sorted_items = sorted(
                        pos_logprobs.items(),
                        key=lambda x: x[1],
                        reverse=True,
                    )[:k]
                    for j, (token_id, logprob) in enumerate(sorted_items):
                        all_top_k_logprobs[i, t, j] = logprob
                        all_top_k_indices[i, t, j] = int(token_id)

                    # Get chosen token logprob
                    chosen_token = response_token_ids[i][t]
                    if chosen_token in pos_logprobs:
                        all_chosen_logprobs[i, t] = pos_logprobs[chosen_token]
        else:
            logger.warning(
                "Teacher scoring returned no prompt_logprobs. "
                "Ensure the teacher vLLM engine supports prompt_logprobs and "
                "SamplingParams(prompt_logprobs=K) is set."
            )

        return TeacherScoringOutput(
            top_k_logprobs=all_top_k_logprobs,
            top_k_indices=all_top_k_indices,
            chosen_token_logprobs=all_chosen_logprobs,
        )

    async def generate(
        self,
        prompt_token_ids: List[List[int]],
        sampling_params: Optional[Dict[str, Any]] = None,
    ) -> InferenceEngineOutput:
        """Generate completions from the teacher model (for off-policy distillation).

        Args:
            prompt_token_ids: List of prompt token ID sequences.
            sampling_params: vLLM sampling parameters. If logprobs not specified,
                defaults to self.top_k_logprobs.

        Returns:
            InferenceEngineOutput with generated responses and logprobs.
        """
        params = dict(sampling_params or {})
        if "logprobs" not in params:
            params["logprobs"] = self.top_k_logprobs

        engine = self._next_engine()
        input_batch: InferenceEngineInput = {
            "prompts": None,
            "prompt_token_ids": prompt_token_ids,
            "sampling_params": params,
            "session_ids": None,
        }
        return await engine.generate(input_batch)
