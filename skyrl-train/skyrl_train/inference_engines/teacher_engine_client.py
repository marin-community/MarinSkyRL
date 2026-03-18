"""
Teacher inference engine client for distillation.

Wraps vLLM inference engines to provide teacher model scoring and generation
for on-policy and off-policy distillation workflows.

Handles cross-model tokenizer mismatches: student-generated token IDs are
decoded to text, re-tokenized with the teacher tokenizer, scored by the
teacher vLLM engine, and the resulting logprobs are mapped back to student
token positions.
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


def _build_student_to_teacher_alignment(
    student_text: str,
    student_token_ids: List[int],
    teacher_token_ids: List[int],
    student_tokenizer,
    teacher_tokenizer,
) -> List[Optional[int]]:
    """Build a mapping from student token positions to teacher token positions.

    Uses character-level offset alignment: for each student token, find the
    teacher token that covers the same starting character position.

    Args:
        student_text: Decoded text from student tokens.
        student_token_ids: Student token IDs.
        teacher_token_ids: Teacher token IDs (from re-encoding student_text).
        student_tokenizer: Student tokenizer.
        teacher_tokenizer: Teacher tokenizer.

    Returns:
        List of length len(student_token_ids). Each entry is the teacher token
        index that aligns with that student token, or None if no alignment.
    """
    # Get character offsets for student tokens
    student_offsets = []
    char_pos = 0
    for tid in student_token_ids:
        token_str = student_tokenizer.decode([tid])
        student_offsets.append(char_pos)
        char_pos += len(token_str)

    # Get character offsets for teacher tokens
    teacher_offsets = []
    char_pos = 0
    for tid in teacher_token_ids:
        token_str = teacher_tokenizer.decode([tid])
        teacher_offsets.append(char_pos)
        char_pos += len(token_str)

    # Map each student token to the teacher token covering its start position
    alignment = []
    teacher_idx = 0
    for s_offset in student_offsets:
        # Advance teacher index until we find the token covering s_offset
        while teacher_idx < len(teacher_offsets) - 1 and teacher_offsets[teacher_idx + 1] <= s_offset:
            teacher_idx += 1
        if teacher_idx < len(teacher_offsets):
            alignment.append(teacher_idx)
        else:
            alignment.append(None)

    return alignment


class TeacherInferenceEngineClient:
    """Client for teacher model inference via vLLM.

    Supports two modes:
    - **Scoring** (on-policy distillation): Score student-generated sequences by
      feeding prompt+response as a "prompt" to teacher vLLM with `prompt_logprobs`.
    - **Generation** (off-policy distillation): Generate completions from the teacher.

    The teacher engines are separate vLLM instances that do NOT participate in
    weight sync (teacher weights are static). They run on their own GPUs.

    When student and teacher use different tokenizers, sequences are detokenized
    from the student vocabulary and re-tokenized with the teacher tokenizer before
    scoring. Logprobs are then mapped back to student token positions.
    """

    def __init__(
        self,
        inference_engines: List[InferenceEngineInterface],
        top_k_logprobs: int = 256,
        student_tokenizer=None,
        teacher_tokenizer=None,
    ):
        self.inference_engines = inference_engines
        self.top_k_logprobs = top_k_logprobs
        self._engine_idx = 0  # simple round-robin
        self.student_tokenizer = student_tokenizer
        self.teacher_tokenizer = teacher_tokenizer

        # Determine if we need cross-tokenizer remapping
        self._needs_retokenization = (
            student_tokenizer is not None
            and teacher_tokenizer is not None
            and student_tokenizer.name_or_path != teacher_tokenizer.name_or_path
        )
        if self._needs_retokenization:
            logger.info(
                f"Cross-tokenizer distillation enabled: "
                f"student={student_tokenizer.name_or_path}, "
                f"teacher={teacher_tokenizer.name_or_path}"
            )

    def _next_engine(self) -> InferenceEngineInterface:
        """Round-robin engine selection."""
        engine = self.inference_engines[self._engine_idx % len(self.inference_engines)]
        self._engine_idx += 1
        return engine

    def _retokenize_for_teacher(
        self,
        prompt_token_ids: List[List[int]],
        response_token_ids: List[List[int]],
    ) -> Tuple[List[List[int]], List[int], List[List[Optional[int]]]]:
        """Decode student sequences and re-tokenize with teacher tokenizer.

        Args:
            prompt_token_ids: Student prompt token IDs.
            response_token_ids: Student response token IDs.

        Returns:
            Tuple of:
            - teacher_full_sequences: Re-tokenized full sequences for teacher scoring.
            - teacher_prompt_lengths: Length of the prompt portion in teacher tokens.
            - response_alignments: For each sample, a list mapping student response
              token positions to teacher token positions (relative to response start).
        """
        teacher_full_sequences = []
        teacher_prompt_lengths = []
        response_alignments = []

        for prompt_ids, response_ids in zip(prompt_token_ids, response_token_ids):
            # Decode student tokens to text
            prompt_text = self.student_tokenizer.decode(prompt_ids, skip_special_tokens=False)
            response_text = self.student_tokenizer.decode(response_ids, skip_special_tokens=False)

            # Re-tokenize with teacher tokenizer
            teacher_prompt_ids = self.teacher_tokenizer.encode(prompt_text, add_special_tokens=False)
            teacher_response_ids = self.teacher_tokenizer.encode(response_text, add_special_tokens=False)

            teacher_full_sequences.append(teacher_prompt_ids + teacher_response_ids)
            teacher_prompt_lengths.append(len(teacher_prompt_ids))

            # Build alignment from student response positions to teacher response positions
            alignment = _build_student_to_teacher_alignment(
                response_text,
                response_ids,
                teacher_response_ids,
                self.student_tokenizer,
                self.teacher_tokenizer,
            )
            response_alignments.append(alignment)

        return teacher_full_sequences, teacher_prompt_lengths, response_alignments

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

        When cross-tokenizer mode is active, sequences are detokenized from student
        vocabulary and re-tokenized with the teacher tokenizer before scoring.

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

        if self._needs_retokenization:
            # Cross-tokenizer path: decode student → re-encode with teacher
            teacher_full_sequences, teacher_prompt_lengths, response_alignments = (
                self._retokenize_for_teacher(prompt_token_ids, response_token_ids)
            )
            full_sequences = teacher_full_sequences
            prompt_lengths = teacher_prompt_lengths
        else:
            # Same-tokenizer path: use student token IDs directly
            full_sequences = []
            prompt_lengths = []
            response_alignments = None
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

        return self._extract_response_logprobs(
            output, prompt_lengths, response_token_ids, k, response_alignments
        )

    def _extract_response_logprobs(
        self,
        output: InferenceEngineOutput,
        prompt_lengths: List[int],
        response_token_ids: List[List[int]],
        k: int,
        response_alignments: Optional[List[List[Optional[int]]]] = None,
    ) -> TeacherScoringOutput:
        """Extract top-K logprobs for the response region from vLLM output.

        When response_alignments is provided (cross-tokenizer mode), uses alignment
        to map teacher token positions back to student token positions.

        Args:
            output: InferenceEngineOutput with prompt_logprobs populated.
            prompt_lengths: Length of each prompt (in teacher tokens if retokenized).
            response_token_ids: The actual student response tokens.
            k: Number of top-K logprobs requested.
            response_alignments: Optional alignment maps (student pos → teacher pos).

        Returns:
            TeacherScoringOutput with padded tensors indexed by student positions.
        """
        B = len(prompt_lengths)
        max_response_len = max(len(r) for r in response_token_ids)

        # Initialize output tensors (indexed by student response positions)
        all_top_k_logprobs = torch.full((B, max_response_len, k), float("-inf"))
        all_top_k_indices = torch.zeros((B, max_response_len, k), dtype=torch.long)
        all_chosen_logprobs = torch.zeros((B, max_response_len))

        prompt_logprobs_batch = output.get("prompt_logprobs")
        if prompt_logprobs_batch is not None:
            for i in range(B):
                prompt_len = prompt_lengths[i]
                response_len = len(response_token_ids[i])
                sample_prompt_logprobs = prompt_logprobs_batch[i]

                for t in range(response_len):
                    if response_alignments is not None:
                        # Cross-tokenizer: use alignment to find teacher position
                        teacher_response_pos = response_alignments[i][t] if t < len(response_alignments[i]) else None
                        if teacher_response_pos is None:
                            continue
                        pos = prompt_len + teacher_response_pos
                    else:
                        # Same-tokenizer: direct position mapping
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
                    # In cross-tokenizer mode, the "chosen token" is the teacher token
                    # at the aligned position (not the student token ID).
                    if response_alignments is not None:
                        # Use the highest-prob token's logprob as a proxy for the
                        # student's chosen token (since student token IDs don't exist
                        # in teacher vocab). The top-K logprobs are still useful for
                        # the KL divergence computation.
                        if sorted_items:
                            all_chosen_logprobs[i, t] = sorted_items[0][1]
                    else:
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
