"""
Base distillation trainer for on-policy and best-of-N distillation.

Extends RayPPOTrainer with teacher model scoring via a separate vLLM inference engine.
The FSDP RefWorker continues to serve its KL-regularization role unchanged.
"""

from typing import List, Optional

import torch
from loguru import logger
from omegaconf import DictConfig

from skyrl_train.trainer import RayPPOTrainer
from skyrl_train.training_batch import TrainingInputBatch
from skyrl_train.inference_engines.teacher_engine_client import (
    TeacherInferenceEngineClient,
    TeacherScoringOutput,
)


class DistillationTrainer(RayPPOTrainer):
    """Base trainer for distillation workflows.

    Adds a teacher vLLM inference engine for scoring student-generated sequences.
    Subclasses should override `apply_reward_kl_penalty` to define how teacher
    logprobs are used as a training signal.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.teacher_client: Optional[TeacherInferenceEngineClient] = None

    def setup_teacher_engine(self, teacher_engines, student_tokenizer=None, teacher_tokenizer=None):
        """Initialize the teacher inference engine client.

        Called by the experiment class after teacher engines are created.

        Args:
            teacher_engines: List of teacher inference engine instances.
            student_tokenizer: Student model tokenizer (for cross-model retokenization).
            teacher_tokenizer: Teacher model tokenizer (for cross-model retokenization).
        """
        # Extract max_model_len from teacher engine_init_kwargs if available
        teacher_max_model_len = getattr(self.cfg.teacher, "engine_init_kwargs", {}).get("max_model_len", None)
        self.teacher_client = TeacherInferenceEngineClient(
            inference_engines=teacher_engines,
            top_k_logprobs=self.cfg.teacher.top_k_logprobs,
            student_tokenizer=student_tokenizer,
            teacher_tokenizer=teacher_tokenizer,
            max_model_len=teacher_max_model_len,
        )
        logger.info(
            f"Teacher engine initialized: model={self.cfg.teacher.model_path}, "
            f"top_k_logprobs={self.cfg.teacher.top_k_logprobs}, "
            f"num_engines={len(teacher_engines)}"
        )

    def fwd_logprobs_values_reward(self, training_input: TrainingInputBatch) -> TrainingInputBatch:
        """Extended forward pass that also scores with the teacher model.

        Runs the standard policy/ref/critic forward passes via the parent class,
        then additionally scores the student sequences with the teacher vLLM engine.
        """
        # Run standard forward passes (policy logprobs, ref logprobs, critic values)
        training_input = super().fwd_logprobs_values_reward(training_input)

        # Score with teacher if configured
        if self.teacher_client is not None:
            training_input = self._score_with_teacher(training_input)

        return training_input

    def _score_with_teacher(self, training_input: TrainingInputBatch) -> TrainingInputBatch:
        """Score student sequences with the teacher model.

        Extracts prompt and response token IDs from the training input,
        sends them to the teacher vLLM engine for scoring, and stores
        the resulting top-K logprobs in the training input.
        """
        import asyncio

        sequences = training_input["sequences"]
        loss_mask = training_input["loss_mask"]
        response_length = training_input.metadata["response_length"]
        B = sequences.shape[0]

        # Split sequences into prompt and response parts
        prompt_token_ids = []
        response_token_ids = []
        for i in range(B):
            seq = sequences[i].tolist()
            prompt_len = len(seq) - response_length
            prompt_token_ids.append(seq[:prompt_len])
            response_token_ids.append(seq[prompt_len:])

        # Score with teacher (synchronous wrapper around async call).
        # Use a dedicated thread with its own event loop to avoid conflicts
        # with Ray's/uvloop's event loop (loop.run_until_complete() inside
        # an existing async context raises RuntimeError).
        import concurrent.futures

        def _run_in_thread():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                return loop.run_until_complete(
                    self.teacher_client.score_sequences(
                        prompt_token_ids=prompt_token_ids,
                        response_token_ids=response_token_ids,
                    )
                )
            finally:
                loop.close()

        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(_run_in_thread)
            teacher_output: TeacherScoringOutput = future.result()

        # Store teacher scoring results in training input
        training_input["teacher_top_k_logprobs"] = teacher_output.top_k_logprobs
        training_input["teacher_top_k_indices"] = teacher_output.top_k_indices

        # Log diagnostics
        chosen_logprobs = teacher_output.chosen_token_logprobs
        if loss_mask is not None:
            masked_logprobs = chosen_logprobs[loss_mask > 0]
            if masked_logprobs.numel() > 0:
                self.all_metrics.update({
                    "teacher/chosen_logprob_mean": masked_logprobs.mean().item(),
                    "teacher/chosen_logprob_std": masked_logprobs.std().item(),
                })

        return training_input
