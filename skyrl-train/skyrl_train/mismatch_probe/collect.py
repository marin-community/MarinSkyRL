"""Synchronous frozen-token probe at the initial and post-sync policy weights."""

from __future__ import annotations

import asyncio
import hashlib
import math
import os
import random
import time
from dataclasses import dataclass, replace
from datetime import UTC, datetime

import numpy as np
import ray
import torch
from finestore import mismatch_probe as mismatch
from loguru import logger

from skyrl_train.config.mismatch_probe import (
    CACHE_BOTH,
    CACHE_OFF,
    CACHE_ON,
    FROZEN_RESCORE_SCORER,
    GENERATION_SCORER,
    RESCORE_AGAIN_SCORER,
    RESCORE_SCORER,
    TRAINER_SCORER,
    rescore_label,
    trainer_label,
)
from skyrl_train.distributed.dispatch import concatenate_outputs_after_mesh_dispatch
from skyrl_train.group_admission import GroupAdvantageInvariant, GroupAdvantageKind
from skyrl_train.io.io import write_bytes_atomic
from skyrl_train.inference_engines.utils import get_sampling_params_for_backend, route_prompts_to_engines
from skyrl_train.inference_engines.vllm_teacher_oracle import tokenizer_vocabulary_fingerprint
from skyrl_train.mismatch_probe.archive import (
    BUILDING_STATUS,
    COMPLETE_STATUS,
    MismatchArchive,
    read_frozen_probe,
)
from skyrl_train.mismatch_probe.protocol import (
    probe_hash,
    request_seed,
    require_token_identity,
)
from skyrl_train.mismatch_probe.modes import NATIVE_MODE, REPEAT_MODE, TRAINER_MODES
from skyrl_train.mismatch_probe.provenance import manifest
from skyrl_train.models.megatron_router_replay import SENTINEL_EXPERT_ID
from skyrl_train.training_batch import TrainingInputBatch
from skyrl_train.trajectory_runners.types import TrajectoryID
from skyrl_train.trajectory_runners.trajectory_processing import (
    concatenate_trajectory_batches,
    prepare_trajectory_request,
    scalar_reward_token_credit,
)


@dataclass(frozen=True)
class EncodedRoutes:
    """Router choices encoded with their NumPy shape and dtype."""

    data: bytes | None
    shape: list[int] | None
    dtype: str | None
    replacements: bytes | None = None


@dataclass(frozen=True)
class ProbeSamples:
    """Trajectory rows and stable request identities for one probe."""

    trajectory: dict
    prompt_ids: list[str]
    sample_ids: list[str]
    seeds: list[int]


@dataclass(frozen=True)
class BatchLayout:
    sample_ids: list[str]
    native_order: list[int]
    repeat_order: list[int]
    padded_rows: int
    native_micro_batch_size: int
    repeat_micro_batch_size: int
    collated_bytes: int
    collated_route_bytes: int
    nonzero_advantage_samples: int


def _encode_routes(routes: torch.Tensor | None, length: int) -> EncodedRoutes:
    if routes is None:
        return EncodedRoutes(None, None, None)
    value = routes[:length].contiguous().cpu().numpy()
    return EncodedRoutes(value.tobytes(), list(value.shape), str(value.dtype))


# Timed forward + backward repetitions per mode, after one warmup pass.
TIMING_REPETITIONS = 3


def _reread_routes(output, rows) -> list[np.ndarray | None]:
    """Return each cache-off re-read's full-sequence routes, or ``None`` when the engine captured none."""
    routes = output.get("routed_experts")
    if routes is None:
        return [None] * len(rows)
    if len(routes) != len(rows):
        raise ValueError("vLLM re-read returned routes for a different number of prefixes")
    result = []
    for row, route in zip(rows, routes, strict=True):
        expected = len(row.prompt_token_ids) + len(row.vllm_output_ids) - 1
        if route is None or route.ndim != 3 or route.shape[0] != expected:
            raise ValueError(f"vLLM re-read routes for {row.sample_id} do not cover prompt and response inputs")
        result.append(np.ascontiguousarray(route.astype(np.uint8 if route.max(initial=0) <= 255 else np.int32)))
    return result


def _reread_route_tensors(rows, sources, response_like: torch.Tensor, prompt_width: int):
    """Place full-sequence re-read routes on the trainer's response and left-padded prompt axes."""
    batch, response_width, layers, top_k = response_like.shape
    response = torch.zeros_like(response_like)
    prompt = torch.zeros((batch, prompt_width, layers, top_k), dtype=response_like.dtype)
    for position, (row, route) in enumerate(zip(rows, sources, strict=True)):
        values = torch.as_tensor(route, dtype=response_like.dtype)
        prompt_length = len(row.prompt_token_ids)
        prompt[position, prompt_width - prompt_length :] = values[:prompt_length]
        # Rows after the prompt are the response inputs 0 .. R-2; the final response token feeds no score.
        response[position, : len(row.vllm_output_ids) - 1] = values[prompt_length:]
    return response, prompt


def _reorder_batch(batch: TrainingInputBatch, order: list[int]) -> TrainingInputBatch:
    index = torch.tensor(order, dtype=torch.long)
    reordered = TrainingInputBatch({key: value[index] if value is not None else None for key, value in batch.items()})
    reordered.metadata = dict(batch.metadata)
    if "uids" in reordered.metadata:
        reordered.metadata["uids"] = [batch.metadata["uids"][i] for i in order]
    return reordered


def _batched_dp_ranks(trainer, prompts: int) -> list[int] | None:
    """The vLLM data-parallel rank the client's even split sends each prompt of one batched request to.

    A single prompt goes to a random engine, so its rank is unknown (``None``).
    """
    if prompts < 2:
        return None
    dp_size = int(trainer.cfg.generator.inference_engine_data_parallel_size)
    placement = route_prompts_to_engines(prompts, len(trainer.inference_engine_client.engines), None)
    ranks = [0] * prompts
    for engine, rows in placement.items():
        for row in rows:
            ranks[row] = engine % dp_size
    return ranks


def _served_dp_ranks(output, trainer, prompts: int) -> list[int] | None:
    """The vLLM data-parallel rank that served each prompt (engine ``i`` is rank ``i % dp``), when reported."""
    served = output.get("engine_indices")
    if served is None:
        return None
    if len(served) != prompts:
        raise ValueError("vLLM re-read reported serving engines for a different number of prompts")
    dp_size = int(trainer.cfg.generator.inference_engine_data_parallel_size)
    ranks = [int(engine) % dp_size for engine in served]
    if ranks != _batched_dp_ranks(trainer, prompts):
        logger.warning("mismatch probe: the re-read left the client's even split (an engine failed over)")
    return ranks


def _generation_dp_ranks(trainer, rows) -> list[int]:
    """The vLLM data-parallel rank that generated each frozen sample.

    ``_generate`` runs one trajectory per request, whose session id is its ``TrajectoryID`` string; the
    client sends a single prompt to engine ``sha256(session id) % engines`` (``route_prompts_to_engines``).
    """
    dp_size = int(trainer.cfg.generator.inference_engine_data_parallel_size)
    engines = len(trainer.inference_engine_client.engines)
    ranks = []
    for row in rows:
        # ``_generate`` names each sample ``<prompt uid>:<repetition>``.
        uid, separator, repetition = row.sample_id.rpartition(":")
        if not separator or uid != row.prompt_id:
            raise ValueError(f"probe sample {row.sample_id} is not named <prompt uid>:<repetition>")
        session = TrajectoryID(instance_id=uid, repetition_id=int(repetition)).to_string()
        ((engine, _),) = route_prompts_to_engines(1, engines, [session]).items()
        ranks.append(engine % dp_size)
    return ranks


def _candidate_logprobs(output, overrides):
    ids = output.get("student_topk_indices")
    values = output.get("behavior_topk_logprobs")
    if ids is None or values is None or len(ids) != len(overrides) or len(values) != len(overrides):
        raise ValueError("vLLM re-read returned incomplete requested candidates")
    result = []
    for override, candidates, scores in zip(overrides, ids, values, strict=True):
        token = override["logprob_token_ids"][0]
        if len(candidates) != 1 or candidates[0] != [token] or len(scores) != 1 or len(scores[0]) != 1:
            raise ValueError("vLLM re-read omitted the frozen response token")
        result.append(float(scores[0][0]))
    return result


def _prompt_group_contract(trainer, samples_per_prompt: int) -> GroupAdvantageInvariant:
    current = trainer.group_advantage_invariant
    minimum = current.minimum_group_size
    if current.kind is GroupAdvantageKind.MINIMUM_BASELINE_ELIGIBLE:
        if samples_per_prompt < 2:
            raise ValueError("mismatch probe requires at least two samples for a baseline-eligible group")
        minimum = min(minimum, samples_per_prompt)
    return replace(current, physical_group_size=samples_per_prompt, minimum_group_size=minimum)


class ProbeCollector:
    """Frozen probe data and engine collection independent of callback scheduling."""

    def __init__(self, cfg):
        self.cfg = cfg
        self.spec = cfg.trainer.mismatch_probe
        self.updates = tuple(self.spec.score_after_updates)
        self.archive_uri = self.spec.archive_uri
        self.archive: MismatchArchive | None = None
        self.probes = None
        self.training_input: TrainingInputBatch | None = None
        self.probe_hash: str | None = None
        self.starting_weights_hash: str | None = None
        self.source_manifest = None
        self.generation_scores: list[np.ndarray] | None = None
        self.timing: dict[str, float] = {}
        self.weights: dict[int, str] = {}
        self.batch_layout: BatchLayout | None = None
        self.metrics: dict[str, object] = {}
        self.created_at = datetime.now(UTC).isoformat()
        self.starting_global_step: int | None = None
        self.scored_global_steps: dict[int, int] = {}
        self.tokenizer_fingerprint: str | None = None
        self.cache_hit_tokens: dict[str, int] = {}
        # Cache-off re-read routes per update, [prompt + response - 1, layer, top_k] per sample.
        self.reread_routes: dict[int, list[np.ndarray]] = {}
        # The reused source's prefill reference rows, keyed by sample.
        self.frozen_rereads: dict[str, mismatch.ScoreRow] = {}
        # The vLLM data-parallel rank that served each cache-off re-read row, per update; for the frozen
        # source, the rank the client's even split gave each row (its re-read was one batched request).
        self.reread_dp_ranks: dict[int, list[int]] = {}
        self.frozen_reread_dp_ranks: list[int] | None = None
        # The vLLM data-parallel rank that generated each frozen sample.
        self.generation_dp_ranks: list[int] | None = None
        # Per vLLM worker: placement, parameter digest and versions (output code is written beside the archive).
        self.vllm_provenance: list[dict] = []

    async def _generate(self, trainer):
        if trainer.eval_dataset is None:
            raise ValueError("mismatch probe requires the run's validation dataset")
        count = int(self.spec.prompts.count)
        samples = int(self.spec.prompts.samples_per_prompt)
        if len(trainer.eval_dataset) < count:
            raise ValueError(
                f"mismatch probe requested {count} validation prompts but only {len(trainer.eval_dataset)} exist"
            )
        prompts = trainer.eval_dataset.collate_fn([trainer.eval_dataset[i] for i in range(count)])
        batches = []
        seeds = []
        sample_ids = []
        uids = []
        await trainer.trajectory_runner.start_eval_session(
            run_name=trainer.cfg.trainer.get("run_name") or "mismatch-probe",
            eval_step=trainer.global_step,
            val_set_name="mismatch-probe",
        )
        try:
            for prompt in prompts:
                for repetition in range(samples):
                    seed = request_seed(int(self.spec.seed), prompt["uid"], repetition)
                    sampling = get_sampling_params_for_backend(
                        trainer.cfg.generator.backend, trainer.cfg.generator.sampling_params
                    )
                    sampling["seed"] = seed
                    request, _ = prepare_trajectory_request(
                        [prompt], 1, sampling, trainer.cfg.environment.env_class, "eval", 0
                    )
                    request["trajectory_ids"][0].repetition_id = repetition
                    batches.append(await trainer.trajectory_runner.run(request))
                    seeds.append(seed)
                    sample_ids.append(f"{prompt['uid']}:{repetition}")
                    uids.append(prompt["uid"])
        finally:
            await trainer.trajectory_runner.stop_eval_session()
        trajectory = concatenate_trajectory_batches(
            batches,
            require_rollout_logprobs=True,
            tis_lcs_alert_threshold=float(trainer.cfg.trainer.algorithm.tis_lcs_alert_threshold),
        )
        return ProbeSamples(trajectory, uids, sample_ids, seeds)

    def _from_source(self):
        source = read_frozen_probe(self.spec.reuse_probe)
        manifest, rows, generations = source.manifest, source.probes, source.generations
        expected = int(self.spec.prompts.count) * int(self.spec.prompts.samples_per_prompt)
        if len(rows) != expected:
            raise ValueError(f"reuse_probe has {len(rows)} samples but this recipe requests {expected}")
        group_counts: dict[str, int] = {}
        for row in rows:
            group_counts[row.prompt_id] = group_counts.get(row.prompt_id, 0) + 1
        if len(group_counts) != int(self.spec.prompts.count) or any(
            count != int(self.spec.prompts.samples_per_prompt) for count in group_counts.values()
        ):
            raise ValueError("reuse_probe prompt groups do not match this recipe")
        self.source_manifest = manifest
        self.frozen_rereads = source.rereads
        self.probes = rows
        self.probe_hash = manifest.probe_hash
        routes = []
        for row in rows:
            if row.routed_experts is None:
                routes.append(None)
            else:
                decoded = np.frombuffer(row.routed_experts, dtype=row.routed_experts_dtype).reshape(
                    row.routed_experts_shape
                )
                routes.append(decoded.copy())
        if any(route is None for route in routes) and any(route is not None for route in routes):
            raise ValueError("reuse_probe source mixes captured and missing routed-expert rows")
        prompt_routes = [
            None
            if row.prompt_routed_experts is None
            else np.frombuffer(row.prompt_routed_experts, dtype=row.prompt_routed_experts_dtype)
            .reshape(row.prompt_routed_experts_shape)
            .copy()
            for row in rows
        ]
        self.generation_scores = [np.asarray(generations[row.sample_id].logprobs, dtype=np.float32) for row in rows]
        trajectory = {
            "prompt_token_ids": [row.prompt_token_ids for row in rows],
            "response_ids": [row.trainer_input_ids for row in rows],
            "rewards": [scalar_reward_token_credit(row.reward or 0.0, row.trainer_input_ids) for row in rows],
            "loss_masks": [[int(value) for value in row.loss_mask] for row in rows],
            "rollout_logprobs": self.generation_scores,
            "rollout_routed_experts": routes if all(route is not None for route in routes) else None,
            **(
                {"rollout_prompt_routed_experts": prompt_routes}
                if all(route is not None for route in prompt_routes)
                else {}
            ),
            "stop_reasons": ["reused"] * len(rows),
            "rollout_metrics": None,
        }
        return ProbeSamples(
            trajectory,
            [row.prompt_id for row in rows],
            [row.sample_id for row in rows],
            [row.request_seed for row in rows],
        )

    def _collate_and_freeze(self, trainer, trajectory, uids, sample_ids, seeds):
        if self.source_manifest is not None:
            source_tokenizer = self.source_manifest.tokenizer_fingerprint
            self.tokenizer_fingerprint = tokenizer_vocabulary_fingerprint(trainer.inference_engine_client.tokenizer)
            if not source_tokenizer or source_tokenizer != self.tokenizer_fingerprint:
                raise ValueError("reuse_probe tokenizer fingerprint differs from the source archive")
        generated = trajectory["response_ids"]
        if generated is None:
            raise ValueError("mismatch probe requires exact vLLM response IDs from the trajectory runner")
        raw_rewards = trajectory["rewards"]
        rewards = [float(sum(value)) if isinstance(value, list) else float(value) for value in raw_rewards]
        original_contract = trainer.group_advantage_invariant
        original_metrics = dict(trainer.all_metrics)
        try:
            trainer.group_advantage_invariant = _prompt_group_contract(
                trainer, int(self.spec.prompts.samples_per_prompt)
            )
            if self.source_manifest is None:
                trajectory = trainer.postprocess_trajectory_batch(trajectory, uids)
            training_input = trainer.convert_to_training_input(trajectory, uids)
            training_input["values"] = None
            if self.source_manifest is None:
                training_input = trainer.compute_advantages_and_returns(training_input)
        finally:
            trainer.group_advantage_invariant = original_contract
            trainer.all_metrics = original_metrics

        response_width = training_input["response_mask"].shape[1]
        prompt_width = training_input["sequences"].shape[1] - response_width
        padded_count = len(training_input["sequences"]) - len(sample_ids)
        if padded_count < 0 or padded_count != training_input.metadata.get("pad_size", 0):
            raise ValueError("mismatch probe training collation changed the number of frozen samples")
        rows = []
        schema = mismatch
        for position, sample_id in enumerate(sample_ids):
            response_length = len(trajectory["response_ids"][position])
            prompt_tokens = training_input["sequences"][position, :prompt_width]
            prompt_mask = training_input["attention_mask"][position, :prompt_width].bool()
            trainer_prompt = prompt_tokens[prompt_mask].tolist()
            trainer_response = training_input["sequences"][
                position, prompt_width : prompt_width + response_length
            ].tolist()
            require_token_identity(
                sample_id=sample_id,
                expected_prompt=trajectory["prompt_token_ids"][position],
                trainer_prompt=trainer_prompt,
                engine_response=generated[position],
                trainer_response=trainer_response,
            )
            routes = training_input.get("rollout_routed_experts")
            encoded_routes = _encode_routes(None if routes is None else routes[position], response_length)
            route_valid_mask = None
            if encoded_routes.shape is not None:
                if len(encoded_routes.shape) != 3 or encoded_routes.shape[0] != response_length:
                    raise ValueError("captured routes must align with the frozen response tokens")
                route_valid_mask = (
                    (routes[position, :response_length] != SENTINEL_EXPERT_ID).any(dim=-1).bool().tolist()
                )
                if self.source_manifest is not None:
                    route_valid_mask = self.probes[position].route_valid_mask
            prompt_routes = training_input.get("rollout_prompt_routed_experts")
            encoded_prompt_routes = (
                EncodedRoutes(None, None, None)
                if prompt_routes is None
                else _encode_routes(prompt_routes[position, prompt_width - len(trainer_prompt) :], len(trainer_prompt))
            )
            advantage = None
            if self.source_manifest is None:
                valid = training_input["loss_mask"][position, :response_length].bool()
                values = training_input["advantages"][position, :response_length][valid]
                advantage = float(values.mean().item()) if values.numel() else 0.0
            else:
                advantage = self.probes[position].advantage
            rows.append(
                schema.ProbeRow(
                    probe_hash="pending",
                    sample_id=sample_id,
                    prompt_id=uids[position],
                    prompt_token_ids=list(trajectory["prompt_token_ids"][position]),
                    trainer_prompt_ids=trainer_prompt,
                    vllm_output_ids=list(generated[position]),
                    trainer_input_ids=trainer_response,
                    response_mask=training_input["response_mask"][position, :response_length].bool().tolist(),
                    loss_mask=training_input["loss_mask"][position, :response_length].bool().tolist(),
                    reward=rewards[position],
                    advantage=advantage,
                    request_seed=seeds[position],
                    batch_position=position,
                    routed_experts=encoded_routes.data,
                    routed_experts_shape=encoded_routes.shape,
                    routed_experts_dtype=encoded_routes.dtype,
                    route_valid_mask=route_valid_mask,
                    prompt_routed_experts=encoded_prompt_routes.data,
                    prompt_routed_experts_shape=encoded_prompt_routes.shape,
                    prompt_routed_experts_dtype=encoded_prompt_routes.dtype,
                )
            )
        digest = probe_hash(
            (row.sample_id, row.prompt_token_ids, row.vllm_output_ids, row.response_mask, row.loss_mask) for row in rows
        )
        if self.source_manifest is not None and digest != self.probe_hash:
            raise ValueError("reuse_probe token hash changed during trainer collation")
        self.probe_hash = digest
        self.probes = [row.model_copy(update={"probe_hash": digest}) for row in rows]
        self.training_input = training_input
        self.generation_scores = trajectory.get("rollout_logprobs")
        if self.generation_scores is None or any(
            len(scores) != len(row.vllm_output_ids) or not all(math.isfinite(score) for score in scores)
            for row, scores in zip(self.probes, self.generation_scores, strict=True)
        ):
            raise ValueError("mismatch probe generation-time logprob vectors are incomplete or nonfinite")
        self.batch_layout = BatchLayout(
            sample_ids=sample_ids,
            native_order=list(range(len(sample_ids))),
            repeat_order=list(reversed(range(len(sample_ids)))),
            padded_rows=padded_count,
            native_micro_batch_size=int(trainer.cfg.trainer.micro_forward_batch_size_per_gpu),
            repeat_micro_batch_size=max(2, 2 * int(trainer.cfg.trainer.micro_forward_batch_size_per_gpu)),
            collated_bytes=sum(
                value.numel() * value.element_size() for value in training_input.values() if value is not None
            ),
            collated_route_bytes=(
                0
                if training_input.get("rollout_routed_experts") is None
                else training_input["rollout_routed_experts"].numel()
                * training_input["rollout_routed_experts"].element_size()
            ),
            nonzero_advantage_samples=(
                0
                if training_input.get("advantages") is None
                else int((training_input["advantages"].abs().sum(dim=1) > 0).sum().item())
            ),
        )

    async def _rescore_vllm(self, trainer, update: int):
        rows = self.probes
        cache_setting = self.spec.rescore_prefix_cache
        cache_modes = (CACHE_OFF, CACHE_ON) if cache_setting == CACHE_BOTH else (cache_setting,)
        passes = [(cache_mode, RESCORE_SCORER) for cache_mode in cache_modes]
        if self.spec.get("reread_again", False) and CACHE_OFF in cache_modes:
            passes.append((CACHE_OFF, RESCORE_AGAIN_SCORER))
        result = []
        for cache_mode, scorer in passes:
            if cache_mode == CACHE_OFF:
                await trainer.inference_engine_client.reset_prefix_cache()
                prefixes = [row.prompt_token_ids + row.vllm_output_ids[:-1] for row in rows]
                overrides = [{"logprob_token_ids": [row.vllm_output_ids[-1]]} for row in rows]
                sampling = {"prompt_logprobs": 1, "logprobs": 1}
            else:
                # Cached decode scores require one prefix per frozen response token.
                prefixes = [
                    row.prompt_token_ids + row.vllm_output_ids[:index]
                    for row in rows
                    for index in range(len(row.vllm_output_ids))
                ]
                overrides = [{"logprob_token_ids": [token]} for row in rows for token in row.vllm_output_ids]
                sampling = {"logprobs": 1}
            started = time.monotonic()
            engine_input = {
                "prompts": None,
                "prompt_token_ids": prefixes,
                "sampling_params": {
                    "max_tokens": 1,
                    "temperature": 1.0,
                    "skip_reading_prefix_cache": cache_mode == CACHE_OFF,
                    "seed": request_seed(int(self.spec.seed), f"reread:{update}:{cache_mode}", 0),
                    **sampling,
                },
                "session_ids": None,
            }
            engine_input["sampling_params_per_prompt"] = overrides
            output = await trainer.inference_engine_client.generate(engine_input)
            duration = time.monotonic() - started
            if cache_mode == CACHE_OFF and scorer == RESCORE_SCORER:
                self.generation_dp_ranks = _generation_dp_ranks(trainer, rows)
                self.reread_dp_ranks[update] = _served_dp_ranks(output, trainer, len(prefixes))
                if self.frozen_rereads:
                    self.frozen_reread_dp_ranks = _batched_dp_ranks(trainer, len(prefixes))
            chosen = []
            candidates = _candidate_logprobs(output, overrides)
            if cache_mode == CACHE_OFF:
                prompt_scores = output.get("prompt_logprobs")
                if prompt_scores is None or len(prompt_scores) != len(rows):
                    raise ValueError("vLLM re-read returned incomplete prompt logprobs")
                for row, values, final_score in zip(rows, prompt_scores, candidates, strict=True):
                    chosen.append(
                        [
                            float(values[len(row.prompt_token_ids) + index][token])
                            for index, token in enumerate(row.vllm_output_ids[:-1])
                        ]
                        + [final_score]
                    )
            else:
                offset = 0
                for row in rows:
                    chosen.append(candidates[offset : offset + len(row.vllm_output_ids)])
                    offset += len(row.vllm_output_ids)
            label = rescore_label(update, cache_mode) + (":again" if scorer == RESCORE_AGAIN_SCORER else "")
            self.timing[f"{label}/seconds"] = duration
            cache_hits = output.get("prefix_cache_hit_tokens")
            if cache_hits is None or len(cache_hits) != len(prefixes):
                raise ValueError("vLLM re-read omitted prefix-cache hit counts")
            self.cache_hit_tokens[label] = sum(cache_hits)
            if cache_mode == CACHE_OFF and self.cache_hit_tokens[label]:
                raise ValueError("cache-off re-read unexpectedly used cached prefix tokens")
            routes = _reread_routes(output, rows) if cache_mode == CACHE_OFF else [None] * len(rows)
            if scorer == RESCORE_SCORER and cache_mode == CACHE_OFF and routes[0] is not None:
                self.reread_routes[update] = routes
            for row, values, route in zip(rows, chosen, routes, strict=True):
                if len(values) != len(row.vllm_output_ids) or not all(math.isfinite(value) for value in values):
                    raise ValueError(f"vLLM re-read returned incomplete or nonfinite scores for {row.sample_id}")
                result.append(
                    mismatch.ScoreRow(
                        probe_hash=self.probe_hash,
                        sample_id=row.sample_id,
                        scorer=scorer,
                        update=update,
                        weights_hash=self.weights[update],
                        cache_mode=cache_mode,
                        logprobs=values,
                        forward_seconds=duration / len(rows),
                        expert_choices=None if route is None else route.tobytes(),
                        expert_choices_shape=None if route is None else list(route.shape),
                        expert_choices_dtype=None if route is None else str(route.dtype),
                    )
                )
        return result

    async def _record_vllm_provenance(self, trainer) -> None:
        """Record each vLLM worker's parameter digest and versions, and write its compiled output code."""
        engines = await trainer.inference_engine_client.probe_numerics_provenance()
        code_root = f"{self.archive_uri.rstrip('/')}-inductor-output-code"
        for engine_index, workers in enumerate(engines):
            for worker in workers:
                placement = worker["placement"]
                worker_dir = f"{code_root}/engine-{engine_index}/dp-{placement['dp_rank']}-ep-{placement['ep_rank']}"
                for relative_path, text in worker["inductor_output_code"].items():
                    write_bytes_atomic(f"{worker_dir}/{relative_path}", text.encode())
                self.vllm_provenance.append(
                    {
                        "engine": engine_index,
                        "placement": placement,
                        "parameter_sha256": worker["parameter_sha256"],
                        "versions": worker["versions"],
                        "output_code_files": len(worker["inductor_output_code"]),
                        "output_code_uri": worker_dir,
                    }
                )

    def _policy_weights_hash(self, trainer) -> str:
        shards = ray.get(trainer.policy_model.async_run_ray_method("pass_through", "probe_weights_digest"))
        if not shards or not all(isinstance(shard, str) and len(shard) == 64 for shard in shards):
            raise ValueError("policy workers did not return complete weight digests")
        return "sha256:" + hashlib.sha256("".join(shards).encode()).hexdigest()

    def _route_observations(self, outputs, mode: str):
        """Gather PP-owned router choices by frozen sample and captured layer."""
        if not self.cfg.trainer.policy.megatron_config.moe_router_replay or all(
            row.routed_experts is None for row in self.probes
        ):
            return [None] * len(self.probes)
        choices = []
        replacements = []
        seen = []
        for row in self.probes:
            if row.routed_experts_shape is None or len(row.routed_experts_shape) != 3:
                raise ValueError("probe route observations require [response, layer, topk] captured routes")
            shape = tuple(row.routed_experts_shape)
            choices.append(np.full(shape, -1, dtype=np.int32))
            replacements.append(np.zeros(shape, dtype=np.bool_))
            seen.append(np.zeros(shape[:2], dtype=np.bool_))
        for output in outputs:
            for observation in output.metadata.get("probe_routes", []):
                sample = int(observation["sample"])
                position = int(observation["position"])
                layer = int(observation["layer"])
                if sample < 0 or sample >= len(self.probes) + self.batch_layout.padded_rows:
                    raise ValueError(f"probe router returned sample index {sample} outside the frozen batch")
                if sample >= len(self.probes):
                    continue
                if position >= choices[sample].shape[0]:
                    if observation["route_valid"]:
                        raise ValueError("probe router marked a padded response position as replayable")
                    continue
                if position < 0 or layer < 0 or layer >= choices[sample].shape[1]:
                    raise ValueError("probe router returned an invalid response position or layer")
                if seen[sample][position, layer]:
                    if not np.array_equal(
                        choices[sample][position, layer], observation["effective"]
                    ) or not np.array_equal(replacements[sample][position, layer], observation["replaced"]):
                        raise ValueError(f"conflicting probe routing observation for {sample}/{position}/{layer}")
                    continue
                seen[sample][position, layer] = True
                choices[sample][position, layer] = observation["effective"]
                replacements[sample][position, layer] = observation["replaced"]
        result = []
        for row, selected, changed, observed in zip(self.probes, choices, replacements, seen, strict=True):
            valid = np.asarray(row.route_valid_mask, dtype=np.bool_)
            if not np.all(observed[valid]):
                missing = int(np.count_nonzero(valid & ~observed))
                raise ValueError(f"{mode} missed {missing} captured routing rows for {row.sample_id}")
            result.append(
                EncodedRoutes(selected.tobytes(), list(selected.shape), str(selected.dtype), changed.tobytes())
            )
        return result

    def _mode_batches(self, update: int, modes: tuple[str, ...]) -> dict[str, TrainingInputBatch]:
        """Prepare each mode's scoring batch: its routes, prompt routes and batch layout."""
        training_input = self.training_input
        rows = self.probes
        n = len(rows)
        route_tensor = training_input.get("rollout_routed_experts")
        prompt_route_tensor = training_input.get("rollout_prompt_routed_experts")
        route_keys = [
            *(["rollout_routed_experts"] if route_tensor is not None else []),
            *(["rollout_prompt_routed_experts"] if prompt_route_tensor is not None else []),
        ]
        reread_tensors = None
        if any(TRAINER_MODES[mode].route_source == "reread" for mode in modes):
            if route_tensor is None:
                raise ValueError("re-read replay modes require captured generation routes for the replay layout")
            if update == 0 and self.frozen_rereads:
                sources = [
                    np.frombuffer(
                        self.frozen_rereads[row.sample_id].expert_choices,
                        dtype=self.frozen_rereads[row.sample_id].expert_choices_dtype,
                    ).reshape(self.frozen_rereads[row.sample_id].expert_choices_shape)
                    for row in rows
                ]
            elif update in self.reread_routes:
                sources = self.reread_routes[update]
            else:
                raise ValueError(f"re-read replay at update {update} has no captured re-read routes")
            prompt_width = training_input["sequences"].shape[1] - route_tensor.shape[1]
            reread_tensors = _reread_route_tensors(rows, sources, route_tensor, prompt_width)
            ranks = (
                self.frozen_reread_dp_ranks if update == 0 and self.frozen_rereads else self.reread_dp_ranks.get(update)
            )
            if ranks is not None and len(ranks) != n:
                raise ValueError(f"re-read replay at update {update} has vLLM data-parallel ranks for other rows")
        batches = {}
        for mode in modes:
            spec = TRAINER_MODES[mode]
            data = training_input.select(["sequences", "attention_mask", "loss_mask", *route_keys], ["response_length"])
            data["probe_row_indices"] = torch.arange(data.batch_size, dtype=torch.long)
            if not spec.requires_routes and route_tensor is not None:
                data["rollout_routed_experts"] = torch.zeros_like(route_tensor)
            if not spec.replays_prompt and prompt_route_tensor is not None:
                data["rollout_prompt_routed_experts"] = torch.zeros_like(prompt_route_tensor)
            # Replay modes carry the vLLM data-parallel rank that served their routes' source; padding rows past
            # the frozen samples take rank 0, and their scores are dropped.
            if spec.route_source == "reread":
                data["rollout_routed_experts"], data["rollout_prompt_routed_experts"] = reread_tensors
                if ranks is not None:
                    data["vllm_dp_rank"] = torch.tensor(ranks + [0] * (data.batch_size - n), dtype=torch.long)
            elif spec.requires_routes and self.generation_dp_ranks is not None:
                padding = [0] * (data.batch_size - n)
                data["vllm_dp_rank"] = torch.tensor(self.generation_dp_ranks + padding, dtype=torch.long)
            if spec.repeat_layout:
                data = _reorder_batch(data, self.batch_layout.repeat_order + list(range(n, data.batch_size)))
            data.metadata.update(
                probe_mode=mode,
                probe_keep_fraction=(
                    float(self.spec.filtered_replay.keep_fraction) if spec.requires_keep_fraction else None
                ),
                probe_micro_batch_size=self.batch_layout.repeat_micro_batch_size if spec.repeat_layout else None,
                global_step=update,
            )
            batches[mode] = data
        return batches

    def _training_pass_timing(self, trainer, update: int) -> None:
        """Time forward + backward on the probe batch for each mode in trainer.mismatch_probe.timing_modes."""
        modes = tuple(self.spec.get("timing_modes") or ())
        for mode, data in self._mode_batches(update, modes).items():
            data.metadata["probe_timing_repetitions"] = TIMING_REPETITIONS
            outputs = ray.get(trainer.policy_model.async_run_ray_method("mesh", "probe_time_training_pass", data=data))
            per_rank = [output for output in outputs]
            label = f"training_pass@{update}:{mode}"
            # The slowest rank sets the pass time; keep each repetition.
            self.timing[f"{label}/seconds"] = [
                max(rank["seconds"][repetition] for rank in per_rank) for repetition in range(TIMING_REPETITIONS)
            ]
            self.timing[f"{label}/peak_memory_bytes"] = max(rank["peak_memory_bytes"] for rank in per_rank)

    def _trainer_scores(self, trainer, update: int):
        rows = self.probes
        n = len(rows)
        modes = (NATIVE_MODE, REPEAT_MODE, *self.spec.extra_trainer_modes)
        batches = self._mode_batches(update, modes)
        result = []
        for mode, data in batches.items():
            order = (
                self.batch_layout.repeat_order if TRAINER_MODES[mode].repeat_layout else self.batch_layout.native_order
            )
            if TRAINER_MODES[mode].captures_layers:
                data.metadata["probe_capture"] = {
                    "uri": f"{self.archive_uri.rstrip('/')}-trainer-capture/update-{update}/{mode}",
                    "layers": list(self.spec.get("capture_layers") or ()),
                }
            started = time.monotonic()
            outputs = ray.get(trainer.policy_model.async_run_ray_method("mesh", "probe_forward", data=data))
            values = concatenate_outputs_after_mesh_dispatch(trainer.policy_model.actor_infos, outputs)["output"][:n]
            route_observations = self._route_observations(outputs, mode)
            duration = time.monotonic() - started
            label = trainer_label(update, mode)
            self.timing[f"{label}/seconds"] = duration
            for ordered_position, original_position in enumerate(order):
                row = rows[original_position]
                length = len(row.vllm_output_ids)
                logprobs = values[ordered_position, :length].float().tolist()
                if len(logprobs) != length or not all(math.isfinite(value) for value in logprobs):
                    raise ValueError(f"{label} returned incomplete or nonfinite scores for {row.sample_id}")
                result.append(
                    mismatch.ScoreRow(
                        probe_hash=self.probe_hash,
                        sample_id=row.sample_id,
                        scorer=TRAINER_SCORER,
                        mode=mode,
                        update=update,
                        weights_hash=self.weights[update],
                        logprobs=logprobs,
                        forward_seconds=duration / n,
                        expert_choices=(
                            None
                            if route_observations[original_position] is None
                            else route_observations[original_position].data
                        ),
                        expert_choices_shape=(
                            None
                            if route_observations[original_position] is None
                            else route_observations[original_position].shape
                        ),
                        expert_choices_dtype=(
                            None
                            if route_observations[original_position] is None
                            else route_observations[original_position].dtype
                        ),
                        replacement_mask=(
                            None
                            if route_observations[original_position] is None
                            else route_observations[original_position].replacements
                        ),
                    )
                )
        return result


async def collect(probe: ProbeCollector, trainer, *, update: int) -> list[mismatch.ScoreRow]:
    python_rng = random.getstate()
    numpy_rng = np.random.get_state()
    cpu_rng = torch.get_rng_state()
    cuda_rng = torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else None
    try:
        started = time.monotonic()
        probe.scored_global_steps[update] = trainer.global_step
        if update == 0:
            if probe.spec.reuse_probe:
                samples = await asyncio.to_thread(probe._from_source)
            else:
                generation_started = time.monotonic()
                original_metrics = dict(trainer.all_metrics)
                try:
                    samples = await probe._generate(trainer)
                finally:
                    trainer.all_metrics = original_metrics
                probe.timing["probe/generation_seconds"] = time.monotonic() - generation_started
            probe._collate_and_freeze(
                trainer, samples.trajectory, samples.prompt_ids, samples.sample_ids, samples.seeds
            )
        elif probe.training_input is None:
            raise RuntimeError("mismatch probe update 0 was not collected")

        # The policy is still offloaded and the inference engine awake after
        # each normal weight sync. Re-read first, then temporarily swap residency.
        rescore_started = time.monotonic()
        probe.weights[update] = "pending"
        rescore_rows = await probe._rescore_vllm(trainer, update)
        probe.timing[f"update@{update}/vllm_total_seconds"] = time.monotonic() - rescore_started
        if update == 0:
            provenance_started = time.monotonic()
            await probe._record_vllm_provenance(trainer)
            probe.timing["probe/vllm_provenance_seconds"] = time.monotonic() - provenance_started
        # Training-pass timing runs backward, which needs the gradient buffers offloaded with the optimizer.
        timing = bool(probe.spec.get("timing_modes"))
        if trainer.colocate_all:
            await trainer.inference_engine_client.sleep()
        try:
            if trainer.colocate_all:
                trainer.policy_model.backload_to_gpu(backload_optimizer=timing, backload_model=True)
            try:
                probe.weights[update] = probe._policy_weights_hash(trainer)
                if update == 0:
                    probe.starting_weights_hash = probe.weights[0]
                    if (
                        probe.source_manifest is not None
                        and probe.source_manifest.starting_weights_hash != probe.weights[0]
                    ):
                        raise ValueError("reuse_probe starting weights differ from the source archive")
                # Re-read rows were delayed until this exact weight identity is known.
                for row in rescore_rows:
                    row.weights_hash = probe.weights[update]
                # Timing first: a failing candidate's training pass then costs no finished scores.
                probe._training_pass_timing(trainer, update)
                trainer_rows = probe._trainer_scores(trainer, update)
                if probe._policy_weights_hash(trainer) != probe.weights[update]:
                    raise ValueError(f"probe scoring changed policy weights or buffers at update {update}")
            finally:
                if trainer.colocate_all:
                    trainer.policy_model.offload_to_cpu(offload_optimizer=timing, offload_model=True)
        finally:
            if trainer.colocate_all:
                await trainer.inference_engine_client.wake_up()

        scores = rescore_rows + trainer_rows
        if update == 0:
            scores.extend(
                probe.frozen_rereads[row.sample_id].model_copy(
                    update={
                        "scorer": FROZEN_RESCORE_SCORER,
                        "probe_hash": probe.probe_hash,
                        "weights_hash": probe.weights[0],
                    }
                )
                for row in probe.probes
                if probe.frozen_rereads
            )
            for row, values in zip(probe.probes, probe.generation_scores, strict=True):
                scores.append(
                    mismatch.ScoreRow(
                        probe_hash=probe.probe_hash,
                        sample_id=row.sample_id,
                        scorer=GENERATION_SCORER,
                        update=0,
                        weights_hash=probe.weights[0],
                        logprobs=values,
                    )
                )
        probe.metrics[f"update@{update}"] = {
            "token_count": sum(sum(row.loss_mask) for row in probe.probes),
            "sample_count": len(probe.probes),
            "route_bytes": sum(len(row.routed_experts or b"") for row in probe.probes),
            "optimizer_steps": int(trainer.all_metrics.get("policy/policy_update_steps", 0)),
            "step_timings": {
                key: float(value) for key, value in trainer.all_timings.items() if isinstance(value, (int, float))
            },
        }
        reference = {
            row.sample_id: row for row in scores if row.scorer == RESCORE_SCORER and row.cache_mode == CACHE_OFF
        }
        if not reference:
            reference = {
                row.sample_id: row for row in scores if row.scorer == RESCORE_SCORER and row.cache_mode == CACHE_ON
            }
        if len(reference) == len(probe.probes):
            for mode in (NATIVE_MODE, REPEAT_MODE, *probe.spec.extra_trainer_modes):
                scored = {row.sample_id: row for row in scores if row.scorer == TRAINER_SCORER and row.mode == mode}
                if len(scored) != len(probe.probes):
                    continue
                delta = np.concatenate(
                    [
                        (
                            np.asarray(scored[row.sample_id].logprobs, dtype=np.float64)
                            - np.asarray(reference[row.sample_id].logprobs, dtype=np.float64)
                        )[np.asarray(row.loss_mask, dtype=np.bool_)]
                        for row in probe.probes
                    ]
                )
                summary = {
                    "abs_p99": float(np.percentile(np.abs(delta), 99)),
                    "k3": float(np.mean(np.expm1(delta) - delta)),
                    "share_beyond_2x": float(np.mean(np.abs(delta) > math.log(2))),
                }
                probe.metrics[f"update@{update}"][mode] = summary
                for metric, value in summary.items():
                    trainer.all_metrics[f"mismatch_probe/update_{update}/{mode}/{metric}"] = value
        trainer.all_metrics[f"mismatch_probe/update_{update}/tokens"] = probe.metrics[f"update@{update}"]["token_count"]
        trainer.all_metrics[f"mismatch_probe/update_{update}/route_bytes"] = probe.metrics[f"update@{update}"][
            "route_bytes"
        ]
        trainer.all_metrics[f"mismatch_probe/update_{update}/collated_bytes"] = probe.batch_layout.collated_bytes
        trainer.all_metrics[f"mismatch_probe/update_{update}/collated_route_bytes"] = (
            probe.batch_layout.collated_route_bytes
        )
        status = COMPLETE_STATUS if update == probe.updates[-1] else BUILDING_STATUS
        if probe.archive is None:
            probe.archive = await asyncio.to_thread(
                MismatchArchive, probe.archive_uri, writer_id=f"probe-{os.getpid()}"
            )
        archive_started = time.monotonic()
        await asyncio.to_thread(
            probe.archive.write,
            probes=probe.probes if update == 0 else None,
            scores=scores,
        )
        probe.timing[f"update@{update}/archive_seconds"] = time.monotonic() - archive_started
        probe.timing[f"update@{update}/total_seconds"] = time.monotonic() - started
        trainer.all_timings[f"mismatch_probe_update_{update}"] = probe.timing[f"update@{update}/total_seconds"]
        await asyncio.to_thread(probe.archive.write, manifest=manifest(probe, trainer, status=status))
        logger.info(
            "Mismatch probe archived update={} samples={} scores={} status={}",
            update,
            len(probe.probes),
            len(scores),
            status,
        )
        return scores
    finally:
        random.setstate(python_rng)
        np.random.set_state(numpy_rng)
        torch.set_rng_state(cpu_rng)
        if cuda_rng is not None:
            torch.cuda.set_rng_state_all(cuda_rng)
