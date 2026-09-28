"""Synchronous frozen-token probe at the initial and post-sync policy weights."""

from __future__ import annotations

import asyncio
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import random
import re
import time
import tomllib
from dataclasses import dataclass
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import ray
import torch
from finestore import mismatch
from loguru import logger
from omegaconf import OmegaConf

from marinskyrl.resource_locator import join_resource_path
from skyrl_train.callbacks.base import TrainerCallback, TrainerControl, TrainerState
from skyrl_train.config.mismatch_probe import (
    CACHE_BOTH,
    CACHE_OFF,
    CACHE_ON,
    FILTERED_REPLAY_MODE,
    GENERATION_SCORING,
    NATIVE_MODE,
    REPEAT_MODE,
    rescore_scoring,
    trainer_scoring,
)
from skyrl_train.distributed.dispatch import concatenate_outputs_after_mesh_dispatch
from skyrl_train.group_admission import GroupAdvantageInvariant, GroupAdvantageKind
from skyrl_train.inference_engines.utils import get_sampling_params_for_backend
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
from skyrl_train.training_batch import TrainingInputBatch
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


def _encode_routes(routes: torch.Tensor | None, length: int) -> EncodedRoutes:
    if routes is None:
        return EncodedRoutes(None, None, None)
    value = routes[:length].contiguous().cpu().numpy()
    return EncodedRoutes(value.tobytes(), list(value.shape), str(value.dtype))


def _reorder_batch(batch: TrainingInputBatch, order: list[int]) -> TrainingInputBatch:
    index = torch.tensor(order, dtype=torch.long)
    reordered = TrainingInputBatch({key: value[index] if value is not None else None for key, value in batch.items()})
    reordered.metadata = dict(batch.metadata)
    if "uids" in reordered.metadata:
        reordered.metadata["uids"] = [batch.metadata["uids"][i] for i in order]
    return reordered


def _prompt_group_contract(trainer, samples_per_prompt: int) -> GroupAdvantageInvariant:
    current = trainer.group_advantage_invariant
    minimum = current.minimum_group_size
    if current.kind is GroupAdvantageKind.MINIMUM_BASELINE_ELIGIBLE:
        if samples_per_prompt < 2:
            raise ValueError("mismatch probe requires at least two samples for a baseline-eligible group")
        minimum = min(minimum, samples_per_prompt)
    return replace(current, physical_group_size=samples_per_prompt, minimum_group_size=minimum)


class MismatchProbeCallback(TrainerCallback):
    """Score one immutable validation probe; never consumes the train loader."""

    error_behavior = "raise"

    def __init__(self, cfg):
        self.cfg = cfg
        self.spec = cfg.trainer.mismatch_probe
        self.updates = tuple(self.spec.score_after_updates)
        self.archive_uri = self.spec.archive_uri or join_resource_path(cfg.trainer.ckpt_path, "mismatch_probe")
        self.archive: MismatchArchive | None = None
        self.probes = None
        self.training_input: TrainingInputBatch | None = None
        self.probe_hash: str | None = None
        self.starting_weights_hash: str | None = None
        self.source_manifest = None
        self.generation_scores: list[list[float]] | None = None
        self.timing: dict[str, float] = {}
        self.weights: dict[int, str] = {}
        self.batch_layout: dict[str, object] = {}
        self.metrics: dict[str, object] = {}
        self.created_at = datetime.now(timezone.utc).isoformat()
        self.tokenizer_fingerprint: str | None = None
        self.cache_hit_tokens: dict[str, int] = {}

    async def on_train_begin_async(self, state: TrainerState, control: TrainerControl, **kwargs):
        trainer = kwargs["trainer"]
        if self.updates[-1] > trainer.total_training_steps:
            raise ValueError("mismatch probe update schedule exceeds the available training batches")
        await self._collect_preserving_rng(trainer, update=0)
        if self.updates[-1] == 0:
            control.should_training_stop = True
        return control

    async def on_step_end_async(self, state: TrainerState, control: TrainerControl, **kwargs):
        if state.global_step in self.updates:
            await self._collect_preserving_rng(kwargs["trainer"], update=state.global_step)
        if state.global_step >= self.updates[-1]:
            control.should_training_stop = True
        return control

    def on_train_end(self, state: TrainerState, control: TrainerControl, **kwargs):
        if self.archive is not None:
            self.archive.close()
            self.archive = None
        return control

    async def _collect_preserving_rng(self, trainer, *, update: int) -> None:
        python_rng = random.getstate()
        numpy_rng = np.random.get_state()
        cpu_rng = torch.get_rng_state()
        cuda_rng = torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else None
        try:
            await self._collect(trainer, update=update)
        finally:
            random.setstate(python_rng)
            np.random.set_state(numpy_rng)
            torch.set_rng_state(cpu_rng)
            if cuda_rng is not None:
                torch.cuda.set_rng_state_all(cuda_rng)

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
                request["probe_capture_token_identity"] = True
                batches.append(await trainer.trajectory_runner.run(request))
                seeds.append(seed)
                sample_ids.append(f"{prompt['uid']}:{repetition}")
                uids.append(prompt["uid"])
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
                routes.append(decoded.tolist())
        if any(route is None for route in routes) and any(route is not None for route in routes):
            raise ValueError("reuse_probe source mixes captured and missing routed-expert rows")
        self.generation_scores = [generations[row.sample_id].logprobs for row in rows]
        trajectory = {
            "prompt_token_ids": [row.prompt_token_ids for row in rows],
            "response_ids": [row.trainer_input_ids for row in rows],
            "engine_response_ids": [row.vllm_output_ids for row in rows],
            "rewards": [scalar_reward_token_credit(row.reward or 0.0, row.trainer_input_ids) for row in rows],
            "loss_masks": [[int(value) for value in row.loss_mask] for row in rows],
            "rollout_logprobs": self.generation_scores,
            "rollout_routed_experts": routes if all(route is not None for route in routes) else None,
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
            source_tokenizer = json.loads(self.source_manifest.software_json).get("tokenizer_fingerprint")
            self.tokenizer_fingerprint = tokenizer_vocabulary_fingerprint(trainer.inference_engine_client.tokenizer)
            if not source_tokenizer or source_tokenizer != self.tokenizer_fingerprint:
                raise ValueError("reuse_probe tokenizer fingerprint differs from the source archive")
        generated = trajectory.get("engine_response_ids")
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
                # vLLM captures a forward for every response token except its final
                # sampled token. Expert ID zero is a valid choice on earlier rows.
                route_valid_mask = [
                    [token_index < response_length - 1] * encoded_routes.shape[1]
                    for token_index in range(response_length)
                ]
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
        self.batch_layout = {
            "sample_ids": sample_ids,
            "native_order": list(range(len(sample_ids))),
            "repeat_order": list(reversed(range(len(sample_ids)))),
            "padded_rows": padded_count,
            "native_micro_batch_size": int(trainer.cfg.trainer.micro_forward_batch_size_per_gpu),
            "repeat_micro_batch_size": max(2, 2 * int(trainer.cfg.trainer.micro_forward_batch_size_per_gpu)),
            "collated_bytes": sum(
                value.numel() * value.element_size() for value in training_input.values() if value is not None
            ),
            "collated_route_bytes": (
                0
                if training_input.get("rollout_routed_experts") is None
                else training_input["rollout_routed_experts"].numel()
                * training_input["rollout_routed_experts"].element_size()
            ),
            "nonzero_advantage_samples": (
                0
                if training_input.get("advantages") is None
                else int((training_input["advantages"].abs().sum(dim=1) > 0).sum().item())
            ),
        }

    async def _rescore_vllm(self, trainer, update: int):
        rows = self.probes
        prefixes = []
        overrides = []
        positions = []
        for row_index, row in enumerate(rows):
            for response_index, token in enumerate(row.vllm_output_ids):
                prefixes.append(row.prompt_token_ids + row.vllm_output_ids[:response_index])
                overrides.append({"logprob_token_ids": [token]})
                positions.append((row_index, token))
        cache_setting = self.spec.rescore_prefix_cache
        cache_modes = (CACHE_OFF, CACHE_ON) if cache_setting == CACHE_BOTH else (cache_setting,)
        result = []
        for cache_mode in cache_modes:
            if cache_mode == CACHE_OFF:
                await trainer.inference_engine_client.reset_prefix_cache()
            started = time.monotonic()
            output = await trainer.inference_engine_client.generate(
                {
                    "prompts": None,
                    "prompt_token_ids": prefixes,
                    "sampling_params": {
                        "max_tokens": 1,
                        "logprobs": 1,
                        "temperature": 1.0,
                        "skip_reading_prefix_cache": cache_mode == CACHE_OFF,
                        "seed": request_seed(int(self.spec.seed), f"reread:{update}:{cache_mode}", 0),
                    },
                    "sampling_params_per_prompt": overrides,
                    "session_ids": None,
                }
            )
            duration = time.monotonic() - started
            requested = output.get("requested_token_logprobs")
            if requested is None or len(requested) != len(positions):
                raise ValueError("vLLM re-read returned incomplete requested-token scores")
            chosen = [[] for _ in rows]
            for position, ((row_index, token), returned) in enumerate(zip(positions, requested, strict=True)):
                if len(returned) != 1 or token not in returned[0]:
                    raise ValueError(f"vLLM re-read omitted frozen token {token} at flattened position {position}")
                score = float(returned[0][token])
                if not math.isfinite(score):
                    raise ValueError(f"vLLM re-read returned a nonfinite score at flattened position {position}")
                chosen[row_index].append(score)
            label = rescore_scoring(update, cache_mode)
            self.timing[f"{label}/seconds"] = duration
            cache_hits = output.get("prefix_cache_hit_tokens")
            if cache_hits is None or len(cache_hits) != len(positions):
                raise ValueError("vLLM re-read omitted prefix-cache hit counts")
            self.cache_hit_tokens[label] = sum(cache_hits)
            if cache_mode == CACHE_OFF and self.cache_hit_tokens[label]:
                raise ValueError("cache-off re-read unexpectedly used cached prefix tokens")
            for row, values in zip(rows, chosen, strict=True):
                if len(values) != len(row.vllm_output_ids):
                    raise ValueError(f"vLLM re-read returned incomplete scores for {row.sample_id}")
                result.append(
                    mismatch.ScoreRow(
                        probe_hash=self.probe_hash,
                        sample_id=row.sample_id,
                        scoring=label,
                        update=update,
                        weights_hash=self.weights[update],
                        cache_mode=cache_mode,
                        logprobs=values,
                        forward_seconds=duration / len(rows),
                    )
                )
        return result

    def _policy_weights_hash(self, trainer) -> str:
        shards = ray.get(trainer.policy_model.async_run_ray_method("pass_through", "probe_weights_digest"))
        if not shards or not all(isinstance(shard, str) and len(shard) == 64 for shard in shards):
            raise ValueError("policy workers did not return complete weight digests")
        return "sha256:" + hashlib.sha256("".join(shards).encode()).hexdigest()

    def _route_observations(self, trainer, outputs, mode: str):
        """Gather PP-owned router choices by frozen sample and captured layer."""
        if all(row.routed_experts is None for row in self.probes):
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
        for actor_info, output in zip(trainer.policy_model.actor_infos, outputs, strict=True):
            rank = actor_info.rank
            if rank.tp != 0 or rank.sp != 0:
                continue
            for observation in output.metadata.get("probe_routes", []):
                sample = int(observation["sample"])
                position = int(observation["position"])
                layer = int(observation["layer"])
                if sample < 0 or sample >= len(self.probes) + self.batch_layout["padded_rows"]:
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
                    raise ValueError(f"duplicate probe routing observation for {sample}/{position}/{layer}")
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

    def _trainer_scores(self, trainer, update: int):
        training_input = self.training_input
        rows = self.probes
        n = len(rows)
        route_tensor = training_input.get("rollout_routed_experts")
        modes = (NATIVE_MODE, REPEAT_MODE, *self.spec.extra_trainer_modes)
        result = []
        for mode in modes:
            order = self.batch_layout["repeat_order"] if mode == REPEAT_MODE else self.batch_layout["native_order"]
            data = training_input.select(
                ["sequences", "attention_mask", *(["rollout_routed_experts"] if route_tensor is not None else [])],
                ["response_length"],
            )
            data["probe_row_indices"] = torch.arange(n, dtype=torch.long)
            if mode in {NATIVE_MODE, REPEAT_MODE} and route_tensor is not None:
                data["rollout_routed_experts"] = torch.zeros_like(route_tensor)
            if mode == REPEAT_MODE:
                data = _reorder_batch(data, order + list(range(n, data.batch_size)))
            micro_batch_size = self.batch_layout["repeat_micro_batch_size"] if mode == REPEAT_MODE else None
            fraction = float(self.spec.filtered_replay.keep_fraction) if mode == FILTERED_REPLAY_MODE else None
            data.metadata.update(
                probe_mode=mode,
                probe_keep_fraction=fraction,
                probe_micro_batch_size=micro_batch_size,
                global_step=update,
            )
            started = time.monotonic()
            outputs = ray.get(trainer.policy_model.async_run_ray_method("mesh", "probe_forward", data=data))
            values = concatenate_outputs_after_mesh_dispatch(trainer.policy_model.actor_infos, outputs)["output"][:n]
            route_observations = self._route_observations(trainer, outputs, mode)
            duration = time.monotonic() - started
            label = trainer_scoring(update, mode)
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
                        scoring=label,
                        update=update,
                        weights_hash=self.weights[update],
                        logprobs=logprobs,
                        forward_seconds=duration / n,
                        expert_choices=None
                        if route_observations[original_position] is None
                        else route_observations[original_position].data,
                        expert_choices_shape=None
                        if route_observations[original_position] is None
                        else route_observations[original_position].shape,
                        expert_choices_dtype=None
                        if route_observations[original_position] is None
                        else route_observations[original_position].dtype,
                        replacement_mask=None
                        if route_observations[original_position] is None
                        else route_observations[original_position].replacements,
                    )
                )
        return result

    def _manifest(self, trainer, *, status: str):
        schema = mismatch
        return schema.ManifestRow(
            archive=self.archive_uri,
            status=status,
            probe_hash=self.probe_hash,
            starting_weights_hash=self.starting_weights_hash,
            source_probe_archive=self.spec.reuse_probe,
            architecture=str(trainer.cfg.trainer.policy.model.path),
            vllm_enforce_eager=bool(trainer.cfg.generator.engine_init_kwargs.get("enforce_eager", False)),
            optimizer_steps_per_update=int(trainer.all_metrics.get("policy/policy_update_steps", 0)),
            seed=int(self.spec.seed),
            bootstrap_seed=request_seed(int(self.spec.seed), "bootstrap", 0),
            created_at_utc=self.created_at,
            config_json=json.dumps(OmegaConf.to_container(self.cfg, resolve=True), sort_keys=True, default=str),
            software_json=json.dumps(self._software_provenance(trainer), sort_keys=True),
            hardware_json=json.dumps(
                {"placement": OmegaConf.to_container(trainer.cfg.trainer.placement)}, sort_keys=True
            ),
            batch_layout_json=json.dumps(self.batch_layout, sort_keys=True),
            timing_json=json.dumps(self.timing, sort_keys=True),
            step_metrics_json=json.dumps(
                self.metrics | {"weights": self.weights, "cache_hit_tokens": self.cache_hit_tokens}, sort_keys=True
            ),
        )

    def _software_provenance(self, trainer) -> dict[str, str | None]:
        def version(package: str) -> str | None:
            try:
                return importlib.metadata.version(package)
            except importlib.metadata.PackageNotFoundError:
                return None

        lock_bytes = None
        for parent in Path(__file__).resolve().parents:
            candidate = parent / "uv.lock"
            if candidate.is_file():
                lock_bytes = candidate.read_bytes()
                break
        vllm_commit = None
        if lock_bytes is not None:
            locked = tomllib.loads(lock_bytes.decode())
            package = next((item for item in locked.get("package", []) if item.get("name") == "vllm"), None)
            source = (package or {}).get("source", {})
            source_text = str(source.get("git", ""))
            matched = re.search(r"[0-9a-f]{40}", source_text)
            if matched:
                vllm_commit = matched.group()
        if self.tokenizer_fingerprint is None:
            self.tokenizer_fingerprint = tokenizer_vocabulary_fingerprint(trainer.inference_engine_client.tokenizer)
        return {
            "python": platform.python_version(),
            "torch": str(torch.__version__),
            "marin_commit": self.spec.get("marin_commit"),
            "marinskyrl_commit": self.spec.get("skyrl_commit"),
            "marinskyrl_version": version("marinskyrl"),
            "marin_finestore_version": version("marin-finestore"),
            "vllm_version": version("vllm"),
            "vllm_commit": vllm_commit,
            "skyrl_lock_sha256": hashlib.sha256(lock_bytes).hexdigest() if lock_bytes is not None else None,
            "tokenizer_fingerprint": self.tokenizer_fingerprint,
        }

    async def _collect(self, trainer, *, update: int) -> None:
        started = time.monotonic()
        if update == 0:
            if self.spec.reuse_probe:
                samples = await asyncio.to_thread(self._from_source)
            else:
                generation_started = time.monotonic()
                original_metrics = dict(trainer.all_metrics)
                try:
                    samples = await self._generate(trainer)
                finally:
                    trainer.all_metrics = original_metrics
                self.timing["probe/generation_seconds"] = time.monotonic() - generation_started
            self._collate_and_freeze(trainer, samples.trajectory, samples.prompt_ids, samples.sample_ids, samples.seeds)
        elif self.training_input is None:
            raise RuntimeError("mismatch probe update 0 was not collected")

        # The policy is still offloaded and the inference engine awake after
        # each normal weight sync. Re-read first, then temporarily swap residency.
        rescore_started = time.monotonic()
        self.weights[update] = "pending"
        rescore_rows = await self._rescore_vllm(trainer, update)
        self.timing[f"update@{update}/vllm_total_seconds"] = time.monotonic() - rescore_started
        if trainer.colocate_all:
            await trainer.inference_engine_client.sleep()
        try:
            if trainer.colocate_all:
                trainer.policy_model.backload_to_gpu(backload_optimizer=False, backload_model=True)
            try:
                self.weights[update] = self._policy_weights_hash(trainer)
                if update == 0:
                    self.starting_weights_hash = self.weights[0]
                    if (
                        self.source_manifest is not None
                        and self.source_manifest.starting_weights_hash != self.weights[0]
                    ):
                        raise ValueError("reuse_probe starting weights differ from the source archive")
                # Re-read rows were delayed until this exact weight identity is known.
                for row in rescore_rows:
                    row.weights_hash = self.weights[update]
                trainer_rows = self._trainer_scores(trainer, update)
                if self._policy_weights_hash(trainer) != self.weights[update]:
                    raise ValueError(f"probe scoring changed policy weights or buffers at update {update}")
            finally:
                if trainer.colocate_all:
                    trainer.policy_model.offload_to_cpu(offload_optimizer=False, offload_model=True)
        finally:
            if trainer.colocate_all:
                await trainer.inference_engine_client.wake_up()

        scores = rescore_rows + trainer_rows
        if update == 0 and self.source_manifest is None:
            for row, values in zip(self.probes, self.generation_scores, strict=True):
                scores.append(
                    mismatch.ScoreRow(
                        probe_hash=self.probe_hash,
                        sample_id=row.sample_id,
                        scoring=GENERATION_SCORING,
                        update=0,
                        weights_hash=self.weights[0],
                        logprobs=values,
                    )
                )
        self.metrics[f"update@{update}"] = {
            "token_count": sum(sum(row.loss_mask) for row in self.probes),
            "sample_count": len(self.probes),
            "route_bytes": sum(len(row.routed_experts or b"") for row in self.probes),
            "optimizer_steps": int(trainer.all_metrics.get("policy/policy_update_steps", 0)),
            "step_timings": {
                key: float(value) for key, value in trainer.all_timings.items() if isinstance(value, (int, float))
            },
        }
        reference_name = rescore_scoring(update, CACHE_OFF)
        reference = {row.sample_id: row for row in scores if row.scoring == reference_name}
        if not reference:
            reference = {row.sample_id: row for row in scores if row.scoring == rescore_scoring(update, CACHE_ON)}
        if len(reference) == len(self.probes):
            for mode in (NATIVE_MODE, REPEAT_MODE, *self.spec.extra_trainer_modes):
                name = trainer_scoring(update, mode)
                scored = {row.sample_id: row for row in scores if row.scoring == name}
                if len(scored) != len(self.probes):
                    continue
                delta = np.concatenate(
                    [
                        (
                            np.asarray(scored[row.sample_id].logprobs, dtype=np.float64)
                            - np.asarray(reference[row.sample_id].logprobs, dtype=np.float64)
                        )[np.asarray(row.loss_mask, dtype=np.bool_)]
                        for row in self.probes
                    ]
                )
                summary = {
                    "abs_p99": float(np.percentile(np.abs(delta), 99)),
                    "k3": float(np.mean(np.expm1(delta) - delta)),
                    "share_beyond_2x": float(np.mean(np.abs(delta) > math.log(2))),
                }
                self.metrics[f"update@{update}"][mode] = summary
                for metric, value in summary.items():
                    trainer.all_metrics[f"mismatch_probe/update_{update}/{mode}/{metric}"] = value
        trainer.all_metrics[f"mismatch_probe/update_{update}/tokens"] = self.metrics[f"update@{update}"]["token_count"]
        trainer.all_metrics[f"mismatch_probe/update_{update}/route_bytes"] = self.metrics[f"update@{update}"][
            "route_bytes"
        ]
        trainer.all_metrics[f"mismatch_probe/update_{update}/collated_bytes"] = self.batch_layout["collated_bytes"]
        trainer.all_metrics[f"mismatch_probe/update_{update}/collated_route_bytes"] = self.batch_layout[
            "collated_route_bytes"
        ]
        status = COMPLETE_STATUS if update == self.updates[-1] else BUILDING_STATUS
        if self.archive is None:
            self.archive = await asyncio.to_thread(MismatchArchive, self.archive_uri, writer_id=f"probe-{os.getpid()}")
        archive_started = time.monotonic()
        await asyncio.to_thread(
            self.archive.write,
            probes=self.probes if update == 0 else None,
            scores=scores,
        )
        self.timing[f"update@{update}/archive_seconds"] = time.monotonic() - archive_started
        self.timing[f"update@{update}/total_seconds"] = time.monotonic() - started
        trainer.all_timings[f"mismatch_probe_update_{update}"] = self.timing[f"update@{update}/total_seconds"]
        await asyncio.to_thread(self.archive.write, manifest=self._manifest(trainer, status=status))
        logger.info(
            "Mismatch probe archived update={} samples={} scores={} status={}",
            update,
            len(self.probes),
            len(scores),
            status,
        )
