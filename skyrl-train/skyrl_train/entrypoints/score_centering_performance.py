"""Finite real-worker alignment and timing probe. Excluded from the merge candidate."""

import asyncio
import cProfile
from dataclasses import replace
import hashlib
import importlib.metadata
import json
import math
import threading
import time
import io
import pstats
import resource
from pathlib import Path

import numpy as np
import psutil
import fsspec
import ray
import torch
from omegaconf import OmegaConf
from rigging.filesystem.s3_compat import configure_coreweave_s3

from skyrl_train.entrypoints.main_base import BasePPOExp
from skyrl_train.objective.correction import compute_correction
from skyrl_train.config.objective_spec import off_policy_correction
from skyrl_train.objective.score_centering import collate_score_centering
from skyrl_train.trainer import RayPPOTrainer, convert_prompts_responses_to_batch_tensors, _token_staleness_metrics
from skyrl_train.training_batch import TrainingInputBatch
from skyrl_train.timing_observability import StepWallTime
from skyrl_train.rollouts.context import TrainingContext, RolloutRequestSpec


_device_memory_peak = 0
_device_memory_lock = threading.Lock()
_device_memory_thread = None


def sample_device_memory(device):
    global _device_memory_peak
    timer = threading.Event()
    while not timer.wait(0.1):
        used = torch.cuda.device_memory_used(device)
        with _device_memory_lock:
            _device_memory_peak = max(_device_memory_peak, used)


def host_memory():
    return {
        "rss_current": psutil.Process().memory_info().rss,
        "rss_cumulative_peak": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
    }


def json_value(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"Unexpected report value: {type(value).__name__}")


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=json_value).encode()).hexdigest()


def profile_summary(profile):
    output = io.StringIO()
    stats = pstats.Stats(profile, stream=output).sort_stats("cumulative")
    stats.print_stats(25)
    selected = {}
    for (_, _, name), (primitive, calls, own, cumulative, _) in stats.stats.items():
        if name in (
            "_response_logprobs",
            "ppo_tis_score_centering_correction",
            "convert_ids_list_to_tokens",
            "_update_sample_logprobs",
            "_postprocess_outputs",
            "collate_score_centering",
        ):
            row = selected.setdefault(name, {"calls": 0, "own_seconds": 0, "cumulative_seconds": 0})
            row["calls"] += calls
            row["own_seconds"] += own
            row["cumulative_seconds"] += cumulative
    return {"selected": selected, "top": output.getvalue()}


class FixedPromptDataset:
    def __init__(self, dataset, size):
        self.dataset, self.size = dataset, size

    def __len__(self):
        return self.size

    def __getitem__(self, index):
        return self.dataset[index]

    def uid(self, index):
        return self.dataset.uid(index)

    def collate_fn(self, rows):
        return self.dataset.collate_fn(rows)


class BenchmarkRequestSpec:
    def __init__(self, original, limits, flat, interval):
        self.original, self.limits, self.flat, self.probe_interval = original, limits, flat, interval

    def request(self, prompt, policy_step, *, probe=False):
        request = (
            self.original.request(prompt, policy_step, probe=True)
            if probe
            else self.original.request(prompt, policy_step)
        )
        limit, seed = self.limits[prompt["uid"]]
        return replace(
            request,
            sampling_params={
                **request.sampling_params,
                "max_tokens": limit,
                "ignore_eos": True,
                "seed": seed,
                "flat_logprobs": self.flat,
            },
        )


def memory_stats(reset=False):
    global _device_memory_peak, _device_memory_thread
    torch.cuda.synchronize()
    used = torch.cuda.device_memory_used()
    with _device_memory_lock:
        _device_memory_peak = used if reset else max(_device_memory_peak, used)
        peak = _device_memory_peak
    if _device_memory_thread is None:
        _device_memory_thread = threading.Thread(
            target=sample_device_memory, args=(torch.cuda.current_device(),), daemon=True
        )
        _device_memory_thread.start()
    if not _device_memory_thread.is_alive():
        raise RuntimeError("Device memory sampler stopped before measurement finished")
    if reset:
        torch.cuda.reset_peak_memory_stats()
    return {
        "device": torch.cuda.get_device_name(),
        "allocated": torch.cuda.memory_allocated(),
        "reserved": torch.cuda.memory_reserved(),
        "peak_allocated": torch.cuda.max_memory_allocated(),
        "peak_reserved": torch.cuda.max_memory_reserved(),
        "device_used": used,
        "sampled_device_peak_used": peak,
        "device_sampling_interval": 0.1,
        "host": host_memory(),
    }


def checked_response_logprobs(original, stats):
    def checked(
        logits, sequences, attention_mask, num_actions, packed_seq_params, candidate_ids=None, candidate_positions=None
    ):
        result = (
            original(logits, sequences, attention_mask, num_actions, packed_seq_params, candidate_ids)
            if candidate_positions is None
            else original(
                logits, sequences, attention_mask, num_actions, packed_seq_params, candidate_ids, candidate_positions
            )
        )
        if candidate_ids is None:
            return result
        for row in range(len(sequences)):
            valid = candidate_ids[row].ge(0).any(-1).nonzero().flatten().tolist()
            if not valid:
                continue
            prompt_length = int(attention_mask[row, :-num_actions].sum())
            for index in sorted({valid[0], valid[len(valid) // 2], valid[-1]}):
                position = index if candidate_positions is None else int(candidate_positions[row, index])
                # Independent causal-position calculation on the real compacted model logits.
                normalized = logits[row, prompt_length - 1 + position].float().log_softmax(-1)
                token = sequences[row, -num_actions + position]
                ids = candidate_ids[row, index].long()
                chosen_error = abs(float(normalized[token] - result.scores[row, position]))
                head_error = float((normalized[ids] - result.selected_logprobs[row, index]).abs().max())
                stats["positions"] += 1
                stats["chosen_max_error"] = max(stats["chosen_max_error"], chosen_error)
                stats["head_max_error"] = max(stats["head_max_error"], head_error)
        return result

    return checked


def reference_forward(worker, data):
    original = worker.model._response_logprobs
    stats = {"positions": 0, "chosen_max_error": 0.0, "head_max_error": 0.0}
    worker.model._response_logprobs = checked_response_logprobs(original, stats)
    try:
        output = worker.forward(data)
    finally:
        worker.model._response_logprobs = original
    output.metadata = {**(output.metadata or {}), "performance_reference": stats}
    return output


class PerformanceExp(BasePPOExp):
    def get_trainer(self, **kwargs):
        trainer = super().get_trainer(**kwargs)
        trainer.__class__ = PerformanceTrainer
        return trainer


class PerformanceTrainer(RayPPOTrainer):
    def save_probe(self):
        self.probe["applied_updates"] = self._performance_applied_updates
        path = Path(self.cfg.trainer.export_path) / "performance.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.probe, indent=2, allow_nan=False, default=json_value) + "\n")
        with fsspec.open(self.cfg.trainer.score_centering_performance.output_uri, "wt") as output:
            json.dump(self.probe, output, allow_nan=False, default=json_value)
        print(
            "READINESS "
            + json.dumps(
                {
                    "complete": self.probe.get("complete", False),
                    "windows": len(self.probe["windows"]),
                    "qualification": list(self.probe["qualification"]),
                }
            ),
            flush=True,
        )

    async def settings(self, width, enabled, telemetry=True, diagnostic_tokens=0, temperature=1.0):
        self.cfg.trainer.algorithm.score_centering_topk = width
        self.cfg.trainer.algorithm.score_centering_enabled = enabled
        self.cfg.trainer.algorithm.score_centering_diagnostic_tokens = diagnostic_tokens
        self.cfg.generator.sampling_params.logprobs = width
        self.cfg.generator.sampling_params.temperature = temperature
        self._training_metrics_enabled = telemetry
        return await asyncio.to_thread(
            ray.get,
            self.policy_model.async_run_ray_method(
                "pass_through", "performance_settings", width, enabled, diagnostic_tokens, temperature
            ),
        )

    async def inference_memory(self, reset=False):
        return await asyncio.gather(
            *(
                engine.inference_engine_actor.performance_memory.remote(reset)
                for engine in self.inference_engine_client.engines
            )
        )

    async def generate_workload(self, width, *, temperature=1.0, short=False, flat=None):
        limits = [8, 24, 64, 128] if short else [64, 256, 1024, 4096]
        outputs = []
        for group, limit in enumerate(limits):
            prompts = self.prompts[group::4]
            result = await self.inference_engine_client.generate(
                {
                    "prompt_token_ids": prompts,
                    "sampling_params": {
                        "temperature": temperature,
                        "top_p": 1.0,
                        "top_k": -1,
                        "min_p": 0.0,
                        "max_tokens": limit,
                        "ignore_eos": True,
                        "seed": 611 + group,
                        "logprobs": width,
                        "flat_logprobs": self.cfg.trainer.score_centering_performance.variant == "candidate"
                        if flat is None
                        else flat,
                    },
                }
            )
            assert all(len(row) == limit for row in result["response_ids"])
            assert all(set(row) == {1} for row in result["response_policy_steps"])
            outputs.extend((prompt, result, index) for index, prompt in enumerate(prompts))
        return outputs

    def batch(self, outputs, width):
        prompts = [prompt for prompt, _, _ in outputs]
        responses = [output["response_ids"][index] for _, output, index in outputs]
        chosen = [np.array(output["response_logprobs"][index]) for _, output, index in outputs]
        masks = [[int(position % 31 != 7) for position in range(len(row))] for row in responses]
        masks[0] = [0] * len(masks[0])
        rewards = [[0.0] * len(row) for row in responses]
        values = convert_prompts_responses_to_batch_tensors(
            self.tokenizer, prompts, responses, rewards, masks, chosen, None, None
        )
        sequences, attention, response_mask, _, loss_mask, sampled, _, _ = values
        policy_steps = torch.full_like(loss_mask, -1, dtype=torch.int32)
        for row, (_, output, index) in enumerate(outputs):
            stamps = output["response_policy_steps"][index]
            policy_steps[row, : len(stamps)] = torch.tensor(stamps, dtype=torch.int32)
        staleness = torch.where(response_mask.bool(), self.global_step - policy_steps, 0)
        payload = {
            "sequences": sequences,
            "attention_mask": attention,
            "response_mask": response_mask,
            "loss_mask": loss_mask,
            "rollout_logprobs": sampled,
            "rollout_staleness": staleness,
            "base_action_log_probs": None,
            "values": None,
            "returns": torch.zeros_like(loss_mask),
            # Fixed signed advantages qualify optimizer execution, without a learning-benefit experiment.
            "advantages": loss_mask * torch.tensor([1.0, -1.0] * (len(prompts) // 2)).unsqueeze(-1),
        }
        if width:
            trajectory = {
                "loss_masks": masks,
                "student_topk_indices": [np.asarray(output["student_topk_indices"][i]) for _, output, i in outputs],
                "behavior_topk_logprobs": [np.asarray(output["behavior_topk_logprobs"][i]) for _, output, i in outputs],
            }
            ids, behavior = collate_score_centering(
                trajectory, responses, response_mask, width, sampled_logprobs=sampled
            )
            payload.update(score_topk_indices=ids, score_behavior_logprobs=behavior)
        batch = TrainingInputBatch(payload)
        if self._training_metrics_enabled:
            self.all_metrics.update(_token_staleness_metrics(staleness, loss_mask, policy_steps))
        batch.metadata = {"response_length": response_mask.shape[1], "uids": list(map(str, range(len(prompts))))}
        return batch

    async def reference(self, batch):
        keys = ["sequences", "attention_mask", "response_mask", "score_topk_indices"]
        data = batch.select(keys, metadata_keys=["response_length"])
        rows = await asyncio.to_thread(
            ray.get, self.policy_model.async_run_ray_method("mesh", "performance_reference", data)
        )
        stats = [
            row.metadata["performance_reference"] for row in rows if row.metadata["performance_reference"]["positions"]
        ]
        assert stats and max(row["chosen_max_error"] for row in stats) < 2e-5
        assert max(row["head_max_error"] for row in stats) < 2e-5
        return stats

    async def prefix_replay(self, outputs, *, temperature=1.0):
        prefixes, overrides, expected, samples = [], [], [], []
        for prompt, output, index in outputs[:: max(1, len(outputs) // 8)]:
            response = output["response_ids"][index]
            for position in sorted({0, len(response) // 2, len(response) - 1}):
                ids = np.asarray(output["student_topk_indices"][index][position]).tolist()
                scores = np.asarray(output["behavior_topk_logprobs"][index][position]).tolist()
                prefixes.append(prompt + response[:position])
                overrides.append({"logprob_token_ids": ids})
                expected.append(scores)
                samples.append(
                    {
                        "prompt_ids": prompt,
                        "response_prefix_ids": response[:position],
                        "chosen_token_id": response[position],
                        "position": position,
                        "captured_support_ids": ids,
                        "captured_support_logprobs": scores,
                        "captured_chosen_logprob": output["response_logprobs"][index][position],
                        "temperature": temperature,
                        "generating_policy_step": output["response_policy_steps"][index][position],
                    }
                )
        replay = await self.inference_engine_client.generate(
            {
                "prompt_token_ids": prefixes,
                "sampling_params": {
                    "temperature": temperature,
                    "top_p": 1.0,
                    "top_k": -1,
                    "max_tokens": 1,
                    "logprobs": 128,
                },
                "sampling_params_per_prompt": overrides,
            }
        )
        shifted = await self.inference_engine_client.generate(
            {
                "prompt_token_ids": [prefix[:-1] for prefix in prefixes],
                "sampling_params": {
                    "temperature": temperature,
                    "top_p": 1.0,
                    "top_k": -1,
                    "max_tokens": 1,
                    "logprobs": 128,
                },
                "sampling_params_per_prompt": overrides,
            }
        )
        chosen_replay = await self.inference_engine_client.generate(
            {
                "prompt_token_ids": prefixes,
                "sampling_params": {
                    "temperature": temperature,
                    "top_p": 1.0,
                    "top_k": -1,
                    "max_tokens": 1,
                    "logprobs": 1,
                },
                "sampling_params_per_prompt": [
                    {"logprob_token_ids": [sample["chosen_token_id"]]} for sample in samples
                ],
            }
        )
        gaps = []
        shifted_gaps = []
        for index, target in enumerate(expected):
            np.testing.assert_array_equal(
                replay["student_topk_indices"][index][0], overrides[index]["logprob_token_ids"]
            )
            gaps.extend(np.asarray(replay["behavior_topk_logprobs"][index][0]) - np.asarray(target))
            shifted_gaps.extend(np.asarray(shifted["behavior_topk_logprobs"][index][0]) - np.asarray(target))
            samples[index]["prefix_replay_support_logprobs"] = replay["behavior_topk_logprobs"][index][0]
            samples[index]["prefix_replay_chosen_logprob"] = float(chosen_replay["behavior_topk_logprobs"][index][0][0])
        assert np.abs(gaps).mean() < np.abs(shifted_gaps).mean()
        return {
            "prefixes": len(prefixes),
            "prefix_ids_sha256": digest(prefixes),
            "support_ids_sha256": digest(overrides),
            "temperature": temperature,
            "same_engine_decode_prefill_gap_abs_mean": float(np.abs(gaps).mean()),
            "same_engine_decode_prefill_gap_abs_max": float(np.abs(gaps).max()),
            "one_token_shift_control_gap_abs_mean": float(np.abs(shifted_gaps).mean()),
            "policy_steps": sorted({step for row in replay["response_policy_steps"] for step in row}),
            "samples": samples,
        }

    async def update(self, batch):
        started = time.perf_counter()
        batch = await asyncio.to_thread(self.fwd_logprobs_values_reward, batch)
        forward = time.perf_counter() - started
        correction = compute_correction(
            batch["action_log_probs"],
            batch["rollout_logprobs"],
            batch["loss_mask"],
            off_policy_correction(self.cfg.trainer.algorithm),
        )
        batch["correction_weights"] = correction.weights
        trained = time.perf_counter()
        status = await asyncio.to_thread(self.train_critic_and_policy, batch)
        update = time.perf_counter() - trained
        assert status["policy_update_steps"] == 1 and status["skipped_steps"] == 0
        assert math.isfinite(status["raw_grad_norm"]) and status["raw_grad_norm"] > 0
        self._performance_applied_updates += int(status["policy_update_steps"])
        self._policy_version_step = self.global_step
        await asyncio.to_thread(self._offload_policy_optimizer, self.all_timings, timer_label="probe_offload")
        memory = await asyncio.to_thread(
            ray.get, self.policy_model.async_run_ray_method("pass_through", "performance_memory")
        )
        return {
            "forward_seconds": forward,
            "update_seconds": update,
            "status": status,
            "learner_memory": memory,
            "mismatch": {k: v for k, v in self.all_metrics.items() if "mismatch" in k},
        }

    def postprocess_trajectory_batch(self, trajectory_batch, uids):
        result = super().postprocess_trajectory_batch(trajectory_batch, uids)
        if "score_probe_positions" in result:
            self._performance_probe_rows = [
                {
                    "uid": uids[row],
                    "response_tokens": len(result["response_ids"][row]),
                    "exported_positions_before_trainer_masks": len(positions),
                }
                for row, positions in enumerate(result["score_probe_positions"])
                if positions is not None
            ]
        if self._performance_fixed_advantages:
            masked_row = uids.index(self._performance_masked_uid)
            result["loss_masks"] = [
                [int(bool(value) and index % 31 != 7 and row != masked_row) for index, value in enumerate(mask)]
                for row, mask in enumerate(result["loss_masks"])
            ]
        return result

    def finalize_advantages_for_training(self, data):
        result = super().finalize_advantages_for_training(data)
        if self._performance_fixed_advantages:
            signs = torch.tensor([1.0 if index % 2 == 0 else -1.0 for index in range(len(data["loss_mask"]))])
            result["advantages"] = data["loss_mask"] * signs.unsqueeze(-1)
        return result

    def _finish_score_probe(self, data):
        if self._performance_record_evidence:
            positions = data["score_probe_positions"]
            valid = (data["score_topk_indices"] >= 0).all(-1) & data["loss_mask"].gather(1, positions).bool()
            self.probe.setdefault("probe_evidence", []).append(
                {
                    "phase": self._performance_phase,
                    "training_step": self.global_step,
                    "synthetic_advantages": self._performance_fixed_advantages,
                    "positions": positions[valid].numpy(),
                    "current_old_head": data["score_old_logprobs"][valid].numpy(),
                    "behavior_head": data["score_behavior_logprobs"][valid].numpy(),
                    "current_old_sampled": data["action_log_probs"].gather(1, positions)[valid].numpy(),
                    "behavior_sampled": data["rollout_logprobs"].gather(1, positions)[valid].numpy(),
                    "advantages": data["advantages"].gather(1, positions)[valid].numpy(),
                    "source_steps": data["score_probe_source_steps"][valid].numpy(),
                    "loss_tokens": int(data["loss_mask"].bool().sum()),
                }
            )
        super()._finish_score_probe(data)

    async def frontend_profile(self, start):
        method = "performance_profile_start" if start else "performance_profile_stop"
        return await asyncio.gather(
            *(
                getattr(engine.inference_engine_actor, method).remote()
                for engine in self.inference_engine_client.engines
            )
        )

    async def learner_profile(self, start):
        return await asyncio.to_thread(
            ray.get, self.policy_model.async_run_ray_method("pass_through", "performance_profile", start)
        )

    async def native_window(self, phase):
        self._performance_phase = phase
        self._performance_probe_rows = []
        await self.inference_memory(reset=True)
        self.all_metrics, self.all_timings = {}, {}
        started = time.perf_counter()
        await self._publish(self.global_step + 1)
        self.global_step += 1
        wall = StepWallTime(OmegaConf.to_container(self.cfg.trainer.step_phase_budgets))
        data = await self._train_step(0, wall)
        elapsed = time.perf_counter() - started
        assert self.all_metrics["policy/policy_update_steps"] == 1
        assert self.all_metrics["policy/skipped_steps"] == 0
        norm = self.all_metrics["policy/raw_grad_norm"]
        assert math.isfinite(norm) and (norm > 0 if self._performance_fixed_advantages else norm >= 0)
        assert self.all_metrics["async/staleness_tokens_max"] == 0
        self._performance_applied_updates += 1
        result = {
            "phase": phase,
            "total_seconds": elapsed,
            "tokens": int(data["response_mask"].sum()),
            "loss_tokens": int(data["loss_mask"].bool().sum()),
            "sequence_shape": list(data["sequences"].shape),
            "prompt_lengths": (data["attention_mask"].sum(-1) - data["response_mask"].sum(-1)).int().tolist(),
            "row_loss_tokens": data["loss_mask"].bool().sum(-1).int().tolist(),
            "optimizer_input_tensor_bytes": {
                name: value.numel() * value.element_size()
                for name, value in data.items()
                if isinstance(value, torch.Tensor)
            },
            "driver_host": host_memory(),
            "response_lengths": data["response_mask"].sum(-1).int().tolist(),
            "sequence_sha256": hashlib.sha256(data["sequences"].numpy().tobytes()).hexdigest(),
            "uids": data.metadata.get("uids"),
            "probe_capture": self._performance_probe_rows,
            "metrics": dict(self.all_metrics),
            "timings": dict(self.all_timings),
            "inference_memory": await self.inference_memory(),
            "learner_memory": await asyncio.to_thread(
                ray.get, self.policy_model.async_run_ray_method("pass_through", "performance_memory")
            ),
            "applied_updates": self._performance_applied_updates,
            "synthetic_advantages": self._performance_fixed_advantages,
        }
        return result

    def convert_to_training_input(self, *args, **kwargs):
        if not self._performance_driver_profile:
            return super().convert_to_training_input(*args, **kwargs)
        profile = cProfile.Profile()
        result = profile.runcall(super().convert_to_training_input, *args, **kwargs)
        self._performance_driver_profiles.append({"phase": "collation", **profile_summary(profile)})
        return result

    def fwd_logprobs_values_reward(self, *args, **kwargs):
        if not self._performance_driver_profile:
            return super().fwd_logprobs_values_reward(*args, **kwargs)
        profile = cProfile.Profile()
        result = profile.runcall(super().fwd_logprobs_values_reward, *args, **kwargs)
        self._performance_driver_profiles.append({"phase": "old_forward_and_collection", **profile_summary(profile)})
        return result

    async def _train_loop(self):
        configure_coreweave_s3()
        spec = self.cfg.trainer.score_centering_performance
        short = spec.workload == "qwen"
        candidate = spec.variant == "candidate"
        self._performance_applied_updates = 0
        self._performance_fixed_advantages = False
        self._performance_record_evidence = False
        self._performance_driver_profile = False
        self._performance_driver_profiles = []
        self._performance_phase = "qualification"
        self.probe = {
            "schema": 2,
            "workload": spec.workload,
            "variant": spec.variant,
            "candidate": spec.candidate,
            "windows": [],
            "qualification": {},
            "complete": False,
        }
        self.probe["runtime"] = {
            "lock_sha256": hashlib.sha256((Path(__file__).resolve().parents[3] / "uv.lock").read_bytes()).hexdigest(),
            "versions": {
                name: importlib.metadata.version(name)
                for name in ("torch", "vllm", "megatron-core", "transformers", "marin-iris")
            },
            "policy_parallelism": OmegaConf.to_container(self.cfg.trainer.policy.megatron_config),
            "sampling": OmegaConf.to_container(self.cfg.generator.sampling_params),
        }
        await self._init_weight_sync()
        await self._sync_policy_for_rollouts(reason="initial")
        self.probe["setup_timings"] = dict(self.all_startup_timings)
        count = 32 if short else 128
        self.prompts = []
        for index in range(count):
            messages, _, extra, _ = self.train_dataset[index // 4]
            prompt = (
                extra["reference_prompt_ids"]
                if "reference_prompt_ids" in extra and not short
                else self.tokenizer.apply_chat_template(messages, add_generation_prompt=True)["input_ids"]
            )
            self.prompts.append(list(prompt))
        self.probe["prompt_sha256"] = digest(self.prompts)
        self.probe["prompt_lengths"] = list(map(len, self.prompts))

        await self.settings(128, False)
        captured = await self.generate_workload(128, short=short)
        self.probe["qualification"]["prefix_replay"] = await self.prefix_replay(captured)
        self.global_step = 1
        batch = self.batch(captured, 128)
        self.probe["qualification"]["reference"] = await self.reference(batch)
        batch = await asyncio.to_thread(self.fwd_logprobs_values_reward, batch)
        matching_logprobs = batch["action_log_probs"].clone()
        valid = batch["loss_mask"] > 0
        gap = (batch["action_log_probs"] - batch["rollout_logprobs"])[valid]
        self.probe["qualification"]["matching_weights_tim"] = {
            "tokens": gap.numel(),
            "abs_mean": gap.abs().mean().item(),
            "abs_p99": torch.quantile(gap.abs(), 0.99).item(),
            "abs_max": gap.abs().max().item(),
        }
        head_gap = (batch["score_old_logprobs"] - batch["score_behavior_logprobs"])[valid]
        self.probe["qualification"]["matching_weights_head_tim"] = {
            "entries": head_gap.numel(),
            "abs_mean": head_gap.abs().mean().item(),
            "abs_max": head_gap.abs().max().item(),
            "sampled_abs_p99": torch.quantile(
                head_gap.flatten()[:: max(1, head_gap.numel() // 65536)].abs(), 0.99
            ).item(),
        }
        self.save_probe()
        await self.settings(128, False, temperature=0.7)
        alternate = await self.generate_workload(128, temperature=0.7, short=True)
        self.probe["qualification"]["temperature_0_7_reference"] = await self.reference(self.batch(alternate, 128))
        self.probe["qualification"]["temperature_0_7_prefix_replay"] = await self.prefix_replay(
            alternate, temperature=0.7
        )
        self.save_probe()

        # Exact response text and numeric evidence parity is checked on the dense
        # startup model. MoE comparisons also retain their own repeat variation.
        if candidate and short:
            engine = self.inference_engine_client.engines[0]
            for temperature in (1.0, 0.7):
                request = {
                    "prompt_token_ids": [self.prompts[0]],
                    "sampling_params": {
                        "temperature": temperature,
                        "top_p": 1.0,
                        "top_k": -1,
                        "max_tokens": 128,
                        "ignore_eos": True,
                        "seed": 7781,
                        "logprobs": 128,
                    },
                }
                # Warm the same engine's prefix cache before comparing formats.
                await engine.generate(
                    {**request, "sampling_params": {**request["sampling_params"], "flat_logprobs": True}}
                )
                old = await engine.generate(
                    {**request, "sampling_params": {**request["sampling_params"], "flat_logprobs": False}}
                )
                new = await engine.generate(
                    {**request, "sampling_params": {**request["sampling_params"], "flat_logprobs": True}}
                )
                assert old["response_ids"] == new["response_ids"]
                assert old["responses"] == new["responses"]
                np.testing.assert_array_equal(old["student_topk_indices"], new["student_topk_indices"])
                np.testing.assert_array_equal(old["behavior_topk_logprobs"], new["behavior_topk_logprobs"])
                np.testing.assert_array_equal(old["response_logprobs"], new["response_logprobs"])
            self.probe["qualification"]["flat_dictionary_parity"] = {
                "same_engine": True,
                "temperatures": [1.0, 0.7],
                "tokens_text_heads_scores_exact": True,
            }
            self.save_probe()
        await self.settings(128, True)
        for _ in range(3):
            self.global_step = self._performance_applied_updates + 1
            stale_batch = self.batch(captured, 128)
            result = await self.update(stale_batch)
            self.probe["qualification"].setdefault("calibration_updates", []).append(result)
            drift = (stale_batch["action_log_probs"] - matching_logprobs)[valid]
            self.probe["qualification"]["trainer_weight_drift_on_identical_tokens"] = {
                "abs_mean": drift.abs().mean().item(),
                "abs_max": drift.abs().max().item(),
                "changed_fraction": (drift != 0).float().mean().item(),
            }
            self.save_probe()
        assert self.probe["qualification"]["trainer_weight_drift_on_identical_tokens"]["abs_max"] > 0
        self.probe["qualification"]["stale_prefix_replay"] = await self.prefix_replay(captured)
        assert self.probe["qualification"]["stale_prefix_replay"]["policy_steps"] == [1]
        self.save_probe()

        # Calibration deliberately kept the sampler frozen. The native loop's
        # first lease must use the latest weights, just as subsequent leases do
        # after _train_step's normal end-of-step sync.
        native_sync_started = time.perf_counter()
        await self._sync_policy_for_rollouts(reason="performance_start")
        self.probe["qualification"]["native_initial_sync_seconds"] = time.perf_counter() - native_sync_started
        self.save_probe()

        # Use the actual rollout coordinator, task workers, object store, old
        # scoring and optimizer path. Only response lengths, signed advantages
        # and masks are controlled for the serialized cost comparison.
        await self.settings(0, False)
        await self.context.close()
        groups = count // 4
        fixed = FixedPromptDataset(self.train_dataset, groups)
        self._performance_masked_uid = fixed.uid(0)
        self.context = TrainingContext.from_config(self.cfg, fixed, self.trajectory_runner)
        self.context.start()
        limits = [8, 24, 64, 128] if short else [64, 256, 1024, 4096]
        placements = {fixed.uid(index): (limits[index % 4], 1701 + index) for index in range(groups)}
        expected_lengths = sorted(limit for index in range(groups) for limit in [limits[index % 4]] * 4)
        self._performance_fixed_advantages = True
        modes = [("ordinary", 0, False), ("capture", 128, False), ("sc", 128, True)]
        if candidate:
            modes.append(("armed", 0, False))
            modes.append(("probe", 0, False))
        repetitions = int(spec.repetitions)
        for repetition in range(repetitions):
            order = modes[repetition % len(modes) :] + modes[: repetition % len(modes)]
            for name, width, enabled in order:
                temperature = 0.7 if name == "probe" and repetition == 0 else 1.0
                await self.settings(width, enabled, temperature=temperature)
                interval = 1 if name == "probe" else 100 if name == "armed" else 0
                if candidate:
                    self.cfg.trainer.algorithm.score_centering_probe.interval = interval
                request_spec = RolloutRequestSpec.from_config(self.cfg)
                self.context._request_spec = BenchmarkRequestSpec(request_spec, placements, candidate, interval)
                if name == "armed":
                    self.context._last_score_probe_step = self.global_step + 1
                self._performance_record_evidence = name == "probe" and repetition == 0
                profile = repetition == 0
                self._performance_driver_profile = profile
                self._performance_driver_profiles = []
                if profile:
                    await self.frontend_profile(True)
                    await self.learner_profile(True)
                compact_reference = name == "probe" and repetition == 0
                if compact_reference:
                    await asyncio.to_thread(
                        ray.get,
                        self.policy_model.async_run_ray_method("pass_through", "performance_check_positions", True),
                    )
                result = await self.native_window(f"{name}/{repetition}")
                assert sorted(result["response_lengths"]) == expected_lengths
                if name == "probe":
                    assert result["metrics"]["score_centering_probe/sampled_tokens"] > 0
                if compact_reference:
                    references = await asyncio.to_thread(
                        ray.get,
                        self.policy_model.async_run_ray_method("pass_through", "performance_check_positions", False),
                    )
                    assert sum(row["positions"] for row in references) > 0
                    assert max(row["chosen_max_error"] for row in references) <= 1e-5
                    assert max(row["head_max_error"] for row in references) <= 1e-5
                    self.probe["qualification"]["compact_temperature_0_7_reference"] = references
                if profile:
                    self.probe.setdefault("profiles", {})[name] = {
                        "frontend": await self.frontend_profile(False),
                        "learner": await self.learner_profile(False),
                        "driver": self._performance_driver_profiles,
                    }
                self._performance_driver_profile = False
                result.update(mode=name, repetition=repetition, warmup=repetition == 0, temperature=temperature)
                self.probe["windows"].append(result)
                self.save_probe()

        if candidate:
            await self.context.close()
            await self.settings(0, False)
            self.cfg.trainer.algorithm.score_centering_probe.interval = 2
            self.context = TrainingContext.from_config(self.cfg, self.train_dataset, self.trajectory_runner)
            self.context.start()
            self._performance_fixed_advantages = False
            self._performance_record_evidence = True
            observed = 0
            self.probe["native_periodic_windows"] = []
            for index in range(6):
                await self.settings(0, False)
                result = await self.native_window(f"native/{index}")
                observed += int(result["metrics"].get("score_centering_probe/sampled_tokens", 0) > 0)
                self.probe["native_periodic_windows"].append(result)
                self.save_probe()
            assert observed >= 3
            self.probe["qualification"]["repeated_periodic_observations"] = observed
        self.probe["complete"] = True
        self.probe["applied_updates"] = self._performance_applied_updates
        self.save_probe()
