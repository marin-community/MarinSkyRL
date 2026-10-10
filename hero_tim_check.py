"""Task-owned, warmed comparison of captured response replay and native routing."""

import hashlib
import io
import json
import os
import statistics
import sys
import time

import numpy as np
import ray
import torch

from hero_fixed_weight_check import distribution, save_arrays
from hero_qualification import s3_client, s3_location
from skyrl_train.distributed.dispatch import concatenate_outputs_after_mesh_dispatch
from skyrl_train.training_batch import TrainingInputBatch
from tests.gpu.grug_serving import rank0_validation_snapshot


class ReplayControl:
    """Temporarily detach replay from an idle task-owned policy worker."""

    def qualification_kernel_state(self):
        module = sys.modules.get("vllm.model_executor.determinism.batch_invariant")
        enabled = module is not None and module._batch_invariant_MODE
        return {
            "rank": torch.distributed.get_rank(),
            "batch_invariant_enabled": enabled,
            "registered_overrides": sorted(module._batch_invariant_LIB._op_impls) if enabled else [],
            "nccl_environment": {
                name: os.environ.get(name)
                for name in (
                    "NCCL_ALGO",
                    "NCCL_PROTO",
                    "NCCL_MIN_NCHANNELS",
                    "NCCL_MAX_NCHANNELS",
                    "NCCL_NTHREADS",
                    "NCCL_LAUNCH_MODE",
                    "NCCL_COLLNET_ENABLE",
                    "NCCL_NVLS_ENABLE",
                    "NCCL_P2P_NET_DISABLE",
                )
            },
            "matmul_bf16_reduced_precision_reduction": torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction,
        }

    def qualification_set_replay(self, enabled):
        from megatron.core.transformer.moe.router import TopKRouter

        if self.model.router_replay is not None:
            self.model.router_replay.assert_drained()
            self._qualification_replay_controller = self.model.router_replay
            self._qualification_replay_routers = [
                (module, module.router_replay)
                for chunk in self.actor_module
                for module in chunk.modules()
                if isinstance(module, TopKRouter)
            ]
        self.model.router_replay = self._qualification_replay_controller if enabled else None
        for module, handle in self._qualification_replay_routers:
            module.router_replay = handle if enabled else None


def read_verified(uri, expected_sha256):
    bucket, key = s3_location(uri)
    data = s3_client().get_object(Bucket=bucket, Key=key)["Body"].read()
    assert hashlib.sha256(data).hexdigest() == expected_sha256, uri
    return data


def check_replay_cost(policy, manifest, output, *, expected_source, required_tokens=50000, repetitions=4):
    assert manifest["source"] == expected_source
    router_name = "model.layers.0.mlp.router.weight"
    before = rank0_validation_snapshot(policy, [router_name])[router_name]
    assert hashlib.sha256(before.numpy().tobytes()).hexdigest() == manifest["router_weight_sha256"]
    bank = []
    for artifact in manifest["input_artifacts"]:
        with np.load(io.BytesIO(read_verified(artifact["uri"], artifact["sha256"])), allow_pickle=False) as arrays:
            values = {
                key: torch.from_numpy(arrays[key].copy())
                for key in ("sequences", "attention", "mask", "behavior", "routes")
            }
        assert values["mask"].dtype == torch.bool
        values["native_routes"] = torch.zeros_like(values["routes"])
        bank.append(values)
    tokens = sum(int(values["mask"].sum()) for values in bank)
    assert tokens >= required_tokens
    modes = ("response_replay", "native_routes", "native_no_replay")
    bucket, prefix = s3_location(output)
    result = {
        "status": "warming",
        "input_manifest": manifest,
        "tokens": tokens,
        "repetitions": repetitions,
        "trials": [],
        "warmup_seconds": {},
        "timing_scope": "Production mesh forward, Ray transport and returned scores; inputs already loaded. Same weights, tokens, positions and captured route bank. native_routes retains the controller with zero sentinels. native_no_replay detaches its handles and sends no route tensor.",
        "limits": "This measures learner replay execution and route transport, but not serving route capture. It does not measure serving eager/batch-invariant/attention alternatives or online PPO clipping.",
    }
    result["learner_kernel_state"] = ray.get(policy.async_run_ray_method("pass_through", "qualification_kernel_state"))

    def score(mode):
        ray.get(policy.async_run_ray_method("pass_through", "qualification_set_replay", mode != "native_no_replay"))
        deltas, scores = [], {}
        started = time.monotonic()
        for index, values in enumerate(bank):
            routes = values["routes"] if mode == "response_replay" else values["native_routes"]
            batch = TrainingInputBatch({"sequences": values["sequences"], "attention_mask": values["attention"]})
            if mode != "native_no_replay":
                batch["rollout_routed_experts"] = routes
            batch.metadata = {"response_length": values["behavior"].shape[1], "global_step": 0}
            outputs = ray.get(policy.async_run_ray_method("mesh", "forward", data=batch))
            measured = concatenate_outputs_after_mesh_dispatch(policy.actor_infos, outputs)["output"].float().cpu()
            assert measured.shape == values["behavior"].shape and torch.isfinite(measured[values["mask"]]).all()
            selected = measured[values["mask"]]
            deltas.append(selected - values["behavior"][values["mask"]])
            scores[f"batch_{index}"] = selected.numpy()
        return time.monotonic() - started, deltas, scores

    # Warm each complete mode, then alternate their order to expose time drift.
    for mode in modes:
        result["warmup_seconds"][mode] = score(mode)[0]
    result["status"] = "measuring"
    previous = {}
    for repetition in range(repetitions):
        for mode in modes if repetition % 2 == 0 else reversed(modes):
            seconds, deltas, scores = score(mode)
            maximum_repeat_gap = (
                max((float(np.max(np.abs(value - previous[mode][key]))) for key, value in scores.items()), default=0.0)
                if mode in previous
                else None
            )
            previous[mode] = scores
            row = {
                "repetition": repetition,
                "mode": mode,
                "seconds": seconds,
                "tokens_per_second": tokens / seconds,
                "distribution": distribution(deltas),
                "max_abs_score_change_from_previous_repeat": maximum_repeat_gap,
                "scores": save_arrays(bucket, f"{prefix}/tim/{mode}-{repetition}.npz", scores),
            }
            result["trials"].append(row)
            s3_client().put_object(Bucket=bucket, Key=f"{prefix}/tim/report.json", Body=json.dumps(result).encode())
    ray.get(policy.async_run_ray_method("pass_through", "qualification_set_replay", True))
    after = rank0_validation_snapshot(policy, [router_name])[router_name]
    assert torch.equal(before, after)
    result["router_weight_unchanged"] = True
    after_kernels = ray.get(policy.async_run_ray_method("pass_through", "qualification_kernel_state"))
    assert after_kernels == result["learner_kernel_state"]
    result["median_seconds"] = {
        mode: statistics.median(row["seconds"] for row in result["trials"] if row["mode"] == mode) for mode in modes
    }
    result["replay_time_ratio_to_native"] = (
        result["median_seconds"]["response_replay"] / result["median_seconds"]["native_no_replay"]
    )
    result["replay_execution_time_ratio_to_sentinels"] = (
        result["median_seconds"]["response_replay"] / result["median_seconds"]["native_routes"]
    )
    result["native_sentinel_vs_disabled_max_abs_difference"] = max(
        float(np.max(np.abs(value - previous["native_no_replay"][key])))
        for key, value in previous["native_routes"].items()
    )
    result["replay_meets_fixed_weight_gate"] = all(
        row["distribution"]["outside_fraction"] < 0.001 for row in result["trials"] if row["mode"] == "response_replay"
    )
    result["status"] = "complete"
    s3_client().put_object(Bucket=bucket, Key=f"{prefix}/tim/report.json", Body=json.dumps(result).encode())
    assert result["replay_meets_fixed_weight_gate"], result
    return result
