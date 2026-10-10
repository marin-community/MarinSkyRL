"""Rescore independent temperature-one completions through the production policy worker."""

import hashlib
import io
import json
from pathlib import Path
import time

import numpy as np
import ray
import torch

from hero_qualification import s3_client, s3_location
from skyrl_train.dataset.preprocess import convert_prompts_responses_to_batch_tensors
from skyrl_train.dataset.routed_expert_batch import RoutedExpertRows
from skyrl_train.training_batch import TrainingInputBatch


def distribution(delta):
    values = torch.cat(delta).double()
    ratios = values.exp()
    return {
        "tokens": values.numel(),
        "ratio_outside_0p8_1p2": ((ratios < 0.8) | (ratios > 1.2)).sum().item(),
        "outside_fraction": ((ratios < 0.8) | (ratios > 1.2)).double().mean().item(),
        "mean_abs_log_gap": values.abs().mean().item(),
        "abs_log_gap_p50_p95_p99_p999_max": [
            *torch.quantile(values.abs(), torch.tensor([0.5, 0.95, 0.99, 0.999], dtype=torch.float64)).tolist(),
            values.abs().max().item(),
        ],
    }


def save_arrays(bucket, key, arrays):
    payload = io.BytesIO()
    np.savez_compressed(payload, **arrays)
    data = payload.getvalue()
    s3_client().put_object(Bucket=bucket, Key=key, Body=data)
    return {"uri": f"s3://{bucket}/{key}", "sha256": hashlib.sha256(data).hexdigest()}


def audit_captured_routes(trajectory_batch, config):
    """Check original response-local routes before any dtype conversion."""
    expected = (config["num_hidden_layers"], config["num_experts_per_tok"])
    digest = hashlib.sha256()
    tokens, response_tokens = 0, 0
    minimum, maximum = config["num_experts"], -1
    for response, mask, routes in zip(
        trajectory_batch["response_ids"],
        trajectory_batch["loss_masks"],
        trajectory_batch["rollout_routed_experts"],
        strict=True,
    ):
        assert routes is not None
        routes = np.asarray(routes)
        assert routes.shape == (len(response), *expected), (routes.shape, len(response), expected)
        assert np.issubdtype(routes.dtype, np.integer), routes.dtype
        assert len(mask) == len(response)
        selected = routes[np.asarray(mask, dtype=bool)]
        response_tokens += len(response)
        tokens += len(selected)
        if not selected.size:
            continue
        minimum, maximum = min(minimum, int(selected.min())), max(maximum, int(selected.max()))
        assert minimum >= 0 and maximum < config["num_experts"]
        ordered = np.sort(selected, axis=-1)
        assert (ordered[..., 1:] > ordered[..., :-1]).all(), "missing or duplicate captured routes"
        digest.update(np.asarray(response, dtype=np.int64).tobytes())
        digest.update(np.asarray(mask, dtype=np.uint8).tobytes())
        digest.update(np.ascontiguousarray(routes).tobytes())
    return {
        "responses": len(trajectory_batch["response_ids"]),
        "response_tokens": response_tokens,
        "valid_response_tokens": tokens,
        "layers": expected[0],
        "top_k": expected[1],
        "minimum_expert_id": minimum if tokens else None,
        "maximum_expert_id": maximum if tokens else None,
        "response_mask_routes_sha256": digest.hexdigest(),
        "passed": True,
    }


def check(trainer, trajectory_batch, output, *, required_tokens=50000):
    from skyrl_train.distributed.dispatch import concatenate_outputs_after_mesh_dispatch

    weight_step = trainer.global_step
    policy = trainer.policy_model
    # The separate learner keeps its model on GPU while its optimizer is offloaded.
    bucket, prefix = s3_location(output)
    config = json.loads((Path(trainer.cfg.trainer.policy.model.path) / "config.json").read_text())
    capture = audit_captured_routes(trajectory_batch, config)
    experts = config["num_experts"]
    valid_indices = [i for i, mask in enumerate(trajectory_batch["loss_masks"]) if sum(mask)]
    deltas = {"response_replay": [], "native_routes": []}
    timings = {key: 0.0 for key in deltas}
    artifacts, input_artifacts, tokens, responses = [], [], 0, 0
    # A multiple of the learner's DP size also bounds the live route arrays.
    world = trainer.cfg.trainer.placement.policy_num_nodes * trainer.cfg.trainer.placement.policy_num_gpus_per_node
    megatron = trainer.cfg.trainer.policy.megatron_config
    dp = world // (megatron.pipeline_model_parallel_size * megatron.context_parallel_size)
    chunk_size = dp * 8
    for start in range(0, len(valid_indices), chunk_size):
        indices = valid_indices[start : start + chunk_size]
        if len(indices) % dp:
            indices = indices[: len(indices) // dp * dp]
        if not indices:
            break

        def select(key):
            return [trajectory_batch[key][i] for i in indices]

        tensors = convert_prompts_responses_to_batch_tensors(
            trainer.tokenizer,
            select("prompt_token_ids"),
            select("response_ids"),
            select("rewards"),
            select("loss_masks"),
            select("rollout_logprobs"),
        )
        sequences, attention, response_mask, _, loss_mask, behavior, _, _ = tensors
        mask = (response_mask * loss_mask).bool()
        route_rows = RoutedExpertRows(tuple(select("rollout_routed_experts")), behavior.shape[1], experts)
        # Validate original IDs before the worker narrows their dtype. Keep validation
        # and transport response-local, as in the production trainer.
        for row, row_mask in zip(route_rows.rows, mask.numpy(), strict=True):
            assert not row_mask[len(row) :].any(), "missing captured response routes"
            selected_routes = row[row_mask[: len(row)]]
            assert selected_routes.size and selected_routes.min() >= 0 and selected_routes.max() < experts
            ordered = np.sort(selected_routes, axis=-1)
            assert (ordered[..., 1:] > ordered[..., :-1]).all(), "missing or duplicate captured routes"
        batch = TrainingInputBatch({"sequences": sequences, "attention_mask": attention}, routed_expert_rows=route_rows)
        batch.metadata = {"response_length": behavior.shape[1], "global_step": weight_step}
        # Only this bounded scoring chunk is made dense for its audit artifact.
        routes = route_rows.materialize()
        saved = {
            "sequences": sequences.numpy(),
            "attention": attention.numpy(),
            "mask": mask.numpy(),
            "behavior": behavior.numpy(),
            "routes": routes.numpy(),
            "indices": np.array(indices),
        }
        input_artifacts.append(save_arrays(bucket, f"{prefix}/fixed-weight/input-batch-{start:04d}.npz", saved))
        s3_client().put_object(
            Bucket=bucket,
            Key=f"{prefix}/fixed-weight/progress.json",
            Body=json.dumps(
                {
                    "status": "scoring",
                    "required_tokens": required_tokens,
                    "weight_step": weight_step,
                    "input_artifacts": input_artifacts,
                }
            ).encode(),
        )
        for mode in deltas:
            batch.routed_expert_rows = (
                route_rows
                if mode == "response_replay"
                else RoutedExpertRows(
                    tuple(np.zeros_like(row) for row in route_rows.rows),
                    route_rows.response_len,
                    experts,
                )
            )
            began = time.monotonic()
            outputs = ray.get(policy.async_run_ray_method("mesh", "forward", data=batch))
            scores = concatenate_outputs_after_mesh_dispatch(policy.actor_infos, outputs)["output"].float().cpu()
            timings[mode] += time.monotonic() - began
            assert scores.shape == behavior.shape and torch.isfinite(scores[mask]).all()
            deltas[mode].append((scores[mask] - behavior[mask]).cpu())
            saved[mode] = scores.numpy()
        artifacts.append(save_arrays(bucket, f"{prefix}/fixed-weight/batch-{start:04d}.npz", saved))
        tokens += int(mask.sum())
        responses += len(indices)
        if tokens >= required_tokens:
            break
    assert tokens >= required_tokens, (
        f"Only {tokens} independently sampled response tokens; need at least {required_tokens}"
    )
    assert trainer.global_step == weight_step
    result = {
        "weight_step": weight_step,
        "harness_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "routed_expert_capture": capture,
        "independent_responses": responses,
        "modes": {key: distribution(value) for key, value in deltas.items()},
        "scoring_seconds": timings,
        "artifacts": artifacts,
        "input_artifacts": input_artifacts,
        "required_tokens": required_tokens,
    }
    result["passed"] = tokens >= required_tokens and result["modes"]["response_replay"]["outside_fraction"] < 0.001
    s3_client().put_object(Bucket=bucket, Key=f"{prefix}/fixed-weight/report.json", Body=json.dumps(result).encode())
    return result
