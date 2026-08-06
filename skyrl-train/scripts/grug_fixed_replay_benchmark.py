#!/usr/bin/env python3
"""Run one warmed Grug policy update on a content-addressed fixed replay.

This is a narrow benchmark driver, not a trainer entry point.  The driver
verifies every replay byte before model initialization, reserves whole nodes,
stages one exact shard per policy rank outside the timer, and warms first-use
work without changing the timed start state.  It can then time either one
production ``ppo_train`` update through Adam or the common token-weighted
next-token CE forward/backward used for cross-stack attribution.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import datetime as dt
import hashlib
import importlib.metadata
import json
import math
import os
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import boto3
import hydra
import ray
import torch
from botocore.config import Config
from omegaconf import OmegaConf
from ray.util.placement_group import placement_group
from safetensors import safe_open
from safetensors.torch import load_file

from skyrl_train.entrypoints.main_base import config_dir
from skyrl_train.training_batch import TrainingInputBatch
from skyrl_train.utils import get_ray_pg_ready_with_timeout, initialize_ray
from skyrl_train.workers.fsdp.fsdp_worker import PolicyWorker
from skyrl_train.workers.worker import PPORayActorGroup

CHUNK_BYTES = 16 * 1024 * 1024
GPUS_PER_NODE = 8
HEADLINE_SEQUENCES = 4096
WORLD_SIZES = {"preflight": 8, "headline": 32}
LOGICAL_SEQUENCES = {"preflight": 8, "headline": HEADLINE_SEQUENCES}
TENSOR_FIELDS = (
    "action_log_probs",
    "advantages",
    "attention_mask",
    "loss_mask",
    "response_mask",
    "returns",
    "sequences",
)
NONE_FIELDS = ("base_action_log_probs", "is_last_step", "rollout_logprobs", "values")
PRACTICAL_GATE = {
    "action_log_probs": {"rtol": 4.0e-2, "atol": 4.0e-3},
    "matched_global_ce": {"rtol": 2.0e-3, "atol": 2.0e-3},
    "representative_gradients": {"rtol": 8.0e-2, "atol": 1.0e-4},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--flash-attn-wheel-sha256", required=True)
    parser.add_argument("--manifest-s3-uri", required=True)
    parser.add_argument("--manifest-sha256", required=True)
    parser.add_argument("--logical-batch-sha256", required=True)
    parser.add_argument("--attention-backend", choices=("eager", "flash_attention_2"), required=True)
    parser.add_argument("--expert-implementation", choices=("eager", "grouped"), required=True)
    parser.add_argument("--expert-parallel-size", choices=(1, 8), type=int, required=True)
    parser.add_argument("--objective", choices=("operational", "matched_ce"), required=True)
    parser.add_argument("--mode", choices=("preflight", "headline"), required=True)
    parser.add_argument("--result-s3-uri", required=True)
    parser.add_argument("--sample", type=int, required=True)
    parser.add_argument("--profile-s3-uri")
    parser.add_argument("--expert-attribution", action="store_true")
    parser.add_argument("--pg-timeout-seconds", type=int, default=900)
    return parser.parse_args()


def s3_client():
    endpoint = os.environ.get("CW_S3_ENDPOINT") or os.environ.get("AWS_ENDPOINT_URL")
    access_key = os.environ.get("CW_KEY_ID") or os.environ.get("AWS_ACCESS_KEY_ID")
    secret_key = os.environ.get("CW_KEY_SECRET") or os.environ.get("AWS_SECRET_ACCESS_KEY")
    kwargs: dict[str, Any] = {
        "config": Config(signature_version="s3v4", s3={"addressing_style": "virtual"}),
    }
    if endpoint:
        kwargs["endpoint_url"] = endpoint
    if access_key:
        kwargs["aws_access_key_id"] = access_key
    if secret_key:
        kwargs["aws_secret_access_key"] = secret_key
    return boto3.client("s3", **kwargs)


def split_s3_uri(uri: str) -> tuple[str, str]:
    parsed = urlparse(uri)
    if parsed.scheme != "s3" or not parsed.netloc or not parsed.path.lstrip("/"):
        raise ValueError(f"not an S3 object URI: {uri}")
    return parsed.netloc, parsed.path.lstrip("/")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(CHUNK_BYTES):
            digest.update(chunk)
    return digest.hexdigest()


def tensor_bytes(tensor: torch.Tensor) -> memoryview:
    cpu = tensor.detach().cpu().contiguous()
    return memoryview(cpu.view(torch.uint8).numpy()).cast("B")


def aggregate_representative_gradients(
    timed: list[dict[str, Any]], expert_parallel_size: int
) -> dict[str, dict[str, float | int]]:
    representative_names = set(timed[0]["representative_gradients"])
    if any(set(item["representative_gradients"]) != representative_names for item in timed):
        raise RuntimeError("ranks returned different representative gradient keys")

    representative_global = {}
    for name in sorted(representative_names):
        replication = 1 if ".mlp.experts." in name else expert_parallel_size
        samples = [item["representative_gradients"][name] for item in timed]
        total_numel = sum(item["local_numel"] for item in samples)
        if total_numel % replication:
            raise RuntimeError(f"gradient shard count for {name} does not divide EP replication")
        representative_global[name] = {
            "numel": total_numel // replication,
            "l2_norm": math.sqrt(sum(item["l2_norm"] ** 2 for item in samples) / replication),
            "max_abs": max(item["max_abs"] for item in samples),
            "ep_replication_divisor": replication,
        }
    return representative_global


def download_object(client, uri: str, destination: Path, expected_bytes: int, expected_sha256: str) -> None:
    bucket, key = split_s3_uri(uri)
    client.download_file(bucket, key, str(destination))
    actual_bytes = destination.stat().st_size
    if actual_bytes != expected_bytes:
        raise RuntimeError(f"{uri}: expected {expected_bytes} bytes, got {actual_bytes}")
    actual_sha256 = sha256_file(destination)
    if actual_sha256 != expected_sha256:
        raise RuntimeError(f"{uri}: expected sha256 {expected_sha256}, got {actual_sha256}")


def load_verified_replay(
    client,
    manifest_uri: str,
    expected_manifest_sha256: str,
    expected_logical_sha256: str,
    work_dir: Path,
) -> tuple[dict[str, Any], list[dict[str, torch.Tensor]]]:
    bucket, key = split_s3_uri(manifest_uri)
    manifest_bytes = client.get_object(Bucket=bucket, Key=key)["Body"].read()
    manifest_sha256 = hashlib.sha256(manifest_bytes).hexdigest()
    if manifest_sha256 != expected_manifest_sha256:
        raise RuntimeError(f"manifest digest mismatch: expected {expected_manifest_sha256}, got {manifest_sha256}")
    manifest = json.loads(manifest_bytes)
    if manifest["schema_version"] != 1:
        raise RuntimeError(f"unsupported replay schema {manifest['schema_version']}")
    if manifest["logical_batch_sha256"] != expected_logical_sha256:
        raise RuntimeError("manifest names a different logical replay")
    if manifest["batch"]["batch_size"] != 4096 or len(manifest["shards"]) != 32:
        raise RuntimeError("headline replay must contain 4096 rows in 32 rank shards")

    paths = [work_dir / record["filename"] for record in manifest["shards"]]
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        futures = [
            pool.submit(
                download_object,
                client,
                record["s3_uri"],
                path,
                int(record["bytes"]),
                record["sha256"],
            )
            for record, path in zip(manifest["shards"], paths)
        ]
        for future in futures:
            future.result()

    shards: list[dict[str, torch.Tensor]] = []
    for record, path in zip(manifest["shards"], paths):
        with safe_open(path, framework="pt", device="cpu") as handle:
            metadata = handle.metadata() or {}
        expected_metadata = {
            "logical_batch_sha256": expected_logical_sha256,
            "rank": str(record["rank"]),
            "start": str(record["start"]),
            "end": str(record["end"]),
        }
        if metadata != expected_metadata:
            raise RuntimeError(f"{path.name}: safetensors metadata mismatch: {metadata}")
        shard = load_file(path, device="cpu")
        if set(shard) != set(TENSOR_FIELDS):
            raise RuntimeError(f"{path.name}: unexpected fields {sorted(shard)}")
        expected_rows = int(record["end"]) - int(record["start"])
        if {int(tensor.shape[0]) for tensor in shard.values()} != {expected_rows}:
            raise RuntimeError(f"{path.name}: row count disagrees with manifest")
        shards.append(shard)

    canonical = hashlib.sha256()
    for name in sorted(manifest["batch"]["fields"]):
        evidence = manifest["batch"]["fields"][name]
        if evidence.get("value", object()) is None:
            if name not in NONE_FIELDS:
                raise RuntimeError(f"unexpected null replay field {name}")
            canonical.update(f"{name}:none\n".encode())
            continue
        if name not in TENSOR_FIELDS:
            raise RuntimeError(f"unexpected tensor replay field {name}")
        header = {key: evidence[key] for key in ("dtype", "shape", "stride", "numel", "nbytes", "sha256")}
        canonical.update(json.dumps({"name": name, **header}, sort_keys=True).encode())
        canonical.update(b"\n")
        field_digest = hashlib.sha256()
        rows = 0
        for shard in shards:
            tensor = shard[name]
            rows += int(tensor.shape[0])
            raw = tensor_bytes(tensor)
            field_digest.update(raw)
            canonical.update(raw)
        if rows != evidence["shape"][0]:
            raise RuntimeError(f"{name}: reconstructed {rows} rows, expected {evidence['shape'][0]}")
        if field_digest.hexdigest() != evidence["sha256"]:
            raise RuntimeError(f"{name}: global field digest mismatch")
    logical_sha256 = canonical.hexdigest()
    if logical_sha256 != expected_logical_sha256:
        raise RuntimeError(f"logical replay digest mismatch: expected {expected_logical_sha256}, got {logical_sha256}")
    return manifest, shards


def make_config(args: argparse.Namespace, sequence_length: int):
    with hydra.initialize_config_dir(config_dir=config_dir, version_base=None):
        cfg = hydra.compose(config_name="ppo_base_config")
    cfg.trainer.logger = "console"
    cfg.trainer.policy.model.path = args.model
    cfg.trainer.critic.model.path = None
    cfg.trainer.strategy = "fsdp2"
    cfg.trainer.flash_attn = args.attention_backend == "flash_attention_2"
    cfg.trainer.attn_backend = "auto"
    cfg.trainer.gradient_checkpointing = True
    cfg.trainer.gradient_checkpointing_use_reentrant = False
    cfg.trainer.use_sample_packing = False
    cfg.trainer.update_epochs_per_batch = 1
    cfg.trainer.micro_train_batch_size_per_gpu = 1
    cfg.trainer.algorithm.policy_loss_type = "regular"
    cfg.trainer.algorithm.loss_reduction = "token_mean"
    cfg.trainer.algorithm.eps_clip_low = 0.2
    cfg.trainer.algorithm.eps_clip_high = 0.2
    cfg.trainer.algorithm.use_kl_loss = False
    cfg.trainer.algorithm.use_entropy_loss = False
    cfg.trainer.algorithm.advantage_batch_normalize = False
    cfg.trainer.algorithm.use_tis = False
    cfg.trainer.algorithm.enable_token_reward_channel = False
    cfg.trainer.algorithm.think_token_weight = 1.0
    cfg.trainer.algorithm.z_clip.enabled = False
    cfg.trainer.algorithm.stale_clip.enabled = False
    algorithm_config = OmegaConf.create(cfg.trainer.algorithm)
    algorithm_config.max_seq_len = sequence_length
    cfg.trainer.algorithm = algorithm_config
    cfg.trainer.placement.colocate_all = False
    cfg.trainer.placement.colocate_policy_ref = False
    cfg.trainer.placement.policy_num_nodes = 1 if args.mode == "preflight" else 4
    cfg.trainer.placement.policy_num_gpus_per_node = 8
    cfg.trainer.placement.policy_strict_spread_pg = True
    cfg.trainer.placement.policy_per_gpu_bundles = False
    cfg.trainer.placement.policy_force_cvd_mask = False
    cfg.trainer.policy.fsdp_config.cpu_offload = False
    cfg.trainer.policy.grug_query_bias_update_mode = "frozen"
    cfg.trainer.policy.fsdp_config.reshard_after_forward = True
    cfg.trainer.policy.fsdp_config.fsdp_size = WORLD_SIZES[args.mode] // args.expert_parallel_size
    cfg.trainer.policy.fsdp_config.expert_model_parallel_size = args.expert_parallel_size
    cfg.trainer.policy.fsdp_config.expert_tensor_parallel_size = 1
    cfg.trainer.policy.fsdp_config.context_parallel_size = 1
    cfg.trainer.policy.fsdp_config.moe_router_replay = False
    cfg.trainer.policy.fsdp_config.moe_grouped_gemm = False
    cfg.trainer.policy.fsdp_config.use_grouped_mm = args.expert_implementation == "grouped"
    cfg.trainer.policy.optimizer_config.optimizer = "AdamW"
    cfg.trainer.policy.optimizer_config.lr = 1.0e-5
    cfg.trainer.policy.optimizer_config.adam_betas = [0.9, 0.999]
    cfg.trainer.policy.optimizer_config.weight_decay = 1.0e-2
    cfg.trainer.policy.optimizer_config.max_grad_norm = 0.5
    cfg.trainer.policy.optimizer_config.offload_after_step = True
    cfg.trainer.policy.optimizer_config.num_warmup_steps = 0
    cfg.trainer.policy.optimizer_config.scheduler = "constant_with_warmup"
    # Eight-way FSDP leaves too little HBM for AdamW's faster CUDA foreach
    # temporaries on this 67B model.  The one-node preflight only proves the
    # operational boundary, so use AdamW's equivalent low-memory loop there.
    # The 32-GPU headline keeps the faster default implementation.
    if args.mode == "preflight":
        cfg.trainer.policy.optimizer_config.optimizer_kwargs = {"foreach": False}
    cfg.generator.n_samples_per_prompt = 1 if args.mode == "preflight" else 16
    cfg.generator.sampling_params.temperature = 1.0
    cfg.generator.weight_sync_backend = "nccl"
    if args.mode == "preflight":
        cfg.trainer.train_batch_size = 8
        cfg.trainer.policy_mini_batch_size = 8
    else:
        cfg.trainer.train_batch_size = 256
        cfg.trainer.policy_mini_batch_size = 256
    return cfg


def expected_microbatches(mode: str, expert_parallel_size: int) -> int:
    data_parallel_size = WORLD_SIZES[mode] // expert_parallel_size
    return LOGICAL_SEQUENCES[mode] // data_parallel_size


def make_rank_batches(
    manifest: dict[str, Any],
    shards: list[dict[str, torch.Tensor]],
    mode: str,
    expert_parallel_size: int,
) -> list[TrainingInputBatch]:
    """Rebuild the logical replay, then mirror normal MeshDispatch EP replication."""

    world_size = WORLD_SIZES[mode]
    data_parallel_size = world_size // expert_parallel_size
    logical_sequences = LOGICAL_SEQUENCES[mode]
    if logical_sequences % data_parallel_size:
        raise RuntimeError("logical replay does not divide over data-parallel ranks")
    rows_per_data_rank = logical_sequences // data_parallel_size
    reconstructed = {name: torch.cat([shard[name] for shard in shards], dim=0) for name in TENSOR_FIELDS}
    if {int(tensor.shape[0]) for tensor in reconstructed.values()} != {HEADLINE_SEQUENCES}:
        raise RuntimeError("reconstructed replay has inconsistent row counts")

    batches = []
    for data_rank in range(data_parallel_size):
        start = data_rank * rows_per_data_rank
        end = start + rows_per_data_rank
        for ep_replica in range(expert_parallel_size):
            data = {name: tensor[start:end].contiguous() for name, tensor in reconstructed.items()}
            for name in NONE_FIELDS:
                data[name] = None
            batch = TrainingInputBatch(data)
            batch.metadata = {
                **manifest["batch_metadata"],
                "grug_benchmark_data_rank": data_rank,
                "grug_benchmark_ep_replica": ep_replica,
                "grug_benchmark_logical_start": start,
                "grug_benchmark_logical_end": end,
            }
            batches.append(batch)
    if len(batches) != world_size:
        raise RuntimeError(f"built {len(batches)} rank batches for world size {world_size}")
    return batches


def field_identity(batch: TrainingInputBatch) -> tuple[dict[str, str | None], dict[str, list[int] | None]]:
    hashes: dict[str, str | None] = {}
    shapes: dict[str, list[int] | None] = {}
    for name, tensor in sorted(batch.items()):
        if tensor is None:
            hashes[name] = None
            shapes[name] = None
        else:
            hashes[name] = hashlib.sha256(tensor_bytes(tensor)).hexdigest()
            shapes[name] = list(tensor.shape)
    return hashes, shapes


def assert_topology(topology: list[dict[str, Any]], mode: str) -> None:
    world_size = 8 if mode == "preflight" else 32
    if sorted(item["rank"] for item in topology) != list(range(world_size)):
        raise RuntimeError("policy ranks are incomplete")
    uuids = [item["phys_uuid"] for item in topology]
    if None in uuids or len(set(uuids)) != world_size:
        raise RuntimeError("policy ranks do not occupy distinct physical GPUs")
    by_host: dict[str, list[dict[str, Any]]] = {}
    for item in topology:
        by_host.setdefault(item["host"], []).append(item)
    expected_hosts = 1 if mode == "preflight" else 4
    if len(by_host) != expected_hosts or {len(items) for items in by_host.values()} != {8}:
        raise RuntimeError(f"expected {expected_hosts} complete 8-GPU hosts, got {by_host}")
    for node_index in range(expected_hosts):
        group = topology[node_index * 8 : (node_index + 1) * 8]
        if len({item["host"] for item in group}) != 1:
            raise RuntimeError(f"rank group {node_index} crosses hosts")


def require_rank_order(items: list[dict[str, Any]], world_size: int, label: str) -> None:
    ranks = [item.get("rank") for item in items]
    if ranks != list(range(world_size)):
        raise RuntimeError(f"{label} returned wrong rank order: {ranks}")


def finite_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(value)


def finite_tree(value: Any) -> bool:
    if isinstance(value, dict):
        return all(finite_tree(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return all(finite_tree(item) for item in value)
    return finite_number(value)


def runtime_git_revision() -> str:
    repo_root = Path(__file__).resolve().parents[2]
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def upload_result(client, uri: str, result: dict[str, Any]) -> str:
    unsigned = json.dumps(result, sort_keys=True, separators=(",", ":")).encode()
    result["result_sha256"] = hashlib.sha256(unsigned).hexdigest()
    payload = (json.dumps(result, indent=2, sort_keys=True) + "\n").encode()
    bucket, key = split_s3_uri(uri)
    client.put_object(Bucket=bucket, Key=key, Body=payload, ContentType="application/json")
    return hashlib.sha256(payload).hexdigest()


def main() -> None:
    args = parse_args()
    actual_source_revision = runtime_git_revision()
    if args.source_revision != actual_source_revision:
        raise RuntimeError(
            f"source revision mismatch: requested {args.source_revision}, running {actual_source_revision}"
        )
    if "@sha256:" not in args.image:
        raise ValueError("--image must pin an immutable sha256 digest")
    if len(args.flash_attn_wheel_sha256) != 64 or any(
        character not in "0123456789abcdef" for character in args.flash_attn_wheel_sha256
    ):
        raise ValueError("--flash-attn-wheel-sha256 must be a lowercase SHA-256 digest")
    if args.attention_backend == "flash_attention_2":
        import flash_attn_2_cuda

        flash_attn_runtime = {
            "version": importlib.metadata.version("flash-attn"),
            "wheel_sha256": args.flash_attn_wheel_sha256,
            "extension_path": flash_attn_2_cuda.__file__,
            "extension_sha256": sha256_file(Path(flash_attn_2_cuda.__file__)),
        }
    else:
        flash_attn_runtime = None
    if args.profile_s3_uri is not None and args.mode != "preflight":
        raise ValueError("profiling is restricted to bounded preflight runs")
    if args.expert_attribution and args.objective != "matched_ce":
        raise ValueError("expert attribution is defined only for the matched_ce boundary")
    if args.mode == "headline" and args.expert_implementation != "grouped":
        raise ValueError("headline rows require native grouped MM; eager is a preflight validity reference")
    if args.expert_parallel_size > 1 and args.expert_implementation != "grouped":
        raise ValueError("expert parallelism requires native grouped MM")
    world_size = WORLD_SIZES[args.mode]
    data_parallel_size = world_size // args.expert_parallel_size
    client = s3_client()
    with tempfile.TemporaryDirectory(prefix="grug-fixed-replay-") as work:
        manifest, shards = load_verified_replay(
            client,
            args.manifest_s3_uri,
            args.manifest_sha256,
            args.logical_batch_sha256,
            Path(work),
        )
        batches = make_rank_batches(manifest, shards, args.mode, args.expert_parallel_size)
        del shards
        primary_batches = batches[:: args.expert_parallel_size]
        global_loss_tokens = sum(int(batch["loss_mask"].sum().item()) for batch in primary_batches)
        if global_loss_tokens <= 0:
            raise RuntimeError("selected replay rows contain no loss tokens")
        for batch in batches:
            batch.metadata["grug_benchmark_global_loss_tokens"] = global_loss_tokens

        sequence_length = int(manifest["batch"]["fields"]["sequences"]["shape"][1])
        cfg = make_config(args, sequence_length)
        initialize_ray(cfg)
        cluster_gpus = int(ray.cluster_resources().get("GPU", 0))
        if cluster_gpus < world_size:
            raise RuntimeError(f"benchmark needs {world_size} Ray GPUs, found {cluster_gpus}")

        pg = None
        policy = None
        try:
            bundles = [{"GPU": GPUS_PER_NODE, "CPU": 8}] * (world_size // GPUS_PER_NODE)
            pg = placement_group(bundles, strategy="STRICT_SPREAD")
            get_ray_pg_ready_with_timeout(pg, timeout=args.pg_timeout_seconds)
            policy = PPORayActorGroup(
                cfg,
                num_nodes=world_size // GPUS_PER_NODE,
                num_gpus_per_node=GPUS_PER_NODE,
                ray_actor_type=PolicyWorker,
                pg=pg,
                num_gpus_per_actor=1,
                colocate_all=False,
                sequence_parallel_size=1,
                record_memory=False,
                pin_to_ray_gpu_id=False,
                force_cvd_mask=False,
            )
            ray.get(policy.async_init_model(args.model, num_training_steps=1, model_revision=args.model_revision))

            topology = ray.get(policy.async_run_ray_method("pass_through", "get_device_placement_diag"))
            assert_topology(topology, args.mode)
            identities = ray.get(policy.async_run_ray_method("pass_through", "grug_benchmark_identity"))
            require_rank_order(identities, world_size, "initial worker identity")
            expected_backend = args.attention_backend
            expected_fsdp_size = world_size // args.expert_parallel_size
            expected_mini = expected_microbatches(args.mode, args.expert_parallel_size)
            for identity in identities:
                if identity["model_revision"] != args.model_revision:
                    raise RuntimeError(f"rank loaded wrong model revision: {identity}")
                if identity["attention_backend"] != expected_backend:
                    raise RuntimeError(f"rank loaded wrong attention backend: {identity}")
                if "H100" not in identity["cuda_device_name"] or identity["cuda_total_memory_bytes"] < 79e9:
                    raise RuntimeError(f"rank is not on an H100-80GB: {identity}")
                if (
                    identity["expert_parallel_size"] != args.expert_parallel_size
                    or identity["fsdp_size"] != expected_fsdp_size
                    or identity["data_parallel_size"] != data_parallel_size
                    or identity["grouped_moe"] is not False
                ):
                    raise RuntimeError(f"wrong expert/FSDP path selected: {identity}")
                if identity["native_grouped_mm"] != (args.expert_implementation == "grouped"):
                    raise RuntimeError(f"rank selected the wrong native grouped-MM state: {identity}")
                if identity["expert_implementation"] != args.expert_implementation:
                    raise RuntimeError(f"rank selected the wrong expert implementation: {identity}")
                if identity["grug_query_bias_update_mode"] != "frozen":
                    raise RuntimeError(f"rank selected mutable query bias: {identity}")
                if identity["sample_packing"] or identity["micro_batch_size"] != 1:
                    raise RuntimeError(f"unexpected batch path selected: {identity}")
                if identity["mini_batch_size_per_gpu"] != expected_mini:
                    raise RuntimeError(f"wrong rank-local mini batch: {identity}")
            for evidence_name in ("runtime_grug_module_sha256", "runtime_worker_sha256"):
                evidence_values = {identity[evidence_name] for identity in identities}
                if len(evidence_values) != 1:
                    raise RuntimeError(f"workers loaded different {evidence_name} values: {evidence_values}")

            stage_refs = [
                actor.grug_benchmark_stage_batch.remote(batch) for actor, batch in zip(policy._actor_handlers, batches)
            ]
            staged = ray.get(stage_refs)
            require_rank_order(staged, world_size, "replay staging")
            for rank, (evidence, batch) in enumerate(zip(staged, batches)):
                expected_hashes, expected_shapes = field_identity(batch)
                if evidence["rank"] != rank:
                    raise RuntimeError(f"actor/rank order mismatch: {evidence}")
                if evidence["field_hashes"] != expected_hashes or evidence["field_shapes"] != expected_shapes:
                    raise RuntimeError(f"rank {rank} staged different replay bytes")
            for data_rank in range(data_parallel_size):
                replica_slice = staged[
                    data_rank * args.expert_parallel_size : (data_rank + 1) * args.expert_parallel_size
                ]
                hashes = [item["field_hashes"] for item in replica_slice]
                shapes = [item["field_shapes"] for item in replica_slice]
                if (
                    len({json.dumps(item, sort_keys=True) for item in hashes}) != 1
                    or len({json.dumps(item, sort_keys=True) for item in shapes}) != 1
                ):
                    raise RuntimeError(f"EP replicas for data rank {data_rank} received different replay rows")
            del batches
            del primary_batches

            warmup_method = (
                "grug_benchmark_warmup_and_restore"
                if args.objective == "operational"
                else "grug_benchmark_warmup_matched_ce"
            )
            warmup = ray.get(policy.async_run_ray_method("pass_through", warmup_method))
            require_rank_order(warmup, world_size, "warmup restore")
            if any(item["state_hash_before"] != item["state_hash_after"] for item in warmup):
                raise RuntimeError("warmup failed to restore a policy shard")
            memory_before = ray.get(policy.async_run_ray_method("pass_through", "grug_benchmark_reset_peak_memory"))
            require_rank_order(memory_before, world_size, "pre-timing memory reset")
            rpc_started = time.perf_counter()
            timed_method = (
                "grug_benchmark_run_staged_ppo"
                if args.objective == "operational"
                else "grug_benchmark_run_staged_matched_ce"
            )
            timed_args = [args.profile_s3_uri is not None]
            if args.objective == "matched_ce":
                timed_args.append(args.expert_attribution)
            timed = ray.get(
                policy.async_run_ray_method(
                    "pass_through",
                    timed_method,
                    *timed_args,
                )
            )
            require_rank_order(timed, world_size, "timed result")
            rpc_elapsed = time.perf_counter() - rpc_started

            profile_artifacts = []
            for item in timed:
                payload = item.pop("profile_artifact_gzip")
                if payload is None:
                    continue
                if profile_artifacts:
                    raise RuntimeError("more than one policy rank returned a profile")
                bucket, key = split_s3_uri(args.profile_s3_uri)
                client.put_object(Bucket=bucket, Key=key, Body=payload, ContentType="application/gzip")
                profile_artifacts.append(
                    {
                        "s3_uri": args.profile_s3_uri,
                        "bytes": len(payload),
                        "sha256": hashlib.sha256(payload).hexdigest(),
                        "rank": item["rank"],
                    }
                )
            if (args.profile_s3_uri is not None) != bool(profile_artifacts):
                raise RuntimeError("profile request and returned artifact disagree")

            if args.objective == "operational":
                raw_grad_norms = []
                for item in timed:
                    status = item["train_status"]
                    if not all(finite_number(value) for value in status.values()):
                        raise RuntimeError(f"rank {item['rank']} produced non-finite status: {status}")
                    if status.get("optimizer_step_succeeded") != 1.0:
                        raise RuntimeError(f"rank {item['rank']} skipped the optimizer boundary: {status}")
                    if status.get("policy_update_steps") != 1.0:
                        raise RuntimeError(f"rank {item['rank']} completed the wrong number of updates: {status}")
                    if status.get("raw_grad_norm", 0.0) <= 0.0:
                        raise RuntimeError(f"rank {item['rank']} did not report a positive gradient norm: {status}")
                    raw_grad_norms.append(float(status["raw_grad_norm"]))
                if not math.isclose(
                    min(raw_grad_norms),
                    max(raw_grad_norms),
                    rel_tol=1.0e-6,
                    abs_tol=1.0e-6,
                ):
                    raise RuntimeError(f"ranks disagree on the all-reduced raw gradient norm: {raw_grad_norms}")
                post_step_state = ray.get(
                    policy.async_run_ray_method("pass_through", "grug_benchmark_validate_finite_state")
                )
                require_rank_order(post_step_state, world_size, "post-step finite state")
                for item in post_step_state:
                    if item["model_tensors"] <= 0 or item["model_numel"] <= 0:
                        raise RuntimeError(f"rank returned no post-step model state: {item}")
                    if item["optimizer_tensors"] <= 0 or item["optimizer_numel"] <= 0:
                        raise RuntimeError(f"rank returned no post-step optimizer state: {item}")
                    nonfinite_counts = {name: value for name, value in item.items() if name.startswith("nonfinite_")}
                    if any(nonfinite_counts.values()):
                        raise RuntimeError(f"rank {item['rank']} retained non-finite state: {nonfinite_counts}")
            else:
                post_step_state = []
                expected_microbatch_count = expected_microbatches(args.mode, args.expert_parallel_size)
                for item, identity in zip(timed, identities):
                    if item["local_loss_tokens"] <= 0 or not finite_number(item["local_loss_sum"]):
                        raise RuntimeError(f"rank returned an invalid matched loss: {item}")
                    if item["microbatches"] != expected_microbatch_count:
                        raise RuntimeError(f"rank completed the wrong matched microbatch count: {item}")
                    if item["gradient_tensors"] <= 0 or item["gradient_numel"] <= 0:
                        raise RuntimeError(f"rank produced no matched gradients: {item}")
                    if item["nonfinite_gradient_tensors"] != 0:
                        raise RuntimeError(f"rank produced non-finite matched gradients: {item}")
                    if not item["representative_action_log_probs"] or not finite_tree(
                        item["representative_action_log_probs"]
                    ):
                        raise RuntimeError(f"rank produced invalid representative log probabilities: {item}")
                    if not item["representative_gradients"] or not finite_tree(item["representative_gradients"]):
                        raise RuntimeError(f"rank produced invalid representative gradients: {item}")
                    if args.expert_attribution:
                        evidence = item["expert_attribution"]
                        expected_blocks = identity["num_hidden_layers"]
                        if evidence is None or evidence["module_count"] != expected_blocks:
                            raise RuntimeError(f"rank did not instrument all {expected_blocks} routed blocks: {item}")
                        expected_calls = expected_blocks * expected_microbatch_count
                        if evidence["call_counts"] != {"forward": expected_calls, "backward": expected_calls}:
                            raise RuntimeError(f"rank returned unexpected routed-block call counts: {item}")
                        if evidence["route_calls_per_layer"] != [expected_microbatch_count] * expected_blocks:
                            raise RuntimeError(f"rank returned unexpected route call counts: {item}")
                        expected_layer_load = (
                            expected_microbatch_count * sequence_length * identity["num_experts_per_tok"]
                        )
                        if any(
                            len(layer_loads) != identity["num_local_experts"] or sum(layer_loads) != expected_layer_load
                            for layer_loads in evidence["route_loads_per_layer"]
                        ):
                            raise RuntimeError(f"rank returned wrong per-layer logical route work: {item}")
                action_gate = PRACTICAL_GATE["action_log_probs"]
                ce_gate = PRACTICAL_GATE["matched_global_ce"]
                for data_rank in range(data_parallel_size):
                    replicas = timed[
                        data_rank * args.expert_parallel_size : (data_rank + 1) * args.expert_parallel_size
                    ]
                    reference = replicas[0]
                    reference_ce = reference["local_loss_sum"] / reference["local_loss_tokens"]
                    for replica in replicas[1:]:
                        replica_ce = replica["local_loss_sum"] / replica["local_loss_tokens"]
                        if replica["local_loss_tokens"] != reference["local_loss_tokens"] or not math.isclose(
                            replica_ce,
                            reference_ce,
                            rel_tol=ce_gate["rtol"],
                            abs_tol=ce_gate["atol"],
                        ):
                            raise RuntimeError(
                                f"EP replicas for data rank {data_rank} disagree on matched CE: {replicas}"
                            )
                        values = replica["representative_action_log_probs"]
                        reference_values = reference["representative_action_log_probs"]
                        if len(values) != len(reference_values) or any(
                            not math.isclose(
                                value,
                                reference_value,
                                rel_tol=action_gate["rtol"],
                                abs_tol=action_gate["atol"],
                            )
                            for value, reference_value in zip(values, reference_values, strict=True)
                        ):
                            raise RuntimeError(
                                f"EP replicas for data rank {data_rank} disagree on representative action logprobs"
                            )

            elapsed_values = [item["elapsed_seconds"] for item in timed]
            if not elapsed_values or not finite_tree(elapsed_values) or min(elapsed_values) <= 0:
                raise RuntimeError(f"workers returned invalid elapsed times: {elapsed_values}")
            synchronized_wall = max(elapsed_values)
            worker_spread = max(elapsed_values) - min(elapsed_values)
            maximum_worker_spread = max(10.0, synchronized_wall * 0.05)
            if worker_spread > maximum_worker_spread:
                raise RuntimeError(
                    f"worker elapsed spread {worker_spread:.3f}s exceeds frozen limit {maximum_worker_spread:.3f}s"
                )
            critical = max(timed, key=lambda item: item["elapsed_seconds"])
            phase_sum = sum(critical["phase_seconds"].values())
            phase_residual = synchronized_wall - phase_sum
            if phase_residual < 0:
                raise RuntimeError(f"phase spans exceed synchronized wall by {-phase_residual:.6f}s")
            primary_staged = staged[:: args.expert_parallel_size]
            allocated_tokens = sum(item["allocated_tokens"] for item in primary_staged)
            nonpad_tokens = sum(item["nonpad_tokens"] for item in primary_staged)
            loss_tokens = sum(item["loss_tokens"] for item in primary_staged)
            logical_sequences = LOGICAL_SEQUENCES[args.mode]
            metrics = {
                "synchronized_wall_seconds": synchronized_wall,
                "rpc_wall_seconds": rpc_elapsed,
                "worker_elapsed_min_seconds": min(elapsed_values),
                "worker_elapsed_max_seconds": max(elapsed_values),
                "worker_elapsed_spread_seconds": worker_spread,
                "worker_elapsed_spread_limit_seconds": maximum_worker_spread,
                "gpu_seconds_per_logical_sequence": synchronized_wall * world_size / logical_sequences,
                "logical_sequences_per_second": logical_sequences / synchronized_wall,
                "allocated_tokens": allocated_tokens,
                "nonpadding_tokens": nonpad_tokens,
                "loss_tokens": loss_tokens,
                "allocated_tokens_per_second": allocated_tokens / synchronized_wall,
                "nonpadding_tokens_per_second": nonpad_tokens / synchronized_wall,
                "allocated_tokens_per_second_per_gpu": allocated_tokens / synchronized_wall / world_size,
                "nonpadding_tokens_per_second_per_gpu": nonpad_tokens / synchronized_wall / world_size,
                "critical_rank": critical["rank"],
                "critical_rank_phase_seconds": critical["phase_seconds"],
                "critical_rank_phase_sum_seconds": phase_sum,
                "phase_residual_seconds": phase_residual,
                "phase_residual_definition": (
                    "synchronized worker wall minus the sum of mutually exclusive CUDA-stream "
                    "events on the rank that set synchronized wall; includes Python/launch/barrier gaps"
                ),
                "peak_allocated_bytes_max": max(item["peak_allocated_bytes"] for item in timed),
                "peak_reserved_bytes_max": max(item["peak_reserved_bytes"] for item in timed),
            }
            if args.objective == "operational":
                metrics["raw_grad_norm"] = raw_grad_norms[0]
            if args.expert_attribution:
                expert_phase_seconds = critical["expert_attribution"]["phase_seconds"]
                expert_seconds = sum(expert_phase_seconds.values())
                expert_residual = synchronized_wall - expert_seconds
                if expert_residual < 0:
                    raise RuntimeError(
                        f"expert spans overlap or exceed synchronized wall: expert={expert_seconds}, wall={synchronized_wall}"
                    )
                metrics.update(
                    {
                        "critical_rank_expert_seconds": expert_seconds,
                        "critical_rank_expert_phase_seconds": expert_phase_seconds,
                        "critical_rank_nonexpert_seconds": expert_residual,
                        "critical_rank_expert_fraction": expert_seconds / synchronized_wall,
                        "expert_partition_definition": (
                            "synchronized wall partitions into routed-block CUDA-stream spans on the critical rank "
                            "and a nonnegative remainder containing every other operation, communication, and idle gap"
                        ),
                    }
                )
            if args.objective == "matched_ce":
                primary_timed = timed[:: args.expert_parallel_size]
                timed_loss_tokens = sum(item["local_loss_tokens"] for item in primary_timed)
                if timed_loss_tokens != global_loss_tokens or timed_loss_tokens != loss_tokens:
                    raise RuntimeError(
                        "matched loss-token accounting disagrees: "
                        f"timed={timed_loss_tokens}, configured={global_loss_tokens}, staged={loss_tokens}"
                    )
                metrics["matched_global_ce_loss"] = (
                    sum(item["local_loss_sum"] for item in primary_timed) / timed_loss_tokens
                )
                metrics["matched_gradient_tensors_min"] = min(item["gradient_tensors"] for item in timed)
                metrics["matched_gradient_numel_min"] = min(item["gradient_numel"] for item in timed)
                metrics["matched_representative_gradients"] = aggregate_representative_gradients(
                    timed, args.expert_parallel_size
                )
            result = {
                "schema_version": 1,
                "created_utc": dt.datetime.now(dt.UTC).isoformat(),
                "benchmark": f"marinskyrl_grug_fixed_replay_{args.objective}",
                "objective": args.objective,
                "mode": args.mode,
                "sample": args.sample,
                "source_revision": args.source_revision,
                "runtime_source_revision": actual_source_revision,
                "image": args.image,
                "model": args.model,
                "model_revision": args.model_revision,
                "attention_backend": args.attention_backend,
                "flash_attn_runtime": flash_attn_runtime,
                "expert_implementation": args.expert_implementation,
                "expert_parallel_size": args.expert_parallel_size,
                "fsdp_size": world_size // args.expert_parallel_size,
                "data_parallel_size": data_parallel_size,
                "ep_replication_factor": args.expert_parallel_size,
                "world_size": world_size,
                "manifest_s3_uri": args.manifest_s3_uri,
                "manifest_sha256": args.manifest_sha256,
                "logical_batch_sha256": args.logical_batch_sha256,
                "manifest_batch": manifest["batch"],
                "manifest_batch_metadata": manifest["batch_metadata"],
                "practical_correctness_gate": PRACTICAL_GATE,
                "config": OmegaConf.to_container(cfg, resolve=True),
                "topology": topology,
                "worker_identities": identities,
                "staging": staged,
                "warmup_restore": warmup,
                "memory_before_timing": memory_before,
                "timed_workers": timed,
                "post_step_finite_state": post_step_state,
                "profile_in_timed_sample": args.profile_s3_uri is not None,
                "expert_attribution_in_timed_sample": args.expert_attribution,
                "runtime_benchmark_path": str(Path(__file__).resolve()),
                "runtime_benchmark_sha256": sha256_file(Path(__file__).resolve()),
                "profile_artifacts": profile_artifacts,
                "headline_eligible": (
                    args.mode == "headline" and args.expert_implementation == "grouped" and args.profile_s3_uri is None
                ),
                "metrics": metrics,
                "data_layout": (
                    "the verified logical replay is split over data_parallel_size unique chunks; each chunk is "
                    "replicated byte-for-byte over expert_parallel_size ranks, matching MeshDispatch"
                ),
                "timing_boundary": (
                    "after verified replay download/staging, model load, warmup, exact state restore, "
                    "and peak reset; includes synchronized production ppo_train forward, policy loss, "
                    "backward, required collectives, and optimizer"
                    if args.objective == "operational"
                    else "after verified replay download/staging, model load, matched forward/backward "
                    "warmup, unchanged-state proof, and peak reset; includes synchronized next-token "
                    "forward, global token-weighted CE, backward, and FSDP collectives; excludes Adam"
                ),
            }
            payload_sha256 = upload_result(client, args.result_s3_uri, result)
            print(
                "GRUG_BENCHMARK_RESULT="
                + json.dumps(
                    {
                        "result_s3_uri": args.result_s3_uri,
                        "payload_sha256": payload_sha256,
                        "result_sha256": result["result_sha256"],
                        "metrics": metrics,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
        finally:
            if policy is not None:
                for actor in policy._actor_handlers:
                    ray.kill(actor, no_restart=True)
            if pg is not None:
                ray.util.remove_placement_group(pg)
            ray.shutdown()


if __name__ == "__main__":
    main()
