"""Bounded, read-only diagnostics for an opt-in initial weight sync."""

import hashlib
import json
import os
import socket
import time

import torch


HASH_CHUNK_BYTES = 1 << 20
RECEIPT_CHUNK_BYTES = 3072
ENVIRONMENT_KEYS = (
    "NCCL_DEBUG",
    "NCCL_DEBUG_SUBSYS",
    "NCCL_IB_HCA",
    "NCCL_SOCKET_IFNAME",
    "NCCL_NET",
    "NCCL_NET_GDR_LEVEL",
    "NCCL_IB_DISABLE",
    "NCCL_PROTO",
    "NCCL_ALGO",
    "NCCL_MIN_NCHANNELS",
    "NCCL_MAX_NCHANNELS",
    "NCCL_NTHREADS",
    "NCCL_P2P_NET_DISABLE",
    "NCCL_LAUNCH_MODE",
    "NCCL_COLLNET_ENABLE",
    "NCCL_NVLS_ENABLE",
    "VLLM_BATCH_INVARIANT",
    "VLLM_ALLREDUCE_USE_SYMM_MEM",
    "CUBLAS_WORKSPACE_CONFIG",
)


def environment_readback() -> dict:
    return {
        "host": socket.gethostname(),
        "pid": os.getpid(),
        "observed_monotonic": time.monotonic(),
        "values": {key: os.environ.get(key) for key in ENVIRONMENT_KEYS},
    }


def tensor_sha256(tensor: torch.Tensor) -> str:
    """Hash exact contiguous storage bytes using at most one MiB of host transfer scratch.

    This diagnostic hash is not the K14 every-byte comparison proof. No tensor
    conversion, full-sized contiguous copy, or device scratch allocation occurs.
    """
    if not tensor.is_contiguous():
        raise ValueError("Diagnostic hashing requires contiguous tensor storage")
    raw = tensor.detach().reshape(-1).view(torch.uint8)
    digest = hashlib.sha256()
    for offset in range(0, raw.numel(), HASH_CHUNK_BYTES):
        chunk = raw[offset : offset + HASH_CHUNK_BYTES].cpu()
        digest.update(memoryview(chunk.numpy()))
    return digest.hexdigest()


def parameter_digests(model_chunks) -> dict:
    """Keep dense and expert-shard identities separate, preserving names and shapes."""
    digests = {kind: hashlib.sha256() for kind in ("dense", "expert")}
    counts = {kind: {"parameters": 0, "bytes": 0} for kind in digests}
    samples = []
    for chunk_index, chunk in enumerate(model_chunks):
        for name, parameter in chunk.named_parameters():
            kind = "dense" if getattr(parameter, "allreduce", True) else "expert"
            receipt = {
                "chunk": chunk_index,
                "name": name,
                "shape": list(parameter.shape),
                "dtype": str(parameter.dtype),
                "sha256": tensor_sha256(parameter),
            }
            digests[kind].update(json.dumps(receipt, sort_keys=True, separators=(",", ":")).encode())
            counts[kind]["parameters"] += 1
            counts[kind]["bytes"] += parameter.numel() * parameter.element_size()
            if ".experts." in name and name.endswith(("linear_fc1.weight0", "linear_fc2.weight0")):
                receipt["stride"] = list(parameter.stride())
                receipt["contiguous"] = parameter.is_contiguous()
                if name.endswith("linear_fc1.weight0"):
                    if parameter.shape[0] % 2:
                        raise ValueError("Expert fc1 cannot be split into equal gate/up halves")
                    halves = parameter.chunk(2, dim=0)
                    receipt["half_sha256"] = [tensor_sha256(half) for half in halves]
                samples.append(receipt)
    return {
        "digests": {kind: {**counts[kind], "sha256": value.hexdigest()} for kind, value in digests.items()},
        "expert_samples": samples,
    }


def validate_replica_digests(rows: list[dict]) -> list[dict]:
    """Compare only ranks in actual Megatron replica groups, never across expert shards."""
    by_rank = {row["rank"]: row for row in rows}
    if len(by_rank) != len(rows):
        raise ValueError("Duplicate policy rank")
    comparisons = []
    for kind in ("dense", "expert"):
        groups = {tuple(row["replica_ranks"][kind]) for row in rows}
        for ranks in sorted(groups):
            if not ranks or len(set(ranks)) != len(ranks) or any(rank not in by_rank for rank in ranks):
                raise ValueError("Incomplete replica group coverage")
            group = [by_rank[rank] for rank in ranks]
            if any(tuple(row["replica_ranks"][kind]) != ranks for row in group):
                raise ValueError("Asymmetric replica group membership")
            first = group[0]["digests"][kind]
            if any(row["digests"][kind] != first for row in group):
                raise ValueError(f"{kind} parameter digest mismatch on replica ranks {ranks}")
            comparisons.append({"kind": kind, "ranks": list(ranks), **first})
    return comparisons


def receipt_chunks(receipt: dict) -> list[dict]:
    """Encode bounded ASCII chunks; JSON escaping preserves arbitrary Unicode exactly."""
    encoded = json.dumps(receipt, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
    digest = hashlib.sha256(encoded.encode()).hexdigest()
    parts = [encoded[offset : offset + RECEIPT_CHUNK_BYTES] for offset in range(0, len(encoded), RECEIPT_CHUNK_BYTES)]
    return [
        {"sha256": digest, "part": index, "parts": len(parts), "bytes": len(encoded), "payload": part}
        for index, part in enumerate(parts)
    ]


def reassemble_receipt(chunks: list[dict]) -> dict:
    if not chunks:
        raise ValueError("Missing diagnostic receipt")
    first = chunks[0]
    if len(chunks) != first["parts"] or {chunk["part"] for chunk in chunks} != set(range(first["parts"])):
        raise ValueError("Incomplete diagnostic receipt")
    if any(any(chunk[key] != first[key] for key in ("sha256", "parts", "bytes")) for chunk in chunks):
        raise ValueError("Mixed diagnostic receipts")
    encoded = "".join(chunk["payload"] for chunk in sorted(chunks, key=lambda chunk: chunk["part"]))
    if len(encoded.encode()) != first["bytes"] or hashlib.sha256(encoded.encode()).hexdigest() != first["sha256"]:
        raise ValueError("Diagnostic receipt hash mismatch")
    return json.loads(encoded)
