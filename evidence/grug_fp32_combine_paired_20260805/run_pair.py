#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 NovaSkyAI
# SPDX-License-Identifier: Apache-2.0

"""Fixed one-node H100 measurement for the Grug FP32 combine candidate.

This is evidence-only code. It runs the exact parent module and a mechanically
verified candidate module in fresh child processes. The real-shape timing arms
use identical state, hidden input, expert IDs, combine weights, and cotangent.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


PARENT_REVISION = "0c213586b5491b8046ca7780e965c4b26dc6a2a2"
CANDIDATE_REVISION = "fbb1fc8378601e0346d00d186809f10d1ad0360d"
PARENT_MODULE_SHA256 = "b1e63368996530dd8fa678ec3b482a1bd63007c0d69901cd13c6a4e42c294d50"
CANDIDATE_MODULE_SHA256 = "2dd51d842afe6f57fe00337c21945923da68664d9c23678633c578c27504ba93"

GPU_COUNT = 8
GPU_NAME = "NVIDIA H100 80GB HBM3"
WARMUP_ITERATIONS = 5
MEASURED_ITERATIONS = 20
REAL_SEED = 20260805

TOKENS = 8192
HIDDEN_SIZE = 2560
INTERMEDIATE_SIZE = 1280
NUM_EXPERTS = 256
TOP_K = 4

CORRECTNESS_TOKENS = 4096
CORRECTNESS_HIDDEN_SIZE = 128
CORRECTNESS_INTERMEDIATE_SIZE = 64
CORRECTNESS_NUM_EXPERTS = 5
OUTPUT_RTOL = 0.0
OUTPUT_ATOL = 0.0
GRADIENT_RTOL = 8e-2
GRADIENT_ATOL = 1e-4

CGROUP_STOP_BYTES = 100 * 1024**3
HOST_AVAILABLE_STOP_BYTES = 256 * 1024**3
MIN_SWAP_FREE_FRACTION = 0.5

ROOT = Path(__file__).resolve().parents[2]
PACKAGE_ROOT = ROOT / "skyrl-train" / "skyrl_train"
MODULE_RELATIVE_PATH = Path("models/grug_moe.py")

PARENT_EAGER = """        output = torch.zeros_like(hidden_states)
        for expert_idx in range(self.num_experts):
            token_idx, slot_idx = torch.where(selected_experts == expert_idx)
            if token_idx.numel() == 0:
                continue
            expert_input = hidden_states.index_select(0, token_idx)
            gate = F.linear(expert_input, self.gate_proj.weight[expert_idx])
            up = F.linear(expert_input, self.up_proj.weight[expert_idx])
            expert_output = F.linear(F.silu(gate) * up, self.down_proj.weight[expert_idx])
            weight = combine_weights[token_idx, slot_idx].unsqueeze(-1).to(expert_output.dtype)
            output.index_add_(0, token_idx, expert_output * weight)
        return output
"""

CANDIDATE_EAGER = """        output = torch.zeros_like(hidden_states, dtype=torch.float32)
        for expert_idx in range(self.num_experts):
            token_idx, slot_idx = torch.where(selected_experts == expert_idx)
            if token_idx.numel() == 0:
                continue
            expert_input = hidden_states.index_select(0, token_idx)
            gate = F.linear(expert_input, self.gate_proj.weight[expert_idx])
            up = F.linear(expert_input, self.up_proj.weight[expert_idx])
            expert_output = F.linear(F.silu(gate) * up, self.down_proj.weight[expert_idx])
            weight = combine_weights[token_idx, slot_idx].unsqueeze(-1).to(expert_output.dtype)
            output.index_add_(0, token_idx, (expert_output * weight).float())
        return output.to(hidden_states.dtype)
"""

PARENT_GROUPED = """        output = torch.zeros((num_tokens, hidden_size), dtype=hidden_states.dtype, device=hidden_states.device)
        output.index_add_(0, sorted_token_indices, routed_output)
        return output
"""

CANDIDATE_GROUPED = """        output = torch.zeros((num_tokens, hidden_size), dtype=torch.float32, device=hidden_states.device)
        output.index_add_(0, sorted_token_indices, routed_output.float())
        return output.to(hidden_states.dtype)
"""


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _candidate_source(parent_source: str) -> str:
    replacements = ((PARENT_EAGER, CANDIDATE_EAGER), (PARENT_GROUPED, CANDIDATE_GROUPED))
    candidate = parent_source
    for old, new in replacements:
        if candidate.count(old) != 1:
            raise RuntimeError("the pinned parent source no longer contains exactly one expected combine block")
        candidate = candidate.replace(old, new)
    return candidate


def _prepare_sources(runtime_dir: Path) -> dict[str, Path]:
    parent_module = PACKAGE_ROOT / MODULE_RELATIVE_PATH
    if _sha256_file(parent_module) != PARENT_MODULE_SHA256:
        raise RuntimeError(f"parent module pin failed: {parent_module}")
    parent_source = parent_module.read_text()
    candidate_source = _candidate_source(parent_source)
    if _sha256_bytes(candidate_source.encode()) != CANDIDATE_MODULE_SHA256:
        raise RuntimeError("mechanically generated candidate does not match fbb1fc8")

    source_roots: dict[str, Path] = {}
    for arm in ("parent", "candidate"):
        destination = runtime_dir / "sources" / arm / "skyrl_train"
        if destination.exists():
            shutil.rmtree(destination)
        shutil.copytree(PACKAGE_ROOT, destination)
        source_roots[arm] = destination.parent
    (source_roots["candidate"] / "skyrl_train" / MODULE_RELATIVE_PATH).write_text(candidate_source)
    if _sha256_file(source_roots["parent"] / "skyrl_train" / MODULE_RELATIVE_PATH) != PARENT_MODULE_SHA256:
        raise RuntimeError("prepared parent source hash mismatch")
    if _sha256_file(source_roots["candidate"] / "skyrl_train" / MODULE_RELATIVE_PATH) != CANDIDATE_MODULE_SHA256:
        raise RuntimeError("prepared candidate source hash mismatch")
    return source_roots


def _import_torch_and_grug():
    import torch
    import torch.nn.functional as functional

    from skyrl_train.models.grug_moe import GrugMoeConfig, GrugMoeSparseMoeBlock

    return torch, functional, GrugMoeConfig, GrugMoeSparseMoeBlock


def _tensor_digest(tensor: Any, *, chunk_bytes: int = 64 * 1024**2) -> str:
    """Hash a CUDA tensor without retaining a host-sized copy."""

    torch, _, _, _ = _import_torch_and_grug()
    detached = tensor.detach().contiguous()
    byte_view = detached.view(torch.uint8).reshape(-1)
    digest = hashlib.sha256()
    digest.update(str(tuple(detached.shape)).encode())
    digest.update(str(detached.dtype).encode())
    for start in range(0, byte_view.numel(), chunk_bytes):
        chunk = byte_view[start : start + chunk_bytes].cpu().numpy()
        digest.update(memoryview(chunk))
    return digest.hexdigest()


def _state_digest(block: Any) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(block.state_dict().items()):
        digest.update(name.encode())
        digest.update(_tensor_digest(tensor).encode())
    return digest.hexdigest()


def _difference(actual: Any, expected: Any, *, rtol: float, atol: float) -> dict[str, Any]:
    absolute = (actual.detach().double() - expected.detach().double()).abs()
    allowance = atol + rtol * expected.detach().double().abs()
    return {
        "exact": bool(actual.dtype == expected.dtype and torch_equal(actual, expected)),
        "allclose": bool((absolute <= allowance).all().item()),
        "max_abs": float(absolute.max().item()),
        "mean_abs": float(absolute.mean().item()),
        "failures": int((absolute > allowance).sum().item()),
        "numel": int(actual.numel()),
    }


def torch_equal(left: Any, right: Any) -> bool:
    torch, _, _, _ = _import_torch_and_grug()
    return bool(torch.equal(left.detach(), right.detach()))


def _config(*, real_shape: bool) -> Any:
    _, _, GrugMoeConfig, _ = _import_torch_and_grug()
    hidden_size = HIDDEN_SIZE if real_shape else CORRECTNESS_HIDDEN_SIZE
    intermediate_size = INTERMEDIATE_SIZE if real_shape else CORRECTNESS_INTERMEDIATE_SIZE
    num_experts = NUM_EXPERTS if real_shape else CORRECTNESS_NUM_EXPERTS
    tokens = TOKENS if real_shape else CORRECTNESS_TOKENS
    return GrugMoeConfig(
        vocab_size=128256 if real_shape else 32,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        shared_expert_intermediate_size=hidden_size,
        num_local_experts=num_experts,
        num_experts_per_tok=TOP_K,
        num_hidden_layers=1,
        num_attention_heads=20 if real_shape else 4,
        num_key_value_heads=5 if real_shape else 2,
        head_dim=128 if real_shape else 32,
        max_position_embeddings=tokens,
        sliding_window=2048 if real_shape else 4,
        initializer_range=0.02,
        qk_mult=1.0,
    )


def _build_block(*, real_shape: bool) -> Any:
    torch, _, _, GrugMoeSparseMoeBlock = _import_torch_and_grug()
    original_dtype = torch.get_default_dtype()
    torch.set_default_dtype(torch.bfloat16)
    try:
        with torch.device("cuda"):
            block = GrugMoeSparseMoeBlock(_config(real_shape=real_shape))
    finally:
        torch.set_default_dtype(original_dtype)
    return block


def _initialize_real_block(block: Any) -> None:
    torch, _, _, _ = _import_torch_and_grug()
    generator = torch.Generator(device="cuda").manual_seed(REAL_SEED)
    with torch.no_grad():
        for _, parameter in sorted(block.named_parameters()):
            parameter.normal_(mean=0.0, std=0.02, generator=generator)
    block.enable_grouped_mm()


def _initialize_correctness_block(block: Any) -> None:
    torch, _, _, _ = _import_torch_and_grug()
    with torch.no_grad():
        block.router.weight.zero_()
        block.experts.gate_proj.weight.zero_()
        block.experts.up_proj.weight.zero_()
        block.experts.down_proj.weight.zero_()
        for expert in range(4):
            block.experts.gate_proj.weight[expert, 0, 0] = 16.0
            block.experts.up_proj.weight[expert, 0, 0] = 1.0
        block.experts.down_proj.weight[0, 0, 0] = 1.0 / 16.0
        block.experts.down_proj.weight[1:4, 0, 0] = 1.0 / 2048.0
    block.enable_grouped_mm()


def _correctness_inputs() -> tuple[Any, Any, Any, Any]:
    torch, _, _, _ = _import_torch_and_grug()
    hidden = torch.zeros((CORRECTNESS_TOKENS, CORRECTNESS_HIDDEN_SIZE), dtype=torch.bfloat16, device="cuda")
    hidden[:, 0] = 1.0
    selected_experts = torch.arange(TOP_K, dtype=torch.long, device="cuda").expand(CORRECTNESS_TOKENS, -1).clone()
    combine_weights = (
        torch.tensor([1.0, 0.5, 0.5, 0.5], dtype=torch.float32, device="cuda")
        .expand(CORRECTNESS_TOKENS, -1)
        .clone()
    )
    cotangent = torch.linspace(
        -1.0,
        1.0,
        steps=hidden.numel(),
        dtype=torch.float32,
        device="cuda",
    ).reshape_as(hidden)
    return hidden, selected_experts, combine_weights, cotangent


def _correctness_fixture() -> tuple[Any, tuple[Any, ...], dict[str, str]]:
    block = _build_block(real_shape=False)
    _initialize_correctness_block(block)
    hidden, selected_experts, combine_weights, cotangent = _correctness_inputs()
    hidden.requires_grad_(True)
    combine_weights.requires_grad_(True)
    hashes = {
        "state": _state_digest(block),
        "hidden": _tensor_digest(hidden),
        "selected_experts": _tensor_digest(selected_experts),
        "combine_weights": _tensor_digest(combine_weights),
        "cotangent": _tensor_digest(cotangent),
    }
    return block, (hidden, selected_experts, combine_weights, cotangent), hashes


def _slot_outputs(block: Any, hidden: Any, selected_experts: Any, *, path: str) -> Any:
    torch, functional, _, _ = _import_torch_and_grug()
    if path == "eager":
        outputs = []
        for slot in range(TOP_K):
            expert = int(selected_experts[0, slot].item())
            gate = functional.linear(hidden, block.experts.gate_proj.weight[expert])
            up = functional.linear(hidden, block.experts.up_proj.weight[expert])
            outputs.append(functional.linear(functional.silu(gate) * up, block.experts.down_proj.weight[expert]))
        return torch.stack(outputs, dim=1)
    if path != "grouped":
        raise ValueError(path)
    flat_experts = selected_experts.reshape(-1)
    order = torch.argsort(flat_experts, stable=True)
    token_indices = (
        torch.arange(hidden.shape[0], device="cuda").unsqueeze(1).expand(-1, TOP_K).reshape(-1)
    )
    sorted_token_indices = token_indices.index_select(0, order)
    routed_input = hidden.index_select(0, sorted_token_indices)
    counts = torch.bincount(flat_experts, minlength=CORRECTNESS_NUM_EXPERTS)
    routed_output = block.experts(routed_input, counts)
    flat_slot_outputs = torch.empty_like(routed_output)
    flat_slot_outputs.index_copy_(0, order, routed_output)
    return flat_slot_outputs.reshape(hidden.shape[0], TOP_K, hidden.shape[1])


def _reference_output(slot_outputs: Any, combine_weights: Any, *, accumulation_dtype: Any) -> tuple[Any, Any]:
    torch, _, _, _ = _import_torch_and_grug()
    weighted = slot_outputs * combine_weights.to(slot_outputs.dtype).unsqueeze(-1)
    accumulator = torch.zeros(
        (slot_outputs.shape[0], slot_outputs.shape[2]),
        dtype=accumulation_dtype,
        device="cuda",
    )
    for slot in range(TOP_K):
        accumulator = accumulator + weighted[:, slot].to(accumulation_dtype)
    return accumulator.to(slot_outputs.dtype), accumulator


def _correctness_run(kind: str, path: str) -> tuple[Any, tuple[Any, ...], dict[str, str], Any | None]:
    torch, _, _, _ = _import_torch_and_grug()
    block, (hidden, selected_experts, combine_weights, cotangent), hashes = _correctness_fixture()
    if kind == "actual":
        if path == "eager":
            output = block.experts.forward_eager(hidden, selected_experts, combine_weights)
        elif path == "grouped":
            output = block._forward_grouped(hidden, selected_experts, combine_weights)
        else:
            raise ValueError(path)
        pre_cast = None
    else:
        slot_outputs = _slot_outputs(block, hidden, selected_experts, path=path)
        accumulation_dtype = torch.float32 if kind == "fp32" else torch.float64
        output, pre_cast = _reference_output(slot_outputs, combine_weights, accumulation_dtype=accumulation_dtype)
    loss = (output.float().square() * (1.0 + cotangent)).mean()
    gradients = torch.autograd.grad(
        loss,
        (
            hidden,
            combine_weights,
            block.experts.gate_proj.weight,
            block.experts.up_proj.weight,
            block.experts.down_proj.weight,
        ),
    )
    return output.detach(), tuple(item.detach() for item in gradients), hashes, None if pre_cast is None else pre_cast.detach()


def _comparison(actual: tuple[Any, tuple[Any, ...]], expected: tuple[Any, tuple[Any, ...]]) -> dict[str, Any]:
    names = ("hidden", "combine_weights", "gate_weight", "up_weight", "down_weight")
    actual_output, actual_gradients = actual
    expected_output, expected_gradients = expected
    return {
        "output": _difference(actual_output, expected_output, rtol=OUTPUT_RTOL, atol=OUTPUT_ATOL),
        "gradients": {
            name: _difference(actual_gradient, expected_gradient, rtol=GRADIENT_RTOL, atol=GRADIENT_ATOL)
            for name, actual_gradient, expected_gradient in zip(names, actual_gradients, expected_gradients)
        },
    }


def _run_correctness(arm: str) -> dict[str, Any]:
    torch, _, _, GrugMoeSparseMoeBlock = _import_torch_and_grug()
    module_path = Path(sys.modules[GrugMoeSparseMoeBlock.__module__].__file__).resolve()
    expected_source = PARENT_MODULE_SHA256 if arm == "parent" else CANDIDATE_MODULE_SHA256
    if _sha256_file(module_path) != expected_source:
        raise RuntimeError(f"{arm} imported the wrong module: {module_path}")
    if torch.cuda.get_device_name() != GPU_NAME:
        raise RuntimeError(f"correctness preflight requires {GPU_NAME}, got {torch.cuda.get_device_name()}")

    paths: dict[str, Any] = {}
    all_fixture_hashes = []
    actual_by_path = {}
    for path in ("eager", "grouped"):
        actual_output, actual_gradients, hashes, _ = _correctness_run("actual", path)
        fp32_output, fp32_gradients, fp32_hashes, fp32_pre_cast = _correctness_run("fp32", path)
        fp64_output, fp64_gradients, fp64_hashes, fp64_pre_cast = _correctness_run("fp64", path)
        if not (hashes == fp32_hashes == fp64_hashes):
            raise RuntimeError(f"{path} correctness fixtures differ")
        all_fixture_hashes.append(hashes)
        actual_by_path[path] = (actual_output, actual_gradients)
        paths[path] = {
            "actual_vs_fp32_slotwise": _comparison(
                (actual_output, actual_gradients),
                (fp32_output, fp32_gradients),
            ),
            "actual_vs_fp64_fixed_order": _comparison(
                (actual_output, actual_gradients),
                (fp64_output, fp64_gradients),
            ),
            "fp64_pre_cast_accuracy": {
                "actual": _difference(
                    actual_output.double(),
                    fp64_pre_cast,
                    rtol=0.0,
                    atol=0.0,
                ),
                "fp32_slotwise": _difference(
                    fp32_pre_cast.double(),
                    fp64_pre_cast,
                    rtol=0.0,
                    atol=0.0,
                ),
            },
        }
    if all_fixture_hashes[0] != all_fixture_hashes[1]:
        raise RuntimeError("eager and grouped correctness fixtures differ")
    return {
        "arm": arm,
        "revision": PARENT_REVISION if arm == "parent" else CANDIDATE_REVISION,
        "module_sha256": expected_source,
        "fixture": {
            "tokens": CORRECTNESS_TOKENS,
            "hidden_size": CORRECTNESS_HIDDEN_SIZE,
            "intermediate_size": CORRECTNESS_INTERMEDIATE_SIZE,
            "num_experts": CORRECTNESS_NUM_EXPERTS,
            "top_k": TOP_K,
            "hashes": all_fixture_hashes[0],
        },
        "rules": {
            "output_rtol": OUTPUT_RTOL,
            "output_atol": OUTPUT_ATOL,
            "gradient_rtol": GRADIENT_RTOL,
            "gradient_atol": GRADIENT_ATOL,
        },
        "paths": paths,
        "eager_vs_grouped": _comparison(actual_by_path["eager"], actual_by_path["grouped"]),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "device": torch.cuda.get_device_name(),
    }


def _real_inputs() -> tuple[Any, Any, Any, Any]:
    torch, _, _, _ = _import_torch_and_grug()
    generator = torch.Generator(device="cuda").manual_seed(REAL_SEED + 1)
    hidden = torch.randn(
        (TOKENS, HIDDEN_SIZE),
        dtype=torch.bfloat16,
        device="cuda",
        generator=generator,
    )
    token = torch.arange(TOKENS, dtype=torch.long, device="cuda").unsqueeze(1)
    offsets = torch.tensor([0, 1, 7, 31], dtype=torch.long, device="cuda")
    selected_experts = (token * 73 + offsets) % NUM_EXPERTS
    token_phase = ((torch.arange(TOKENS, dtype=torch.float32, device="cuda") % 97) - 48.0) / 32.0
    slot_phase = torch.tensor([0.75, 0.25, -0.25, -0.75], dtype=torch.float32, device="cuda")
    unnormalized = torch.sigmoid(token_phase.unsqueeze(1) + slot_phase)
    combine_weights = unnormalized * (2.5 / unnormalized.sum(dim=-1, keepdim=True))
    cotangent = torch.linspace(-1.0, 1.0, steps=hidden.numel(), dtype=torch.float32, device="cuda")
    cotangent = cotangent.reshape_as(hidden).to(torch.bfloat16)
    hidden.requires_grad_(True)
    combine_weights.requires_grad_(True)
    return hidden, selected_experts, combine_weights, cotangent


def _cuda_iteration(function: Any, cotangent: Any) -> tuple[float, float]:
    torch, _, _, _ = _import_torch_and_grug()
    start = torch.cuda.Event(enable_timing=True)
    forward_done = torch.cuda.Event(enable_timing=True)
    backward_done = torch.cuda.Event(enable_timing=True)
    start.record()
    output = function()
    forward_done.record()
    torch.autograd.backward(output, cotangent)
    backward_done.record()
    backward_done.synchronize()
    forward_ms = float(start.elapsed_time(forward_done))
    backward_ms = float(forward_done.elapsed_time(backward_done))
    del output
    return forward_ms, backward_ms


def _measure_boundary(
    function: Any,
    clear_gradients: Any,
    cotangent: Any,
) -> dict[str, Any]:
    torch, _, _, _ = _import_torch_and_grug()
    for _ in range(WARMUP_ITERATIONS):
        clear_gradients()
        _cuda_iteration(function, cotangent)
    clear_gradients()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    baseline = int(torch.cuda.memory_allocated())
    torch.cuda.reset_peak_memory_stats()
    forward_ms = []
    backward_ms = []
    for _ in range(MEASURED_ITERATIONS):
        clear_gradients()
        forward, backward = _cuda_iteration(function, cotangent)
        forward_ms.append(forward)
        backward_ms.append(backward)
    peak = int(torch.cuda.max_memory_allocated())
    clear_gradients()
    return {
        "warmup_iterations": WARMUP_ITERATIONS,
        "measured_iterations": MEASURED_ITERATIONS,
        "forward_ms": forward_ms,
        "backward_ms": backward_ms,
        "forward_backward_ms": [forward + backward for forward, backward in zip(forward_ms, backward_ms)],
        "baseline_allocated_bytes": baseline,
        "peak_allocated_bytes": peak,
        "incremental_peak_allocated_bytes": peak - baseline,
    }


def _combine_function(
    arm: str,
    routed_output: Any,
    combine_weights: Any,
    order: Any,
    sorted_token_indices: Any,
) -> Any:
    torch, _, _, _ = _import_torch_and_grug()
    sorted_weights = combine_weights.reshape(-1).index_select(0, order).to(routed_output.dtype)
    weighted = routed_output * sorted_weights.unsqueeze(-1)
    if arm == "parent":
        output = torch.zeros((TOKENS, HIDDEN_SIZE), dtype=torch.bfloat16, device="cuda")
        output.index_add_(0, sorted_token_indices, weighted)
        return output
    output = torch.zeros((TOKENS, HIDDEN_SIZE), dtype=torch.float32, device="cuda")
    output.index_add_(0, sorted_token_indices, weighted.float())
    return output.to(torch.bfloat16)


def _run_performance(arm: str, physical_gpu: int, repetition: int) -> dict[str, Any]:
    torch, _, _, GrugMoeSparseMoeBlock = _import_torch_and_grug()
    module_path = Path(sys.modules[GrugMoeSparseMoeBlock.__module__].__file__).resolve()
    expected_source = PARENT_MODULE_SHA256 if arm == "parent" else CANDIDATE_MODULE_SHA256
    if _sha256_file(module_path) != expected_source:
        raise RuntimeError(f"{arm} imported the wrong module: {module_path}")
    if torch.cuda.device_count() != 1:
        raise RuntimeError(f"worker must see one GPU, found {torch.cuda.device_count()}")
    if torch.cuda.get_device_name() != GPU_NAME:
        raise RuntimeError(f"performance worker requires {GPU_NAME}, got {torch.cuda.get_device_name()}")

    torch.manual_seed(REAL_SEED)
    torch.cuda.manual_seed_all(REAL_SEED)
    block = _build_block(real_shape=True)
    _initialize_real_block(block)
    hidden, selected_experts, combine_weights, cotangent = _real_inputs()
    fixture_hashes = {
        "state": _state_digest(block),
        "hidden": _tensor_digest(hidden),
        "selected_experts": _tensor_digest(selected_experts),
        "combine_weights": _tensor_digest(combine_weights),
        "cotangent": _tensor_digest(cotangent),
    }
    counts = torch.bincount(selected_experts.reshape(-1), minlength=NUM_EXPERTS)
    if int(counts.min().item()) != 128 or int(counts.max().item()) != 128:
        raise RuntimeError("the frozen real-shape route must assign exactly 128 rows per expert")

    def clear_full() -> None:
        block.zero_grad(set_to_none=True)
        hidden.grad = None
        combine_weights.grad = None

    full_block = _measure_boundary(
        lambda: block._forward_grouped(hidden, selected_experts, combine_weights),
        clear_full,
        cotangent,
    )

    clear_full()
    torch.cuda.empty_cache()
    flat_experts = selected_experts.reshape(-1)
    order = torch.argsort(flat_experts, stable=True)
    token_indices = torch.arange(TOKENS, device="cuda").unsqueeze(1).expand(-1, TOP_K).reshape(-1)
    sorted_token_indices = token_indices.index_select(0, order)
    routed_input = hidden.detach().index_select(0, sorted_token_indices)
    with torch.no_grad():
        routed_output = block.experts(routed_input, counts)
        sorted_weights = combine_weights.detach().reshape(-1).index_select(0, order).to(torch.bfloat16)
        weighted_summands = routed_output * sorted_weights.unsqueeze(-1)
    fixture_hashes["weighted_bf16_summands"] = _tensor_digest(weighted_summands)
    routed_output = routed_output.detach().requires_grad_(True)

    def clear_combine() -> None:
        routed_output.grad = None
        combine_weights.grad = None

    combine = _measure_boundary(
        lambda: _combine_function(
            arm,
            routed_output,
            combine_weights,
            order,
            sorted_token_indices,
        ),
        clear_combine,
        cotangent,
    )

    properties = torch.cuda.get_device_properties(0)
    return {
        "arm": arm,
        "revision": PARENT_REVISION if arm == "parent" else CANDIDATE_REVISION,
        "module_sha256": expected_source,
        "physical_gpu": physical_gpu,
        "repetition": repetition,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "device": properties.name,
        "device_total_memory_bytes": int(properties.total_memory),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "fixture_hashes": fixture_hashes,
        "routes": {
            "minimum_rows_per_expert": int(counts.min().item()),
            "maximum_rows_per_expert": int(counts.max().item()),
            "total_rows": int(counts.sum().item()),
        },
        "full_block": full_block,
        "combine_boundary": combine,
    }


def _meminfo() -> dict[str, int]:
    values: dict[str, int] = {}
    for line in Path("/proc/meminfo").read_text().splitlines():
        key, raw = line.split(":", 1)
        values[key] = int(raw.strip().split()[0]) * 1024
    return values


def _read_int(path: Path) -> int | None:
    try:
        raw = path.read_text().strip()
    except FileNotFoundError:
        return None
    if raw == "max":
        return None
    return int(raw)


def _memory_sample() -> dict[str, Any]:
    meminfo = _meminfo()
    return {
        "time_unix": time.time(),
        "cgroup_current_bytes": _read_int(Path("/sys/fs/cgroup/memory.current")),
        "cgroup_limit_bytes": _read_int(Path("/sys/fs/cgroup/memory.max")),
        "host_mem_available_bytes": meminfo.get("MemAvailable", 0),
        "host_swap_total_bytes": meminfo.get("SwapTotal", 0),
        "host_swap_free_bytes": meminfo.get("SwapFree", 0),
    }


def _memory_stop_reason(sample: dict[str, Any]) -> str | None:
    current = sample["cgroup_current_bytes"]
    if current is not None and current >= CGROUP_STOP_BYTES:
        return f"cgroup memory reached {current} bytes"
    if sample["host_mem_available_bytes"] < HOST_AVAILABLE_STOP_BYTES:
        return f"host MemAvailable fell to {sample['host_mem_available_bytes']} bytes"
    swap_total = sample["host_swap_total_bytes"]
    if swap_total and sample["host_swap_free_bytes"] / swap_total < MIN_SWAP_FREE_FRACTION:
        return (
            f"host swap free fell below {MIN_SWAP_FREE_FRACTION:.0%}: "
            f"{sample['host_swap_free_bytes']} / {swap_total} bytes"
        )
    return None


def _run_wave(
    commands: list[tuple[list[str], dict[str, str], Path, Path]],
    memory_samples: list[dict[str, Any]],
) -> None:
    processes = []
    handles = []
    for command, environment, stdout_path, stderr_path in commands:
        stdout_handle = stdout_path.open("w")
        stderr_handle = stderr_path.open("w")
        handles.extend((stdout_handle, stderr_handle))
        processes.append(
            subprocess.Popen(
                command,
                env=environment,
                stdout=stdout_handle,
                stderr=stderr_handle,
            )
        )
    try:
        while any(process.poll() is None for process in processes):
            sample = _memory_sample()
            memory_samples.append(sample)
            reason = _memory_stop_reason(sample)
            if reason is not None:
                for process in processes:
                    if process.poll() is None:
                        process.terminate()
                for process in processes:
                    try:
                        process.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        process.kill()
                raise RuntimeError(f"memory stop rule triggered: {reason}")
            time.sleep(1)
        return_codes = [process.wait() for process in processes]
        if any(code != 0 for code in return_codes):
            raise RuntimeError(f"worker wave failed with return codes {return_codes}")
    finally:
        for handle in handles:
            handle.close()


def _worker_environment(source_root: Path, physical_gpu: int) -> dict[str, str]:
    environment = os.environ.copy()
    current_pythonpath = environment.get("PYTHONPATH", "")
    environment["PYTHONPATH"] = f"{source_root}:{current_pythonpath}" if current_pythonpath else str(source_root)
    environment["CUDA_VISIBLE_DEVICES"] = str(physical_gpu)
    environment["PYTHONHASHSEED"] = "0"
    return environment


def _correctness_gate(parent: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
    parent_discriminates = all(
        not parent["paths"][path]["actual_vs_fp32_slotwise"]["output"]["exact"]
        for path in ("eager", "grouped")
    )
    candidate_contract = all(
        candidate["paths"][path]["actual_vs_fp32_slotwise"]["output"]["exact"]
        and all(
            gradient["allclose"]
            for gradient in candidate["paths"][path]["actual_vs_fp32_slotwise"]["gradients"].values()
        )
        for path in ("eager", "grouped")
    )
    candidate_fp64 = all(
        candidate["paths"][path]["actual_vs_fp64_fixed_order"]["output"]["exact"]
        and all(
            gradient["allclose"]
            for gradient in candidate["paths"][path]["actual_vs_fp64_fixed_order"]["gradients"].values()
        )
        for path in ("eager", "grouped")
    )
    candidate_paths = candidate["eager_vs_grouped"]["output"]["exact"] and all(
        gradient["allclose"] for gradient in candidate["eager_vs_grouped"]["gradients"].values()
    )
    candidate_more_accurate = all(
        candidate["paths"][path]["fp64_pre_cast_accuracy"]["actual"]["max_abs"]
        < parent["paths"][path]["fp64_pre_cast_accuracy"]["actual"]["max_abs"]
        for path in ("eager", "grouped")
    )
    checks = {
        "parent_fixture_discriminates": parent_discriminates,
        "candidate_matches_fp32_contract": candidate_contract,
        "candidate_matches_fp64_post_cast": candidate_fp64,
        "candidate_eager_grouped_parity": candidate_paths,
        "candidate_reduces_fp64_error": candidate_more_accurate,
    }
    return {"checks": checks, "pass": all(checks.values())}


def _orchestrate(args: argparse.Namespace) -> None:
    runtime_dir = Path(args.runtime_dir).resolve()
    output_path = Path(args.output).resolve()
    protocol_path = Path(args.protocol).resolve()
    runtime_dir.mkdir(parents=True, exist_ok=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    source_roots = _prepare_sources(runtime_dir)

    protocol = json.loads(protocol_path.read_text())
    driver_sha256 = _sha256_file(Path(__file__).resolve())
    expected_pins = {
        "parent_revision": PARENT_REVISION,
        "candidate_revision": CANDIDATE_REVISION,
        "parent_module_sha256": PARENT_MODULE_SHA256,
        "candidate_module_sha256": CANDIDATE_MODULE_SHA256,
        "driver_sha256": driver_sha256,
    }
    if protocol["pins"] != expected_pins:
        raise RuntimeError(f"frozen protocol pins do not match the executed driver: {protocol['pins']}")

    package_inventory = sorted(
        line
        for line in subprocess.check_output(
            ["uv", "pip", "freeze", "--python", sys.executable],
            text=True,
        ).splitlines()
        if line
    )
    package_inventory_sha256 = _sha256_bytes(("\n".join(package_inventory) + "\n").encode())
    frozen_runtime = protocol["runtime"]
    runtime_checks = {
        "cluster": args.cluster,
        "iris_job_id": args.iris_job_id,
        "pod_name": args.pod_name,
        "container_image_id": args.container_image_id,
        "workspace_base_head": args.workspace_head,
        "python_package_inventory_sha256": package_inventory_sha256,
    }
    for field, actual in runtime_checks.items():
        if frozen_runtime[field] != actual:
            raise RuntimeError(
                f"frozen runtime mismatch for {field}: expected {frozen_runtime[field]!r}, got {actual!r}"
            )

    gpu_lines = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=index,uuid,name,memory.total,pci.bus_id",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    ).strip().splitlines()
    if len(gpu_lines) != GPU_COUNT:
        raise RuntimeError(f"measurement requires exactly {GPU_COUNT} visible GPUs, found {len(gpu_lines)}")
    if any(GPU_NAME not in line for line in gpu_lines):
        raise RuntimeError(f"measurement requires eight {GPU_NAME} devices: {gpu_lines}")
    driver_versions = set(
        subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=driver_version",
                "--format=csv,noheader,nounits",
            ],
            text=True,
        ).strip().splitlines()
    )
    if len(driver_versions) != 1:
        raise RuntimeError(f"expected one NVIDIA driver version, got {sorted(driver_versions)}")

    memory_samples: list[dict[str, Any]] = []
    correctness: dict[str, Any] = {}
    for arm in ("parent", "candidate"):
        result_path = runtime_dir / f"correctness-{arm}.json"
        command = [
            sys.executable,
            str(Path(__file__).resolve()),
            "worker",
            "--kind",
            "correctness",
            "--arm",
            arm,
            "--output",
            str(result_path),
        ]
        _run_wave(
            [
                (
                    command,
                    _worker_environment(source_roots[arm], 0),
                    runtime_dir / f"correctness-{arm}.stdout",
                    runtime_dir / f"correctness-{arm}.stderr",
                )
            ],
            memory_samples,
        )
        correctness[arm] = json.loads(result_path.read_text())
    gate = _correctness_gate(correctness["parent"], correctness["candidate"])
    if not gate["pass"]:
        partial = {
            "schema_version": 1,
            "status": "correctness_gate_failed",
            "correctness": correctness,
            "correctness_gate": gate,
            "memory_monitor": memory_samples,
        }
        output_path.write_text(json.dumps(partial, indent=2, sort_keys=True) + "\n")
        raise RuntimeError(f"correctness gate failed: {gate}")

    schedules = {
        str(gpu): (["parent", "candidate", "parent", "candidate"] if gpu % 2 == 0 else ["candidate", "parent", "candidate", "parent"])
        for gpu in range(GPU_COUNT)
    }
    performance: list[dict[str, Any]] = []
    arm_repetitions = {(gpu, arm): 0 for gpu in range(GPU_COUNT) for arm in ("parent", "candidate")}
    for wave in range(4):
        commands = []
        result_paths = []
        for gpu in range(GPU_COUNT):
            arm = schedules[str(gpu)][wave]
            repetition = arm_repetitions[(gpu, arm)]
            arm_repetitions[(gpu, arm)] += 1
            result_path = runtime_dir / f"performance-gpu{gpu}-{arm}-r{repetition}.json"
            result_paths.append(result_path)
            command = [
                sys.executable,
                str(Path(__file__).resolve()),
                "worker",
                "--kind",
                "performance",
                "--arm",
                arm,
                "--physical-gpu",
                str(gpu),
                "--repetition",
                str(repetition),
                "--output",
                str(result_path),
            ]
            commands.append(
                (
                    command,
                    _worker_environment(source_roots[arm], gpu),
                    runtime_dir / f"performance-wave{wave}-gpu{gpu}.stdout",
                    runtime_dir / f"performance-wave{wave}-gpu{gpu}.stderr",
                )
            )
        _run_wave(commands, memory_samples)
        performance.extend(json.loads(path.read_text()) for path in result_paths)

    raw = {
        "schema_version": 1,
        "status": "complete",
        "created_at_unix": time.time(),
        "pins": {
            "parent_revision": PARENT_REVISION,
            "candidate_revision": CANDIDATE_REVISION,
            "parent_module_sha256": PARENT_MODULE_SHA256,
            "candidate_module_sha256": CANDIDATE_MODULE_SHA256,
            "driver_sha256": driver_sha256,
            "protocol_sha256": _sha256_file(protocol_path),
        },
        "runtime": {
            "cluster": args.cluster,
            "iris_job_id": args.iris_job_id,
            "pod_name": args.pod_name,
            "container_image_id": args.container_image_id,
            "workspace_head": args.workspace_head,
            "gpu_inventory": gpu_lines,
            "nvidia_driver_version": next(iter(driver_versions)),
            "python_executable": sys.executable,
            "python_version": sys.version,
            "python_package_inventory": package_inventory,
            "python_package_inventory_sha256": package_inventory_sha256,
            "resource_request": frozen_runtime["resource_request"],
        },
        "shape": {
            "tokens": TOKENS,
            "hidden_size": HIDDEN_SIZE,
            "intermediate_size": INTERMEDIATE_SIZE,
            "num_experts": NUM_EXPERTS,
            "top_k": TOP_K,
        },
        "schedule": {
            "warmup_iterations_per_process": WARMUP_ITERATIONS,
            "measured_iterations_per_process": MEASURED_ITERATIONS,
            "arms_by_gpu": schedules,
            "verdict": "estimate_only_no_materiality_threshold",
        },
        "correctness": correctness,
        "correctness_gate": gate,
        "performance": performance,
        "memory_stop_rules": {
            "cgroup_stop_bytes": CGROUP_STOP_BYTES,
            "host_available_stop_bytes": HOST_AVAILABLE_STOP_BYTES,
            "minimum_swap_free_fraction": MIN_SWAP_FREE_FRACTION,
        },
        "memory_monitor": memory_samples,
    }
    output_path.write_text(json.dumps(raw, indent=2, sort_keys=True) + "\n")


def _worker(args: argparse.Namespace) -> None:
    if args.kind == "correctness":
        result = _run_correctness(args.arm)
    else:
        result = _run_performance(args.arm, args.physical_gpu, args.repetition)
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")


def _self_test() -> None:
    parent_path = PACKAGE_ROOT / MODULE_RELATIVE_PATH
    assert _sha256_file(parent_path) == PARENT_MODULE_SHA256
    candidate = _candidate_source(parent_path.read_text())
    assert _sha256_bytes(candidate.encode()) == CANDIDATE_MODULE_SHA256
    assert len({tuple(value) for value in (
        ["parent", "candidate", "parent", "candidate"],
        ["candidate", "parent", "candidate", "parent"],
    )}) == 2
    assert statistics.median([1.0, 2.0, 3.0]) == 2.0
    print("self-test passed")


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    self_test = subparsers.add_parser("self-test")
    self_test.set_defaults(function=lambda _: _self_test())

    worker = subparsers.add_parser("worker")
    worker.add_argument("--kind", choices=("correctness", "performance"), required=True)
    worker.add_argument("--arm", choices=("parent", "candidate"), required=True)
    worker.add_argument("--physical-gpu", type=int, default=0)
    worker.add_argument("--repetition", type=int, default=0)
    worker.add_argument("--output", required=True)
    worker.set_defaults(function=_worker)

    orchestrate = subparsers.add_parser("orchestrate")
    orchestrate.add_argument("--runtime-dir", required=True)
    orchestrate.add_argument("--output", required=True)
    orchestrate.add_argument("--protocol", required=True)
    orchestrate.add_argument("--cluster", required=True)
    orchestrate.add_argument("--iris-job-id", required=True)
    orchestrate.add_argument("--pod-name", required=True)
    orchestrate.add_argument("--container-image-id", required=True)
    orchestrate.add_argument("--workspace-head", required=True)
    orchestrate.set_defaults(function=_orchestrate)

    args = parser.parse_args()
    args.function(args)


if __name__ == "__main__":
    main()
