import asyncio
import gzip
import hashlib
import inspect
import io
import math
import os
import shutil
import tempfile
import time
from dataclasses import dataclass

import ray
import torch
import torch.distributed
import torch.nn.functional as F
from loguru import logger
from torch.distributed.fsdp.api import ShardedStateDictConfig, StateDictType
from torch.distributed.fsdp.fully_sharded_data_parallel import FullyShardedDataParallel as FSDP
from transformers import AutoConfig

from skyrl_train.utils.trainer_utils import get_rope_scaling_config, get_rope_theta_config

try:
    # for torch 2.5+
    from torch.distributed.tensor import DTensor
except ImportError:
    from torch.distributed._tensor import DTensor

from skyrl_train.distributed import collective_phase_diagnostics as _phase_diagnostics
from skyrl_train.distributed.fsdp_strategy import FSDPStrategy
from skyrl_train.distributed.fsdp_utils import fsdp_version, get_init_weight_context_manager
from skyrl_train.model_wrapper import HFModelWrapper, get_llm_for_sequence_regression
from skyrl_train.models.grug_moe import (
    GRUG_MOE_MODEL_TYPE,
    GrugMoeRouter,
    GrugMoeSparseMoeBlock,
    validate_grug_expert_parallel_options,
)
from skyrl_train.training_batch import TrainingInputBatch, TrainingOutputBatch
from skyrl_train.utils import get_physical_gpu_id, str_to_torch_dtype, torch_dtype_to_str
from skyrl_train.weight_sync import WeightChunk, WeightExtractor
from skyrl_train.weight_sync.weight_extractor import (
    prepare_weight_sync_tensor,
    validate_weight_sync_mode,
    weight_sync_dtype,
)
from skyrl_train.weight_sync.weight_extractor_utils import yield_module_grouped_chunks
from skyrl_train.workers.worker import (
    CriticWorkerBase,
    PolicyWorkerBase,
    RefWorkerBase,
)
from skyrl_train.workers.worker_utils import BatchIterator


@dataclass(frozen=True)
class GrugValidationSnapshot:
    """Test-only snapshot of one policy rank's loaded Grug state."""

    rank: int
    attention_backend: str
    weights: dict[str, torch.Tensor]


class FSDPWeightExtractor(WeightExtractor):
    """Extracts weights from FSDP-sharded models.

    Args:
        model: FSDP model to extract weights from
        group_by_module: If True, group parameters by module (e.g., for FlashRL QKV fusion).
            Grug always uses separate chunks to preserve its FP32 router-bias buffers.
        batch_size_threshold_gb: If > 0, batch complete modules together until threshold is reached
        moe_grouped_gemm: If True, the model was grouped-swapped (Stage 3b) so its MoE
            blocks are ``GroupedMoEShim`` instances holding grouped ``experts.w1/w2/w3``
            tensors. The extracted state dict is then name/shape-remapped back to the
            per-expert HF layout the inference engine expects (Stage 4b). Default False
            keeps the path byte-identical to the non-grouped (a3-production) extractor.
        fuse_weights: Whether the inference engine expects fused FP8 weight transfer.
            Unsupported model types fail during extractor initialization.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        group_by_module: bool = False,
        batch_size_threshold_gb: float = 0.0,
        moe_grouped_gemm: bool = False,
        fuse_weights: bool = False,
    ):
        self.model = model
        self.batch_size_threshold_gb = batch_size_threshold_gb
        self.moe_grouped_gemm = moe_grouped_gemm
        # Per-arch inference-engine (vLLM) weight-NAME translation. Most grouped-MoE
        # arches (qwen3_moe/qwen3_next/olmoe) emit broadcast names that already match
        # vLLM's stock params_dict, so this stays the identity for them. Mixtral is the
        # exception (transformers-5.x ``mlp.*`` vs vLLM's stock ``block_sparse_moe.*``);
        # ``translate_moe_name_to_vllm`` renames ONLY Mixtral keys (see moe_weight_remap).
        _cfg = getattr(model, "config", None)
        self._model_type = getattr(_cfg, "model_type", "") or "" if _cfg is not None else ""
        validate_weight_sync_mode(self._model_type, fuse_weights=fuse_weights)
        self.group_by_module = group_by_module and self._model_type != GRUG_MOE_MODEL_TYPE
        # Qwen3.5/3.6 VLM-shell weight-sync (tmax Stage 2): the RL policy is the
        # unwrapped TEXT tower (``Qwen3_5MoeForCausalLM``, names ``model.*``) but the
        # vLLM rollout engine instantiates the multimodal SHELL
        # (``Qwen3_5MoeForConditionalGeneration``), whose ``load_weights`` expects the
        # text decoder under the HF namespace ``model.language_model.*``. When the
        # policy config is the hybrid text tower, the broadcast names must be
        # remapped ``model.X`` -> ``model.language_model.X`` (see
        # ``map_text_name_to_vlm_engine``). Identity for every other arch.
        from skyrl_train.models.qwen3_5_vlm import is_qwen3_5_text_tower

        self._is_qwen3_5_text_tower = is_qwen3_5_text_tower(_cfg)

    def _target_dtype(self, name: str, default: torch.dtype) -> torch.dtype:
        return weight_sync_dtype(self._model_type, name, default)

    def _translate_name(self, name: str) -> str:
        """Apply the per-arch inference-engine name translation (identity for all
        arches except Mixtral, and the Qwen3.5/3.6 VLM-shell namespace). Scoped via
        ``self._model_type`` / ``self._is_qwen3_5_text_tower``."""
        from skyrl_train.models.layers.moe_weight_remap import translate_moe_name_to_vllm

        name = translate_moe_name_to_vllm(name, self._model_type)
        if self._is_qwen3_5_text_tower:
            from skyrl_train.models.qwen3_5_vlm import map_text_name_to_vlm_engine

            name = map_text_name_to_vlm_engine(name)
        return name

    def extract_weights(self, dtype: torch.dtype):
        """Extract weights from FSDP model.

        Args:
            dtype: Target dtype for inference

        Yields:
            WeightChunk objects (one per parameter, or grouped by module)
        """
        # Configure state_dict type for FSDP v1
        if fsdp_version(self.model) == 1:
            FSDP.set_state_dict_type(
                self.model,
                state_dict_type=StateDictType.SHARDED_STATE_DICT,
                state_dict_config=ShardedStateDictConfig(),
            )

        # Get state dict (handles FSDP sharding)
        params = self.model.state_dict()

        # Stage 7 (80B) — STREAMED grouped gather. For grouped-swapped models the old
        # path eagerly `full_tensor()`-gathered EVERY layer's grouped `experts.w1/w2/w3`
        # (the 512-expert stacks) into a single remapped dict before the broadcast loop,
        # materializing the whole unsharded MoE on ONE GPU → OOM at 80B (job 602650,
        # 93.78/95 GiB). When grouped + the (disaggregated/NCCL-broadcast) per-tensor
        # transport, stream instead: gather → remap → yield → FREE one MoE layer (and one
        # non-MoE param) at a time, so peak GPU memory is a single layer's expert stack,
        # not all 48. Byte-identical to the eager remap (same converter, same tensors);
        # only the materialization order/lifetime changes.
        # Gated: non-grouped models (a3: moe_grouped_gemm=False) skip this entirely and
        # take the unchanged simple/grouped-by-module paths below — code-path identical.
        if self.moe_grouped_gemm and not self.group_by_module:
            yield from self._extract_weights_streamed(params, dtype)
            return

        # Stage 4b: if the trainer was grouped-swapped (Stage 3b) AND on the CUDA-IPC /
        # FlashRL module-grouping path (colocated NCCL IPC — not the 80B disaggregated
        # broadcast), fall back to the eager whole-model remap. This combination is not
        # on the 80B path; left unchanged.
        if self.moe_grouped_gemm:
            params = self._remap_grouped_state_dict(params)

        if not self.group_by_module:
            # Simple path: yield one chunk per parameter
            for name, param in params.items():
                target_dtype = self._target_dtype(name, dtype)
                tensor = self._gather_tensor(param)
                tensor = prepare_weight_sync_tensor(self._model_type, name, tensor, target_dtype)
                tensor = tensor.detach().contiguous()
                name = self._translate_name(name)
                yield WeightChunk(
                    names=[name],
                    dtypes=[torch_dtype_to_str(target_dtype)],
                    shapes=[list(tensor.shape)],
                    tensors=[tensor],
                )
        else:
            for chunk in yield_module_grouped_chunks(
                params=params,
                dtype=dtype,
                gather_tensor_fn=self._gather_tensor,
                get_shape_fn=lambda name, param, tensor: list(tensor.shape),
                batch_size_threshold_gb=self.batch_size_threshold_gb,
            ):
                yield chunk

    def _gather_tensor(self, param: torch.Tensor) -> torch.Tensor:
        """Gather sharded tensor into full tensor.

        For EP+FSDP-composed grouped-expert params (placement
        ``(_StridedShard(fsdp), Shard(ep))``) ``full_tensor()`` reassembles the
        expert ROWS in the WRONG global order on torch 2.11 (the
        ``_StridedShard.is_shard()==False`` / non-ascending-all_gather quirk that
        silently corrupted the r2–r7 MoE weight sync). ``gather_dtensor_strided_safe``
        gathers via each placement's own ``_split_tensor`` instead, so the global
        expert order is correct and version-independent. It is byte-identical to
        ``full_tensor()`` for every non-strided (a3 / non-EP / plain-Shard) param.
        """
        from skyrl_train.distributed.fsdp_utils import gather_dtensor_strided_safe

        device = torch.cuda.current_device()
        if not isinstance(param, DTensor):
            return param
        out = gather_dtensor_strided_safe(param.to(device, non_blocking=True))
        return out

    def _extract_weights_streamed(self, params, dtype: torch.dtype):
        """Streamed grouped-MoE weight extraction (Stage 7 / 80B OOM fix).

        Yields one ``WeightChunk`` per HF parameter, gathering + remapping LAZILY so
        peak GPU memory is bounded by a single MoE layer's grouped expert stack (3 ×
        ``[num_experts, moe_dim, dim]``) rather than the whole unsharded 80B model. The
        emitted tensors are byte-identical to the eager ``_remap_grouped_state_dict``
        path — same ``full_tensor()`` gather, same ``convert_tt_layer_to_hf`` per-expert
        split, same dtype/contiguity — only their lifetime is per-layer.

        IMPORTANT (collective correctness): ``full_tensor()`` is a collective over the
        FSDP/EP mesh, so EVERY rank must drive this generator and reach each gather in
        the SAME order. Iteration order is the deterministic ``state_dict()`` ordering on
        all ranks, so the gather sequence is identical across ranks (matches the eager
        path, which also gathered in dict order).
        """
        from skyrl_train.models.layers.moe_weight_remap import convert_tt_layer_to_hf

        # Post-prefix-strip suffixes of the grouped-block tensors the converter consumes.
        grouped_suffixes = (
            ".mlp.experts.w1",
            ".mlp.experts.w2",
            ".mlp.experts.w3",
            ".mlp.router.gate.weight",
            ".mlp.shared_expert.w1.weight",
            ".mlp.shared_expert.w2.weight",
            ".mlp.shared_expert.w3.weight",
        )

        def _layer_of(name: str):
            # ``model.layers.{i}.mlp.experts.w1`` -> i ; None for non-layer keys.
            parts = name.split(".")
            if len(parts) >= 3 and parts[0] == "model" and parts[1] == "layers":
                try:
                    return int(parts[2])
                except ValueError:
                    return None
            return None

        # First pass: strip prefixes (cheap, no gather) and partition into per-layer
        # grouped-MoE tensors vs. everything else, preserving state_dict() order.
        # ``layer_groups[i]`` = list of (stripped_name, dtensor_param) for that layer's
        # grouped MoE keys; ``passthrough`` = ordered (stripped_name, param) for the rest.
        from collections import OrderedDict

        layer_groups: "OrderedDict[int, list]" = OrderedDict()
        # Ordered plan of work items: ("moe", layer_idx) flushes that layer once, "param"
        # yields a single non-MoE tensor. Emitted in first-encounter order so the gather
        # sequence is deterministic and identical on every rank.
        plan = []
        seen_moe_layer = set()
        passthrough = OrderedDict()

        for name, param in params.items():
            new_name = self._strip_grouped_prefix(name)
            if new_name.endswith(grouped_suffixes):
                li = _layer_of(new_name)
                layer_groups.setdefault(li, []).append((new_name, param))
                if li not in seen_moe_layer:
                    seen_moe_layer.add(li)
                    plan.append(("moe", li))
            else:
                passthrough[new_name] = param
                plan.append(("param", new_name))

        for kind, key in plan:
            # Re-sync all policy ranks on the default WORLD PG (SKYRL_WORKER_NCCL_TIMEOUT_IN_S)
            # before each plan item's mesh_fsdp gather. init_device_mesh's mesh_fsdp submesh PG
            # inherits torch's default 600s timeout; the GIL-heavy convert_tt_layer_to_hf (prev
            # item) + the plan-build can skew one rank's arrival at the next `_all_gather_base`
            # past 600s -> #6936 SIGABRT (the r4h gs1 death). `plan` is built in deterministic
            # state_dict() order identical on every rank (see the docstring above), so this
            # barrier is hit the same count + order on all ranks = deadlock-free; it lifts the
            # effective per-gather sync window from the 600s submesh default to the WORLD
            # timeout. Mirrors Fix-1(a) (worker.py ppo_train entry: cuda.synchronize + barrier).
            if torch.distributed.is_initialized() and torch.distributed.get_world_size() > 1:
                torch.cuda.synchronize()
                torch.distributed.barrier()
            if kind == "param":
                param = passthrough[key]
                tensor = self._gather_tensor(param).to(dtype).detach().contiguous()
                key = self._translate_name(key)
                yield WeightChunk(
                    names=[key],
                    dtypes=[str(dtype)],
                    shapes=[list(tensor.shape)],
                    tensors=[tensor],
                )
                del tensor
            else:
                # Gather ONLY this layer's grouped MoE tensors, remap per-expert, yield
                # each, then free the layer's grouped stack before moving on.
                layer_sd = {}
                for sname, sparam in layer_groups[key]:
                    layer_sd[sname] = self._gather_tensor(sparam).detach().contiguous()
                # In-place grouped -> per-expert HF split for THIS layer only. The
                # per-expert entries are views into w1/w2/w3 (no extra alloc); we
                # .contiguous() each on yield so the parent stack can free after the loop.
                convert_tt_layer_to_hf(layer_sd, key)
                for ename, etensor in layer_sd.items():
                    out = etensor.to(dtype).detach().contiguous()
                    ename = self._translate_name(ename)
                    yield WeightChunk(
                        names=[ename],
                        dtypes=[str(dtype)],
                        shapes=[list(out.shape)],
                        tensors=[out],
                    )
                    del out
                # Drop all references to this layer's gathered tensors + per-expert views
                # so the (large) grouped expert stack is freed before the next layer.
                del layer_sd
                torch.cuda.empty_cache()

    # MoE grouped-block (GroupedMoEShim.moe) segment that sits between the HF
    # `...mlp.` prefix and the grouped `experts.w1/...`/`router.gate` keys the
    # `convert_tt_to_hf_moe` converter matches on. FSDP2 `fully_shard` does not add a
    # `_fsdp_wrapped_module` segment to state_dict keys, but FSDP1 (and nested wraps)
    # can — strip it defensively so the remap is layout-agnostic.
    _SHIM_SEG = ".mlp.moe."
    _FSDP_SEG = "._fsdp_wrapped_module."

    @staticmethod
    def _strip_grouped_prefix(name: str) -> str:
        """Normalize a live grouped-swapped key to the converter's expected form.

        ``...layers.{i}.mlp.moe.experts.w1`` -> ``...layers.{i}.mlp.experts.w1``
        ``...layers.{i}.mlp.moe.router.gate.weight`` -> ``...mlp.router.gate.weight``
        Also drops any FSDP ``_fsdp_wrapped_module`` segments.
        """
        name = name.replace(FSDPWeightExtractor._FSDP_SEG, ".")
        name = name.replace(FSDPWeightExtractor._SHIM_SEG, ".mlp.")
        return name

    def _remap_grouped_state_dict(self, params):
        """Strip the GroupedMoEShim/FSDP prefix + run ``convert_tt_to_hf_moe`` in place.

        Only the grouped MoE tensors (``experts.w1/w2/w3``, ``router.gate``, the shared
        expert ``w1/w2/w3``) need to be materialized to full tensors before the converter
        slices them per-expert (``w1[j]``) — a DTensor ``Shard(0)`` on the expert dim would
        otherwise give a partial slice. Non-MoE params are left as-is (gathered lazily in the
        existing broadcast loop). After the converter runs, expert keys become the per-expert
        HF names the inference engine already loads.
        """
        from skyrl_train.models.layers.moe_weight_remap import convert_tt_to_hf_moe

        # Grouped-block tensors the converter consumes (post-prefix-strip suffixes).
        grouped_suffixes = (
            ".mlp.experts.w1",
            ".mlp.experts.w2",
            ".mlp.experts.w3",
            ".mlp.router.gate.weight",
            ".mlp.shared_expert.w1.weight",
            ".mlp.shared_expert.w2.weight",
            ".mlp.shared_expert.w3.weight",
        )

        remapped = {}
        for name, param in params.items():
            new_name = self._strip_grouped_prefix(name)
            if new_name.endswith(grouped_suffixes):
                # Materialize before the converter slices per-expert.
                remapped[new_name] = self._gather_tensor(param).detach().contiguous()
            else:
                remapped[new_name] = param

        # In-place grouped -> per-expert HF remap (splits w1/w2/w3 into experts.{j}.*).
        convert_tt_to_hf_moe(remapped)
        return remapped


class FSDPPolicyWorkerBase(PolicyWorkerBase):
    @staticmethod
    def _grug_benchmark_file_sha256(path: str) -> str:
        digest = hashlib.sha256()
        with open(path, "rb") as source:
            while chunk := source.read(16 * 1024 * 1024):
                digest.update(chunk)
        return digest.hexdigest()

    @staticmethod
    def _grug_benchmark_local_tensor(tensor):
        """Return the rank-local storage for a plain tensor or DTensor."""

        return tensor.to_local() if isinstance(tensor, DTensor) else tensor

    @classmethod
    def _grug_benchmark_state_hash(cls, state):
        """Hash exact local model state without gathering any FSDP shards."""

        digest = hashlib.sha256()
        for name, tensor in sorted(state.items()):
            local = cls._grug_benchmark_local_tensor(tensor).detach().to("cpu").contiguous()
            digest.update(name.encode("utf-8"))
            digest.update(b"\0")
            digest.update(str(local.dtype).encode("ascii"))
            digest.update(b"\0")
            digest.update(str(tuple(local.shape)).encode("ascii"))
            digest.update(b"\0")
            digest.update(local.view(torch.uint8).numpy().tobytes())
        return digest.hexdigest()

    def _grug_benchmark_phase(self, action: str, name: str):
        """Record benchmark-only CUDA events without synchronizing the loop."""

        events = getattr(self, "_grug_benchmark_phase_events", None)
        if events is None:
            return
        if action == "begin":
            if getattr(self, "_grug_benchmark_open_phase", None) is not None:
                raise RuntimeError("nested Grug benchmark phases are not supported")
            event = torch.cuda.Event(enable_timing=True)
            event.record()
            annotation = None
            if getattr(self, "_grug_benchmark_profile_enabled", False):
                annotation = torch.profiler.record_function(f"grug::{name}")
                annotation.__enter__()
            self._grug_benchmark_open_phase = (name, event, annotation)
            self._grug_benchmark_parent_phase = name
            return
        if action != "end":
            raise ValueError(f"unknown Grug benchmark phase action: {action}")
        opened = getattr(self, "_grug_benchmark_open_phase", None)
        if opened is None or opened[0] != name:
            raise RuntimeError(f"Grug benchmark phase mismatch: open={opened}, closing={name}")
        event = torch.cuda.Event(enable_timing=True)
        event.record()
        events.setdefault(name, []).append((opened[1], event))
        if opened[2] is not None:
            opened[2].__exit__(None, None, None)
        self._grug_benchmark_open_phase = None
        self._grug_benchmark_parent_phase = None

    def _grug_benchmark_install_expert_hooks(self):
        """Time routed blocks and retain exact per-layer route loads.

        The sparse block is a common module boundary for both the eager and
        native-grouped paths. Forward hooks run only inside the outer
        model-forward phase. During
        gradient-checkpoint recompute the outer phase is ``matched_backward``;
        those forward hooks are deliberately ignored because the enclosing
        module-backward hook already includes recompute. This makes the two
        reported categories nonoverlapping.
        """

        modules = [module for module in self.model.modules() if isinstance(module, GrugMoeSparseMoeBlock)]
        if not modules:
            raise RuntimeError("routed-block attribution found no GrugMoeSparseMoeBlock modules")
        self._grug_benchmark_expert_events = {"forward": [], "backward": []}
        self._grug_benchmark_open_expert_span = None
        self._grug_benchmark_route_loads = [None] * len(modules)
        self._grug_benchmark_route_calls = [0] * len(modules)

        def begin(kind: str):
            parent = getattr(self, "_grug_benchmark_parent_phase", None)
            expected = "matched_model_forward" if kind == "forward" else "matched_backward"
            if parent != expected:
                return
            if self._grug_benchmark_open_expert_span is not None:
                raise RuntimeError(
                    f"nested Grug expert attribution spans: {self._grug_benchmark_open_expert_span}, {kind}"
                )
            event = torch.cuda.Event(enable_timing=True)
            event.record()
            self._grug_benchmark_open_expert_span = (kind, event)

        def end(kind: str):
            parent = getattr(self, "_grug_benchmark_parent_phase", None)
            expected = "matched_model_forward" if kind == "forward" else "matched_backward"
            if parent != expected:
                return
            opened = self._grug_benchmark_open_expert_span
            if opened is None or opened[0] != kind:
                raise RuntimeError(f"Grug expert attribution span mismatch: open={opened}, closing={kind}")
            event = torch.cuda.Event(enable_timing=True)
            event.record()
            self._grug_benchmark_expert_events[kind].append((opened[1], event))
            self._grug_benchmark_open_expert_span = None

        def capture_routes(layer_index: int, router: GrugMoeRouter, output):
            if getattr(self, "_grug_benchmark_parent_phase", None) != "matched_model_forward":
                return
            selected_experts = output.selected_experts
            loads = torch.bincount(selected_experts.reshape(-1), minlength=router.num_experts)
            accumulated = self._grug_benchmark_route_loads[layer_index]
            if accumulated is None:
                accumulated = torch.zeros_like(loads)
                self._grug_benchmark_route_loads[layer_index] = accumulated
            accumulated.add_(loads)
            self._grug_benchmark_route_calls[layer_index] += 1

        handles = []
        for layer_index, module in enumerate(modules):
            handles.extend(
                (
                    module.register_forward_pre_hook(lambda _module, _args: begin("forward")),
                    module.register_forward_hook(lambda _module, _args, _output: end("forward")),
                    module.register_full_backward_pre_hook(lambda _module, _grad_output: begin("backward")),
                    module.register_full_backward_hook(lambda _module, _grad_input, _grad_output: end("backward")),
                    module.router.register_forward_hook(
                        lambda router, _args, output, index=layer_index: capture_routes(index, router, output)
                    ),
                )
            )
        return handles, len(modules)

    def _grug_benchmark_finish_expert_hooks(self, handles, module_count: int):
        for handle in handles:
            handle.remove()
        if self._grug_benchmark_open_expert_span is not None:
            raise RuntimeError(f"unclosed Grug expert attribution span: {self._grug_benchmark_open_expert_span[0]}")
        events = self._grug_benchmark_expert_events
        self._grug_benchmark_expert_events = None
        phase_seconds = {
            name: sum(start.elapsed_time(end) for start, end in spans) / 1000.0 for name, spans in events.items()
        }
        call_counts = {name: len(spans) for name, spans in events.items()}
        if any(count == 0 for count in call_counts.values()):
            raise RuntimeError(f"routed-block attribution missed a required phase: {call_counts}")
        if any(loads is None for loads in self._grug_benchmark_route_loads):
            raise RuntimeError("routed-block attribution missed route loads")
        route_loads = [loads.detach().cpu().tolist() for loads in self._grug_benchmark_route_loads]
        route_calls = list(self._grug_benchmark_route_calls)
        self._grug_benchmark_route_loads = None
        self._grug_benchmark_route_calls = None
        return {
            "module_count": module_count,
            "phase_seconds": phase_seconds,
            "call_counts": call_counts,
            "route_calls_per_layer": route_calls,
            "route_loads_per_layer": route_loads,
            "boundary": (
                "CUDA-stream time inside GrugMoeSparseMoeBlock initial forwards plus full module backward "
                "spans; includes routing, route materialization, dispatch/sort, routed expert kernels, and "
                "combine; backward includes checkpoint recompute; layer-level FSDP communication and the "
                "separate shared expert are excluded"
            ),
        }

    def _grug_benchmark_start_profile(self, enabled: bool):
        """Start one rank-zero profiler for a bounded, non-headline run."""

        self._grug_benchmark_profile_enabled = bool(enabled and torch.distributed.get_rank() == 0)
        if not self._grug_benchmark_profile_enabled:
            return None
        profiler = torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
            record_shapes=False,
            profile_memory=True,
            with_stack=False,
        )
        profiler.__enter__()
        return profiler

    def _grug_benchmark_finish_profile(self, profiler):
        """Stop and return a compressed Chrome trace from rank zero."""

        self._grug_benchmark_profile_enabled = False
        if profiler is None:
            return None
        profiler.__exit__(None, None, None)
        with tempfile.TemporaryDirectory(prefix="grug-torch-profile-") as directory:
            trace_path = os.path.join(directory, "trace.json")
            compressed_path = trace_path + ".gz"
            profiler.export_chrome_trace(trace_path)
            with open(trace_path, "rb") as source, gzip.open(compressed_path, "wb", compresslevel=1) as target:
                shutil.copyfileobj(source, target, length=16 * 1024 * 1024)
            with open(compressed_path, "rb") as source:
                return source.read()

    def grug_benchmark_stage_batch(self, batch: TrainingInputBatch):
        """Stage one rank-local fixed replay shard outside the timed update.

        The benchmark driver sends one already-sharded batch directly to every
        policy actor. The stored batch is the exact shard that ``ppo_train``
        consumes. Per-field hashes prove row identity without moving it again.
        """

        self._grug_benchmark_batch = batch
        field_hashes = {}
        field_shapes = {}
        for name, tensor in sorted(batch.items()):
            if tensor is None:
                field_hashes[name] = None
                field_shapes[name] = None
                continue
            cpu_tensor = tensor.detach().to("cpu").contiguous()
            field_hashes[name] = hashlib.sha256(cpu_tensor.numpy().tobytes()).hexdigest()
            field_shapes[name] = list(cpu_tensor.shape)
        return {
            "rank": int(torch.distributed.get_rank()),
            "batch_size": int(batch.batch_size),
            "field_hashes": field_hashes,
            "field_shapes": field_shapes,
            "allocated_tokens": int(batch["attention_mask"].numel()),
            "nonpad_tokens": int(batch["attention_mask"].sum().item()),
            "loss_tokens": int(batch["loss_mask"].sum().item()),
        }

    def grug_benchmark_reset_peak_memory(self):
        """Reset CUDA peak accounting after model and replay staging."""

        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        return {
            "rank": int(torch.distributed.get_rank()),
            "allocated_bytes": int(torch.cuda.memory_allocated()),
            "reserved_bytes": int(torch.cuda.memory_reserved()),
        }

    @classmethod
    def _grug_benchmark_element_counts(cls, tensor):
        """Count local values without gathering an FSDP or optimizer shard."""

        local = cls._grug_benchmark_local_tensor(tensor).detach()
        if not (torch.is_floating_point(local) or torch.is_complex(local)):
            return local.numel(), 0
        nonfinite = int((~torch.isfinite(local)).sum().item())
        return local.numel(), nonfinite

    def grug_benchmark_validate_finite_state(self):
        """Prove the timed optimizer boundary left finite local state.

        This runs in a separate actor call after timing and therefore cannot
        inflate the measured update wall or peak-memory boundary.
        """

        model_tensors = 0
        model_numel = 0
        nonfinite_model_tensors = 0
        nonfinite_model_elements = 0
        for tensor in self.model.state_dict().values():
            numel, nonfinite = self._grug_benchmark_element_counts(tensor)
            model_tensors += 1
            model_numel += numel
            nonfinite_model_tensors += int(nonfinite > 0)
            nonfinite_model_elements += nonfinite

        optimizer_tensors = 0
        optimizer_numel = 0
        nonfinite_optimizer_tensors = 0
        nonfinite_optimizer_elements = 0
        nonfinite_optimizer_scalars = 0
        for parameter_state in self.optimizer.state.values():
            for value in parameter_state.values():
                if torch.is_tensor(value) or isinstance(value, DTensor):
                    numel, nonfinite = self._grug_benchmark_element_counts(value)
                    optimizer_tensors += 1
                    optimizer_numel += numel
                    nonfinite_optimizer_tensors += int(nonfinite > 0)
                    nonfinite_optimizer_elements += nonfinite
                elif isinstance(value, (int, float)):
                    nonfinite_optimizer_scalars += int(not math.isfinite(value))
                else:
                    raise TypeError(f"cannot validate optimizer state value of type {type(value)}")

        torch.cuda.synchronize()
        return {
            "rank": int(torch.distributed.get_rank()),
            "model_tensors": model_tensors,
            "model_numel": model_numel,
            "nonfinite_model_tensors": nonfinite_model_tensors,
            "nonfinite_model_elements": nonfinite_model_elements,
            "optimizer_tensors": optimizer_tensors,
            "optimizer_numel": optimizer_numel,
            "nonfinite_optimizer_tensors": nonfinite_optimizer_tensors,
            "nonfinite_optimizer_elements": nonfinite_optimizer_elements,
            "nonfinite_optimizer_scalars": nonfinite_optimizer_scalars,
        }

    def grug_benchmark_warmup_and_restore(self):
        """Warm kernels and optimizer allocation, then restore the exact start state.

        The timed update must not include lazy optimizer-state allocation or first-use
        kernels.  A one-row production training step warms both.  We snapshot each
        rank's local FSDP shard, then restore it byte-for-byte, reset every Adam state
        tensor to zero (the mathematical state before AdamW step 1), restore the LR
        scheduler and RNG, and verify the model-state hash.
        """

        if not hasattr(self, "_grug_benchmark_batch"):
            raise RuntimeError("call grug_benchmark_stage_batch before warmup")
        if self.cfg.trainer.strategy != "fsdp2":
            raise RuntimeError("the fixed-replay warmup restoration is implemented only for FSDP2")

        torch.cuda.synchronize()
        if torch.distributed.is_initialized():
            torch.distributed.barrier()

        state = self.model.state_dict()
        state_hash_before = self._grug_benchmark_state_hash(state)
        state_snapshot = {
            name: self._grug_benchmark_local_tensor(tensor).detach().to("cpu", copy=True)
            for name, tensor in state.items()
        }
        scheduler_state = self.scheduler.state_dict()
        cpu_rng_state = torch.get_rng_state()
        cuda_rng_state = torch.cuda.get_rng_state()

        warm_batch = self._grug_benchmark_batch[:1]
        experience = BatchIterator.batch_to_experience(warm_batch)
        causal_lm = self._grug_causal_lm()
        if causal_lm is None:
            raise RuntimeError("the fixed-replay benchmark requires a Grug policy model")
        self._begin_grug_query_bias_window(causal_lm, int(warm_batch["attention_mask"].sum().item()))
        warm_status = self.training_step(experience, warm_batch.metadata.get("global_step", 0), 0, 1)
        if not self.strategy.last_optimizer_step_succeeded:
            raise RuntimeError("the warmup optimizer step was skipped")
        self.strategy.all_reduce(warm_status)
        torch.cuda.synchronize()

        with torch.no_grad():
            restored_state = self.model.state_dict()
            if set(restored_state) != set(state_snapshot):
                raise RuntimeError("model state keys changed during benchmark warmup")
            for name, tensor in restored_state.items():
                local = self._grug_benchmark_local_tensor(tensor)
                local.copy_(state_snapshot[name].to(device=local.device, dtype=local.dtype))

        optimizer_state_tensors = 0
        optimizer_state_numel = 0
        for parameter_state in self.optimizer.state.values():
            for key, value in parameter_state.items():
                if torch.is_tensor(value):
                    value.zero_()
                    optimizer_state_tensors += 1
                    optimizer_state_numel += value.numel()
                elif isinstance(value, (int, float)):
                    parameter_state[key] = type(value)(0)
                else:
                    raise TypeError(f"cannot reset optimizer state {key!r} of type {type(value)}")
        if optimizer_state_tensors == 0:
            raise RuntimeError("warmup did not materialize optimizer state")

        self.scheduler.load_state_dict(scheduler_state)
        self.optimizer.zero_grad(set_to_none=True)
        self.strategy.last_optimizer_step_succeeded = False
        torch.set_rng_state(cpu_rng_state)
        torch.cuda.set_rng_state(cuda_rng_state)
        for name in (
            "_grug_query_bias_accumulator",
            "_grug_query_bias_candidate_count",
            "_ratio_diag_acc",
        ):
            if hasattr(self, name):
                delattr(self, name)

        torch.cuda.synchronize()
        if torch.distributed.is_initialized():
            torch.distributed.barrier()
        torch.cuda.synchronize()
        state_hash_after = self._grug_benchmark_state_hash(self.model.state_dict())
        if state_hash_after != state_hash_before:
            raise RuntimeError(
                "warmup restoration changed the local model state: "
                f"before={state_hash_before}, after={state_hash_after}"
            )
        return {
            "rank": int(torch.distributed.get_rank()),
            "state_hash_before": state_hash_before,
            "state_hash_after": state_hash_after,
            "optimizer_state_tensors": optimizer_state_tensors,
            "optimizer_state_numel": optimizer_state_numel,
            "scheduler_last_epoch": int(self.scheduler.last_epoch),
        }

    def _grug_benchmark_matched_ce(self, batch: TrainingInputBatch):
        """Run the benchmark's common token-weighted next-token CE backward.

        The replay ``loss_mask`` is aligned to SkyRL's action-log-probability
        slice.  Multiplying each rank-local loss sum by world_size/global_tokens
        compensates for FSDP's DP gradient averaging, so the resulting gradient
        is the gradient of one global token mean over the fixed logical batch.
        This intentionally omits PPO diagnostics, entropy, Grug query-bias
        capture, gradient clipping, and the optimizer boundary.
        """

        global_loss_tokens = int(batch.metadata["grug_benchmark_global_loss_tokens"])
        if global_loss_tokens <= 0:
            raise RuntimeError("matched CE needs a positive global loss-token count")

        world_size = torch.distributed.get_world_size()
        local_loss_sum = torch.zeros((), dtype=torch.float64, device=torch.cuda.current_device())
        local_loss_tokens = torch.zeros((), dtype=torch.int64, device=torch.cuda.current_device())
        representative_action_log_probs = []
        microbatches = 0
        self.model.train()
        for experience in BatchIterator(batch, sample_batch_size=1, drop_last=False):
            experience.to_device(torch.cuda.current_device())
            phase = getattr(self, "_grug_benchmark_phase", None)
            if phase is not None:
                phase("begin", "matched_model_forward")
            with torch.autocast(dtype=torch.bfloat16, device_type="cuda"):
                action_log_probs = self.model(
                    experience.sequences,
                    experience.num_actions,
                    attention_mask=experience.attention_mask,
                    temperature=1.0,
                    return_output=False,
                    compute_entropy=False,
                    rollout_routed_experts=None,
                )
                flat_action_log_probs = action_log_probs.detach().reshape(-1)
                representative_indices = torch.tensor(
                    (0, flat_action_log_probs.numel() // 2, flat_action_log_probs.numel() - 1),
                    dtype=torch.int64,
                    device=flat_action_log_probs.device,
                )
                representative_action_log_probs.append(flat_action_log_probs.index_select(0, representative_indices))
                if phase is not None:
                    phase("end", "matched_model_forward")
                    phase("begin", "matched_ce_loss")
                loss_mask = experience.loss_mask.to(torch.float32)
                microbatch_loss_sum = (-action_log_probs.to(torch.float32) * loss_mask).sum()
                loss = microbatch_loss_sum * (world_size / global_loss_tokens)
            if phase is not None:
                phase("end", "matched_ce_loss")
                phase("begin", "matched_backward")
            self.strategy.backward(loss, self.model, self.optimizer)
            if phase is not None:
                phase("end", "matched_backward")
            teardown_replay = getattr(self.model, "teardown_router_replay", None)
            if teardown_replay is not None:
                teardown_replay()
            local_loss_sum += microbatch_loss_sum.detach().to(torch.float64)
            local_loss_tokens += loss_mask.sum(dtype=torch.int64)
            microbatches += 1

        return (
            local_loss_sum,
            int(local_loss_tokens.item()),
            microbatches,
            torch.cat(representative_action_log_probs),
        )

    def grug_benchmark_warmup_matched_ce(self):
        """Warm the common forward/backward path without changing model state."""

        if not hasattr(self, "_grug_benchmark_batch"):
            raise RuntimeError("call grug_benchmark_stage_batch before warmup")
        if self.cfg.trainer.strategy != "fsdp2":
            raise RuntimeError("the fixed-replay warmup is implemented only for FSDP2")

        torch.cuda.synchronize()
        if torch.distributed.is_initialized():
            torch.distributed.barrier()
        state_hash_before = self._grug_benchmark_state_hash(self.model.state_dict())
        cpu_rng_state = torch.get_rng_state()
        cuda_rng_state = torch.cuda.get_rng_state()
        self.optimizer.zero_grad(set_to_none=True)
        warm_batch = self._grug_benchmark_batch[:1]
        _, warm_loss_tokens, microbatches, _ = self._grug_benchmark_matched_ce(warm_batch)
        if warm_loss_tokens <= 0 or microbatches != 1:
            raise RuntimeError("matched CE warmup did not exercise one nonempty microbatch")
        gradient_tensors = sum(parameter.grad is not None for parameter in self.model.parameters())
        if gradient_tensors == 0:
            raise RuntimeError("matched CE warmup produced no gradients")
        self.optimizer.zero_grad(set_to_none=True)
        torch.set_rng_state(cpu_rng_state)
        torch.cuda.set_rng_state(cuda_rng_state)
        torch.cuda.synchronize()
        if torch.distributed.is_initialized():
            torch.distributed.barrier()
        torch.cuda.synchronize()
        state_hash_after = self._grug_benchmark_state_hash(self.model.state_dict())
        if state_hash_after != state_hash_before:
            raise RuntimeError(
                f"matched CE warmup changed the local model state: before={state_hash_before}, after={state_hash_after}"
            )
        return {
            "rank": int(torch.distributed.get_rank()),
            "state_hash_before": state_hash_before,
            "state_hash_after": state_hash_after,
            "gradient_tensors": gradient_tensors,
            "warmup_loss_tokens": warm_loss_tokens,
        }

    def grug_benchmark_identity(self):
        """Return the exact policy path selected by this benchmark rank."""

        config = getattr(self.model.model, "config", None)
        if getattr(config, "model_type", None) != GRUG_MOE_MODEL_TYPE:
            raise ValueError("the fixed-replay benchmark requires a Grug policy model")
        first_layer = self.model.model.model.layers[0]
        grouped_mm_enabled = bool(first_layer.mlp.experts.use_grouped_mm)
        grug_module_path = os.path.realpath(inspect.getsourcefile(GrugMoeSparseMoeBlock))
        worker_path = os.path.realpath(__file__)
        return {
            "rank": int(torch.distributed.get_rank()),
            "model_type": config.model_type,
            "model_revision": getattr(config, "_commit_hash", None),
            "num_hidden_layers": int(config.num_hidden_layers),
            "num_local_experts": int(config.num_local_experts),
            "num_experts_per_tok": int(config.num_experts_per_tok),
            "cuda_device_name": torch.cuda.get_device_name(),
            "cuda_total_memory_bytes": int(torch.cuda.get_device_properties(torch.cuda.current_device()).total_memory),
            "cuda_compute_capability": list(torch.cuda.get_device_capability()),
            "attention_backend": config._attn_implementation,
            "attention_module": type(first_layer.self_attn).__qualname__,
            "moe_module": type(first_layer.mlp).__qualname__,
            "strategy": type(self.strategy).__qualname__,
            "optimizer": type(self.optimizer).__qualname__,
            "scheduler": type(self.scheduler).__qualname__,
            "gradient_checkpointing": bool(self.cfg.trainer.gradient_checkpointing),
            "sample_packing": bool(self.cfg.trainer.use_sample_packing),
            "fsdp_size": int(self.cfg.trainer.policy.fsdp_config.fsdp_size),
            "expert_parallel_size": int(self.cfg.trainer.policy.fsdp_config.expert_model_parallel_size),
            "grouped_moe": bool(self.cfg.trainer.policy.fsdp_config.moe_grouped_gemm),
            "native_grouped_mm": grouped_mm_enabled,
            "expert_implementation": "grouped" if grouped_mm_enabled else "eager",
            "runtime_grug_module_path": grug_module_path,
            "runtime_grug_module_sha256": self._grug_benchmark_file_sha256(grug_module_path),
            "runtime_worker_path": worker_path,
            "runtime_worker_sha256": self._grug_benchmark_file_sha256(worker_path),
            "micro_batch_size": int(self.cfg.trainer.micro_train_batch_size_per_gpu),
            "mini_batch_size_per_gpu": int(self.policy_mini_batch_size_per_gpu),
        }

    def grug_benchmark_localize_staged(self):
        """Compare eager and grouped Grug blocks on identical live FSDP inputs.

        This is disposable measurement instrumentation.  One no-grad eager
        model forward supplies the input to every sparse block.  A pre-hook
        runs the grouped branch on that same tensor before the eager branch,
        so later layers cannot hide where the first block-local difference
        begins.  Layer zero additionally records projection-level FP32
        references and a route-conditioned input/weight-gradient probe.
        """

        if not hasattr(self, "_grug_benchmark_batch"):
            raise RuntimeError("call grug_benchmark_stage_batch before localization")
        if self.cfg.trainer.strategy != "fsdp2":
            raise RuntimeError("Grug localization is implemented only for FSDP2")

        modules = [module for module in self.model.modules() if isinstance(module, GrugMoeSparseMoeBlock)]
        if not modules:
            raise RuntimeError("Grug localization found no sparse blocks")
        if any(module.experts.use_grouped_mm for module in modules):
            raise RuntimeError("Grug localization must start from the eager expert path")

        def tensor_sha256(tensor: torch.Tensor) -> str:
            cpu = tensor.detach().to("cpu").contiguous()
            return hashlib.sha256(cpu.view(torch.uint8).numpy().tobytes()).hexdigest()

        def tensor_summary(tensor: torch.Tensor) -> dict[str, object]:
            value = tensor.detach().float()
            return {
                "shape": list(tensor.shape),
                "dtype": str(tensor.dtype),
                "sha256": tensor_sha256(tensor),
                "l2_norm": float(torch.linalg.vector_norm(value).item()),
                "max_abs": float(value.abs().max().item()),
                "mean": float(value.mean().item()),
                "finite": bool(torch.isfinite(value).all().item()),
            }

        def difference(
            actual: torch.Tensor,
            reference: torch.Tensor,
            *,
            rtol: float = 0.0,
            atol: float = 0.0,
        ) -> dict[str, object]:
            if actual.shape != reference.shape:
                raise RuntimeError(f"cannot compare shapes {tuple(actual.shape)} and {tuple(reference.shape)}")
            actual_float = actual.detach().float()
            reference_float = reference.detach().float()
            absolute = (actual_float - reference_float).abs()
            allowance = atol + rtol * reference_float.abs()
            scale = torch.linalg.vector_norm(reference_float)
            return {
                "numel": actual.numel(),
                "exact": bool(torch.equal(actual.detach(), reference.detach())),
                "allclose": bool((absolute <= allowance).all().item()),
                "rtol": rtol,
                "atol": atol,
                "max_abs": float(absolute.max().item()),
                "mean_abs": float(absolute.mean().item()),
                "relative_l2": float((torch.linalg.vector_norm(absolute) / scale.clamp_min(1e-30)).item()),
                "nonfinite": int((~torch.isfinite(actual_float)).sum().item()),
            }

        def local_weight(weight: torch.Tensor) -> torch.Tensor:
            return self._grug_benchmark_local_tensor(weight).detach()

        def eager_routed_stages(
            hidden_states: torch.Tensor,
            counts: torch.Tensor,
            gate_weight: torch.Tensor,
            down_weight: torch.Tensor,
            up_weight: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
            rows = hidden_states.shape[0]
            intermediate = gate_weight.shape[1]
            gate = hidden_states.new_empty((rows, intermediate))
            up = hidden_states.new_empty((rows, intermediate))
            hidden = hidden_states.new_empty((rows, intermediate))
            down = hidden_states.new_empty((rows, hidden_states.shape[1]))
            start = 0
            for expert_index, count in enumerate(counts.tolist()):
                end = start + count
                if count:
                    expert_input = hidden_states[start:end]
                    expert_gate = F.linear(expert_input, gate_weight[expert_index])
                    expert_up = F.linear(expert_input, up_weight[expert_index])
                    expert_hidden = F.silu(expert_gate) * expert_up
                    gate[start:end] = expert_gate
                    up[start:end] = expert_up
                    hidden[start:end] = expert_hidden
                    down[start:end] = F.linear(expert_hidden, down_weight[expert_index])
                start = end
            if start != rows:
                raise RuntimeError(f"eager routed rows disagree with loads: rows={rows}, loads={start}")
            return gate, up, hidden, down

        def grouped_routed_stages(
            hidden_states: torch.Tensor,
            counts: torch.Tensor,
            gate_weight: torch.Tensor,
            down_weight: torch.Tensor,
            up_weight: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
            from torchtitan.distributed.expert_parallel import TOKEN_GROUP_ALIGN_SIZE_M
            from torchtitan.experiments.kernels.moe.indices import generate_permute_indices

            experts = gate_weight.shape[0]
            with torch.no_grad():
                permuted_indices, padded_counts, _ = generate_permute_indices(
                    counts,
                    experts,
                    1,
                    hidden_states.shape[0] + experts * TOKEN_GROUP_ALIGN_SIZE_M,
                    TOKEN_GROUP_ALIGN_SIZE_M,
                )
            source = torch.vstack((hidden_states, hidden_states.new_zeros(hidden_states.shape[-1])))
            # Torchtitan leaves padded slots at -1 so they select the appended
            # zero row through normal Python indexing.  index_select rejects
            # those negative indices on CUDA and is not equivalent here.
            padded_input = source[permuted_indices.long()]
            offsets = torch.cumsum(padded_counts, dim=0, dtype=torch.int32)
            padded_gate = torch._grouped_mm(
                padded_input.bfloat16(), gate_weight.bfloat16().transpose(-2, -1), offs=offsets
            )
            padded_up = torch._grouped_mm(padded_input.bfloat16(), up_weight.bfloat16().transpose(-2, -1), offs=offsets)
            padded_hidden = F.silu(padded_gate) * padded_up
            padded_down = torch._grouped_mm(
                padded_hidden, down_weight.bfloat16().transpose(-2, -1), offs=offsets
            ).type_as(hidden_states)

            def unpermute(padded: torch.Tensor) -> torch.Tensor:
                restored = padded.new_empty((source.shape[0], padded.shape[-1]))
                restored[permuted_indices.long()] = padded
                return restored[:-1]

            return tuple(unpermute(value) for value in (padded_gate, padded_up, padded_hidden, padded_down))

        def combine_routed(
            routed_output: torch.Tensor,
            combine_weights: torch.Tensor,
            order: torch.Tensor,
            sorted_token_indices: torch.Tensor,
            num_tokens: int,
        ) -> torch.Tensor:
            sorted_weights = combine_weights.reshape(-1).index_select(0, order).to(routed_output.dtype)
            output = routed_output.new_zeros((num_tokens, routed_output.shape[-1]))
            output.index_add_(0, sorted_token_indices, routed_output * sorted_weights.unsqueeze(-1))
            return output

        def fp32_routed_reference(
            hidden_states: torch.Tensor,
            counts: torch.Tensor,
            gate_weight: torch.Tensor,
            down_weight: torch.Tensor,
            up_weight: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
            rows = hidden_states.shape[0]
            intermediate = gate_weight.shape[1]
            gate = torch.empty((rows, intermediate), dtype=torch.float32, device=hidden_states.device)
            up = torch.empty_like(gate)
            hidden = torch.empty_like(gate)
            down = torch.empty((rows, hidden_states.shape[1]), dtype=torch.float32, device=hidden_states.device)
            start = 0
            with torch.autocast(device_type="cuda", enabled=False):
                for expert_index, count in enumerate(counts.tolist()):
                    end = start + count
                    if count:
                        expert_input = hidden_states[start:end].float()
                        expert_gate = F.linear(expert_input, gate_weight[expert_index].float())
                        expert_up = F.linear(expert_input, up_weight[expert_index].float())
                        expert_hidden = F.silu(expert_gate) * expert_up
                        gate[start:end] = expert_gate
                        up[start:end] = expert_up
                        hidden[start:end] = expert_hidden
                        down[start:end] = F.linear(expert_hidden, down_weight[expert_index].float())
                    start = end
            return gate, up, hidden, down

        def single_expert_probe(
            expert_input: torch.Tensor,
            gate_weight: torch.Tensor,
            down_weight: torch.Tensor,
            up_weight: torch.Tensor,
        ) -> dict[str, object]:
            from skyrl_train.models.grug_moe import _run_grug_grouped_mm

            output_count = expert_input.shape[0] * down_weight.shape[0]
            cotangent = torch.linspace(
                -1.0,
                1.0,
                steps=output_count,
                dtype=torch.float32,
                device=expert_input.device,
            ).reshape(expert_input.shape[0], down_weight.shape[0])

            def run(path: str, dtype: torch.dtype):
                x = expert_input.detach().to(dtype).clone().requires_grad_(True)
                gate = gate_weight.detach().to(dtype).clone().requires_grad_(True)
                down = down_weight.detach().to(dtype).clone().requires_grad_(True)
                up = up_weight.detach().to(dtype).clone().requires_grad_(True)
                if path == "eager":
                    output = F.linear(F.silu(F.linear(x, gate)) * F.linear(x, up), down)
                elif path == "grouped":
                    counts = torch.tensor([x.shape[0]], dtype=torch.int64, device=x.device)
                    output = _run_grug_grouped_mm(gate.unsqueeze(0), down.unsqueeze(0), up.unsqueeze(0), x, counts)
                else:
                    raise ValueError(path)
                scalar = (output.float() * cotangent).mean()
                gradients = torch.autograd.grad(scalar, (x, gate, down, up))
                return output.detach(), tuple(gradient.detach() for gradient in gradients)

            with torch.enable_grad():
                eager_output, eager_gradients = run("eager", torch.bfloat16)
                grouped_output, grouped_gradients = run("grouped", torch.bfloat16)
                with torch.autocast(device_type="cuda", enabled=False):
                    fp32_output, fp32_gradients = run("eager", torch.float32)
            names = ("input", "gate_weight", "down_weight", "up_weight")
            return {
                "rows": expert_input.shape[0],
                "output_grouped_vs_eager": difference(grouped_output, eager_output, rtol=4e-2, atol=4e-3),
                "output_eager_vs_fp32": difference(eager_output, fp32_output, rtol=4e-2, atol=4e-3),
                "output_grouped_vs_fp32": difference(grouped_output, fp32_output, rtol=4e-2, atol=4e-3),
                "gradients_grouped_vs_eager": {
                    name: difference(grouped, eager, rtol=8e-2, atol=1e-4)
                    for name, grouped, eager in zip(names, grouped_gradients, eager_gradients)
                },
                "gradients_eager_vs_fp32": {
                    name: difference(eager, fp32, rtol=8e-2, atol=1e-4)
                    for name, eager, fp32 in zip(names, eager_gradients, fp32_gradients)
                },
                "gradients_grouped_vs_fp32": {
                    name: difference(grouped, fp32, rtol=8e-2, atol=1e-4)
                    for name, grouped, fp32 in zip(names, grouped_gradients, fp32_gradients)
                },
            }

        block_evidence: list[dict[str, object] | None] = [None] * len(modules)
        shadow_by_layer: dict[int, dict[str, object]] = {}
        hook_handles = []

        def before_block(layer_index: int, module: GrugMoeSparseMoeBlock, args):
            if layer_index in shadow_by_layer:
                raise RuntimeError(f"nested Grug localization at layer {layer_index}")
            hidden_states = args[0]
            shape = hidden_states.shape
            flat = hidden_states.reshape(-1, shape[-1])
            with torch.no_grad():
                router_output = module.router(flat)
                selected_experts = router_output.selected_experts
                combine_weights = router_output.combine_weights
                flat_experts = selected_experts.reshape(-1)
                order = torch.argsort(flat_experts, stable=True)
                token_indices = (
                    torch.arange(flat.shape[0], device=flat.device)
                    .unsqueeze(1)
                    .expand(-1, selected_experts.shape[-1])
                    .reshape(-1)
                )
                sorted_token_indices = token_indices.index_select(0, order)
                routed_input = flat.index_select(0, sorted_token_indices)
                counts = torch.bincount(flat_experts, minlength=module.experts.num_experts)

                product_routed: dict[str, object] = {}

                def capture_product_routed(_experts, expert_args, expert_output):
                    product_routed["input"] = expert_args[0]
                    product_routed["counts"] = expert_args[1]
                    product_routed["output"] = expert_output

                expert_handle = module.experts.register_forward_hook(capture_product_routed)
                was_grouped = module.experts.use_grouped_mm
                module.experts.use_grouped_mm = True
                try:
                    grouped_output = module._forward_grouped(flat, selected_experts, combine_weights).reshape(shape)
                finally:
                    module.experts.use_grouped_mm = was_grouped
                    expert_handle.remove()
                if set(product_routed) != {"input", "counts", "output"}:
                    raise RuntimeError(f"layer {layer_index} did not capture the product grouped expert output")

                sorted_logits = torch.topk(
                    router_output.router_logits + module.router.bias,
                    k=module.router.top_k + 1,
                    dim=-1,
                    sorted=True,
                ).values
                margin = sorted_logits[:, module.router.top_k - 1] - sorted_logits[:, module.router.top_k]
                evidence: dict[str, object] = {
                    "layer": layer_index,
                    "input_shape": list(flat.shape),
                    "input_sha256": tensor_sha256(flat),
                    "selected_experts_sha256": tensor_sha256(selected_experts),
                    "route_loads": counts.cpu().tolist(),
                    "route_margin": {
                        "min": float(margin.min().item()),
                        "p01": float(torch.quantile(margin.float(), 0.01).item()),
                        "median": float(torch.quantile(margin.float(), 0.5).item()),
                        "p99": float(torch.quantile(margin.float(), 0.99).item()),
                        "max": float(margin.max().item()),
                        "count_le_1e-4": int((margin <= 1e-4).sum().item()),
                        "count_le_1e-3": int((margin <= 1e-3).sum().item()),
                        "count_le_1e-2": int((margin <= 1e-2).sum().item()),
                    },
                }

                if layer_index == 0:
                    gate_weight = local_weight(module.experts.gate_proj.weight)
                    down_weight = local_weight(module.experts.down_proj.weight)
                    up_weight = local_weight(module.experts.up_proj.weight)
                    eager_stages = eager_routed_stages(routed_input, counts, gate_weight, down_weight, up_weight)
                    grouped_stages = grouped_routed_stages(routed_input, counts, gate_weight, down_weight, up_weight)
                    old_tf32 = torch.backends.cuda.matmul.allow_tf32
                    old_precision = torch.get_float32_matmul_precision()
                    torch.backends.cuda.matmul.allow_tf32 = False
                    torch.set_float32_matmul_precision("highest")
                    try:
                        fp32_stages = fp32_routed_reference(routed_input, counts, gate_weight, down_weight, up_weight)
                    finally:
                        torch.backends.cuda.matmul.allow_tf32 = old_tf32
                        torch.set_float32_matmul_precision(old_precision)
                    stage_names = ("gate_projection", "up_projection", "swiglu", "down_projection")
                    detailed = {
                        "input": tensor_summary(flat),
                        "router_logits": tensor_summary(router_output.router_logits),
                        "route_order_exact": bool(
                            torch.equal(
                                order,
                                torch.cat([torch.where(flat_experts == expert)[0] for expert in range(counts.numel())]),
                            )
                        ),
                        "stages_grouped_vs_eager": {
                            name: difference(grouped, eager)
                            for name, grouped, eager in zip(stage_names, grouped_stages, eager_stages)
                        },
                        "stages_eager_vs_fp32": {
                            name: difference(eager, fp32)
                            for name, eager, fp32 in zip(stage_names, eager_stages, fp32_stages)
                        },
                        "stages_grouped_vs_fp32": {
                            name: difference(grouped, fp32)
                            for name, grouped, fp32 in zip(stage_names, grouped_stages, fp32_stages)
                        },
                    }
                    manual_grouped_output = combine_routed(
                        grouped_stages[-1],
                        combine_weights,
                        order,
                        sorted_token_indices,
                        flat.shape[0],
                    )
                    product_routed_output = product_routed["output"]
                    detailed["product_routed_input_exact"] = bool(
                        torch.equal(product_routed["input"], routed_input)
                    )
                    detailed["product_routed_counts_exact"] = bool(
                        torch.equal(product_routed["counts"], counts)
                    )
                    detailed["product_routed_vs_manual_grouped"] = difference(
                        product_routed_output, grouped_stages[-1]
                    )
                    detailed["manual_grouped_vs_product_grouped"] = difference(
                        manual_grouped_output, grouped_output.reshape_as(manual_grouped_output)
                    )
                    repeated_product_combine = combine_routed(
                        product_routed_output,
                        combine_weights,
                        order,
                        sorted_token_indices,
                        flat.shape[0],
                    )
                    detailed["repeated_combine_vs_product_grouped"] = difference(
                        repeated_product_combine, grouped_output.reshape_as(repeated_product_combine)
                    )
                    probe_expert = int(torch.argmax(counts).item())
                    probe_start = int(counts[:probe_expert].sum().item())
                    probe_end = probe_start + int(counts[probe_expert].item())
                    detailed["gradient_probe_expert"] = probe_expert
                    detailed["gradient_probe"] = single_expert_probe(
                        routed_input[probe_start:probe_end],
                        gate_weight[probe_expert],
                        down_weight[probe_expert],
                        up_weight[probe_expert],
                    )
                    evidence["detailed"] = detailed
                    del eager_stages, grouped_stages, fp32_stages

            actual_router: dict[str, object] = {}

            def capture_actual_router(_router, _args, output):
                actual_router["logits"] = difference(output.router_logits, router_output.router_logits)
                actual_router["selected_experts_exact"] = bool(
                    torch.equal(output.selected_experts, router_output.selected_experts)
                )
                actual_router["combine_weights"] = difference(output.combine_weights, router_output.combine_weights)

            router_handle = module.router.register_forward_hook(capture_actual_router)
            shadow_by_layer[layer_index] = {
                "grouped_output": grouped_output,
                "actual_router": actual_router,
                "router_handle": router_handle,
                "evidence": evidence,
            }

        def after_block(layer_index: int, _module: GrugMoeSparseMoeBlock, _args, eager_output):
            shadow = shadow_by_layer.pop(layer_index)
            shadow["router_handle"].remove()
            actual_router = shadow["actual_router"]
            if set(actual_router) != {"logits", "selected_experts_exact", "combine_weights"}:
                raise RuntimeError(f"layer {layer_index} did not capture the eager router output")
            evidence = shadow["evidence"]
            evidence["router_shadow_vs_eager"] = actual_router
            evidence["block_output_grouped_vs_eager"] = difference(
                shadow["grouped_output"], eager_output, rtol=4e-2, atol=4e-3
            )
            block_evidence[layer_index] = evidence

        for layer_index, module in enumerate(modules):
            hook_handles.extend(
                (
                    module.register_forward_pre_hook(
                        lambda module, args, index=layer_index: before_block(index, module, args)
                    ),
                    module.register_forward_hook(
                        lambda module, args, output, index=layer_index: after_block(index, module, args, output)
                    ),
                )
            )

        torch.cuda.synchronize()
        if torch.distributed.is_initialized():
            torch.distributed.barrier()
        state_hash_before = self._grug_benchmark_state_hash(self.model.state_dict())
        cpu_rng_state = torch.get_rng_state()
        cuda_rng_state = torch.cuda.get_rng_state()
        gradients_before = sum(parameter.grad is not None for parameter in self.model.parameters())
        self.optimizer.zero_grad(set_to_none=True)
        diagnostic_batch = self._grug_benchmark_batch[:1]
        experience = BatchIterator.batch_to_experience(diagnostic_batch)
        experience.to_device(torch.cuda.current_device())
        try:
            self.model.train()
            with torch.no_grad(), torch.autocast(dtype=torch.bfloat16, device_type="cuda"):
                action_log_probs = self.model(
                    experience.sequences,
                    experience.num_actions,
                    attention_mask=experience.attention_mask,
                    temperature=1.0,
                    return_output=False,
                    compute_entropy=False,
                    rollout_routed_experts=None,
                )
        finally:
            for handle in hook_handles:
                handle.remove()
            for shadow in shadow_by_layer.values():
                shadow["router_handle"].remove()
            torch.set_rng_state(cpu_rng_state)
            torch.cuda.set_rng_state(cuda_rng_state)
            self.optimizer.zero_grad(set_to_none=True)
        if shadow_by_layer:
            raise RuntimeError(f"Grug localization left open layers: {sorted(shadow_by_layer)}")
        if any(item is None for item in block_evidence):
            raise RuntimeError("Grug localization did not observe every sparse block")
        if any(module.experts.use_grouped_mm for module in modules):
            raise RuntimeError("Grug localization did not restore the eager path")
        torch.cuda.synchronize()
        if torch.distributed.is_initialized():
            torch.distributed.barrier()
        state_hash_after = self._grug_benchmark_state_hash(self.model.state_dict())
        if state_hash_after != state_hash_before:
            raise RuntimeError(
                f"Grug localization changed model state: before={state_hash_before}, after={state_hash_after}"
            )
        gradients_after = sum(parameter.grad is not None for parameter in self.model.parameters())
        if gradients_after:
            raise RuntimeError(f"Grug localization left {gradients_after} model gradients")
        first_nonexact_layer = next(
            (item["layer"] for item in block_evidence if not item["block_output_grouped_vs_eager"]["exact"]),
            None,
        )
        return {
            "rank": int(torch.distributed.get_rank()),
            "state_hash_before": state_hash_before,
            "state_hash_after": state_hash_after,
            "gradients_cleared_before": gradients_before,
            "gradients_after": gradients_after,
            "cpu_rng_restored": bool(torch.equal(torch.get_rng_state(), cpu_rng_state)),
            "cuda_rng_restored": bool(torch.equal(torch.cuda.get_rng_state(), cuda_rng_state)),
            "action_log_probs": tensor_summary(action_log_probs),
            "first_nonexact_block_output_layer": first_nonexact_layer,
            "blocks": block_evidence,
        }

    def grug_benchmark_run_staged_ppo(self, profile: bool = False):
        """Time one production PPO update on the previously staged replay shard."""

        if not hasattr(self, "_grug_benchmark_batch"):
            raise RuntimeError("call grug_benchmark_stage_batch before the timed update")
        self._grug_benchmark_phase_events = {}
        self._grug_benchmark_open_phase = None
        profiler = self._grug_benchmark_start_profile(profile)
        torch.cuda.synchronize()
        if torch.distributed.is_initialized():
            torch.distributed.barrier()
        torch.cuda.synchronize()
        started = time.perf_counter()
        output = self.ppo_train(self._grug_benchmark_batch)
        torch.cuda.synchronize()
        if torch.distributed.is_initialized():
            torch.distributed.barrier()
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - started
        if self._grug_benchmark_open_phase is not None:
            raise RuntimeError(f"unclosed Grug benchmark phase: {self._grug_benchmark_open_phase[0]}")
        phase_seconds = {
            name: sum(start.elapsed_time(end) for start, end in spans) / 1000.0
            for name, spans in self._grug_benchmark_phase_events.items()
        }
        self._grug_benchmark_phase_events = None
        profile_artifact_gzip = self._grug_benchmark_finish_profile(profiler)
        return {
            "rank": int(torch.distributed.get_rank()),
            "elapsed_seconds": elapsed,
            "phase_seconds": phase_seconds,
            "train_status": output.metadata["train_status"],
            "peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
            "peak_reserved_bytes": int(torch.cuda.max_memory_reserved()),
            "allocated_bytes": int(torch.cuda.memory_allocated()),
            "reserved_bytes": int(torch.cuda.memory_reserved()),
            "profile_artifact_gzip": profile_artifact_gzip,
        }

    def grug_benchmark_run_staged_matched_ce(self, profile: bool = False, expert_attribution: bool = False):
        """Time the common fixed-replay CE forward and backward, without Adam."""

        if not hasattr(self, "_grug_benchmark_batch"):
            raise RuntimeError("call grug_benchmark_stage_batch before the timed update")
        self._grug_benchmark_phase_events = {}
        self._grug_benchmark_open_phase = None
        self._grug_benchmark_parent_phase = None
        profiler = self._grug_benchmark_start_profile(profile)
        expert_handles = []
        expert_module_count = 0
        if expert_attribution:
            expert_handles, expert_module_count = self._grug_benchmark_install_expert_hooks()
        self.optimizer.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        if torch.distributed.is_initialized():
            torch.distributed.barrier()
        torch.cuda.synchronize()
        started = time.perf_counter()
        local_loss_sum, local_loss_tokens, microbatches, representative_action_log_probs = (
            self._grug_benchmark_matched_ce(self._grug_benchmark_batch)
        )
        torch.cuda.synchronize()
        if torch.distributed.is_initialized():
            torch.distributed.barrier()
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - started
        if self._grug_benchmark_open_phase is not None:
            raise RuntimeError(f"unclosed Grug benchmark phase: {self._grug_benchmark_open_phase[0]}")
        phase_seconds = {
            name: sum(start.elapsed_time(end) for start, end in spans) / 1000.0
            for name, spans in self._grug_benchmark_phase_events.items()
        }
        self._grug_benchmark_phase_events = None
        expert_evidence = (
            self._grug_benchmark_finish_expert_hooks(expert_handles, expert_module_count)
            if expert_attribution
            else None
        )
        profile_artifact_gzip = self._grug_benchmark_finish_profile(profiler)
        peak_allocated_bytes = int(torch.cuda.max_memory_allocated())
        peak_reserved_bytes = int(torch.cuda.max_memory_reserved())

        gradient_tensors = 0
        gradient_numel = 0
        nonfinite_gradient_tensors = 0
        representative_suffixes = (
            "model.embed_tokens.weight",
            "model.layers.0.self_attn.q_proj.weight",
            "model.layers.0.mlp.router.weight",
            "model.layers.0.mlp.experts.gate_proj.weight",
            "model.layers.0.mlp.experts.up_proj.weight",
            "model.layers.0.mlp.experts.down_proj.weight",
            "model.layers.0.shared_expert.gate_proj.weight",
            "model.norm.weight",
            "lm_head.weight",
        )
        representative_gradients = {}
        for name, parameter in self.model.named_parameters():
            if parameter.grad is None:
                continue
            gradient = self._grug_benchmark_local_tensor(parameter.grad)
            gradient_tensors += 1
            gradient_numel += gradient.numel()
            if not bool(torch.isfinite(gradient).all().item()):
                nonfinite_gradient_tensors += 1
            matched_suffix = next((suffix for suffix in representative_suffixes if name.endswith(suffix)), None)
            if matched_suffix is None:
                continue
            flat_gradient = gradient.detach().float().reshape(-1)
            sample_count = min(16, flat_gradient.numel())
            sample_indices = (
                torch.linspace(
                    0,
                    flat_gradient.numel() - 1,
                    steps=sample_count,
                    dtype=torch.float64,
                    device=flat_gradient.device,
                )
                .round()
                .to(torch.int64)
            )
            representative_gradients[matched_suffix] = {
                "local_numel": flat_gradient.numel(),
                "l2_norm": float(torch.linalg.vector_norm(flat_gradient).item()),
                "max_abs": float(flat_gradient.abs().max().item()),
                "samples": flat_gradient.index_select(0, sample_indices).cpu().tolist(),
            }
        missing_representative_gradients = set(representative_suffixes) - set(representative_gradients)
        if missing_representative_gradients:
            raise RuntimeError(
                f"matched CE missed representative gradients: {sorted(missing_representative_gradients)}"
            )
        self.optimizer.zero_grad(set_to_none=True)
        return {
            "rank": int(torch.distributed.get_rank()),
            "elapsed_seconds": elapsed,
            "phase_seconds": phase_seconds,
            "local_loss_sum": float(local_loss_sum.item()),
            "local_loss_tokens": local_loss_tokens,
            "microbatches": microbatches,
            "gradient_tensors": gradient_tensors,
            "gradient_numel": gradient_numel,
            "nonfinite_gradient_tensors": nonfinite_gradient_tensors,
            "representative_action_log_probs": representative_action_log_probs.float().cpu().tolist(),
            "representative_gradients": representative_gradients,
            "expert_attribution": expert_evidence,
            "peak_allocated_bytes": peak_allocated_bytes,
            "peak_reserved_bytes": peak_reserved_bytes,
            "allocated_bytes": int(torch.cuda.memory_allocated()),
            "reserved_bytes": int(torch.cuda.memory_reserved()),
            "profile_artifact_gzip": profile_artifact_gzip,
        }

    def offload_to_cpu(self, pin_memory=True, non_blocking=True, offload_optimizer=True, offload_model=True):
        self._set_numa_affinity(torch.distributed.get_rank() % torch.cuda.device_count())
        self.strategy.offload_to_cpu(
            self.model, self.optimizer, pin_memory, non_blocking, offload_optimizer, offload_model
        )

    def backload_to_gpu(self, non_blocking=True, backload_optimizer=True, backload_model=True):
        self.strategy.backload_to_gpu(self.model, self.optimizer, non_blocking, backload_optimizer, backload_model)

    def read_post_step_weights(self, names):
        """TEST-ONLY (Stage 6 weight-equality gate): return the post-step HF-named
        weight tensors the broadcast would send, for a representative ``names`` set.

        Runs the SAME ``extract_weights`` path used by ``broadcast_to_inference_engines``
        (grouped->HF remap + FSDP ``full_tensor()`` gather), so the returned tensors
        are byte-identical to what the engine receives. ``extract_weights`` /
        ``full_tensor()`` are collective over the full mesh, so EVERY rank must run
        the generator; only rank 0 returns the (full) tensors as CPU fp32 — other
        ranks return an empty dict to keep the payload small.
        """
        wanted = set(names)
        collected = {}
        generator_dtype = str_to_torch_dtype(self.cfg.generator.model_dtype)
        is_rank0 = torch.distributed.get_rank() == 0
        # Must drive the full generator on every rank (the per-tensor full_tensor()
        # gather is a collective); harvest only the requested names on rank 0.
        for chunk in self.weight_extractor.extract_weights(generator_dtype):
            for name, tensor in zip(chunk.names, chunk.tensors):
                if is_rank0 and name in wanted:
                    collected[name] = tensor.detach().to("cpu", dtype=torch.float32).contiguous()
        return collected

    def grug_validation_snapshot(self, names=()):
        """Return the calling rank and requested Grug weights gathered on rank 0.

        Every rank must call this with the same names because DTensor
        materialization is collective. The result includes the rank, loaded
        attention backend, and requested ``weights``; the weights mapping is
        empty on nonzero ranks.
        """
        config = getattr(self.model.model, "config", None)
        if getattr(config, "model_type", None) != GRUG_MOE_MODEL_TYPE:
            raise ValueError("grug_validation_snapshot is only valid for Grug models")
        state = self.model.model.state_dict()
        missing = set(names).difference(state)
        if missing:
            raise KeyError(f"missing Grug state entries: {sorted(missing)}")
        is_rank0 = torch.distributed.get_rank() == 0
        weights = {}
        for name in names:
            tensor = self.weight_extractor._gather_tensor(state[name])
            if is_rank0:
                weights[name] = tensor.detach().to("cpu", dtype=torch.float32).contiguous()
        return GrugValidationSnapshot(
            rank=torch.distributed.get_rank(),
            attention_backend=config._attn_implementation,
            weights=weights,
        )

    def diag_ep8_geometry(self):
        """TEST-ONLY (EP=8 cross-node diag): return this rank's mesh geometry +
        physical-node identity so the driver can PROVE an EP group straddles >=2
        nodes. No collectives, no gather — pure introspection.

        Returns a dict with global rank, hostname, mesh shape/dim-names, this rank's
        per-mesh-dim coordinate, and the EP submesh coordinate (the index of this rank
        within its 8-rank EP group).
        """
        import socket

        mesh = self.strategy.device_mesh
        dim_names = list(mesh.mesh_dim_names)
        shape = tuple(mesh.shape)
        coord = list(mesh.get_coordinate())
        ep_dim = dim_names.index("ep") if "ep" in dim_names else None
        # The EP-group identity = the coord with the ep dim removed (all ranks sharing
        # this tuple form one 8-way EP group). The ep coord = position within the group.
        group_key = tuple(c for i, c in enumerate(coord) if i != ep_dim)
        return {
            "rank": int(torch.distributed.get_rank()),
            "host": socket.gethostname(),
            "mesh_dim_names": dim_names,
            "mesh_shape": shape,
            "coord": coord,
            "ep_dim": ep_dim,
            "ep_coord": (coord[ep_dim] if ep_dim is not None else None),
            "ep_group_key": group_key,
        }

    def diag_ep8_disk_ref_compare(self, model_path, layer_idx=0, n_rep_gather=2):
        """TEST-ONLY (EP=8 cross-node, NON-CIRCULAR weight-equality assert).

        Captures the value-corruption signature of the FSDP->vLLM MoE weight gather
        WITHOUT any inference engine, rollout, or engine-readback. For ``layer_idx``'s
        grouped expert stacks (w1/w2/w3) this rank:

          1. ON-GPU: runs the REAL ``self._gather_tensor`` (= ``gather_dtensor_strided_safe``
             over the ``(_StridedShard(fsdp), Shard(ep))`` composite) ``n_rep_gather``
             times, keeping every result on the CUDA device (NO ``.cpu().float()``
             round-trip, which would hide a W3 stream race).
          2. REFERENCE (non-circular): rank 0 loads the BASE model's per-expert
             weights independently from the on-disk HF checkpoint shards via
             ``safetensors.safe_open`` — a path that NEVER touches the EP gather.
          3. DIFFS each gathered expert row j vs the disk reference row j on GPU:
             max_abs, a cross-expert nearest-match (find disk m with gathered[j]==ref[m],
             m!=j => W1 swap), a prefix-block Δ test (rows off by a fixed shift => W2),
             gather-repeat determinism (gather1 vs gather2 => W3), and a dtype/byte check (W4).

        Collective contract: ``_gather_tensor`` is an all_gather over the (fsdp,ep)
        submesh, so EVERY rank must call it in the SAME order. All ranks gather; only
        rank 0 loads the disk reference + emits the signature (returns {} elsewhere).
        """
        import hashlib
        import json
        import os
        import socket

        rank = int(torch.distributed.get_rank())
        host = socket.gethostname()

        # --- locate THIS layer's grouped expert DTensor params (shim layout) ---
        # Keys look like ``...layers.{i}.mlp.moe.experts.w1`` (GroupedMoEShim) possibly
        # with an ``_fsdp_wrapped_module`` segment. Match by suffix on the stripped name.
        named = dict(self.model.model.named_parameters())

        def _strip(n):
            return n.replace("._fsdp_wrapped_module.", ".").replace(".mlp.moe.", ".mlp.")

        want_suffix = {
            "w1": f".layers.{layer_idx}.mlp.experts.w1",
            "w3": f".layers.{layer_idx}.mlp.experts.w3",
            "w2": f".layers.{layer_idx}.mlp.experts.w2",
        }
        found = {}
        for n, p in named.items():
            sn = _strip(n)
            for tag, suf in want_suffix.items():
                if sn.endswith(suf):
                    found[tag] = p
        # ---- ON-GPU gather, repeated, kept on device ----
        # ``_gather_tensor`` (= gather_dtensor_strided_safe) lives on the weight
        # EXTRACTOR, not the worker — it is the EXACT gather the broadcast path uses.
        gather_fn = self.weight_extractor._gather_tensor
        gathered = {tag: [] for tag in found}  # tag -> list of n_rep CUDA tensors
        placements_info = {}
        for rep in range(n_rep_gather):
            # Deterministic, all-ranks-identical iteration order over w1,w2,w3.
            for tag in ("w1", "w2", "w3"):
                if tag not in found:
                    continue
                p = found[tag]
                if rep == 0 and isinstance(p, DTensor):
                    placements_info[tag] = (str(p.placements), tuple(p.shape), str(p.dtype))
                g = gather_fn(p)  # ON-GPU gather (gather_dtensor_strided_safe)
                gathered[tag].append(g)  # keep on CUDA

        if rank != 0:
            # free + return (non-rank-0 still had to drive the collectives above)
            return {}

        # ---------------- rank 0: disk reference + signature ----------------
        out = {
            "rank": rank,
            "host": host,
            "layer": layer_idx,
            "placements": placements_info,
            "n_rep_gather": n_rep_gather,
            "lines": [],
            "verdict": None,
            "wrong_expert_map": {},
        }

        # Resolve the on-disk HF checkpoint shards (local cache or download).
        from huggingface_hub import snapshot_download
        from safetensors import safe_open

        local_dir = model_path
        if not (os.path.isdir(model_path) and os.path.exists(os.path.join(model_path, "config.json"))):
            local_dir = snapshot_download(
                model_path,
                allow_patterns=["*.safetensors", "*.json"],
            )

        # Build name -> shard-file index.
        idx_path = os.path.join(local_dir, "model.safetensors.index.json")
        if os.path.exists(idx_path):
            with open(idx_path) as f:
                weight_map = json.load(f)["weight_map"]
        else:
            # single-shard model
            single = os.path.join(local_dir, "model.safetensors")
            weight_map = None
            single_file = single

        proj_for = {"w1": "gate_proj", "w3": "up_proj", "w2": "down_proj"}

        def _load_disk_expert(tag, j):
            """Load base disk weight for expert j of this layer/tag as a CUDA tensor."""
            key = f"model.layers.{layer_idx}.mlp.experts.{j}.{proj_for[tag]}.weight"
            if weight_map is not None:
                shard = os.path.join(local_dir, weight_map[key])
            else:
                shard = single_file
            with safe_open(shard, framework="pt", device="cpu") as fp:
                t = fp.get_tensor(key)
            return t.to(gathered[tag][0].device)

        def _row_hash(t):
            return hashlib.md5(t.detach().to(torch.float32).cpu().contiguous().numpy().tobytes()).hexdigest()[:12]

        n_experts = gathered["w1"][0].shape[0] if "w1" in gathered else 0
        out["num_experts"] = n_experts

        # Pre-load ALL disk gate_proj rows once (for the cross-expert nearest-match on w1).
        disk_w1_rows = None
        EPS = 1e-6  # bf16 round-trip epsilon; gathered base == disk base should be exact-ish

        for tag in ("w1", "w2", "w3"):
            if tag not in gathered:
                continue
            g0 = gathered[tag][0]
            g1 = gathered[tag][min(1, len(gathered[tag]) - 1)]
            # --- W3: gather determinism (gather0 vs gather1) ---
            det_max = float((g0.float() - g1.float()).abs().max().item())
            out["lines"].append(
                f"[L{layer_idx}.{tag}] shape={tuple(g0.shape)} dtype={g0.dtype} "
                f"gather-repeat max_abs(g0-g1)={det_max:.3e} "
                f"({'NON-DETERMINISTIC=>W3' if det_max > EPS else 'deterministic'})"
            )
            if tag == "w1":
                disk_w1_rows = [_load_disk_expert("w1", j).float() for j in range(n_experts)]

            n_corrupt = 0
            worst = (None, -1.0)
            for j in range(n_experts):
                gj = g0[j].float()
                ref = _load_disk_expert(tag, j).float()
                if tuple(gj.shape) != tuple(ref.shape):
                    out["lines"].append(f"    {tag}[{j}] SHAPE_MISMATCH g={tuple(gj.shape)} ref={tuple(ref.shape)}")
                    n_corrupt += 1
                    continue
                ma = float((gj - ref).abs().max().item())
                if ma > worst[1]:
                    worst = (j, ma)
                if ma <= EPS:
                    continue
                n_corrupt += 1
                extra = ""
                # --- W1: cross-expert nearest-match (does gathered[j] == disk[m], m!=j?) ---
                if tag == "w1" and disk_w1_rows is not None:
                    best_m, best_e = None, float("inf")
                    for m in range(n_experts):
                        e = float((disk_w1_rows[m] - gj).abs().max().item())
                        if e < best_e:
                            best_e, best_m = e, m
                    if best_m is not None and best_m != j and best_e <= EPS:
                        extra += f"  WRONG_EXPERT(carries disk expert {best_m})"
                        out["wrong_expert_map"][j] = best_m
                    elif best_m is not None:
                        extra += f"  closest_disk_expert={best_m}@{best_e:.2e}"
                # --- W2: prefix-block Δ (is gathered row j == disk row j+Δ for a fixed Δ?) ---
                gh, rh = _row_hash(gj), _row_hash(ref)
                extra += f"  ghash={gh} refhash={rh}"
                out["lines"].append(f"    {tag}[{j}] max_abs={ma:.3e}{extra}")
            out["lines"].append(
                f"[L{layer_idx}.{tag}] CORRUPT {n_corrupt}/{n_experts}  worst=expert{worst[0]}@{worst[1]:.3e}"
            )

        # --- W2 contiguous-shift detector: if wrong_expert_map is a constant offset Δ ---
        if out["wrong_expert_map"]:
            deltas = {(m - j) % n_experts for j, m in out["wrong_expert_map"].items()}
            if len(deltas) == 1:
                d = next(iter(deltas))
                out["lines"].append(
                    f"[L{layer_idx}] CONSTANT row shift Δ={d} across ALL wrong experts => W2-style block shift"
                )
            else:
                out["lines"].append(
                    f"[L{layer_idx}] wrong-expert offsets are NON-uniform (deltas={sorted(deltas)}) => W1 strided permutation"
                )

        # Verdict
        total_corrupt = sum(1 for l in out["lines"] if l.strip().startswith(("w1[", "w2[", "w3[")))
        out["total_corrupt_rows"] = total_corrupt
        out["verdict"] = (
            ("CLEAN (gathered==disk at EP=8 on-GPU => corruption is DOWNSTREAM: NCCL broadcast or vLLM load_weights)")
            if total_corrupt == 0
            else (f"CORRUPT ({total_corrupt} expert rows differ from disk reference at EP=8 cross-node)")
        )
        return out

    def init_model(self, model_path, num_training_steps: int = None, model_revision: str | None = None):
        assert self.cfg.trainer.strategy in ("fsdp", "fsdp2")
        strategy = FSDPStrategy(
            fsdp_config=self.cfg.trainer.policy.fsdp_config,
            optimizer_config=self.cfg.trainer.policy.optimizer_config,
            model_config=self.cfg.trainer.policy.model,
            fsdp_strategy=self.cfg.trainer.strategy,
            seed=self.cfg.trainer.seed,
            micro_train_batch_size_per_gpu=self.cfg.trainer.micro_train_batch_size_per_gpu,
            num_training_steps=num_training_steps,
        )
        strategy.setup_distributed()
        self.strategy = strategy

        # Stage 3: surface the CP submesh/group on the worker so the Stage-4 forward wrap
        # can read it. cp_size==1 leaves both None (flag-off path untouched).
        self.cp_mesh = getattr(strategy, "cp_mesh", None)
        self.cp_group = getattr(strategy, "cp_group", None)

        self._is_lora = self.cfg.trainer.policy.model.lora.rank > 0

        # Update per-gpu mini batch size based on device mesh
        self._normalize_mini_batch_size()

        model_config = AutoConfig.from_pretrained(
            model_path,
            trust_remote_code=True,
            revision=model_revision,
        )
        validate_grug_expert_parallel_options(
            getattr(model_config, "model_type", None),
            expert_model_parallel_size=getattr(strategy, "ep_size", 1),
            use_grouped_mm=bool(self.cfg.trainer.policy.fsdp_config.get("use_grouped_mm", False)),
            ep_comm_backend=str(self.cfg.trainer.policy.fsdp_config.get("ep_comm_backend", "torch")),
        )
        init_context = get_init_weight_context_manager(
            use_meta_tensor=not model_config.tie_word_embeddings, mesh=self.strategy.device_mesh
        )
        with init_context():
            wrapped_model = HFModelWrapper(
                model_path,
                use_flash_attention_2=self.cfg.trainer.flash_attn,
                # Marin stores MuonH parameters and optimizer state in FP32;
                # FSDP2's mixed-precision policy still casts forward compute to BF16.
                # Every pre-existing optimizer keeps its BF16 load path.
                bf16=not strategy.is_muonh_optimizer,
                lora_rank=self.cfg.trainer.policy.model.lora.rank,
                lora_alpha=self.cfg.trainer.policy.model.lora.alpha,
                lora_dropout=self.cfg.trainer.policy.model.lora.dropout,
                target_modules=self.cfg.trainer.policy.model.lora.target_modules,
                exclude_modules=self.cfg.trainer.policy.model.lora.exclude_modules,
                sequence_parallel_size=self.cfg.trainer.policy.sequence_parallel_size,
                use_sample_packing=self.cfg.trainer.use_sample_packing,
                use_torch_compile=self.cfg.trainer.policy.use_torch_compile,
                rope_scaling=get_rope_scaling_config(self.cfg.trainer),
                rope_theta=get_rope_theta_config(self.cfg.trainer),
                moe_router_replay=bool(self.cfg.trainer.policy.fsdp_config.get("moe_router_replay", False)),
                moe_grouped_gemm=bool(self.cfg.trainer.policy.fsdp_config.get("moe_grouped_gemm", False)),
                use_grouped_mm=bool(self.cfg.trainer.policy.fsdp_config.get("use_grouped_mm", False)),
                attn_backend=self.cfg.trainer.get("attn_backend", "auto"),
                context_parallel_size=int(self.cfg.trainer.policy.fsdp_config.get("context_parallel_size", 1)),
                # Stage 4: surface the CP submesh + rotate method so the forward
                # enters torch-native context_parallel (ring SDPA). None at cp=1.
                cp_mesh=self.cp_mesh,
                cp_rotate_method=str(self.cfg.trainer.policy.fsdp_config.get("cp_rotate_method", "allgather")),
                training_strategy=self.cfg.trainer.strategy,
                revision=model_revision,
            )
            # in-place patch
            self._seq_parallel_monkey_patch(model=wrapped_model.model)

            if self.cfg.trainer.gradient_checkpointing:
                wrapped_model.gradient_checkpointing_enable(
                    gradient_checkpointing_kwargs={
                        "use_reentrant": self.cfg.trainer.gradient_checkpointing_use_reentrant
                    }
                )

        self.model, self.optimizer, self.scheduler = strategy.prepare(
            (wrapped_model, None, None),
        )
        assert self.optimizer is not None and self.scheduler is not None, (
            "FSDP preparation should create optimizer and scheduler"
        )

        # Initialize weight extractor
        self.use_cuda_ipc = self.cfg.generator.weight_sync_backend == "nccl" and self.cfg.trainer.placement.colocate_all
        # TODO(haochen): Now module grouping (in order to support FlashRL) is only enabled for the CUDA IPC
        # transfer strategy, we can enable it for other strategies as well.
        self.weight_extractor = FSDPWeightExtractor(
            self.model.model,
            group_by_module=self.use_cuda_ipc,
            batch_size_threshold_gb=(
                self.cfg.generator.weight_transfer_threshold_cuda_ipc_GB if self.use_cuda_ipc else 0.0
            ),
            moe_grouped_gemm=bool(self.cfg.trainer.policy.fsdp_config.get("moe_grouped_gemm", False)),
            fuse_weights=(bool(self.cfg.generator.fuse_weights) or os.environ.get("SKYRL_FUSE_WEIGHTS", "0") == "1"),
        )

        self._maybe_start_host_ram_monitor()

    def _maybe_start_host_ram_monitor(self):
        """Start the periodic host-RAM / cgroup-mem reporter on the POLICY worker.

        Mirrors the fd-monitor already running on the RolloutCoordinator (gen) and
        the skyrl_entrypoint (head); the policy/training workers were the one path
        with NO host-RAM telemetry, so a cgroup-OOM there (e.g. the 80B naive-map
        GDN chunked-scan forward, which host-RAM-OOM'd a policy node at
        global_step 0) left the peak RAM UNMEASURED. This closes that gap: every
        policy NODE now emits peak RSS + cgroup mem vs the `--memory` cap.

        Gating (low-overhead, best-effort, never raises into init_model):
          * ONE rank per node only (``self._local_rank == 0``) — node mem/cgroup
            is shared across the 8 ranks on a node, so 8x logging would be spam.
          * env ``SKYRL_POLICY_HOST_RAM_MONITOR`` (default "1"; set "0" to disable).
          * env ``SKYRL_POLICY_HOST_RAM_MONITOR_INTERVAL`` seconds (default 60 —
            tighter than the fd-monitor's 120s default so the fast GDN-scan peak
            is sampled; <=0 disables).
        """
        try:
            if int(os.environ.get("SKYRL_POLICY_HOST_RAM_MONITOR", "1")) == 0:
                return
            if getattr(self, "_local_rank", None) != 0:
                return
            interval = int(os.environ.get("SKYRL_POLICY_HOST_RAM_MONITOR_INTERVAL", "60"))
            import socket

            from examples.terminal_bench.fd_monitor import start_fd_monitor

            logger.info(
                f"[policy-host-ram-monitor] starting on rank={self._rank} "
                f"host={socket.gethostname()} interval={interval}s"
            )
            # breakdown=True: emit the HOST_RAM_BREAKDOWN attribution line (cgroup
            # total bucketed across pids + shm + pinned) — the decisive signal for
            # the 80B gs1 host-RAM OOM. Only the policy monitor (this gated path)
            # requests it; the gen/head monitors stay byte-identical.
            self._host_ram_monitor_stop = start_fd_monitor(interval, breakdown=True)
        except Exception as e:  # pragma: no cover - best-effort telemetry
            logger.warning(f"[policy-host-ram-monitor] failed to start: {e}")

    async def _save_lora_adapters_and_sync(self, peft_model, lora_sync_path, inference_engine_client):
        """Collect LoRA parameters, save and call inference engine to load."""
        import json
        import os
        from dataclasses import asdict

        from safetensors.torch import save_file

        from skyrl_train.distributed.fsdp_utils import collect_lora_params

        lora_params = collect_lora_params(module=self.model.model)

        if torch.distributed.get_rank() == 0:
            os.makedirs(lora_sync_path, exist_ok=True)

            peft_config = asdict(peft_model.peft_config.get("default", {}))
            peft_config["task_type"] = peft_config["task_type"].value
            peft_config["peft_type"] = peft_config["peft_type"].value
            peft_config["target_modules"] = list(peft_config["target_modules"])

            # Save LoRA parameters and config
            save_file(lora_params, os.path.join(lora_sync_path, "adapter_model.safetensors"))
            with io.open(os.path.join(lora_sync_path, "adapter_config.json"), "w", encoding="utf-8") as f:
                json.dump(peft_config, f, ensure_ascii=False, indent=4)

            # Send LoRA disk loading request to inference engine. `lora_disk_load` is a specific identifier
            # to tell the inference engine to extract the `lora_disk_path`.
            lora_request = {
                "names": ["lora_disk_load"],
                "extras": [{"lora_disk_path": lora_sync_path}],
            }
            await inference_engine_client.update_named_weights(lora_request)

        torch.distributed.barrier()

    async def broadcast_to_inference_engines(self, inference_engine_client):
        use_prefix_cache = self.cfg.generator.enable_prefix_caching
        generator_dtype = str_to_torch_dtype(self.cfg.generator.model_dtype)
        cache_reset_task = None
        if use_prefix_cache and torch.distributed.get_rank() == 0:
            # clear prefix cache
            cache_reset_task = inference_engine_client.reset_prefix_cache()

        torch.cuda.empty_cache()

        # Check if this is a LoRA model
        peft_model = getattr(self.model.model, "_fsdp_wrapped_module", self.model.model)

        if self._is_lora:
            assert hasattr(peft_model, "peft_config"), "LoRA model should have peft_config"

            # assume base model is already synced, sync LoRA adapters
            lora_sync_path = self.cfg.trainer.policy.model.lora.lora_sync_path
            await self._save_lora_adapters_and_sync(peft_model, lora_sync_path, inference_engine_client)
            return

        # Extract weights using the initialized extractor
        import os

        _fuse_weights = os.environ.get("SKYRL_FUSE_WEIGHTS", "0") == "1"

        # #1685 fix (FlashInfer-CUTLASS w13 swap skipped on RL update -> MoE token-salad):
        # bracket the WHOLE multi-chunk sync with vLLM's layerwise reload so per-chunk
        # model.load_weights DEFER processing and a single finalize re-runs
        # process_weights_after_loading (re-applying swap_w13_to_w31) EXACTLY once. PROVEN
        # by the disagg kernel-format diag: without this the engine holds checkpoint
        # [gate;up] while the FlashInfer CUTLASS kernel reads [up;gate]. Inert (swap-wise)
        # on triton/dense backends, so byte-identical there. Gated by env for safety.
        _w13_bracket = (
            not self.use_cuda_ipc and not _fuse_weights and os.environ.get("SKYRL_W13_RELOAD_BRACKET", "1") == "1"
        )

        if not self.use_cuda_ipc:
            # Signal engines to start accumulating weights (for FP8 batched quantization)
            if _fuse_weights and torch.distributed.get_rank() == 0:
                await inference_engine_client.begin_weight_update()

            # Open the layerwise-reload bracket (rank 0 drives the engine RPC).
            if _w13_bracket and torch.distributed.get_rank() == 0:
                await inference_engine_client.begin_weight_reload()
            if _w13_bracket:
                torch.distributed.barrier()

            # Broadcast path: one chunk per parameter
            for chunk in self.weight_extractor.extract_weights(generator_dtype):
                # Each chunk contains one parameter
                assert len(chunk) == 1
                name = chunk.names[0]
                tensor = chunk.tensors[0]

                if torch.distributed.get_rank() == 0:
                    # Create legacy update request
                    update_weight_task = asyncio.create_task(
                        inference_engine_client.update_named_weights(
                            {
                                "names": [name],
                                "dtypes": chunk.dtypes,
                                "shapes": [list(tensor.shape)],
                            }
                        )
                    )

                # Broadcast tensor
                def broadcast_tensor(tensor):
                    if torch.distributed.get_rank() == 0:
                        torch.distributed.broadcast(tensor.data, 0, group=self._model_update_group)

                await asyncio.to_thread(broadcast_tensor, tensor)
                if torch.distributed.get_rank() == 0:
                    await update_weight_task
                torch.distributed.barrier()

            # Flush accumulated weights (triggers FP8 quantization on receiver)
            if _fuse_weights and torch.distributed.get_rank() == 0:
                await inference_engine_client.end_weight_update()

            # Close the layerwise-reload bracket: finalize_layerwise_reload re-runs
            # process_weights_after_loading over every layer ONCE -> re-applies the
            # FlashInfer-CUTLASS w13 [gate;up]->[up;gate] swap the per-chunk loads skipped.
            if _w13_bracket:
                torch.distributed.barrier()
                if torch.distributed.get_rank() == 0:
                    await inference_engine_client.finish_weight_reload()
        else:
            # CUDA IPC path: batched chunks (batching handled by extractor)
            from torch.multiprocessing.reductions import reduce_tensor

            # Iterate over batched chunks
            for chunk in self.weight_extractor.extract_weights(generator_dtype):
                weights_update_request = {"names": [], "dtypes": [], "shapes": [], "extras": [], "packed": False}

                # Process all parameters in this batch
                # TODO(haochen): Pack tensors into contiguous buffer before creating IPC handle
                # (like Megatron does) to reduce number of IPC handles and file descriptors
                for name, dtype_str, tensor, shape in zip(
                    chunk.names, chunk.dtypes, chunk.tensors, chunk.shapes, strict=True
                ):
                    # Create IPC handle for tensor
                    ipc_handle = reduce_tensor(tensor)
                    ipc_handle = {get_physical_gpu_id(): ipc_handle}
                    ipc_handle_list = [None] * torch.distributed.get_world_size()
                    torch.distributed.all_gather_object(ipc_handle_list, ipc_handle)

                    if torch.distributed.get_rank() == 0:
                        ipc_handles = {}
                        for d in ipc_handle_list:
                            ipc_handles.update(d)

                        weights_update_request["names"].append(name)
                        weights_update_request["dtypes"].append(dtype_str)
                        weights_update_request["shapes"].append(shape)
                        weights_update_request["extras"].append({"ipc_handles": ipc_handles})

                    torch.distributed.barrier()
                    torch.cuda.synchronize()

                # Send batch
                if torch.distributed.get_rank() == 0:
                    await inference_engine_client.update_named_weights(weights_update_request)
                    torch.cuda.ipc_collect()
                torch.distributed.barrier()
                torch.cuda.synchronize()

        if cache_reset_task is not None:
            await cache_reset_task
        torch.cuda.empty_cache()
        torch.distributed.barrier()

    def get_weight_statistics(self):
        """Compute lightweight statistics for model weights"""
        raise NotImplementedError()

    def _set_pad_token_id(self, pad_token_id):
        # NOTE (sumanthrh): self.model -> HFModelWrapper; self.model -> DeepSpeedEngine, self.model.module -> AutoModelForCausalLM
        self.model.model.config.pad_token_id = pad_token_id

    def _forward_impl(self, data: TrainingInputBatch) -> TrainingOutputBatch:
        """SYNC forward body (the heavy FSDP unshard + micro-batch forward + reshard).

        This is the exact pre-fix `forward` body. It is invoked either inline (flag
        OFF -> byte-identical to the old sync `forward`) or off the event-loop thread
        via `asyncio.to_thread` from the async `forward` entry (flag ON).
        """
        _phase_diagnostics.start_phase(_phase_diagnostics.CollectivePhase.FORWARD_IMPL_ENTER)
        output = super().forward(data)
        # unshard the root FSDP module (https://pytorch.org/docs/stable/notes/fsdp.html#fsdp-notes)
        if self._world_size > 1 and fsdp_version(self.model.model) == 1:
            self.model.model._handle.reshard(True)
        _phase_diagnostics.log_phase(_phase_diagnostics.CollectivePhase.FORWARD_IMPL_EXIT)
        return output

    async def forward(
        self,
        data: TrainingInputBatch,
    ) -> TrainingOutputBatch:
        """COOPERATIVELY-SCHEDULED forward entry (the 131k forward-dispatch wedge fix).

        FR-proven root cause (2026-06-30, EP8xFSDP2xCP2 @ 131k, gs-1): the FSDP policy
        worker is a Ray ASYNC actor (it defines `async def` methods like
        `broadcast_to_inference_engines`), so EVERY actor method runs on ONE asyncio
        event-loop thread. The dispatched per-step `forward` was a plain SYNC `def`.
        The prior `bdae17bb` weight-sync drain (`barrier_all` async) FIXED the 8k wedge
        (peers' loops were occupied by the broadcast coroutine; the drain unwound them
        -> all 32 ran the forward). At 131k it WEDGES one layer down: the drain still
        frees every peer loop (py-spy: ranks 16-23 IDLE in `select`, NOT in the
        broadcast coroutine), yet only rank 0 RAN the forward. `MeshDispatch.dispatch`
        provably issues `forward.remote()` to all 32 actors (code-proven, and now logged
        via MESH_DISPATCH), so this is NOT a keying bug — it is a SCHEDULING RACE: the
        fully-async trainer keeps OTHER coroutines (generator loops, staleness-driven
        weight-sync) running concurrently, so between the driver's drain completing and
        the SYNC `forward` task being scheduled on a peer's loop, that loop can be
        RE-OCCUPIED by another dispatched coroutine. A sync method that arrives while
        the loop is mid-coroutine is queued-not-scheduled and HOL-blocked exactly like
        the pre-drain forward; only rank 0 (loop still free) ran it, hit the lonely
        mesh_fsdp `_all_gather_base` unshard, and the 1800s NCCL watchdog SIGABRTed.

        WHY THE ASYNC ENTRY IS ROBUST TO BOTH MODES (the directly-analogous fix to the
        proven `barrier_all`): as an `async def`, this method is a COROUTINE TASK on the
        actor loop the instant Ray delivers it. The loop's ready-queue schedules a
        pending coroutine task regardless of what other coroutine is running (a running
        coroutine yields at its own `await` points — the broadcast/generator coroutines
        all `await`), so the forward task CANNOT be permanently head-of-line-blocked the
        way a sync method is. `await asyncio.sleep(0)` forces it through one full loop
        turn so it is provably picked up even if it arrived mid-cycle. The heavy sync
        body then runs via `asyncio.to_thread` so the blocking FSDP unshard collective
        does not re-occupy the single event-loop thread (mirrors `barrier_all`). If the
        next run's MESH_DISPATCH log were instead to reveal a keying bug, the async entry
        is still correct (the dispatch already targets all 32) and harmless.

        CORRECTNESS / THREAD-SAFETY: the forward body is INFERENCE-ONLY
        (`torch.no_grad()` + `torch.autocast`, no autograd graph is built -> no
        cross-thread autograd-engine affinity concern; backward/optimizer live in the
        separate `ppo_train` method, untouched). The whole forward (all micro-batches +
        reshard) runs in ONE `to_thread` call, so any per-call thread-local CUDA/autocast
        state is set and used on the same worker thread. CUDA ops enqueue to the device's
        current stream irrespective of host thread, and they are ordered on that stream;
        the forward is the SOLE in-flight op per the upstream drain. This off-event-loop
        pattern is already established in this codebase: the driver runs
        `fwd_logprobs_values_reward` (which `ray.get`s these forwards) in
        `asyncio.to_thread`, and `barrier_all` runs its CUDA sync+barrier in `to_thread`.
        Ray transparently awaits async actor methods, so the driver's `ray.get(refs)`
        still returns the `TrainingOutputBatch` unchanged -> no driver-side change.

        Gated behind SKYRL_FORWARD_DISPATCH_FIX (default ON). Flag OFF -> the sync body
        runs INLINE -> byte-identical to the pre-fix forward (still an async method, but
        no yield / no thread hop, so a single-rank / non-racy run is unaffected).
        """
        # UNGATED per-rank rendezvous marker: we MUST see this on all 32 ranks on the
        # next run (mirrors WORKER_FORWARD_ENTER, which fires deeper inside the sync
        # body). Pre-fix only rank 0 reached the forward; with this fix every dispatched
        # rank's coroutine task is scheduled, so all 32 must log this.
        logger.info(f"WORKER_FORWARD_DISPATCH_RDV rank={self._rank}")
        metadata = data.metadata or {}
        diagnostic_metadata = _phase_diagnostics.CollectiveRegionMetadata(global_step=metadata.get("global_step"))
        diagnostic_mesh = self.strategy.device_mesh if _phase_diagnostics.enabled() else None
        with _phase_diagnostics.region(
            diagnostic_mesh,
            kind=_phase_diagnostics.CollectiveRegionKind.POLICY_INFERENCE_FORWARD,
            rank=self._rank,
            metadata=diagnostic_metadata,
        ):
            _phase_diagnostics.log_phase(_phase_diagnostics.CollectivePhase.FORWARD_ENTER)
            try:
                if os.environ.get("SKYRL_FORWARD_DISPATCH_FIX", "1") != "1":
                    return self._forward_impl(data)
                # Yield so this dispatched coroutine is guaranteed a loop turn and is scheduled
                # even if it arrived while the loop was mid-cycle servicing another coroutine.
                await asyncio.sleep(0)
                # Run the heavy sync FSDP forward off the event-loop thread so the blocking
                # unshard collective cannot re-occupy the loop (head-of-line-block a peer).
                return await asyncio.to_thread(self._forward_impl, data)
            finally:
                _phase_diagnostics.log_phase(_phase_diagnostics.CollectivePhase.FORWARD_EXIT)


class FSDPCriticWorkerBase(CriticWorkerBase):
    def offload_to_cpu(self, pin_memory=True, non_blocking=True, offload_optimizer=True, offload_model=True):
        self._set_numa_affinity(torch.distributed.get_rank() % torch.cuda.device_count())
        self.strategy.offload_to_cpu(
            self.model, self.optimizer, pin_memory, non_blocking, offload_optimizer, offload_model
        )

    def backload_to_gpu(self, non_blocking=True, backload_optimizer=True, backload_model=True):
        self.strategy.backload_to_gpu(self.model, self.optimizer, non_blocking, backload_optimizer, backload_model)

    def init_model(self, model_path, num_training_steps: int = None):
        assert self.cfg.trainer.strategy in ("fsdp", "fsdp2")
        strategy = FSDPStrategy(
            fsdp_config=self.cfg.trainer.critic.fsdp_config,
            optimizer_config=self.cfg.trainer.critic.optimizer_config,
            fsdp_strategy=self.cfg.trainer.strategy,
            seed=self.cfg.trainer.seed,
            micro_train_batch_size_per_gpu=self.cfg.trainer.micro_train_batch_size_per_gpu,
            num_training_steps=num_training_steps,
        )
        strategy.setup_distributed()
        self.strategy = strategy

        # Stage 3: surface the CP submesh/group on the worker (None when cp_size==1).
        self.cp_mesh = getattr(strategy, "cp_mesh", None)
        self.cp_group = getattr(strategy, "cp_group", None)

        # Update per-gpu mini batch size based on device mesh
        self._normalize_mini_batch_size()

        model_config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
        init_context = get_init_weight_context_manager(
            use_meta_tensor=not model_config.tie_word_embeddings, mesh=self.strategy.device_mesh
        )
        with init_context():
            critic = get_llm_for_sequence_regression(
                model_path,
                "critic",
                use_flash_attention_2=self.cfg.trainer.flash_attn,
                # NOTE (sumanthrh): Model initialization should always be in fp32
                # during training
                bf16=True,
                lora_rank=self.cfg.trainer.critic.model.lora.rank,
                lora_alpha=self.cfg.trainer.critic.model.lora.alpha,
                lora_dropout=self.cfg.trainer.critic.model.lora.dropout,
                target_modules=self.cfg.trainer.critic.model.lora.target_modules,
                exclude_modules=self.cfg.trainer.critic.model.lora.exclude_modules,
                value_head_prefix=self.cfg.trainer.algorithm.value_head_prefix,
                init_value_head=self.cfg.trainer.policy.model.path == self.cfg.trainer.critic.model.path,
                sequence_parallel_size=self.cfg.trainer.critic.sequence_parallel_size,
                use_sample_packing=self.cfg.trainer.use_sample_packing,
                attn_backend=self.cfg.trainer.get("attn_backend", "auto"),
                context_parallel_size=int(self.cfg.trainer.critic.fsdp_config.get("context_parallel_size", 1)),
                # Stage 4: value forward must CP-shard identically (G3). None at cp=1.
                cp_mesh=self.cp_mesh,
                cp_rotate_method=str(self.cfg.trainer.critic.fsdp_config.get("cp_rotate_method", "allgather")),
            )
            self._seq_parallel_monkey_patch(model=critic, use_parent_class=True)

            if self.cfg.trainer.gradient_checkpointing:
                critic.gradient_checkpointing_enable(
                    gradient_checkpointing_kwargs={
                        "use_reentrant": self.cfg.trainer.gradient_checkpointing_use_reentrant
                    }
                )

        # prepare models/optimizers...
        self.model, self.optimizer, self.scheduler = strategy.prepare(
            (critic, None, None),
        )
        assert self.optimizer is not None

    def _forward_impl(self, data: TrainingInputBatch) -> TrainingOutputBatch:
        """SYNC critic forward body (pre-fix `forward`). See FSDPPolicyWorkerBase.forward."""
        output = super().forward(data)
        # unshard the root FSDP module (https://pytorch.org/docs/stable/notes/fsdp.html#fsdp-notes)
        if self._world_size > 1 and fsdp_version(self.model.model) == 1:
            self.model.model._handle.reshard(True)
        return output

    async def forward(
        self,
        data: TrainingInputBatch,
    ) -> TrainingOutputBatch:
        """Cooperatively-scheduled critic forward entry (131k forward-dispatch wedge fix).

        Identical mechanism/justification to FSDPPolicyWorkerBase.forward (see its
        docstring): async entry + `await asyncio.sleep(0)` yield + `asyncio.to_thread`
        of the sync body so the dispatched forward task on this Ray async actor cannot
        be head-of-line-blocked by a concurrently-running coroutine. Inference-only body
        (no autograd graph). Gated SKYRL_FORWARD_DISPATCH_FIX (default ON); flag OFF ->
        inline sync body (byte-identical to pre-fix).
        """
        logger.info(f"WORKER_FORWARD_DISPATCH_RDV rank={self._rank}")
        if os.environ.get("SKYRL_FORWARD_DISPATCH_FIX", "1") != "1":
            return self._forward_impl(data)
        await asyncio.sleep(0)
        return await asyncio.to_thread(self._forward_impl, data)


class FSDPRefWorkerBase(RefWorkerBase):
    def offload_to_cpu(self, pin_memory=True, non_blocking=True, **kwargs):
        self._set_numa_affinity(torch.distributed.get_rank() % torch.cuda.device_count())
        self.strategy.offload_to_cpu(self.model, None, pin_memory, non_blocking)

    def backload_to_gpu(self, non_blocking=True, **kwargs):
        self.strategy.backload_to_gpu(self.model, None, non_blocking)

    def init_model(self, model_path):
        assert self.cfg.trainer.strategy in ("fsdp", "fsdp2")
        strategy = FSDPStrategy(
            fsdp_config=self.cfg.trainer.ref.fsdp_config,
            fsdp_strategy=self.cfg.trainer.strategy,
            seed=self.cfg.trainer.seed,
            micro_train_batch_size_per_gpu=self.cfg.trainer.micro_train_batch_size_per_gpu,
        )
        strategy.setup_distributed()
        self.strategy = strategy

        # Stage 3: surface the CP submesh/group on the worker (None when cp_size==1).
        self.cp_mesh = getattr(strategy, "cp_mesh", None)
        self.cp_group = getattr(strategy, "cp_group", None)

        model_config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
        init_context = get_init_weight_context_manager(
            use_meta_tensor=not model_config.tie_word_embeddings, mesh=self.strategy.device_mesh
        )

        with init_context():
            wrapped_model = HFModelWrapper(
                model_path,
                use_flash_attention_2=self.cfg.trainer.flash_attn,
                bf16=self.cfg.trainer.bf16,
                sequence_parallel_size=self.cfg.trainer.ref.sequence_parallel_size,
                use_sample_packing=self.cfg.trainer.use_sample_packing,
                rope_scaling=get_rope_scaling_config(self.cfg.trainer),
                rope_theta=get_rope_theta_config(self.cfg.trainer),
                attn_backend=self.cfg.trainer.get("attn_backend", "auto"),
                context_parallel_size=int(self.cfg.trainer.ref.fsdp_config.get("context_parallel_size", 1)),
                # Stage 4: ref-logprob forward must CP-shard identically to the
                # policy so KL aligns post-unshard (G3). None at cp=1.
                cp_mesh=self.cp_mesh,
                cp_rotate_method=str(self.cfg.trainer.ref.fsdp_config.get("cp_rotate_method", "allgather")),
                training_strategy=self.cfg.trainer.strategy,
            )
            self._seq_parallel_monkey_patch(model=wrapped_model.model)

        self.model = strategy.prepare(wrapped_model)
        self.model.eval()

    def _forward_impl(self, data: TrainingInputBatch) -> TrainingOutputBatch:
        """SYNC ref forward body (pre-fix `forward`). See FSDPPolicyWorkerBase.forward."""
        output = super().forward(data)
        # unshard the root FSDP module (https://pytorch.org/docs/stable/notes/fsdp.html#fsdp-notes)
        if self._world_size > 1 and fsdp_version(self.model.model) == 1:
            self.model.model._handle.reshard(True)
        return output

    async def forward(
        self,
        data: TrainingInputBatch,
    ) -> TrainingOutputBatch:
        """Cooperatively-scheduled ref forward entry (131k forward-dispatch wedge fix).

        Identical mechanism/justification to FSDPPolicyWorkerBase.forward (see its
        docstring): async entry + `await asyncio.sleep(0)` yield + `asyncio.to_thread`
        of the sync body. Inference-only body. Gated SKYRL_FORWARD_DISPATCH_FIX
        (default ON); flag OFF -> inline sync body (byte-identical to pre-fix).
        """
        logger.info(f"WORKER_FORWARD_DISPATCH_RDV rank={self._rank}")
        if os.environ.get("SKYRL_FORWARD_DISPATCH_FIX", "1") != "1":
            return self._forward_impl(data)
        await asyncio.sleep(0)
        return await asyncio.to_thread(self._forward_impl, data)


# Ray remote actors
PolicyWorker = ray.remote(num_gpus=1)(FSDPPolicyWorkerBase)
CriticWorker = ray.remote(num_gpus=1)(FSDPCriticWorkerBase)
RefWorker = ray.remote(num_gpus=1)(FSDPRefWorkerBase)
