# This code is adapted from VERL
# https://github.com/volcengine/verl/blob/main/verl/utils/fsdp_utils.py
# The original copyright is reproduced below:
# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import functools
from contextlib import nullcontext
from typing import Union

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed import DeviceMesh
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp._runtime_utils import _lazy_init
from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy, transformer_auto_wrap_policy
from transformers.trainer_pt_utils import get_module_class_from_name
from torch.distributed.device_mesh import init_device_mesh
from collections import OrderedDict

from packaging import version
from peft.utils.save_and_load import get_peft_model_state_dict

if version.parse(torch.__version__) >= version.parse("2.6"):
    from torch.distributed.fsdp import CPUOffloadPolicy, FSDPModule, MixedPrecisionPolicy, fully_shard
elif version.parse(torch.__version__) >= version.parse("2.4"):
    from torch.distributed._composable.fsdp import CPUOffloadPolicy, FSDPModule, MixedPrecisionPolicy, fully_shard
else:
    fully_shard, MixedPrecisionPolicy, FSDPModule, CPUOffloadPolicy = None, None, None, None


def init_fn(x: torch.nn.Module):
    if torch.distributed.get_rank() != 0:
        x = x.to_empty(device=torch.cuda.current_device(), recurse=False)
        torch.cuda.empty_cache()
    return x


def get_init_weight_context_manager(use_meta_tensor=True, mesh: DeviceMesh = None):
    from accelerate import init_empty_weights

    def cpu_init_weights():
        return torch.device("cpu")

    if use_meta_tensor:
        if mesh is None:
            init_context = init_empty_weights if torch.distributed.get_rank() != 0 else cpu_init_weights
        else:
            init_context = init_empty_weights if mesh.get_coordinate()[-1] != 0 else cpu_init_weights
    else:
        init_context = cpu_init_weights
    return init_context


def get_fsdp_wrap_policy(module, config=None, is_lora=False):
    """Get FSDP wrap policy for the module.

    Args:
        module: The module to get wrap policy for
        config: Configuration for wrap policy
        is_lora: Whether to enable lambda policy for LoRA modules
    """
    if config is None:
        config = {}

    def _get_attr(attr_name, default_value=None):
        if hasattr(config, "get"):
            return config.get(attr_name, default_value)
        else:
            return getattr(config, attr_name, default_value)

    if _get_attr("disable", False):
        return None

    default_transformer_cls_names_to_wrap = getattr(module, "_no_split_modules", None)
    fsdp_transformer_layer_cls_to_wrap = _get_attr(
        "transformer_layer_cls_to_wrap", default_transformer_cls_names_to_wrap
    )
    min_num_params = _get_attr("min_num_params", 0)
    auto_wrap_policy = None

    policies = []

    from torch.distributed.fsdp.wrap import _or_policy, lambda_auto_wrap_policy

    # Add lambda policy for LoRA modules if is_lora is True
    if is_lora:

        def lambda_policy_fn(module):
            return bool(
                len(list(module.named_children())) == 0
                and getattr(module, "weight", None) is not None
                and module.weight.requires_grad
            )

        lambda_policy = functools.partial(lambda_auto_wrap_policy, lambda_fn=lambda_policy_fn)
        policies.append(lambda_policy)

    if min_num_params > 0:
        size_policy = functools.partial(size_based_auto_wrap_policy, min_num_params=min_num_params)
        policies.append(size_policy)
    elif fsdp_transformer_layer_cls_to_wrap is not None:
        transformer_cls_to_wrap = set()
        for layer_class in fsdp_transformer_layer_cls_to_wrap:
            transformer_cls = get_module_class_from_name(module, layer_class)
            if transformer_cls is None:
                raise Exception("Could not find the transformer layer class to wrap in the model.")
            else:
                transformer_cls_to_wrap.add(transformer_cls)

        transformer_policy = functools.partial(
            transformer_auto_wrap_policy,
            transformer_layer_cls=transformer_cls_to_wrap,
        )
        policies.append(transformer_policy)

    if len(policies) > 0:
        auto_wrap_policy = functools.partial(_or_policy, policies=policies)

    return auto_wrap_policy


@torch.no_grad()
def offload_fsdp_model_to_cpu(model: FSDP, empty_cache: bool = True):
    if fsdp_version(model) == 2:
        offload_fsdp2_model_to_cpu(model, empty_cache)
        return

    assert isinstance(model, FSDP)
    # lazy init FSDP model
    _lazy_init(model, model)
    assert model._is_root, "Only support root model offloading to CPU"
    for handle in model._all_handles:
        if handle._offload_params:
            continue
        flat_param = handle.flat_param
        assert (
            flat_param.data.data_ptr() == flat_param._local_shard.data_ptr()
            and id(flat_param.data) != id(flat_param._local_shard)
            and flat_param.data.size() == flat_param._local_shard.size()
        )
        handle.flat_param_to(torch.device("cpu"), non_blocking=True)
        # the following still keeps id(._local_shard) != id(.data)
        flat_param._local_shard = flat_param.data
        assert id(flat_param._local_shard) != id(flat_param.data)
    if empty_cache:
        torch.cuda.empty_cache()


@torch.no_grad()
def offload_fsdp2_model_to_cpu(model, empty_cache: bool = True):
    model.to("cpu", non_blocking=True)
    if empty_cache:
        torch.cuda.empty_cache()


@torch.no_grad()
def load_fsdp_model_to_gpu(model: FSDP):
    if fsdp_version(model) == 2:
        load_fsdp2_model_to_gpu(model)
        return

    assert isinstance(model, FSDP)
    # lazy init FSDP model
    _lazy_init(model, model)
    assert model._is_root, "Only support root model loading to GPU"
    device_id = torch.cuda.current_device()
    for handle in model._all_handles:
        if handle._offload_params:
            continue
        flat_param = handle.flat_param
        handle.flat_param_to(torch.device(f"cuda:{device_id}"), non_blocking=True)
        # the following still keeps id(._local_shard) != id(.data)
        flat_param._local_shard = flat_param.data


@torch.no_grad()
def load_fsdp2_model_to_gpu(model):
    device = torch.cuda.current_device()
    model.to(device, non_blocking=True)


@torch.no_grad()
def offload_fsdp_optimizer(optimizer):
    if not optimizer.state:
        return
    for param_group in optimizer.param_groups:
        for param in param_group["params"]:
            state = optimizer.state[param]
            for key, value in state.items():
                if isinstance(value, torch.Tensor):
                    state[key] = value.to("cpu", non_blocking=True)


@torch.no_grad()
def load_fsdp_optimizer(optimizer, device_id):
    if not optimizer.state:
        return
    for param_group in optimizer.param_groups:
        for param in param_group["params"]:
            state = optimizer.state[param]
            for key, value in state.items():
                if isinstance(value, torch.Tensor):
                    state[key] = value.to(device_id, non_blocking=True)


def fsdp_version(model):
    if isinstance(model, FSDP):
        return 1
    elif FSDPModule is not None and isinstance(model, FSDPModule):
        return 2
    else:
        return 0


def get_fsdp_state_ctx(model, state_type, state_cfg, optim_cfg):
    if fsdp_version(model) == 1:
        return FSDP.state_dict_type(model, state_type, state_cfg, optim_cfg)
    else:
        return nullcontext()


# Fsdp2 load full state dict from `accelerate`
# Reference: https://github.com/huggingface/accelerate/blob/0af621bbecc0e43f5d43766a4945d3d2236bb8a9/src/accelerate/utils/fsdp_utils.py#L455
# NOTE (sumanthrh): The original code from `accelerate` assumes init on meta device - with cpu init only on rank 0, but the code is compatible with cpu init on all ranks.
def fsdp2_load_full_state_dict(model: torch.nn.Module, full_sd: dict, cpu_offload=None, ep_enabled=False):
    """
    Loads the full state dict (could be only on rank 0) into the sharded model. This is done by broadcasting the
    parameters from rank 0 to all other ranks. This function modifies the model in-place.

    Args:
        model (`torch.nn.Module`):
            The model to load the state dict into, expected to be on meta device or a VRAM spike can occur
        full_sd (`dict`): The full state dict to load, can be only on rank 0
        ep_enabled (`bool`): Whether expert parallelism is active. When True, the model has a MIX of params
            sharded over the global FSDP mesh AND expert params sharded over a (fsdp, ep) submesh. The naive
            per-param `broadcast(global)` + `distribute_tensor(submesh)` path used for the non-EP case
            deadlocks on that mix: per param it interleaves a global broadcast with a submesh-scoped collective
            (distribute_tensor scatters from mesh-coordinate 0), so ranks that are coordinate-0 on one mesh but
            not another desync → NCCL store->get wait timeout. When ep_enabled we delegate to the documented
            FSDP2 full-state-dict loader `torch.distributed.checkpoint.state_dict.set_model_state_dict`
            (broadcast_from_rank0=True), the same robust loader torchtitan uses: it broadcasts the rank-0 full
            state dict and re-shards EACH param to its OWN DTensor mesh / placement automatically, so mixed
            global + (fsdp,ep)-submesh params are handled uniformly with no manual per-param
            broadcast/distribute_tensor/set_data dance.

            NOTE on the historical deadlock: an earlier attempt at this `set_model_state_dict` path hung
            because, at that time, rank 0 held REAL CPU-initialized weights while the other ranks were
            meta-initialized. `_load_model_state_dict` infers the broadcast device from the MODEL's local
            params, so the rank0/non-rank0 real-vs-meta split made device inference asymmetric (rank0 → CPU,
            others → CUDA) → mismatched gloo/nccl backend in broadcast_object_list → timeout. That is no longer
            possible: the caller (`_fsdp_init_model`) now meta-izes ALL ranks' params uniformly before
            apply_ep/apply_fsdp2, so every rank's local params are meta DTensors at load time and the inferred
            broadcast device is identical (the default PG device) across ranks.

            DEFAULT False keeps the a3 (non-EP) production path byte-identical.
    """
    import torch.distributed as dist
    from torch.distributed.tensor import distribute_tensor

    if ep_enabled:
        # Documented, robust FSDP2 full-state-dict loader (torchtitan-style). It broadcasts the
        # rank-0 full state dict and re-shards each param to its OWN DTensor mesh / placement
        # automatically — handling mixed global + (fsdp,ep)-submesh + meta-init params — which
        # eliminates the manual per-param broadcast / distribute_tensor / set_data dance and the
        # whole set_data/meta-copy error class that came with it.
        #
        # Precondition (guaranteed by the caller _fsdp_init_model): ALL ranks' model params are
        # uniformly meta DTensors at this point, so set_model_state_dict's broadcast-device
        # inference (which reads the MODEL's local params) is symmetric across ranks. full_sd holds
        # the real weights on rank 0 and is empty ({}) on the other ranks, which is exactly what
        # broadcast_from_rank0=True expects.
        import os as _os
        import sys as _sys

        _dbg = _os.environ.get("SKYRL_EP_LOADER_DEBUG", "") == "1"

        # ------------------------------------------------------------------
        # STREAMED EP full-state-dict load (80B GPU-0 init OOM fix).
        #
        # WHY NOT torch's set_model_state_dict(broadcast_from_rank0=True):
        # that loader is already per-param (it comments "Broadcast every tensor
        # to avoid OOM for now"), but for EACH param it does
        # `full_state[key].detach().to(cuda)` to stage the WHOLE param on GPU-0
        # before the global broadcast. For an 80B grouped-MoE expert param (dim-0
        # = num_experts, fused w1/w2/w3) that single tensor alone exceeds GPU-0's
        # free VRAM (<1 GiB observed on jobs 605185/606619/607073) → init OOM at
        # _broadcast_state_dict / _broadcast_tensors.
        #
        # THIS loader assembles each param's FULL tensor on CPU one dim-0 CHUNK at
        # a time (rank-0 stages one chunk → GPU, GLOBAL-PG broadcast, every rank
        # copies the chunk into a CPU full buffer, frees the GPU chunk), then
        # extracts ONLY this rank's local shard from the CPU full tensor using
        # each placement's OWN `_split_tensor` in mesh order — the exact local
        # decomposition `distribute_tensor` performs, but WITHOUT its collective
        # and WITHOUT ever putting the full tensor on GPU. `_split_tensor` is
        # overridden by `_StridedShard` (the placement FSDP2 emits when a tensor
        # dim is sharded by BOTH the ep and fsdp mesh dims, as the grouped experts
        # are), so the shard layout is byte-identical to `distribute_tensor` for
        # plain Shard, _StridedShard, AND Replicate. Peak GPU usage is therefore
        # ONE dim-0 chunk + this rank's (already-sharded) local shard — the full
        # ~2 GiB unsharded grouped-expert tensor is NEVER on GPU on any rank.
        #
        # Collective-safe: the ONLY collective is the per-chunk global-PG
        # broadcast; every rank drives it in identical key+chunk order (the
        # _split_tensor extraction is purely local). No submesh-scoped collective
        # is interleaved, so the historical global-vs-submesh desync cannot recur.
        # ------------------------------------------------------------------
        from torch.distributed.tensor import DTensor

        rank = dist.get_rank()
        device = torch.device("cuda", torch.cuda.current_device())

        # Per-broadcast row budget along dim 0. The grouped-expert params are the
        # only ones large enough to matter; a small budget caps the GPU transient.
        # Override via env for finer granularity if even one chunk is too large.
        max_rows = int(_os.environ.get("SKYRL_EP_LOADER_CHUNK_ROWS", "8"))
        if max_rows < 1:
            max_rows = 1

        def _extract_local_shard(full_cpu, dtensor_meta):
            """Reproduce distribute_tensor's LOCAL scatter result for this rank.

            Walks mesh dims in order; for each, splits the running tensor with the
            placement's own `_split_tensor` (so _StridedShard is honored) and keeps
            this rank's coordinate slice. Returns the local-shard CPU tensor.
            """
            mesh = dtensor_meta.device_mesh
            placements = dtensor_meta.placements
            coord = mesh.get_coordinate()  # this rank's coord per mesh dim
            cur = full_cpu
            for mesh_dim, placement in enumerate(placements):
                if placement.is_shard():
                    num_chunks = mesh.size(mesh_dim)
                    shards, _ = placement._split_tensor(
                        cur, num_chunks, with_padding=False, contiguous=True
                    )
                    cur = shards[coord[mesh_dim]]
                # Replicate / Partial: no narrowing on this mesh dim.
            return cur.contiguous()

        meta_sharded_sd = model.state_dict()
        # LIVE registered params, keyed by the SAME names load_state_dict(assign=True)
        # validates against. Used by the loader shape assert (B1) below so a stale /
        # aliased state_dict() snapshot that disagrees with the live param (the
        # ep-only-vs-composed divergence behind `start+length exceeds`) is caught with
        # a precise message instead of opaquely at `assign`.
        live_params = dict(model.named_parameters())
        new_sd = {}

        # Deterministic, all-ranks-identical iteration order.
        for key in meta_sharded_sd.keys():
            local_state = meta_sharded_sd[key]

            # Rank 0 holds the real (CPU) source; other ranks have nothing.
            src = full_sd.get(key, None) if rank == 0 else None
            if rank == 0 and src is None:
                raise RuntimeError(f"[EP-LOADER] missing key on rank 0: {key}")

            # Shape/dtype come from the local (meta) param — identical on all ranks.
            full_shape = tuple(local_state.shape)
            dtype = local_state.dtype
            is_dt = isinstance(local_state, DTensor)

            # Assemble the FULL tensor on CPU, chunk by chunk, broadcasting from
            # rank 0. GPU only ever holds one chunk at a time.
            full_cpu = torch.empty(full_shape, dtype=dtype, device="cpu")
            if len(full_shape) == 0:  # 0-D scalar
                gpu = torch.empty((), device=device, dtype=dtype)
                if rank == 0:
                    gpu.copy_(src.detach().to(device=device, dtype=dtype))
                dist.broadcast(gpu, src=0)
                full_cpu.copy_(gpu.cpu())
                del gpu
            else:
                nrows = full_shape[0]
                rows_per_chunk = max(1, min(max_rows, nrows))
                start = 0
                while start < nrows:
                    end = min(start + rows_per_chunk, nrows)
                    if rank == 0:
                        gpu = src[start:end].detach().to(device=device, dtype=dtype, copy=True)
                    else:
                        gpu = torch.empty((end - start,) + full_shape[1:], device=device, dtype=dtype)
                    dist.broadcast(gpu, src=0)
                    full_cpu[start:end].copy_(gpu.cpu())
                    del gpu
                    start = end
            torch.cuda.empty_cache()

            # Extract this rank's local shard (LOCAL, no collective) and place on GPU.
            if is_dt:
                local_cpu = _extract_local_shard(full_cpu, local_state)
                # Loader shape assert (B1): the assembled local shard MUST match the
                # LIVE registered param's local shape — i.e. exactly what
                # `load_state_dict(assign=True)` narrows the new tensor into. We compare
                # against `live_params[key]` (named_parameters), NOT the possibly stale /
                # aliased `state_dict()` snapshot `local_state` we extracted with, so a
                # snapshot-vs-live placement divergence (the ep-only 1-D snapshot vs the
                # 2-D composed live param that produces `start(0)+length(N) exceeds N//fsdp`)
                # is caught with a precise, keyed message here instead of opaquely at the
                # `assign` narrow. No-op on the correct Qwen / 80B paths (snapshot==live,
                # shapes already match) and on the non-DTensor branch below.
                live_p = live_params.get(key, None)
                expected_local_shape = (
                    tuple(live_p.to_local().shape)
                    if isinstance(live_p, DTensor)
                    else tuple(local_state.to_local().shape)
                )
                assert tuple(local_cpu.shape) == expected_local_shape, (
                    f"[EP-LOADER] {key}: assembled local shard {tuple(local_cpu.shape)} != "
                    f"live registered param local shape {expected_local_shape}; "
                    f"snapshot placements={local_state.placements}, live placements="
                    f"{getattr(live_p, 'placements', None)}, mesh_dims="
                    f"{getattr(local_state.device_mesh, 'mesh_dim_names', None)}. "
                    f"Expert param is likely ep-sharded but not fsdp-composed (1-D meta) "
                    f"— see apply_ep's composition assert (A)."
                )
                local_gpu = local_cpu.to(device=device, dtype=dtype)
                new_sd[key] = DTensor.from_local(
                    local_gpu,
                    local_state.device_mesh,
                    local_state.placements,
                    shape=local_state.shape,
                    stride=local_state.stride(),
                )
                del local_cpu
            else:
                new_sd[key] = full_cpu.to(device=device, dtype=dtype)

            del full_cpu

        if _dbg:
            print(
                f"[EP-LOADER-DBG] rank={rank} streamed-load assembled {len(new_sd)} params "
                f"(chunk_rows={max_rows}); calling load_state_dict(assign=True)",
                file=_sys.stderr,
                flush=True,
            )

        # assign=True: params are meta DTensors, replace storage in-place.
        model.load_state_dict(new_sd, assign=True)
        del new_sd

        if _dbg:
            print(f"[EP-LOADER-DBG] rank={rank} streamed-load returned cleanly", file=_sys.stderr, flush=True)

        # Mirror the non-EP path's CPU<->GPU offload dance to keep reserved memory bounded.
        offload_fsdp2_model_to_cpu(model)
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        if not cpu_offload:
            load_fsdp2_model_to_gpu(model)
        return model

    # Model was previously copied to meta device
    meta_sharded_sd = model.state_dict()
    sharded_sd = {}

    # Rank 0 distributes the full state dict to other ranks
    def _infer_parameter_dtype(model, param_name, empty_param):
        try:
            old_param = model.get_parameter_or_buffer(param_name)
        except AttributeError:
            # Need this for LORA, as there some params are not *parameters* of sorts
            base_param_name, local_param_name = param_name.rsplit(".", 1)
            submodule = model.get_submodule(base_param_name)
            old_param = getattr(submodule, local_param_name)

        is_torch_e4m3fn_available = hasattr(torch, "float8_e4m3fn")
        casting_dtype = None
        is_param_float8_e4m3fn = is_torch_e4m3fn_available and empty_param.dtype == torch.float8_e4m3fn

        if empty_param.dtype.is_floating_point and not is_param_float8_e4m3fn:
            casting_dtype = old_param.dtype

        return old_param is not None and old_param.is_contiguous(), casting_dtype

    def _cast_and_contiguous(tensor, to_contiguous, dtype):
        if dtype is not None:
            tensor = tensor.to(dtype=dtype)
        if to_contiguous:
            tensor = tensor.contiguous()
        return tensor

    if dist.get_rank() == 0:
        for (param_name, full_param), sharded_param in zip(full_sd.items(), meta_sharded_sd.values()):
            full_param = full_param.detach().cuda()
            mesh = sharded_param.device_mesh
            dist.broadcast(full_param, src=0)
            sharded_tensor = distribute_tensor(full_param, mesh, sharded_param.placements)
            to_contiguous, casting_dtype = _infer_parameter_dtype(
                model,
                param_name,
                full_param,
            )
            sharded_tensor = _cast_and_contiguous(sharded_tensor, to_contiguous, casting_dtype)
            sharded_sd[param_name] = sharded_tensor
    # We need this else to have a matching `broadcast` for all of the ranks, else we deadlock
    else:
        for param_name, sharded_param in meta_sharded_sd.items():
            full_tensor = torch.empty(sharded_param.size(), device="cuda", dtype=sharded_param.dtype)
            mesh = sharded_param.device_mesh
            dist.broadcast(full_tensor, src=0)
            sharded_tensor = distribute_tensor(full_tensor, mesh, sharded_param.placements)
            to_contiguous, casting_dtype = _infer_parameter_dtype(
                model,
                param_name,
                full_tensor,
            )
            sharded_tensor = _cast_and_contiguous(sharded_tensor, to_contiguous, casting_dtype)
            sharded_sd[param_name] = sharded_tensor

    # we set `assign=True` because our params can be on meta device
    model.load_state_dict(sharded_sd, assign=True)

    # If we don't offload FSDP2 Module to CPU and then back to GPU,
    # it will occupy a large amount of reserved GPU memory，which can not be released using torch.cuda.empty_cache()
    # even if we are using cpu_offload
    # TODO (erictang000): this requires an additional offload + backload, see if this can be avoided
    # Credit: https://github.com/volcengine/verl/pull/1667
    offload_fsdp2_model_to_cpu(model)

    torch.cuda.synchronize()
    torch.cuda.empty_cache()

    if not cpu_offload:
        load_fsdp2_model_to_gpu(model)
    return model


def fsdp2_get_full_state_dict(model: torch.nn.Module, cpu_offload=True, rank0_only=True):
    """
    Get the full state dict from an FSDP2 model using proper PyTorch FSDP2 APIs.
    This function will gather the complete state dict on rank 0 only by default.

    Args:
        model (`torch.nn.Module`): The FSDP2 model to get state dict from
        cpu_offload (`bool`): Whether to offload to CPU
        rank0_only (`bool`): Whether to gather full state dict only on rank 0

    Returns:
        dict: The full state dict (only on rank 0 if rank0_only=True, empty dict on other ranks)
    """
    from torch.distributed.checkpoint.state_dict import StateDictOptions, get_model_state_dict

    # All ranks must participate in the collective operation
    options = StateDictOptions(
        full_state_dict=True, cpu_offload=cpu_offload, broadcast_from_rank0=False  # We want to get, not set
    )

    # This must be called on all ranks for the collective operation to work
    state_dict = get_model_state_dict(model, options=options)

    # If rank0_only is True, clear the state_dict on non-rank-0 processes
    if rank0_only and dist.get_rank() != 0:
        # Clear the state dict on non-rank-0 processes to save memory
        state_dict.clear()

    return state_dict


def apply_fsdp2(model, fsdp_kwargs, config):
    """model: AutoModelForCausalLM"""
    assert CPUOffloadPolicy is not None, "PyTorch version >= 2.4 is required for using fully_shard API (FSDP2)"
    default_transformer_cls_names_to_wrap = getattr(model, "_no_split_modules", None)
    fsdp_transformer_layer_cls_to_wrap = config.get("wrap_policy", {}).get(
        "transformer_layer_cls_to_wrap", default_transformer_cls_names_to_wrap
    )

    if isinstance(fsdp_transformer_layer_cls_to_wrap, str):
        fsdp_transformer_layer_cls_to_wrap = [fsdp_transformer_layer_cls_to_wrap]
    # HF returns `_no_split_modules` as a SET for some archs (e.g. Qwen3-Next); the
    # small-MoE models tested earlier returned a list. Normalize any non-str iterable
    # to a list so the indexing below (and `in` membership later) is well-defined.
    elif fsdp_transformer_layer_cls_to_wrap is not None and not isinstance(
        fsdp_transformer_layer_cls_to_wrap, (list, tuple)
    ):
        fsdp_transformer_layer_cls_to_wrap = list(fsdp_transformer_layer_cls_to_wrap)

    assert len(fsdp_transformer_layer_cls_to_wrap) > 0 and fsdp_transformer_layer_cls_to_wrap[0] is not None

    modules = []
    for name, module in model.named_modules():
        if module.__class__.__name__ in fsdp_transformer_layer_cls_to_wrap or (
            isinstance(module, nn.Embedding) and not model.config.tie_word_embeddings
        ):
            modules.append(module)

    for idx, module in enumerate(modules):
        fully_shard(module, **fsdp_kwargs)
    fully_shard(model, **fsdp_kwargs)  # fsdp2 will not reshard_after_forward for root module


def apply_ep(model, device_mesh, ep_comm_backend="torch", sequence_parallel_size=1, fsdp_kwargs=None):
    """Shard MoE experts across the ``ep`` submesh via torchtitan ``ExpertParallel``.

    Stage 4a — torch ``all_to_all`` backend only (NO DeepEP; that is Stage 5) and
    ETP==1 (plain ``ExpertParallel``, not ``ExpertTensorParallel``). For each lifted
    ``GroupedMoEShim.moe.experts`` (the ``GroupedExperts`` w1/w2/w3 holder) this:

      * ``Shard(0)``-s every expert param over ``device_mesh["ep"]`` (each rank holds
        ``num_experts // ep_size`` experts) via ``parallelize_module`` — while the
        params are still PLAIN tensors (torchtitan's ``_partition_fn`` calls
        ``distribute_tensor`` onto the ep mesh, which rejects an already-DTensor input);
      * when ``fsdp_kwargs`` is given, immediately ``fully_shard``-s the same experts
        module on the ``fsdp`` submesh, composing a second ``Shard`` dim of the SAME
        root mesh → net 2-D expert DTensors ``[Shard(0)_ep, Shard_fsdp]``. Doing the
        experts' ``fully_shard`` here (not leaving it to the parent decoder layer's
        ``fully_shard`` in ``apply_fsdp2``) makes the 2-D composition explicit and lets
        the parent layer's wrap nest correctly (FSDP2 excludes already-wrapped children);
      * installs ``ExpertParallel._token_dispatch`` / ``_token_combine`` all_to_all
        hooks on the ``experts`` module boundary (the autograd ``_A2A`` carries grads
        symmetrically on the backward).

    The router gate + the forced-index override fire BEFORE any token movement, so
    router replay is preserved by construction (scope §3). Returns the number of
    expert modules sharded.

    Must be called BEFORE ``apply_fsdp2`` (so EP runs on plain params) and before the
    full-state-dict load so the load distributes weights into their final EP+FSDP
    placement.
    """
    assert ep_comm_backend in ("torch", "deepep"), (
        f"ep_comm_backend must be 'torch' or 'deepep'; got {ep_comm_backend!r}"
    )
    assert sequence_parallel_size == 1, (
        "SP+EP is deferred (scope §5): apply_ep requires sequence_parallel_size==1"
    )

    from torch.distributed.tensor.parallel import parallelize_module

    # torch (Stage 4) → torchtitan ExpertParallel (installs all_to_all hooks +
    # @expert_parallel grouped-mm). deepep (Stage 5) → DeepEPExpertParallel (Shard(0)
    # only; dispatch/combine is driven from MoE.forward). Imported lazily so the base
    # / torch path never imports deep_ep and the deepep path never needs torchtitan.
    if ep_comm_backend == "deepep":
        from skyrl_train.distributed.expert_parallel import DeepEPExpertParallel

        ep_plan = DeepEPExpertParallel()
    else:
        from torchtitan.distributed.expert_parallel import ExpertParallel

        ep_plan = ExpertParallel()

    # Matcher relaxation (EP=2xFSDP=2 OLMoE grouped-expert load bug): match the
    # expert holder by `isinstance(experts, GroupedExperts)` AS WELL AS the legacy
    # `__class__.__name__ == "GroupedExperts"` string check. ALL supported archs
    # (Qwen3-MoE, Qwen3-Next, OLMoE, Mixtral) build their expert holder as the SAME
    # `skyrl_train.models.layers.moe.GroupedExperts` (MoE.__init__ -> self.experts =
    # GroupedExperts(...)); there is NO sibling/subclass holder today. We use the
    # `isinstance OR name` UNION (not isinstance alone) deliberately so the match is
    # a STRICT SUPERSET of the prior name-check and cannot regress on either axis:
    #   * isinstance also catches any FUTURE GroupedExperts subclass (a name-only
    #     check would silently miss a subclass -> ep-only 1-D leak -> the
    #     `length(N) exceeds N/fsdp` load crash this fix targets);
    #   * the name fallback survives module-import duplication (two import paths for
    #     GroupedExperts would defeat a bare isinstance but keep the name equal).
    # It never broadens to non-expert modules (only `.moe.experts` that are
    # GroupedExperts / subclasses match), so it is byte-identical on EP=1 (apply_ep
    # not called) and on the working Qwen EP x FSDP paths (match already fired).
    from skyrl_train.models.layers.moe import GroupedExperts

    ep_mesh = device_mesh["ep"]
    fsdp_mesh = device_mesh["fsdp"]
    sharded = 0
    for module in model.modules():
        # The lifted grouped block exposes `moe.experts` (a GroupedExperts holding
        # w1/w2/w3). Match the shim's `moe` attribute to find expert holders.
        moe = getattr(module, "moe", None)
        if moe is None:
            continue
        experts = getattr(moe, "experts", None)
        if experts is None or not (
            isinstance(experts, GroupedExperts) or experts.__class__.__name__ == "GroupedExperts"
        ):
            continue
        parallelize_module(experts, device_mesh=ep_mesh, parallelize_plan=ep_plan)
        # Compose the FSDP Shard dim on the fsdp submesh → 2-D expert DTensors.
        if fsdp_kwargs is not None:
            # FAIL-FAST: when EP AND FSDP both shard the expert dim, each EP-rank
            # holds (num_experts // ep_size) experts, which FSDP then shards over
            # fsdp_size. If that is uneven, FSDP2 even-pads the param/optimizer
            # local shard while the EP-backward grad stays unpadded → the Adam
            # `lerp_` raises `size of tensor a (N) must match b (N-1) at dim 0` at
            # the step-1 optimizer step (job 674574: fsdp_size=6, 64/6 uneven).
            # Catch the invalid geometry at init with a clear message instead.
            num_experts = getattr(experts, "num_experts", None)
            ep_size = ep_mesh.size()
            fsdp_size = fsdp_mesh.size()
            if num_experts is not None and ep_size > 1 and fsdp_size > 1:
                experts_per_ep_rank = num_experts // ep_size
                assert num_experts % ep_size == 0, (
                    f"num_experts={num_experts} must be divisible by ep_size={ep_size}"
                )
                assert experts_per_ep_rank % fsdp_size == 0, (
                    f"fsdp_size={fsdp_size} must divide num_experts//ep_size="
                    f"{experts_per_ep_rank} (num_experts={num_experts}, ep_size={ep_size}); "
                    f"uneven expert shard → FSDP2 pads the local optimizer shard but the "
                    f"EP-backward grad is unpadded → Adam dim-0 mismatch at the step-1 "
                    f"optimizer step. Choose an fsdp_size that divides {experts_per_ep_rank}."
                )
            ep_fsdp_kwargs = {k: v for k, v in fsdp_kwargs.items() if k != "mesh"}
            fully_shard(experts, mesh=fsdp_mesh, **ep_fsdp_kwargs)
            # Composition assert (A): when EP AND FSDP both shard the expert dim,
            # `fully_shard(experts)` MUST have composed a 2-D (fsdp, ep) DTensor on
            # top of the ep `parallelize_module` Shard(0). If a future arch's holder
            # leaves the param EP-sharded-only (1-D `(Shard(0),)`, num_experts//ep
            # rows), the streamed loader `fsdp2_load_full_state_dict` would faithfully
            # assemble that 1-D shard and crash opaquely at `load_state_dict(assign=True)`
            # with `start(0)+length(num_experts//ep) exceeds dimension size(num_experts//ep//fsdp)`.
            # Fail LOUD here at wrap time instead. No-op on the working Qwen / 80B
            # paths (always 2-D) and skipped entirely unless ep>1 AND fsdp>1.
            if ep_size > 1 and fsdp_size > 1:
                e_per = None
                if num_experts is not None:
                    e_per = num_experts // ep_size // fsdp_size
                for _pn, _p in experts.named_parameters(recurse=False):
                    _pls = getattr(_p, "placements", ())
                    assert len(_pls) == 2 and all(getattr(pl, "is_shard", lambda: False)() for pl in _pls), (
                        f"EP+FSDP expert param {_pn} did not compose to a 2-D (fsdp,ep) "
                        f"sharded DTensor (got placements={_pls}); apply_ep's "
                        f"fully_shard(experts) did not reach this holder for this arch. "
                        f"This is the EP-only 1-D leak that triggers the loader "
                        f"`length(...) exceeds ...` crash."
                    )
                    if e_per is not None:
                        _local_rows = _p.to_local().shape[0]
                        assert _local_rows == e_per, (
                            f"EP+FSDP expert param {_pn} local rows {_local_rows} != "
                            f"num_experts//ep//fsdp = {e_per} "
                            f"(num_experts={num_experts}, ep_size={ep_size}, fsdp_size={fsdp_size})."
                        )
        # Tell the grouped block which comm backend to run. For deepep this also
        # switches GroupedExperts.forward to the local-experts (.to_local) path and
        # MoE.forward to the DeepEP dispatch/combine branch.
        moe.set_ep_comm_backend(ep_comm_backend)
        # Flag the grouped block so its forward selects the EP-decorated compute path.
        moe._ep_enabled = True
        sharded += 1
    return sharded


def fsdp2_clip_grad_norm_(parameters, max_norm, norm_type=2.0, error_if_nonfinite=False, foreach=None):
    """torch.nn.utils.clip_grad_norm_ can't run on cpu parameter DTensor.

    Stage 6: under expert parallelism the parameter set spans MULTIPLE device
    meshes — EP-sharded grouped-expert grads are DTensors on the 2-D
    ``(fsdp, ep)`` mesh, while non-expert grads are on the 1-D ``(fsdp)`` mesh.
    ``_get_total_norm`` ultimately ``torch.stack``-s the per-grad partial norms,
    and ``aten.stack`` rejects operands on different meshes
    (``ValueError: All operands in aten.stack.default must have the same mesh``).
    So we group grads by their grad's device mesh, reduce each group to a plain
    *replicated* scalar via ``full_tensor()`` (which all-reduces the
    ``_NormPartial`` across that group's mesh), then combine the per-group
    ``p``-norms into one global scalar: ``total = (sum_g norm_g ** p) ** (1/p)``
    (and ``max`` for ``p == inf``). The single combined scalar then clips every
    grad. The non-EP path (all grads on one mesh, or plain tensors) is unchanged.
    """
    from torch.nn.utils.clip_grad import _clip_grads_with_norm_, _get_total_norm

    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    else:
        # prevent generators from being exhausted
        parameters = list(parameters)
    grads = [p.grad for p in parameters if p.grad is not None]

    # cpu_offload (CPUOffloadPolicy, required to fit the 80B): params/grads are
    # CPU-resident. The norm computation below all-reduces a DTensor _NormPartial
    # (and full_tensor()) over the param's device mesh; with CPU tensors that
    # collective hits the process group's CPU backend, which is not registered
    # (the worker pg is nccl-only) → "No backend type associated with device
    # type cpu". So when grads live on CPU, compute the NORM over CUDA copies of
    # the grads (the all-reduce then runs on nccl). DTensor.to(cuda) moves the
    # local shard while preserving the mesh/placement, so the reduction semantics
    # are identical — only the backend differs. The CLIP is still applied to the
    # ORIGINAL (cpu) grads in-place: _clip_grads_with_norm_ moves the scalar clip
    # coefficient to each grad's device, so a cuda total_norm scales cpu grads
    # correctly. cpu_offload=false (8B / a3 / ablation policy paths) keeps grads
    # on cuda, ``grads_on_cpu`` is False, and this is byte-identical to before.
    grads_on_cpu = any(g.device.type == "cpu" for g in grads)
    if grads_on_cpu:
        cuda_device = torch.cuda.current_device()
        norm_grads = [g.to(cuda_device) for g in grads]
    else:
        norm_grads = grads

    # Group grads by device mesh (DTensors only). Plain tensors / non-DTensors
    # collect under a single ``None`` key. EP introduces >1 distinct mesh.
    # Grouping uses ``norm_grads`` (cuda copies under cpu_offload) so the
    # per-mesh norm reductions below run on the nccl backend.
    mesh_groups: dict = {}
    for g in norm_grads:
        mesh = getattr(g, "device_mesh", None)
        mesh_groups.setdefault(mesh, []).append(g)

    if len(mesh_groups) <= 1:
        # Single mesh (or all plain tensors): today's path, byte-identical when
        # grads are on cuda (norm_grads is grads). Under cpu_offload norm_grads
        # are the cuda copies so the _NormPartial all-reduce runs on nccl.
        total_norm = _get_total_norm(norm_grads, norm_type, error_if_nonfinite, foreach)
        total_norm = total_norm.to(torch.cuda.current_device(), non_blocking=True)
        # Clip the ORIGINAL parameters' grads (cpu under cpu_offload);
        # _clip_grads_with_norm_ moves the cuda total_norm to each grad's device.
        _clip_grads_with_norm_(parameters, max_norm, total_norm, foreach)
        return total_norm

    # Multi-mesh (EP): reduce each mesh-group to a replicated plain scalar, then
    # combine the per-group p-norms into a single global scalar.
    norm_type = float(norm_type)
    device = torch.cuda.current_device()
    group_norms = []
    for group_grads in mesh_groups.values():
        gn = _get_total_norm(group_grads, norm_type, error_if_nonfinite, foreach)
        # full_tensor() all-reduces the _NormPartial across THIS group's mesh,
        # yielding a plain (non-DTensor) replicated scalar.
        gn_full = gn.full_tensor() if hasattr(gn, "full_tensor") else gn
        group_norms.append(gn_full.to(device, non_blocking=True))

    stacked = torch.stack([gn.reshape(()) for gn in group_norms])
    if norm_type == float("inf"):
        total_norm = stacked.max()
    else:
        total_norm = stacked.pow(norm_type).sum().pow(1.0 / norm_type)

    # Apply the clip PER mesh-group. _clip_grads_with_norm_ (foreach) batches the
    # scale-multiply into a single aten._foreach_mul_ over ALL grads, which again
    # mixes the (fsdp) and (fsdp,ep) meshes ("Could not run pointwise computation
    # across different mesh"). Clipping each group's params separately keeps every
    # _foreach_mul_ within a single mesh. The scale factor is the SAME global
    # total_norm for all groups (correct: it's one global clip coefficient).
    params_by_mesh: dict = {}
    for p in parameters:
        if p.grad is None:
            continue
        mesh = getattr(p.grad, "device_mesh", None)
        params_by_mesh.setdefault(mesh, []).append(p)
    for group_params in params_by_mesh.values():
        _clip_grads_with_norm_(group_params, max_norm, total_norm, foreach)
    return total_norm


def create_device_mesh(world_size, fsdp_size, ep_size=1, device_type="cuda"):
    """Build the FSDP2 device mesh.

    ``ep_size <= 1`` (the default / a3-production path) is UNCHANGED — the today
    1-D ``["fsdp"]`` or 2-D ``["ddp","fsdp"]`` mesh, byte-identical to before EP
    (Stage 4 flag-off requirement G4-0).

    ``ep_size > 1`` (Stage 4a expert parallelism) builds a 3-D
    ``["ddp","fsdp","ep"]`` mesh of shape ``(ddp, fsdp_size, ep_size)`` where
    ``ddp = world_size // (ep_size * fsdp_size)``. Experts shard over the ``ep``
    submesh; non-expert params shard over the ``fsdp`` submesh. E.g.
    ``create_device_mesh(4, 2, ep_size=2)`` → ``(1, 2, 2)``.

    The ``fsdp`` dim is placed BEFORE ``ep`` deliberately: an EP-sharded expert param
    is later ``fully_shard``-ed on the ``fsdp`` submesh, producing a 2-D expert DTensor
    that FSDP2 internally slices as ``("fsdp", "ep")``. ``DeviceMesh._get_slice_mesh_dims``
    requires those root-dim indices to be ascending, so ``fsdp`` (idx 1) must precede
    ``ep`` (idx 2). The reverse order raised
    ``KeyError: ... Mesh dim indices should be in ascending order``.
    """
    if ep_size <= 1:
        if fsdp_size < 0 or fsdp_size >= world_size:
            device_mesh = init_device_mesh(device_type, mesh_shape=(world_size,), mesh_dim_names=["fsdp"])
        else:
            device_mesh = init_device_mesh(
                device_type, mesh_shape=(world_size // fsdp_size, fsdp_size), mesh_dim_names=["ddp", "fsdp"]
            )
        return device_mesh

    # Expert parallelism (Stage 4a): 3-D ["ddp", "fsdp", "ep"] mesh (fsdp before ep so
    # the composed 2-D expert DTensor slices in ascending root-dim order).
    fsdp = world_size if (fsdp_size < 0 or fsdp_size >= world_size) else fsdp_size
    assert world_size % ep_size == 0, f"world_size={world_size} not divisible by ep_size={ep_size}"
    assert world_size % fsdp == 0, f"world_size={world_size} not divisible by fsdp_size={fsdp}"
    assert (world_size % (ep_size * fsdp)) == 0, (
        f"world_size={world_size} not divisible by ep_size*fsdp_size={ep_size * fsdp}"
    )
    ddp = world_size // (ep_size * fsdp)
    device_mesh = init_device_mesh(
        device_type, mesh_shape=(ddp, fsdp, ep_size), mesh_dim_names=["ddp", "fsdp", "ep"]
    )
    return device_mesh


def get_sharding_strategy(device_mesh):
    from torch.distributed.fsdp import ShardingStrategy

    if device_mesh.ndim == 1:
        sharding_strategy = ShardingStrategy.FULL_SHARD
    elif device_mesh.ndim in (2, 3):
        sharding_strategy = ShardingStrategy.HYBRID_SHARD
    else:
        raise NotImplementedError(f"Get device mesh ndim={device_mesh.ndim}, but only support 1, 2 or 3")
    return sharding_strategy


"""
Adapted from Cruise.
"""

HALF_LIST = [16, "16", "fp16", "float16", torch.float16]
FLOAT_LIST = [32, "32", "fp32", "float32", torch.float32]
BFLOAT_LIST = ["bf16", "bfloat16", torch.bfloat16]


class PrecisionType:
    """Type of precision used.

    >>> PrecisionType.HALF == 16
    True
    >>> PrecisionType.HALF in (16, "16")
    True
    """

    HALF = "16"
    FLOAT = "32"
    FULL = "64"
    BFLOAT = "bf16"
    MIXED = "mixed"

    @staticmethod
    def supported_type(precision: Union[str, int]) -> bool:
        return any(x == precision for x in PrecisionType)

    @staticmethod
    def supported_types() -> list[str]:
        return [x.value for x in PrecisionType]

    @staticmethod
    def is_fp16(precision):
        return precision in HALF_LIST

    @staticmethod
    def is_fp32(precision):
        return precision in FLOAT_LIST

    @staticmethod
    def is_bf16(precision):
        return precision in BFLOAT_LIST

    @staticmethod
    def to_dtype(precision):
        if precision in HALF_LIST:
            return torch.float16
        elif precision in FLOAT_LIST:
            return torch.float32
        elif precision in BFLOAT_LIST:
            return torch.bfloat16
        else:
            raise RuntimeError(f"unexpected precision: {precision}")

    @staticmethod
    def to_str(precision):
        if precision == torch.float16:
            return "fp16"
        elif precision == torch.float32:
            return "fp32"
        elif precision == torch.bfloat16:
            return "bf16"
        else:
            raise RuntimeError(f"unexpected precision: {precision}")


# Reference: https://github.com/volcengine/verl/blob/main/verl/utils/fsdp_utils.py
def layered_summon_lora_params(fsdp_module) -> OrderedDict:

    def __prefix_submodules(module, prefix):
        for name, submodule in module.named_modules():
            if name.startswith(prefix) and "." not in name[len(prefix) :]:
                yield name, submodule

    lora_params = OrderedDict()
    prefix_list = [
        # fsdp
        "_fsdp_wrapped_module.base_model.model.",
        "_fsdp_wrapped_module.base_model.model.model.",
        "_fsdp_wrapped_module.base_model.model.model.layers.",
        "_fsdp_wrapped_module.base_model.model.model.language_model.layers.",
        # fsdp2
        "base_model.model.",
        "base_model.model.model.",
        "base_model.model.model.layers.",
        "base_model.model.model.language_model.layers.",
    ]
    peft_model = getattr(fsdp_module, "_fsdp_wrapped_module", fsdp_module)
    for prefix in prefix_list:
        for name, submodule in __prefix_submodules(fsdp_module, prefix):
            prefix = name.replace("_fsdp_wrapped_module.base_model.model.", "base_model.model.")
            if name.endswith(".model") or name.endswith(".layers"):
                continue
            if fsdp_version(submodule) > 0:
                with FSDP.summon_full_params(submodule, writeback=False):
                    sub_lora_params = get_peft_model_state_dict(peft_model, state_dict=submodule.state_dict())
                    sub_lora_params = {
                        f"{prefix}.{name}": (
                            param.full_tensor().detach().cpu()
                            if hasattr(param, "full_tensor")
                            else param.detach().cpu()
                        )
                        for name, param in sub_lora_params.items()
                    }
                    lora_params.update(sub_lora_params)
                    submodule._is_root = False
                torch.cuda.empty_cache()
    return lora_params


def collect_lora_params(module: FSDP) -> OrderedDict:
    """
    collect lora params or full params if base model is not ready in vllm
    requires `module._fsdp_wrapped_module` to be a `PeftModel`
    """
    lora_params = OrderedDict()
    peft_model = getattr(module, "_fsdp_wrapped_module", module)
    if fsdp_version(module) > 0:
        with FSDP.summon_full_params(module, writeback=False):
            # If base model is synced, we can get the full state dict from peft model
            lora_params = get_peft_model_state_dict(peft_model)
            lora_params = {
                name: param.full_tensor().detach().cpu() if hasattr(param, "full_tensor") else param.detach().cpu()
                for name, param in lora_params.items()
            }
        torch.cuda.empty_cache()
    else:
        lora_params = get_peft_model_state_dict(peft_model)
    return lora_params
