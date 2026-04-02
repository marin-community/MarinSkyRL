"""Utility functions for weight extraction."""

import os
from collections import defaultdict
from typing import Dict, List, Callable, Iterator, Any
import torch

from skyrl_train.weight_sync import WeightChunk

import logging
logger = logging.getLogger(__name__)

# vLLM fuses certain layers (gate+up → gate_up_proj, q+k+v → qkv_proj).
# When SKYRL_FUSE_WEIGHTS=1, we fuse policy weights before syncing so shapes
# match the inference engine.  This is required for FP8 quantized engines
# but safe to enable for BF16 too (vLLM always uses fused layers).
_FUSE_WEIGHTS = os.environ.get("SKYRL_FUSE_WEIGHTS", "0") == "1"

# Mapping: fused_name_suffix -> list of source suffixes (in concat order)
_FUSE_RULES = {
    "qkv_proj": ["q_proj", "k_proj", "v_proj"],
    "gate_up_proj": ["gate_proj", "up_proj"],
}


def _maybe_fuse_module_weights(
    module_names: List[str],
    module_tensors: List[torch.Tensor],
    module_shapes: List[List[int]],
    module_dtypes: List[str],
) -> tuple:
    """Fuse separate proj weights into vLLM's packed format.

    For example, gate_proj.weight + up_proj.weight → gate_up_proj.weight
    Only active when SKYRL_FUSE_WEIGHTS=1.
    Returns (names, tensors, shapes, dtypes) with fused entries replacing originals.
    """
    if not _FUSE_WEIGHTS:
        return module_names, module_tensors, module_shapes, module_dtypes

    # Index tensors by their short name (e.g. "gate_proj.weight")
    name_to_idx = {}
    for i, name in enumerate(module_names):
        parts = name.split(".")
        if len(parts) >= 2:
            short = f"{parts[-2]}.{parts[-1]}"  # e.g. "gate_proj.weight"
            name_to_idx[short] = i

    fused_names = []
    fused_tensors = []
    fused_shapes = []
    fused_dtypes = []
    consumed = set()

    for fused_suffix, source_suffixes in _FUSE_RULES.items():
        # Check weight and bias separately
        for param_type in ["weight", "bias"]:
            source_keys = [f"{s}.{param_type}" for s in source_suffixes]
            indices = [name_to_idx.get(k) for k in source_keys]

            if all(idx is not None for idx in indices):
                # All source tensors present — fuse them
                tensors_to_cat = [module_tensors[idx] for idx in indices]
                fused_tensor = torch.cat(tensors_to_cat, dim=0).contiguous()

                # Build fused name: replace source suffix with fused suffix
                base = ".".join(module_names[indices[0]].split(".")[:-2])
                fused_name = f"{base}.{fused_suffix}.{param_type}"

                fused_names.append(fused_name)
                fused_tensors.append(fused_tensor)
                fused_shapes.append(list(fused_tensor.shape))
                fused_dtypes.append(module_dtypes[indices[0]])

                for idx in indices:
                    consumed.add(idx)

    # Add non-fused params unchanged
    for i in range(len(module_names)):
        if i not in consumed:
            fused_names.append(module_names[i])
            fused_tensors.append(module_tensors[i])
            fused_shapes.append(module_shapes[i])
            fused_dtypes.append(module_dtypes[i])

    if consumed:
        logger.debug(
            f"Fused {len(consumed)} params into {len(consumed) - len([n for n in fused_names if any(s in n for s in _FUSE_RULES)])} "
            f"packed params for module {module_names[0].rsplit('.', 2)[0]}"
        )

    return fused_names, fused_tensors, fused_shapes, fused_dtypes


def yield_module_grouped_chunks(
    params: Dict[str, Any],
    dtype: torch.dtype,
    gather_tensor_fn: Callable[[Any], torch.Tensor],
    get_shape_fn: Callable[[str, Any, torch.Tensor], List[int]],
    batch_size_threshold_gb: float = 0.0,
) -> Iterator[WeightChunk]:
    """Yield WeightChunk objects grouped by module.

    This helper function eliminates duplication between different weight extractors
    that need to group parameters by module (e.g., for FlashRL QKV fusion).

    Groups parameters by their parent module by removing the last two components
    from the parameter name. For example:
    "model.layers.0.self_attn.q_proj.weight" -> "model.layers.0.self_attn"

    When SKYRL_FUSE_WEIGHTS=1, fuses gate_proj+up_proj→gate_up_proj and
    q_proj+k_proj+v_proj→qkv_proj to match vLLM's packed module format.
    This enables FP8 quantized inference with weight sync.

    Args:
        params: Dictionary mapping parameter names to parameter objects
        dtype: Target dtype for inference
        gather_tensor_fn: Backend-specific function to gather sharded tensors into full tensors
        get_shape_fn: Function to extract shape from param_name, param, and prepared tensor
        batch_size_threshold_gb: If > 0, batch complete modules together until threshold is reached

    Yields:
        WeightChunk objects containing all parameters for each module (or batched modules if threshold set)
    """
    if _FUSE_WEIGHTS:
        logger.info("SKYRL_FUSE_WEIGHTS=1: will fuse gate/up and q/k/v weights for vLLM packed format")

    # Group parameters by module for FlashRL
    # NOTE (sumanthrh): We sync weights module by module. Ex: weights for self attn together, weights for mlp together
    # For FlashRL integration, we allocate new storage for each param. Since q, k and v layer weights are fused internally by vllm,
    # we need to pass the weights for all of these together.
    # Overall, this doesn't hurt perf even in the general case
    module_to_params: Dict[str, List[str]] = defaultdict(list)
    for param_name in params.keys():
        # Extract module name (e.g., "model.layers.0.self_attn" from "model.layers.0.self_attn.q_proj.weight")
        # TODO (sumanthrh): When would this fail? Works for many AutoModelForCausalLM models for now
        module_name = ".".join(param_name.split(".")[:-2])
        module_to_params[module_name].append(param_name)

    # Accumulate complete modules until threshold reached
    batch_tensors = []
    batch_names = []
    batch_shapes = []
    batch_dtypes = []
    current_size = 0
    threshold_bytes = batch_size_threshold_gb * 1024**3

    for module_name, param_names in module_to_params.items():
        module_tensors = []
        module_names = []
        module_shapes = []
        module_dtypes = []
        module_size = 0

        # Prepare all tensors for this module
        # TODO: Allow gather_tensor_fn to accept a list of params for batched gathering.
        # This would be more efficient for DeepSpeed ZeRO-3 where GatheredParameters
        # can gather multiple params in a single all-gather collective.
        for param_name in param_names:
            param = params[param_name]
            tensor = gather_tensor_fn(param)
            tensor = tensor.to(dtype).detach().contiguous()
            shape = get_shape_fn(param_name, param, tensor)
            module_tensors.append(tensor)
            module_names.append(param_name)
            module_shapes.append(shape)
            module_dtypes.append(str(dtype))
            module_size += tensor.nbytes

        # Fuse weights if enabled (gate+up → gate_up_proj, q+k+v → qkv_proj)
        module_names, module_tensors, module_shapes, module_dtypes = _maybe_fuse_module_weights(
            module_names, module_tensors, module_shapes, module_dtypes
        )
        module_size = sum(t.nbytes for t in module_tensors)

        # Check if adding this module would exceed threshold
        if current_size > 0 and current_size + module_size > threshold_bytes:
            # Yield current batch before adding this module
            yield WeightChunk(
                names=batch_names,
                dtypes=batch_dtypes,
                shapes=batch_shapes,
                tensors=batch_tensors,
            )
            # Start new batch
            batch_tensors = []
            batch_names = []
            batch_shapes = []
            batch_dtypes = []
            current_size = 0

        # Add module to current batch
        batch_tensors.extend(module_tensors)
        batch_names.extend(module_names)
        batch_shapes.extend(module_shapes)
        batch_dtypes.extend(module_dtypes)
        current_size += module_size

    # Yield final batch if non-empty
    if batch_tensors:
        yield WeightChunk(
            names=batch_names,
            dtypes=batch_dtypes,
            shapes=batch_shapes,
            tensors=batch_tensors,
        )
