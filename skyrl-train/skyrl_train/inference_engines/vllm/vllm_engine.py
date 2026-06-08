import os
import threading
from typing import List, Any, Dict, Optional, Tuple, Iterator
from dataclasses import dataclass
from loguru import logger
from http import HTTPStatus
import ray
import torch
import asyncio
import vllm
from types import SimpleNamespace
from vllm import SamplingParams
from vllm.inputs import TokensPrompt
# vLLM 0.16+ reorganized entrypoints into sub-packages.
# Try new paths first, fall back to old paths for backwards compatibility.
try:
    # vLLM >= 0.16
    from vllm.entrypoints.openai.chat_completion.serving import OpenAIServingChat
    from vllm.entrypoints.openai.completion.serving import OpenAIServingCompletion
    from vllm.entrypoints.openai.models.serving import OpenAIServingModels
    from vllm.entrypoints.openai.models.protocol import BaseModelPath
    from vllm.entrypoints.openai.chat_completion.protocol import (
        ChatCompletionRequest,
        ChatCompletionResponse,
    )
    from vllm.entrypoints.openai.completion.protocol import (
        CompletionRequest,
        CompletionResponse,
    )
    from vllm.entrypoints.openai.engine.protocol import ErrorResponse
except ImportError:
    # vLLM < 0.16 (old flat layout)
    from vllm.entrypoints.openai.serving_chat import OpenAIServingChat
    from vllm.entrypoints.openai.serving_completion import OpenAIServingCompletion
    from vllm.entrypoints.openai.serving_models import BaseModelPath, OpenAIServingModels
    from vllm.entrypoints.openai.protocol import (
        ChatCompletionRequest,
        ChatCompletionResponse,
        ErrorResponse,
        CompletionRequest,
        CompletionResponse,
    )

try:
    from vllm.v1.metrics.loggers import LoggingStatLogger
except ImportError:
    LoggingStatLogger = None  # Not available in all vLLM versions
from vllm.lora.request import LoRARequest
from torch.distributed import destroy_process_group
from skyrl_train.distributed.utils import init_custom_process_group
from uuid import uuid4
import warnings
from skyrl_train.inference_engines.base import (
    InferenceEngineInterface,
    InferenceEngineInput,
    InferenceEngineOutput,
    NamedWeightsUpdateRequest,
)
from skyrl_train.weight_sync import WeightLoader
from skyrl_train.inference_engines.vllm.utils import pop_openai_kwargs
from loguru import logger
from skyrl_train.utils import str_to_torch_dtype, get_tcp_url
import time
from packaging import version


def _parse_vllm_version() -> version.Version:
    """Parse vllm.__version__, treating 'dev' or other invalid strings as 999.0.0."""
    try:
        return version.Version(vllm.__version__)
    except version.InvalidVersion:
        return version.parse("999.0.0")


def _build_error_response(message: str, type_phrase: str, code: int) -> Dict[str, Any]:
    """Build an OpenAI-style ErrorResponse dict, robust to vLLM's ErrorInfo move.

    vLLM >= 0.10 wraps the error fields in a nested ``ErrorInfo``; older vLLM put
    them flat on ``ErrorResponse``. vLLM 0.16 ALSO relocated ``ErrorInfo`` out of
    the flat ``vllm.entrypoints.openai.protocol`` module (which no longer exists)
    into ``vllm.entrypoints.openai.engine.protocol`` — importing the old path
    raised ``ModuleNotFoundError`` inside the engine's request-error handler
    (vllm_engine.py:1591), turning every recoverable per-request error into an
    unhandled crash. Try the new sub-package path first, then the old flat path,
    then fall back to the flat-field ErrorResponse for pre-0.10 vLLM.
    """
    ErrorInfo = None
    try:  # vLLM >= 0.16 (sub-package layout, same module as ErrorResponse)
        from vllm.entrypoints.openai.engine.protocol import ErrorInfo  # type: ignore
    except ImportError:
        try:  # vLLM 0.10–0.15 (flat layout)
            from vllm.entrypoints.openai.protocol import ErrorInfo  # type: ignore
        except ImportError:
            ErrorInfo = None

    if ErrorInfo is not None:
        return ErrorResponse(
            error=ErrorInfo(message=message, type=type_phrase, code=code),
        ).model_dump()
    # pre-0.10 vLLM: flat fields directly on ErrorResponse.
    return ErrorResponse(message=message, type=type_phrase, code=code).model_dump()


@dataclass
class Logprob:
    logprob: float
    rank: int
    token_id: str


def setup_envvars_for_vllm(kwargs, bundle_indices):
    noset_visible_devices = kwargs.pop("noset_visible_devices")
    os.environ["VLLM_USE_FLASHINFER_SAMPLER"] = "0"  # TODO(Charlie): may not be needed.

    # When custom all-reduce is disabled (e.g. for TP=2 on H100 where
    # SymmMemCommunicator rendezvous fails), also disable symmetric memory
    # via env var — the engine arg alone doesn't prevent SymmMemCommunicator
    # from being instantiated.
    if kwargs.get("disable_custom_all_reduce"):
        os.environ["VLLM_ALLREDUCE_USE_SYMM_MEM"] = "0"
        logger.info("setup_envvars_for_vllm: set VLLM_ALLREDUCE_USE_SYMM_MEM=0 (disable_custom_all_reduce=True)")
    if kwargs.get("distributed_executor_backend") == "ray":
        # a hack to make the script work.
        # stop ray from manipulating *_VISIBLE_DEVICES
        # at the top-level when the distributed_executor_backend is ray.
        os.environ.pop("CUDA_VISIBLE_DEVICES", None)
        os.environ.pop("ROCR_VISIBLE_DEVICES", None)
        os.environ.pop("HIP_VISIBLE_DEVICES", None)
    elif noset_visible_devices:
        # We need to set CUDA_VISIBLE_DEVICES to the ray assigned GPU
        # when the distributed_executor_backend is not rayargs and
        # RAY_EXPERIMENTAL_NOSET_*_VISIBLE_DEVICES is set.
        os.environ["CUDA_VISIBLE_DEVICES"] = str(ray.get_gpu_ids()[0])

    num_gpus = kwargs.pop("num_gpus")
    if bundle_indices is not None:
        os.environ["VLLM_RAY_PER_WORKER_GPUS"] = str(num_gpus)
        os.environ["VLLM_RAY_BUNDLE_INDICES"] = ",".join(map(str, bundle_indices))
        logger.info(f"creating LLM with bundle_indices={bundle_indices}")

    # Set NUMA CPU affinity for single-GPU (TP=1) inference actors.
    # For TP>1, affinity is set per-worker via WorkerWrap.set_numa_affinity().
    #
    # When NUMA affinity is enabled, we also disable vLLM V1 multiprocessing.
    # vLLM's V1 engine spawns EngineCore as a separate subprocess using
    # multiprocessing with start_method="spawn" (forced when running inside a
    # Ray actor — see vllm.utils._maybe_force_spawn). Spawned processes do NOT
    # inherit the parent's CPU affinity, so NUMA binding set here would be lost.
    # Disabling V1 multiprocessing forces EngineCore to run in the same process,
    # where our affinity settings take effect.
    executor_backend = kwargs.get("distributed_executor_backend")
    logger.info(
        f"setup_envvars_for_vllm: distributed_executor_backend={executor_backend}, "
        f"SKYRL_ENABLE_NUMA_AFFINITY={os.environ.get('SKYRL_ENABLE_NUMA_AFFINITY', '<unset>')}, "
        f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', '<unset>')}, "
        f"VLLM_ENABLE_V1_MULTIPROCESSING={os.environ.get('VLLM_ENABLE_V1_MULTIPROCESSING', '<unset>')}"
    )
    # NOTE: the `mp` executor backend (Qwen3-Next R3 capture path) MUST keep v1
    # multiprocessing ENABLED — it spawns its TP worker subprocesses via the v1 mp
    # path, and disabling it cancels the shm message queue at warm-up
    # ("RuntimeError: cancelled"). NUMA single-GPU pinning does not apply to the
    # multi-GPU mp engine anyway, so skip this branch for mp.
    if executor_backend not in ("ray", "mp"):
        try:
            from skyrl_train.utils.numa import is_numa_affinity_enabled, set_numa_affinity_for_gpu
            numa_enabled = is_numa_affinity_enabled()
            logger.info(f"setup_envvars_for_vllm: numa_enabled={numa_enabled}")
            if numa_enabled:
                os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
                logger.info("setup_envvars_for_vllm: set VLLM_ENABLE_V1_MULTIPROCESSING=0 for NUMA affinity")
                cuda_devs = os.environ.get("CUDA_VISIBLE_DEVICES", "")
                if cuda_devs:
                    gpu_ids = [int(x) for x in cuda_devs.split(",")]
                    if len(gpu_ids) == 1:
                        set_numa_affinity_for_gpu(gpu_ids[0])
        except Exception as e:
            logger.warning(f"setup_envvars_for_vllm: NUMA affinity setup failed: {e}")


class WorkerWrap:
    def set_numa_affinity(self):
        """Set CPU affinity to match this worker's GPU NUMA node.

        Called via collective_rpc for TP>1 configurations so each
        vLLM EngineCore process binds to its GPU's local CPUs.
        """
        try:
            from skyrl_train.utils.numa import set_numa_affinity_for_gpu
            gpu_id = self.device.index if self.device is not None else 0
            set_numa_affinity_for_gpu(gpu_id)
        except Exception:
            pass

    def test_rpc(self, *args, **kwargs):
        """Test RPC call to worker"""
        return args, kwargs

    def init_weight_update_communicator(
        self,
        master_address,
        master_port,
        rank_offset,
        world_size,
        group_name,
        backend="nccl",
        override_existing: bool = False,
    ):
        """Init torch process group for model weights update"""
        assert torch.distributed.is_initialized(), "default torch process group must be initialized"
        assert group_name != "", "group name must not be empty"

        if getattr(self, "_model_update_group", None):
            if override_existing:
                logger.info("Destroying existing model update group")
                destroy_process_group(self._model_update_group)
                self._model_update_group = None
            else:
                warnings.warn(
                    "Detected an existing weights update group. For overriding, use `generator.override_existing_update_group=True`"
                )

        rank = torch.distributed.get_rank() + rank_offset
        logger.info(
            f"torch.distributed.get_rank(): {torch.distributed.get_rank()}, rank_offset: {rank_offset}, rank: {rank}, world_size: {world_size}, group_name: {group_name}"
        )

        self._model_update_group = init_custom_process_group(
            backend=backend,
            init_method=get_tcp_url(master_address, master_port),
            world_size=world_size,
            rank=rank,
            group_name=group_name,
        )
        logger.info(
            f"init_weight_update_communicator: master_address={master_address}, master_port={master_port}, ",
            f"rank={rank}, world_size={world_size}, group_name={group_name}",
        )

        # Create receiver now that we have all the state
        self._weight_receiver = VLLMWeightTransferReceiver(
            model_update_group=self._model_update_group,
            model_config=self.model_config,
            device=self.device,
        )

    @staticmethod
    def _apply_fp8_weight_loader_patches():
        """Patch Fp8LinearMethod.process_weights_after_loading to preserve weight_loader.

        Following verl's approach: after FP8 processing creates new Parameter objects,
        copy custom attributes (weight_loader, output_dim, input_dim, subclass_type)
        from the original specialized parameter so weight sync can reload weights.
        """
        import os
        if os.environ.get("SKYRL_FUSE_WEIGHTS", "0") != "1":
            return

        try:
            from vllm.model_executor.layers.quantization.fp8 import Fp8LinearMethod
        except ImportError:
            return

        original_process = Fp8LinearMethod.process_weights_after_loading

        def patched_process(self_method, layer, *args, **kwargs):
            # Save original param attributes before processing
            saved_attrs = {}
            for pname, param in layer.named_parameters():
                attrs = {}
                for attr in ('weight_loader', 'output_dim', 'input_dim',
                             '_output_dim', '_input_dim', 'packed_dim',
                             'packed_factor', 'tp_rank', 'tp_size',
                             'logical_widths', 'output_sizes'):
                    if hasattr(param, attr):
                        attrs[attr] = getattr(param, attr)
                attrs['subclass_type'] = type(param)
                saved_attrs[pname] = attrs

            # Call original process_weights_after_loading
            result = original_process(layer, *args, **kwargs)

            # Restore attributes on new parameters
            for pname, param in layer.named_parameters():
                if pname in saved_attrs:
                    for attr, value in saved_attrs[pname].items():
                        try:
                            setattr(param, attr, value)
                        except (AttributeError, TypeError):
                            pass

            return result

        Fp8LinearMethod.process_weights_after_loading = patched_process

    def begin_weight_update(self) -> None:
        """Start accumulating weights for batched load_weights call.

        When SKYRL_FUSE_WEIGHTS=1, weights are accumulated instead of loaded
        immediately. Call end_weight_update() to flush and apply them all at once
        via model.load_weights(), which handles packed module mapping (qkv_proj, gate_up_proj).
        Weights are stored on CPU to avoid GPU OOM during accumulation.
        """
        self._accumulated_weights = []

    def _is_fp8_model(self):
        """Check if the model uses FP8 quantization."""
        quant_config = getattr(self.model_runner.model, 'quant_config', None)
        if quant_config is None:
            return False
        from vllm.model_executor.layers.quantization.fp8 import Fp8Config
        return isinstance(quant_config, Fp8Config)

    def _quantize_weights_for_fp8(self, weights):
        """Quantize BF16 weights to FP8 before loading into FP8 model.

        Follows verl's approach: quantize each weight tensor to FP8 with
        per-tensor scale, then yield (name, fp8_tensor) and (name_scale, scale).
        Non-linear weights (layernorm, embedding) are passed through as-is.
        """
        import torch
        from vllm._custom_ops import scaled_fp8_quant

        model = self.model_runner.model
        # Build set of parameter names that are FP8 quantized
        # These are the linear layer weights (not biases, not layernorms, not embeddings)
        fp8_param_names = set()
        for name, module in model.named_modules():
            from vllm.model_executor.layers.quantization.fp8 import Fp8LinearMethod
            if hasattr(module, 'quant_method') and isinstance(module.quant_method, Fp8LinearMethod):
                for pname, _ in module.named_parameters():
                    if 'weight' in pname and 'scale' not in pname:
                        full_name = f"{name}.{pname}" if name else pname
                        fp8_param_names.add(full_name)

        for name, tensor in weights:
            # Check if this weight maps to an FP8-quantized parameter
            # The name might be "layers.0.self_attn.q_proj.weight" but the
            # FP8 param is "layers.0.self_attn.qkv_proj.weight"
            # We need to check the ORIGINAL unfused name against the fused params
            is_fp8 = False
            packed_mapping = getattr(model, 'packed_modules_mapping', {})
            # Reverse mapping: q_proj -> qkv_proj, gate_proj -> gate_up_proj
            reverse_map = {}
            for fused, originals in packed_mapping.items():
                for orig in originals:
                    reverse_map[orig] = fused

            # Try to find the FP8 param name
            check_name = name
            parts = name.rsplit('.', 2)
            if len(parts) >= 2:
                module_part = parts[-2]  # e.g. "q_proj"
                if module_part in reverse_map:
                    check_name = name.replace(module_part, reverse_map[module_part])

            if check_name in fp8_param_names or name in fp8_param_names:
                is_fp8 = True

            if is_fp8 and tensor.dtype != torch.float8_e4m3fn:
                # Move to GPU, quantize, move back to CPU
                gpu_tensor = tensor.to(device='cuda', dtype=torch.bfloat16)
                fp8_tensor, scale = scaled_fp8_quant(gpu_tensor)
                yield (name, fp8_tensor.cpu())
                # Yield the scale with the FUSED param name
                scale_name = check_name.replace('.weight', '.weight_scale')
                yield (scale_name, scale.cpu())
                del gpu_tensor, fp8_tensor
            else:
                yield (name, tensor)

    def _restore_param_subclasses(self, model):
        """Temporarily restore param __class__ to subclass_type for weight loading.

        After process_weights_after_loading, params are plain Parameter but have
        subclass_type saved. Restoring __class__ makes weight_loader dispatch work.
        Returns list of (param, original_class) for cleanup.
        """
        patched = []
        for name, param in model.named_parameters():
            subclass_type = getattr(param, 'subclass_type', None)
            if subclass_type is not None and type(param) != subclass_type:
                original_class = type(param)
                param.__class__ = subclass_type
                patched.append((param, original_class))
        return patched

    def _undo_param_subclasses(self, patched):
        """Undo the temporary __class__ patching."""
        for param, original_class in patched:
            param.__class__ = original_class

    def end_weight_update(self) -> None:
        """Flush accumulated weights via model.load_weights().

        For FP8 models: quantizes BF16 weights to FP8 before loading,
        following verl's approach. Also temporarily restores param subclass
        types so weight_loader dispatch works correctly with FP8 params.
        """
        import gc
        if hasattr(self, "_accumulated_weights") and self._accumulated_weights:
            model = self.model_runner.model
            if self._is_fp8_model():
                import torch
                import gc
                from vllm.model_executor.layers.quantization.fp8 import Fp8LinearMethod
                from vllm._custom_ops import scaled_fp8_quant

                # Receiver-side FP8 quantization: BF16 weights arrive via NCCL,
                # fuse stacked params, quantize to FP8, write directly to model.
                weight_index = {name: tensor for name, tensor in self._accumulated_weights}
                stacked = [
                    ("qkv_proj", "q_proj", "q"),
                    ("qkv_proj", "k_proj", "k"),
                    ("qkv_proj", "v_proj", "v"),
                    ("gate_up_proj", "gate_proj", 0),
                    ("gate_up_proj", "up_proj", 1),
                ]

                for mname, module in model.named_modules():
                    if not (hasattr(module, 'quant_method') and isinstance(module.quant_method, Fp8LinearMethod)):
                        continue
                    param = module.weight
                    device = param.device
                    is_stacked = any(mname.endswith(pn) for pn, _, _ in stacked)

                    if is_stacked:
                        shard_list = []
                        for param_name, weight_name, shard_id in stacked:
                            if not mname.endswith(param_name):
                                continue
                            src_name = mname.replace(param_name, weight_name) + ".weight"
                            if src_name in weight_index:
                                shard_list.append(weight_index[src_name])
                        if shard_list:
                            full_bf16 = torch.cat(shard_list, dim=0).to(
                                device=device, dtype=torch.bfloat16, non_blocking=True)
                            torch.cuda.current_stream().synchronize()
                            fp8_full, scale = scaled_fp8_quant(full_bf16)
                            param.data.copy_(fp8_full)
                            if hasattr(module, 'weight_scale'):
                                module.weight_scale.data.copy_(scale.squeeze())
                            del full_bf16, fp8_full, scale, shard_list
                    else:
                        src_name = mname + ".weight"
                        if src_name in weight_index:
                            bf16_w = weight_index[src_name].to(
                                device=device, dtype=torch.bfloat16, non_blocking=True)
                            torch.cuda.current_stream().synchronize()
                            fp8_w, scale = scaled_fp8_quant(bf16_w)
                            param.data.copy_(fp8_w)
                            if hasattr(module, 'weight_scale'):
                                module.weight_scale.data.copy_(scale.squeeze())
                            del bf16_w, fp8_w, scale

                # Load non-FP8 params (layernorms, embeddings)
                params_dict = dict(model.named_parameters())
                for name, tensor in self._accumulated_weights:
                    if name in params_dict:
                        param = params_dict[name]
                        if param.dtype != torch.float8_e4m3fn:
                            param.data.copy_(tensor.to(device=param.device, dtype=param.dtype))

                del weight_index

                gc.collect()
                torch.cuda.empty_cache()
            else:
                model.load_weights(weights=iter(self._accumulated_weights))
            self._accumulated_weights.clear()
            del self._accumulated_weights
            gc.collect()
            import torch
            torch.cuda.empty_cache()

    def load_weights(self, request: NamedWeightsUpdateRequest) -> None:
        """Load weights using the receiver.

        This method is called via collective_rpc from VLLMWeightLoader.

        When SKYRL_FUSE_WEIGHTS=1 and begin_weight_update() was called,
        weights are accumulated on CPU instead of loaded immediately.

        Args:
            request: Weight update request with names, dtypes, shapes, etc.
        """
        weight_list = []
        for name, tensor in self._weight_receiver.receive_weights(request):
            weight_list.append((name, tensor))

        if hasattr(self, "_accumulated_weights"):
            # Batched mode: move to CPU and accumulate for later flush
            for name, tensor in weight_list:
                self._accumulated_weights.append((name, tensor.cpu()))
            del weight_list
        else:
            # Immediate mode (default): load right away
            self.model_runner.model.load_weights(weights=weight_list)
            for weight in weight_list:
                del weight

    # TODO (sumanthrh): Add destroy process group RPC as a atexit handler to Trainer code.
    def destroy_weights_update_group(self):
        if not getattr(self, "_model_update_group", None):
            warnings.warn("No model update group to destroy")
            return
        destroy_process_group(self._model_update_group)

    def read_named_weights(self, hf_names, dump_inventory: bool = False):
        """TEST-ONLY (Stage 6 weight-equality gate): read engine-side weights back
        from the live vLLM model, reconstructed under the HF parameter names the
        trainer broadcasts.

        This is the symmetric inverse of ``load_weights`` (vLLM consumes HF-named
        tensors in ``model.load_weights`` and maps them into its internal
        fused/sharded params; here we read those internal params back and rebuild
        the HF view so the trainer's post-step HF tensors can be compared
        tensor-by-tensor). Returns, per requested HF name, this worker's
        contribution as a CPU fp32 tensor plus the rank coordinates so the caller
        can assemble across TP/EP shards.

        Supported HF name forms (Qwen1.5-MoE / Qwen2MoE vLLM layout):
          * ``model.embed_tokens.weight``                       -> VocabParallelEmbedding (TP vocab-sharded)
          * ``model.layers.{i}.mlp.gate.weight`` (router)       -> ReplicatedLinear (full copy every rank)
          * ``model.layers.{i}.self_attn.o_proj.weight``        -> RowParallelLinear (TP input-sharded)
          * ``model.layers.{i}.mlp.experts.{j}.gate_proj.weight`` -> FusedMoE w13_weight[local_e, :I]  (EP expert-sharded)
          * ``...experts.{j}.up_proj.weight``                   -> FusedMoE w13_weight[local_e, I:]
          * ``...experts.{j}.down_proj.weight``                 -> FusedMoE w2_weight[local_e]

        Args:
            hf_names: list of HF parameter names to read back.
            dump_inventory: if True, also returns the full ``named_parameters()``
                name->shape inventory under key ``__inventory__`` (first run aid).
        """
        import re
        import torch as _torch

        model = self.model_runner.model
        params = dict(model.named_parameters())
        buffers = dict(model.named_buffers())
        all_params = {**params, **buffers}

        try:
            from vllm.distributed import parallel_state as _ps
            tp_rank = _ps.get_tensor_model_parallel_rank()
            tp_size = _ps.get_tensor_model_parallel_world_size()
        except Exception:
            tp_rank, tp_size = 0, 1
        try:
            ep_rank = _ps.get_ep_group().rank_in_group
            ep_size = _ps.get_ep_group().world_size
        except Exception:
            ep_rank, ep_size = 0, 1

        def _cpu(t):
            return t.detach().to("cpu", dtype=_torch.float32).contiguous()

        out = {}
        if dump_inventory:
            out["__inventory__"] = {n: list(p.shape) for n, p in all_params.items()}
        out["__ranks__"] = {"tp_rank": tp_rank, "tp_size": tp_size, "ep_rank": ep_rank, "ep_size": ep_size}

        expert_re = re.compile(r"^(model\.layers\.\d+\.mlp)\.experts\.(\d+)\.(gate_proj|up_proj|down_proj)\.weight$")

        for name in hf_names:
            entry = {"found": False}
            try:
                # 1. Direct (replicated) match: router gate, norms, etc.
                if name in all_params:
                    entry = {"found": True, "mode": "direct", "tensor": _cpu(all_params[name])}
                    out[name] = entry
                    continue

                # 2. Routed expert -> FusedMoE fused weights.
                m = expert_re.match(name)
                if m is not None:
                    prefix, gj, proj = m.group(1), int(m.group(2)), m.group(3)
                    # vLLM FusedMoE stores w13_weight [n_local_experts, 2*I, H] and
                    # w2_weight [n_local_experts, H, I]. Local experts are a contiguous
                    # EP slice: global expert gj lives on ep_rank == gj // n_local.
                    w13 = all_params.get(f"{prefix}.experts.w13_weight")
                    w2 = all_params.get(f"{prefix}.experts.w2_weight")
                    if w13 is None or w2 is None:
                        # Fallback: scan for any experts.*weight tensor under this prefix.
                        cand = {k: v for k, v in all_params.items() if k.startswith(f"{prefix}.experts.") and k.endswith("weight")}
                        entry = {"found": False, "note": f"no w13/w2; candidates={list(cand.keys())}"}
                        out[name] = entry
                        continue
                    n_local = w13.shape[0]
                    owner_ep = gj // n_local
                    if owner_ep != ep_rank:
                        entry = {"found": False, "mode": "expert", "owner_ep": owner_ep, "skip": True}
                        out[name] = entry
                        continue
                    local_e = gj - owner_ep * n_local
                    if proj == "down_proj":
                        t = w2[local_e]
                    else:
                        inter = w13.shape[1] // 2
                        t = w13[local_e, :inter] if proj == "gate_proj" else w13[local_e, inter:]
                    entry = {"found": True, "mode": "expert", "owner_ep": owner_ep, "local_e": local_e, "tensor": _cpu(t)}
                    out[name] = entry
                    continue

                # 3. Unknown / unsupported name.
                entry = {"found": False, "note": "no mapping"}
                out[name] = entry
            except Exception as e:  # never crash the collective_rpc
                out[name] = {"found": False, "error": repr(e)}
        return out


class BaseVLLMInferenceEngine(InferenceEngineInterface):
    """Base class containing shared logic between sync and async VLLM engines."""

    def __init__(self, *args, bundle_indices: list = None, **kwargs):
        setup_envvars_for_vllm(kwargs, bundle_indices)
        vllm_v1_disable_multiproc = kwargs.pop("vllm_v1_disable_multiproc", False)
        logger.info(
            f"BaseVLLMInferenceEngine: vllm_v1_disable_multiproc={vllm_v1_disable_multiproc}, "
            f"vllm.__version__={vllm.__version__}, "
            f"VLLM_ENABLE_V1_MULTIPROCESSING={os.environ.get('VLLM_ENABLE_V1_MULTIPROCESSING', '<unset>')}"
        )
        if vllm_v1_disable_multiproc or vllm.__version__ == "0.8.2":
            # https://github.com/vllm-project/vllm/blob/effc5d24fae10b29996256eb7a88668ff7941aed/examples/offline_inference/reproduciblity.py#L11
            os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
            logger.info("BaseVLLMInferenceEngine: set VLLM_ENABLE_V1_MULTIPROCESSING=0")

        # Store common attributes
        self._tp_size = kwargs.get("tensor_parallel_size", 1)
        self._pp_size = kwargs.get("pipeline_parallel_size", 1)
        self._dp_size = kwargs.get("data_parallel_size", 1)
        self._is_lora = kwargs.get("enable_lora", False)

        if "rope_scaling" in kwargs:
            kwargs.pop("rope_scaling")
        # Let subclass create the appropriate engine
        self.llm = self._create_engine(*args, **kwargs)

        # Set NUMA affinity for TP>1 workers via collective_rpc
        if self._tp_size > 1 or self._pp_size > 1:
            try:
                self.llm.collective_rpc("set_numa_affinity")
            except Exception:
                pass

        # Weight loader is created by subclass after engine initialization
        self._weight_loader = None

    def tp_size(self):
        return self._tp_size

    def pp_size(self):
        return self._pp_size

    def dp_size(self):
        return self._dp_size

    def _create_engine(self, *args, **kwargs):
        """Abstract method for subclasses to implement engine creation."""
        raise NotImplementedError("Subclasses must implement _create_engine")

    def _preprocess_prompts(self, input_batch: InferenceEngineInput):
        """Common prompt preprocessing logic."""
        prompts = input_batch.get("prompts")
        prompt_token_ids = input_batch.get("prompt_token_ids")
        request_sampling_params = input_batch.get("sampling_params")

        assert (
            prompts is None and prompt_token_ids is not None
        ), "VLLMInferenceEngine only accepts `prompt_token_ids`, not `prompts`."

        sampling_params = (
            SamplingParams(**request_sampling_params) if request_sampling_params is not None else SamplingParams()
        )

        return prompt_token_ids, sampling_params

    def _postprocess_outputs(self, outputs):
        """Common output processing logic."""
        responses: List[str] = []
        stop_reasons: List[str] = []
        response_ids: List[List[int]] = []
        response_logprobs: Optional[List[List[float]]] = []
        all_prompt_logprobs: Optional[List] = None

        for output in outputs:
            # TODO(tgriggs): Support n>1 sampling.
            assert (
                len(output.outputs) == 1
            ), "Each prompt should have only one responses. n>1 sampling is supported by copying prompts."
            resp = output.outputs[0]
            responses.append(resp.text)
            stop_reasons.append(resp.finish_reason)
            response_ids.append(resp.token_ids)
            _logprobs = None
            if resp.logprobs:
                _logprobs = []
                for i, token_logprobs in enumerate(resp.logprobs):
                    token_logprobs: Dict[str, Logprob]
                    token_id = resp.token_ids[i]
                    logprob = token_logprobs[token_id].logprob
                    _logprobs.append(logprob)
                    del token_logprobs
            response_logprobs.append(_logprobs)

            # Extract prompt_logprobs if available (used for teacher scoring)
            if hasattr(output, "prompt_logprobs") and output.prompt_logprobs is not None:
                if all_prompt_logprobs is None:
                    all_prompt_logprobs = []
                # Convert vLLM's List[Optional[Dict[int, Logprob]]] to
                # List[Optional[Dict[int, float]]] (extract .logprob from Logprob objects)
                prompt_lps = []
                for pos_logprobs in output.prompt_logprobs:
                    if pos_logprobs is None:
                        prompt_lps.append(None)
                    else:
                        prompt_lps.append({
                            token_id: lp.logprob if hasattr(lp, "logprob") else lp
                            for token_id, lp in pos_logprobs.items()
                        })
                all_prompt_logprobs.append(prompt_lps)

        if len(response_logprobs) and response_logprobs[0] is None:
            response_logprobs = None  # hack: assume uniform sampling params

        return InferenceEngineOutput(
            responses=responses,
            stop_reasons=stop_reasons,
            response_ids=response_ids,
            response_logprobs=response_logprobs,
            prompt_logprobs=all_prompt_logprobs,
        )

    def _get_engine(self):
        """Get the underlying engine for RPC calls."""
        return self.llm.engine if hasattr(self.llm, "engine") else self.llm

    def _is_lora_disk_loading_request(self, request: NamedWeightsUpdateRequest) -> bool:
        """Check if this is a LoRA disk loading request."""
        is_lora = request["names"][0] == "lora_disk_load"
        if is_lora:
            assert request.get("extras") and len(request["extras"]) > 0 and "lora_disk_path" in request["extras"][0], (
                "vLLM LoRA weight update requests must contain the disk load " "path under key `lora_disk_path`"
            )
        return is_lora

    def reset_prefix_cache(self):
        """Reset the prefix cache. Subclasses override for async version."""
        return self.llm.llm_engine.reset_prefix_cache()

    async def abort_generation(self) -> None:
        raise NotImplementedError("Abort generation is only supported for AsyncVLLMInferenceEngine.")


class VLLMInferenceEngine(BaseVLLMInferenceEngine):
    """Synchronous VLLM engine."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._weight_loader = VLLMWeightLoader(self.llm, is_async=False)

    def _create_engine(self, *args, **kwargs):
        # Pipeline parallelism requires AsyncLLMEngine
        if kwargs.get("pipeline_parallel_size", 1) > 1:
            raise ValueError(
                "Pipeline parallelism is only supported with AsyncVLLMInferenceEngine. "
                "Please set `generator.async_engine=true` in your config."
            )
        # Strip OpenAI-serving-only kwargs (e.g. openai_sampling_params, tool
        # parser) that the config layer injects for all engines. The sync
        # vllm.LLM/EngineArgs path does not accept these — only the async
        # OpenAI server consumes them. Mirror the async engine's pop so the
        # sync engine (async_engine=false, used by the batched OPD path) does
        # not pass them through to EngineArgs and raise TypeError.
        openai_kwargs = pop_openai_kwargs(kwargs)
        self._openai_sampling_params = openai_kwargs.pop("openai_sampling_params", {})
        # Pop enable_ray_prometheus_stats - only supported for async engine
        enable_ray_prometheus_stats = kwargs.pop("enable_ray_prometheus_stats", False)
        if enable_ray_prometheus_stats:
            logger.warning(
                "enable_ray_prometheus_stats is only supported with AsyncVLLMInferenceEngine. "
                "Set `generator.async_engine=true` to enable Ray Prometheus stats logging."
            )
        return vllm.LLM(*args, **kwargs)

    async def generate(self, input_batch: InferenceEngineInput) -> InferenceEngineOutput:
        prompt_token_ids, sampling_params = self._preprocess_prompts(input_batch)

        # Check if LoRA is enabled and create LoRA requests
        lora_requests = None
        if self._is_lora:
            lora_int_ids = list(self.llm.llm_engine.list_loras())
            if len(lora_int_ids) > 0:
                lora_int_id = lora_int_ids[0]
                batch_size = len(prompt_token_ids)
                # dummy_lora_path for placeholder (actual loading done in add_lora())
                lora_requests = [
                    LoRARequest(lora_name=f"{lora_int_id}", lora_int_id=lora_int_id, lora_path="/dummy_lora_path")
                ] * batch_size

        outputs = await asyncio.to_thread(
            self.llm.generate,
            prompts=[TokensPrompt(prompt_token_ids=r) for r in prompt_token_ids],
            sampling_params=sampling_params,
            lora_request=lora_requests,
        )

        return self._postprocess_outputs(outputs)

    async def chat_completion(self, request_payload: Dict[str, Any]) -> Dict[str, Any]:
        """Only supported in AsyncVLLMInferenceEngine."""
        raise NotImplementedError()

    async def completion(self, request_payload: Dict[str, Any]) -> Dict[str, Any]:
        """Only supported in AsyncVLLMInferenceEngine."""
        raise NotImplementedError()

    async def wake_up(self, *args: Any, **kwargs: Any):
        await asyncio.to_thread(self.llm.wake_up, tags=kwargs.get("tags", None))

    async def sleep(self, *args: Any, **kwargs: Any):
        engine = self._get_engine().llm_engine
        output_processor = engine.output_processor
        if output_processor.has_unfinished_requests():
            logger.warning(
                "Calling sleep() with unfinished requests in vLLM engine. This is unexpected since all "
                "generation should be done before sleep() is called. Check for potential failures or "
                "dangling requests in your Generator/Env. Aborting all unfinished requests."
            )
            unfinished_request_ids = list(output_processor.request_states.keys())
            await asyncio.to_thread(engine.abort_request, unfinished_request_ids)

        level = 1 if self._is_lora else kwargs.get("level", 2)
        await asyncio.to_thread(self.llm.sleep, level=level)

    async def init_weight_update_communicator(
        self, master_addr, master_port, rank_offset, world_size, group_name, backend, override_existing: bool = False
    ):
        engine = self._get_engine()
        return await asyncio.to_thread(
            engine.collective_rpc,
            "init_weight_update_communicator",
            args=(master_addr, master_port, rank_offset, world_size, group_name, backend, override_existing),
        )

    async def _load_lora_from_disk(self, lora_path: str):
        """Load LoRA adapters from disk using vLLM's native add_lora method."""
        lora_id = int(time.time_ns() % 0x7FFFFFFF)
        lora_request = LoRARequest(lora_name=f"{lora_id}", lora_int_id=lora_id, lora_path=lora_path)
        result = self.llm.llm_engine.add_lora(lora_request)
        return result

    async def update_named_weights(self, request: NamedWeightsUpdateRequest):
        if "names" not in request:
            raise ValueError(f"Expected update weight request with 'names' entry, got keys: {request.keys()}")

        if not len(request["names"]):
            raise ValueError("Update weight request should have at least one entry in 'names'")

        # Handle LoRA disk loading request
        if self._is_lora_disk_loading_request(request):
            lora_path = request["extras"][0]["lora_disk_path"]
            return await self._load_lora_from_disk(lora_path)

        # Use the weight loader to coordinate weight transfer
        return await self._weight_loader.load_weights(request)

    async def teardown(self):
        await self._destroy_weights_update_group()

    async def reset_prefix_cache(self):
        return await asyncio.to_thread(self.llm.llm_engine.reset_prefix_cache)

    async def _destroy_weights_update_group(self):
        engine = self._get_engine()
        return await asyncio.to_thread(engine.collective_rpc, "destroy_weights_update_group")

class V1LoggingStatLoggerFixed(LoggingStatLogger):
    """
    A fixed version of LoggingStatLogger that actually logs during the record method.
    The log method is otherwise not called in the VLLM codebase.

    Also stores aggregated stats in a class-level registry for programmatic access
    (used by VLLMStatsCallback to bypass Ray log-to-driver unreliability).

    Stats are accumulated throughout a step:
    - Request counts (running, waiting): track peak and median values
    - Throughput metrics: track peak and median values observed during active periods
    - Cache metrics: track peak and median usage
    """

    # Class-level registry mapping engine IDs to their accumulated stats
    _stats_registry: Dict[int, Dict[str, Any]] = {}
    _registry_lock = threading.Lock()

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.log_interval = 5
        self._engine_id: Optional[int] = None

    def set_engine_id(self, engine_id: int) -> None:
        """Set the engine ID for this stat logger instance."""
        self._engine_id = engine_id

    def record(self, *args: Any, **kwargs: Any) -> None:
        # Call parent with original arguments - important to preserve vLLM's calling convention
        super().record(*args, **kwargs)

        # Accumulate stats in registry if engine ID is set
        if self._engine_id is not None:
            # Extract scheduler_stats from vLLM v1 API:
            # vLLM calls record(scheduler_stats, iteration_stats, ...) with positional args
            # or record(scheduler_stats=..., iteration_stats=...) with keyword args
            scheduler_stats = None
            if args:
                scheduler_stats = args[0]
            elif "scheduler_stats" in kwargs:
                scheduler_stats = kwargs["scheduler_stats"]

            current_running = 0
            current_waiting = 0
            current_cache_usage = 0.0
            current_prefix_hit = 0.0

            if scheduler_stats is not None:
                current_running = getattr(scheduler_stats, "num_running_reqs", 0)
                current_waiting = getattr(scheduler_stats, "num_waiting_reqs", 0)
                current_cache_usage = getattr(scheduler_stats, "kv_cache_usage", 0.0) * 100.0  # Convert to percentage

                # Extract prefix cache hit rate from prefix_cache_stats
                prefix_cache_stats = getattr(scheduler_stats, "prefix_cache_stats", None)
                if prefix_cache_stats is not None:
                    hits = getattr(prefix_cache_stats, "hits", 0)
                    misses = getattr(prefix_cache_stats, "misses", 0)
                    total = hits + misses
                    current_prefix_hit = (hits / total * 100.0) if total > 0 else 0.0

            # Extract iteration_stats (second positional arg) for per-request latency data
            iteration_stats = None
            if len(args) > 1:
                iteration_stats = args[1]
            elif "iteration_stats" in kwargs:
                iteration_stats = kwargs["iteration_stats"]

            # Collect per-request latency samples from finished requests
            finished_prefill_times: List[float] = []
            finished_decode_times: List[float] = []
            finished_e2e_latencies: List[float] = []
            finished_queued_times: List[float] = []
            finished_ttfts: List[float] = []
            finished_num_preempted = 0
            if iteration_stats is not None:
                # Time-to-first-token samples from this iteration
                ttft_iter = getattr(iteration_stats, "time_to_first_tokens_iter", None)
                if ttft_iter:
                    finished_ttfts.extend(ttft_iter)
                # Preemption count
                finished_num_preempted = getattr(iteration_stats, "num_preempted_reqs", 0)
                # Per-request stats from completed requests
                for req in getattr(iteration_stats, "finished_requests", []):
                    prefill_t = getattr(req, "prefill_time", 0.0)
                    decode_t = getattr(req, "decode_time", 0.0)
                    e2e_t = getattr(req, "e2e_latency", 0.0)
                    queued_t = getattr(req, "queued_time", 0.0)
                    if prefill_t > 0:
                        finished_prefill_times.append(prefill_t)
                    if decode_t > 0:
                        finished_decode_times.append(decode_t)
                    if e2e_t > 0:
                        finished_e2e_latencies.append(e2e_t)
                    if queued_t > 0:
                        finished_queued_times.append(queued_t)

            # Throughput is computed by parent class LoggingStatLogger after super().record()
            # These are stored as instance attributes
            current_prompt_tp = getattr(self, "last_prompt_throughput", 0.0) or 0.0
            current_gen_tp = getattr(self, "last_generation_throughput", 0.0) or 0.0

            is_active = current_running > 0 or current_waiting > 0

            with V1LoggingStatLoggerFixed._registry_lock:
                existing = V1LoggingStatLoggerFixed._stats_registry.get(self._engine_id)

                if existing is None:
                    # Initialize with sample lists for median calculation
                    V1LoggingStatLoggerFixed._stats_registry[self._engine_id] = {
                        # Sample lists for computing median (only active samples)
                        "_samples_prompt_tp": [current_prompt_tp] if is_active else [],
                        "_samples_gen_tp": [current_gen_tp] if is_active else [],
                        "_samples_running": [current_running] if is_active else [],
                        "_samples_waiting": [current_waiting] if is_active else [],
                        "_samples_cache": [current_cache_usage] if is_active else [],
                        "_samples_prefix_hit": [current_prefix_hit] if is_active else [],
                        # Per-request latency samples (accumulated from finished requests)
                        "_samples_prefill_time": list(finished_prefill_times),
                        "_samples_decode_time": list(finished_decode_times),
                        "_samples_e2e_latency": list(finished_e2e_latencies),
                        "_samples_queued_time": list(finished_queued_times),
                        "_samples_ttft": list(finished_ttfts),
                        "_total_preempted": finished_num_preempted,
                        # Peak values
                        "_peak_prompt_tp": current_prompt_tp,
                        "_peak_gen_tp": current_gen_tp,
                        "_peak_running": current_running,
                        "_peak_waiting": current_waiting,
                        "_peak_cache": current_cache_usage,
                        "_peak_prefix_hit": current_prefix_hit,
                        # Counters
                        "_num_samples": 1,
                        "_num_active_samples": 1 if is_active else 0,
                        "timestamp": time.time(),
                    }
                else:
                    # Update peak values
                    existing["_peak_prompt_tp"] = max(existing["_peak_prompt_tp"], current_prompt_tp)
                    existing["_peak_gen_tp"] = max(existing["_peak_gen_tp"], current_gen_tp)
                    existing["_peak_running"] = max(existing["_peak_running"], current_running)
                    existing["_peak_waiting"] = max(existing["_peak_waiting"], current_waiting)
                    existing["_peak_cache"] = max(existing["_peak_cache"], current_cache_usage)
                    existing["_peak_prefix_hit"] = max(existing["_peak_prefix_hit"], current_prefix_hit)

                    # Accumulate per-request latency samples
                    existing["_samples_prefill_time"].extend(finished_prefill_times)
                    existing["_samples_decode_time"].extend(finished_decode_times)
                    existing["_samples_e2e_latency"].extend(finished_e2e_latencies)
                    existing["_samples_queued_time"].extend(finished_queued_times)
                    existing["_samples_ttft"].extend(finished_ttfts)
                    existing["_total_preempted"] += finished_num_preempted

                    # Append to sample lists (only for active samples to get meaningful medians)
                    if is_active:
                        existing["_samples_prompt_tp"].append(current_prompt_tp)
                        existing["_samples_gen_tp"].append(current_gen_tp)
                        existing["_samples_running"].append(current_running)
                        existing["_samples_waiting"].append(current_waiting)
                        existing["_samples_cache"].append(current_cache_usage)
                        existing["_samples_prefix_hit"].append(current_prefix_hit)
                        existing["_num_active_samples"] += 1

                    existing["_num_samples"] += 1
                    existing["timestamp"] = time.time()

        now = time.monotonic()
        if now - self.last_log_time > self.log_interval:
            self.log()
            self.last_log_time = now

    @staticmethod
    def _compute_median(samples: List[float]) -> float:
        """Compute median of a list of samples."""
        if not samples:
            return 0.0
        sorted_samples = sorted(samples)
        n = len(sorted_samples)
        mid = n // 2
        if n % 2 == 0:
            return (sorted_samples[mid - 1] + sorted_samples[mid]) / 2.0
        return sorted_samples[mid]

    @classmethod
    def get_stats_by_engine_id(cls, engine_id: int, reset: bool = True) -> Optional[Dict[str, Any]]:
        """Get the accumulated stats for a given engine ID.

        Args:
            engine_id: The engine ID to get stats for.
            reset: If True, reset the accumulated stats after reading (default True).
                   This ensures each training step gets fresh stats.

        Returns:
            Dict with accumulated stats, or None if no stats recorded yet.
            Includes peak values, median values, and computed averages.
        """
        with cls._registry_lock:
            stats = cls._stats_registry.get(engine_id)
            if stats is None:
                return None

            # Compute medians from sample lists
            median_prompt_tp = cls._compute_median(stats["_samples_prompt_tp"])
            median_gen_tp = cls._compute_median(stats["_samples_gen_tp"])
            median_running = cls._compute_median(stats["_samples_running"])
            median_waiting = cls._compute_median(stats["_samples_waiting"])
            median_cache = cls._compute_median(stats["_samples_cache"])
            median_prefix_hit = cls._compute_median(stats["_samples_prefix_hit"])

            # Compute means from sample lists
            num_active = stats["_num_active_samples"]
            if num_active > 0:
                mean_prompt_tp = sum(stats["_samples_prompt_tp"]) / num_active
                mean_gen_tp = sum(stats["_samples_gen_tp"]) / num_active
            else:
                mean_prompt_tp = 0.0
                mean_gen_tp = 0.0

            # Compute per-request latency statistics
            prefill_samples = stats["_samples_prefill_time"]
            decode_samples = stats["_samples_decode_time"]
            e2e_samples = stats["_samples_e2e_latency"]
            queued_samples = stats["_samples_queued_time"]
            ttft_samples = stats["_samples_ttft"]

            def _mean(s: List[float]) -> float:
                return sum(s) / len(s) if s else 0.0

            def _p90(s: List[float]) -> float:
                if not s:
                    return 0.0
                sorted_s = sorted(s)
                idx = int(len(sorted_s) * 0.9)
                return sorted_s[min(idx, len(sorted_s) - 1)]

            result = {
                # Peak values
                "peak_prompt_throughput": stats["_peak_prompt_tp"],
                "peak_generation_throughput": stats["_peak_gen_tp"],
                "peak_running_reqs": stats["_peak_running"],
                "peak_waiting_reqs": stats["_peak_waiting"],
                "peak_gpu_cache_usage_perc": stats["_peak_cache"],
                "peak_prefix_cache_hit_rate": stats["_peak_prefix_hit"],
                # Median values
                "median_prompt_throughput": median_prompt_tp,
                "median_generation_throughput": median_gen_tp,
                "median_running_reqs": median_running,
                "median_waiting_reqs": median_waiting,
                "median_gpu_cache_usage_perc": median_cache,
                "median_prefix_cache_hit_rate": median_prefix_hit,
                # Mean values
                "mean_prompt_throughput": mean_prompt_tp,
                "mean_generation_throughput": mean_gen_tp,
                # Per-request latency stats (seconds)
                "latency_prefill_mean": _mean(prefill_samples),
                "latency_prefill_median": cls._compute_median(prefill_samples),
                "latency_prefill_p90": _p90(prefill_samples),
                "latency_decode_mean": _mean(decode_samples),
                "latency_decode_median": cls._compute_median(decode_samples),
                "latency_decode_p90": _p90(decode_samples),
                "latency_e2e_mean": _mean(e2e_samples),
                "latency_e2e_median": cls._compute_median(e2e_samples),
                "latency_e2e_p90": _p90(e2e_samples),
                "latency_queued_mean": _mean(queued_samples),
                "latency_queued_median": cls._compute_median(queued_samples),
                "latency_queued_p90": _p90(queued_samples),
                "latency_ttft_mean": _mean(ttft_samples),
                "latency_ttft_median": cls._compute_median(ttft_samples),
                "latency_ttft_p90": _p90(ttft_samples),
                "latency_num_finished_requests": len(e2e_samples),
                "total_preempted_reqs": stats["_total_preempted"],
                # Legacy field names for backwards compatibility (use peak values)
                "avg_prompt_throughput": stats["_peak_prompt_tp"],
                "avg_generation_throughput": stats["_peak_gen_tp"],
                "num_running_reqs": stats["_peak_running"],
                "num_waiting_reqs": stats["_peak_waiting"],
                "gpu_cache_usage_perc": stats["_peak_cache"],
                "prefix_cache_hit_rate": stats["_peak_prefix_hit"],
                # Metadata
                "timestamp": stats["timestamp"],
                "num_samples": stats["_num_samples"],
                "num_active_samples": stats["_num_active_samples"],
            }

            if reset:
                # Reset for next step
                del cls._stats_registry[engine_id]

            return result

class AsyncVLLMInferenceEngine(BaseVLLMInferenceEngine):
    """Asynchronous VLLM engine."""

    def __init__(self, *args, **kwargs):
        # Generate unique engine ID before calling super().__init__() which calls _create_engine
        self._stats_engine_id = id(self)
        super().__init__(*args, **kwargs)
        self._weight_loader = VLLMWeightLoader(self.llm, is_async=True)

    def _create_stat_logger_factory(self):
        """Create a factory that produces stat loggers with the engine ID set."""
        engine_id = self._stats_engine_id

        def factory(*args, **kwargs):
            logger_instance = V1LoggingStatLoggerFixed(*args, **kwargs)
            logger_instance.set_engine_id(engine_id)
            return logger_instance

        return factory

    def _create_engine(self, *args, **kwargs):
        openai_kwargs = pop_openai_kwargs(kwargs)
        # Store sampling params for OpenAI-style requests (Harbor rollouts)
        self._openai_sampling_params = openai_kwargs.pop("openai_sampling_params", {})
        if self._openai_sampling_params:
            logger.warning(
                f"OpenAI API sampling params overridden: "
                f"temperature={self._openai_sampling_params.get('temperature', 1.0)}, "
                f"top_p={self._openai_sampling_params.get('top_p', 1.0)}, "
                f"top_k={self._openai_sampling_params.get('top_k', -1)}"
            )
        enable_ray_prometheus_stats = kwargs.pop("enable_ray_prometheus_stats", False)

        # TODO (erictang000): potentially enable log requests for a debugging mode
        custom_chat_template_path = kwargs.pop("custom_chat_template_chat_completion_path", None)
        # Use factory to inject engine ID into stat logger
        stat_loggers = [self._create_stat_logger_factory()]
        engine_args = vllm.AsyncEngineArgs(**kwargs)

        if _parse_vllm_version() >= version.parse("0.10.0"):
            engine_args = vllm.AsyncEngineArgs(enable_log_requests=False, **kwargs)
        else:
            engine_args = vllm.AsyncEngineArgs(disable_log_requests=True, **kwargs)

        # Add Ray Prometheus stat loggers if enabled
        if enable_ray_prometheus_stats:
            ray_loggers = self._create_ray_prometheus_stat_loggers()
            if ray_loggers:
                stat_loggers.extend(ray_loggers)

        # Stagger engine startup to avoid TOCTOU port collisions (EADDRINUSE).
        # vLLM's get_open_port() queries a free port then releases the socket;
        # if multiple engines on the same node call it simultaneously, they can
        # get the same port. A random pre-startup delay desynchronises the
        # within-job case.
        #
        # The retry loop below additionally addresses the *cross-job* race
        # we hit on Jupiter A3 RL chain restarts (job 485102, 2026-05-23):
        # Slurm reaps the prior chain leader on TIMEOUT, allocates the same
        # nodes to the next-in-chain immediately, but kernel socket TIME_WAIT
        # can hold the prior holder's bound port for up to ~60 s. A 1.5-3 s
        # stagger doesn't bridge that gap, so without a retry the new chain
        # head exits 1 in ~20 min with EADDRINUSE and the chain visibly
        # "loses" a restart slot until the next dependency-satisfied slot
        # finally gets a fresh port.
        #
        # 5 attempts with exponential backoff (15→30→60→120→240 s) bridges
        # the TIME_WAIT window cleanly while staying well under the outer
        # wait_for_engine_startup deadline.
        import random, time
        from torch.distributed import DistNetworkError

        _MAX_INIT_ATTEMPTS = 5
        _BACKOFF_BASE_SEC = 15.0
        engine = None
        for _attempt in range(_MAX_INIT_ATTEMPTS):
            _stagger = random.uniform(1.5, 3.0)
            logger.info(
                f"Engine startup stagger: sleeping {_stagger:.2f}s "
                f"(attempt {_attempt + 1}/{_MAX_INIT_ATTEMPTS}) to avoid port collisions"
            )
            time.sleep(_stagger)
            try:
                engine = vllm.AsyncLLMEngine.from_engine_args(engine_args, stat_loggers=stat_loggers)
                break
            except DistNetworkError as e:
                msg = str(e)
                if "EADDRINUSE" not in msg and "already in use" not in msg.lower():
                    raise
                if _attempt == _MAX_INIT_ATTEMPTS - 1:
                    logger.error(
                        f"Engine init still hit EADDRINUSE after {_MAX_INIT_ATTEMPTS} attempts; giving up"
                    )
                    raise
                _backoff = _BACKOFF_BASE_SEC * (2 ** _attempt)
                logger.warning(
                    f"Engine init hit EADDRINUSE on attempt {_attempt + 1}/{_MAX_INIT_ATTEMPTS}; "
                    f"retrying in {_backoff:.0f}s (likely kernel socket TIME_WAIT from prior job): {msg.splitlines()[0]}"
                )
                time.sleep(_backoff)
        assert engine is not None  # loop either breaks with engine set or raises

        # Adapted from https://github.com/volcengine/verl/blob/e90f18c40aa639cd25092b78a5ff7e2d2508c088/verl/workers/rollout/vllm_rollout/vllm_async_server.py#L327
        model_config = engine.model_config
        model_path = kwargs.get("model")
        # Allow overriding the served model name (similar to vLLM's --served-model-name flag).
        # Useful for Harbor/LiteLLM compatibility where model names must have exactly one '/'.
        # See https://github.com/NovaSky-AI/SkyRL/pull/238#discussion_r2326561295
        served_model_name = kwargs.get("served_model_name")
        model_name = served_model_name if served_model_name else model_path

        base_model_paths = [BaseModelPath(name=model_name, model_path=model_path)]

        # vLLM API compatibility via try/except:
        # - vLLM >= 0.13: model_config removed (obtained internally from engine_client)
        # - vLLM < 0.13: model_config is required as a parameter
        # Try newer API first, fall back to older API if TypeError
        try:
            models = OpenAIServingModels(
                engine_client=engine,
                base_model_paths=base_model_paths,
            )
        except TypeError:
            logger.info(f"vLLM {vllm.__version__}: using legacy API with model_config")
            models = OpenAIServingModels(
                engine_client=engine,
                model_config=model_config,
                base_model_paths=base_model_paths,
            )

        # TODO(Charlie): adding custom chat template for chat completion. Hacky!
        if custom_chat_template_path:
            with open(custom_chat_template_path, "r") as f:
                custom_chat_template_content = f.read()
            logger.info(f"Initializing OpenAIServingChat with custom_chat_template read from: {custom_chat_template_path}")
        else:
            custom_chat_template_content = None

        # TODO(Charlie): revisit kwargs `enable_auto_tools` and `tool_parser` when we need to
        # support OAI-style tool calling; and `request_logger` for better debugging.
        # Try newer API first, fall back to older API if TypeError
        try:
            self.openai_serving_chat = OpenAIServingChat(
                engine_client=engine,
                models=models,
                response_role="assistant",
                request_logger=None,
                chat_template=custom_chat_template_content,
                chat_template_content_format="auto",
                **openai_kwargs,
            )
        except TypeError:
            self.openai_serving_chat = OpenAIServingChat(
                engine_client=engine,
                model_config=model_config,
                models=models,
                response_role="assistant",
                request_logger=None,
                chat_template=custom_chat_template_content,
                chat_template_content_format="auto",
                **openai_kwargs,
            )

        # TODO(Charlie): revisit kwargs `return_tokens_as_token_ids`,
        # `enable_prompt_tokens_details`, `enable_force_include_usage`.
        # Try newer API first, fall back to older API if TypeError
        try:
            self.openai_serving_completion = OpenAIServingCompletion(
                engine_client=engine,
                models=models,
                request_logger=None,
            )
        except TypeError:
            self.openai_serving_completion = OpenAIServingCompletion(
                engine_client=engine,
                model_config=model_config,
                models=models,
                request_logger=None,
            )
        return engine

    def _create_ray_prometheus_stat_loggers(self):
        """Create Ray Prometheus stat loggers for vLLM metrics.

        Returns stat_loggers in the format expected by vLLM's from_engine_args().
        For vLLM v1 (0.9.0+), this returns a list of StatLoggerFactory callables.
        For older versions where the v1 API is not available, this returns `None`.

        See: https://docs.vllm.ai/en/latest/api/vllm/v1/metrics/ray_wrappers/
        """
        try:
            # Try vLLM v1 API first (0.9.0+)
            from vllm.v1.metrics.ray_wrappers import RayPrometheusStatLogger

            logger.info("Enabling RayPrometheusStatLogger for vLLM inference engine metrics")
            # For v1, stat_loggers is a list of factory callables
            return [RayPrometheusStatLogger]
        except ImportError:
            logger.warning(
                "RayPrometheusStatLogger not available in this vLLM version. "
                "For Ray-integrated metrics, upgrade to vLLM >= 0.9.0. "
                "Stat logging will be disabled."
            )
            return None

    async def _load_lora_from_disk(self, lora_path: str):
        """Load LoRA adapters from disk using vLLM's native add_lora method."""
        lora_id = int(time.time_ns() % 0x7FFFFFFF)
        lora_request = LoRARequest(lora_name=f"{lora_id}", lora_int_id=lora_id, lora_path=lora_path)
        result = await self.llm.add_lora(lora_request)
        return result

    async def _collect_outputs(self, prompt_token_ids, request_id: str, sampling_params: SamplingParams):
        """Collect outputs for a single prompt."""
        # Check if LoRA is enabled and create LoRA request
        final_output = None
        lora_request = None

        if self._is_lora:
            lora_int_ids = list(await self.llm.list_loras())
            if len(lora_int_ids) > 0:
                lora_int_id = lora_int_ids[0]
                # dummy_lora_path for placeholder (actual loading done in add_lora())
                lora_request = LoRARequest(
                    lora_name=f"{lora_int_id}", lora_int_id=lora_int_id, lora_path="/dummy_lora_path"
                )

        async for request_output in self.llm.generate(
            prompt=TokensPrompt(prompt_token_ids=prompt_token_ids),
            sampling_params=sampling_params,
            request_id=request_id,
            lora_request=lora_request,
        ):
            final_output = request_output

        return final_output

    async def generate(self, input_batch: InferenceEngineInput) -> InferenceEngineOutput:
        """Generate responses using vLLM's async engine.

        v2 (2026-05-26): wrapped in try/except to explicitly abort all
        sibling in-flight vLLM requests when any task in this batch raises
        (typical case: vLLM serving_chat ValueError on 32k-token validation).
        Without this, the failed task unwinds Python but its sibling tasks'
        Ray ObjectRefs leak into the entrypoint actor's distributed-refcount
        state — that's the `reference_count.cc:1619` SIGABRT pattern that's
        been killing the v3 maxgn09_hint chain links one after another even
        after the harbor rollback_on_exception hook landed (the hook fires
        AFTER the trial, but the ObjectRefs leak DURING the generate batch).
        See agent_logs/2026-05-25_v6a-agrs_507771_moe_combine_stack_pinned.md
        and project_ray_workercrashed_harbor_rollback.md for the chain of
        evidence.
        """
        prompt_token_ids, sampling_params = self._preprocess_prompts(input_batch)

        tasks = []
        request_ids: list[str] = []
        for prompt in prompt_token_ids:
            # Schedule the collection of outputs for each prompt.
            # Avoid duplicate request_ids
            request_id = str(uuid4().hex)
            request_ids.append(request_id)
            task = asyncio.create_task(self._collect_outputs(prompt, request_id, sampling_params))
            tasks.append(task)
        try:
            outputs = await asyncio.gather(*tasks)
        except BaseException as e:
            # Cancel any sibling asyncio tasks still in flight.
            for t in tasks:
                if not t.done():
                    t.cancel()
            # Abort their vLLM-side request state so Ray releases the
            # ObjectRefs and the entrypoint's distributed-refcount table
            # doesn't accumulate orphan entries. We tolerate failures of
            # the abort itself — the goal is best-effort cleanup, not a
            # second hard exception.
            try:
                engine = self._get_engine()
                await engine.abort(request_ids)
            except Exception as abort_exc:
                logger.warning(
                    "generate() failed with %r and vllm engine.abort cleanup "
                    "also failed with %r — Ray ObjectRefs may leak",
                    e, abort_exc,
                )
            raise

        return self._postprocess_outputs(outputs)

    async def wake_up(self, *args: Any, **kwargs: Any):
        await self.llm.wake_up(tags=kwargs.get("tags", None))

    async def sleep(self, *args: Any, **kwargs: Any):
        engine = self._get_engine()
        output_processor = engine.output_processor
        # make sure that the engine is alive
        engine.engine_core.ensure_alive()
        if output_processor.has_unfinished_requests():
            logger.warning(
                "Calling sleep() with unfinished requests in vLLM engine. This is unexpected since all "
                "generation should be done before sleep() is called. Check for potential failures or "
                "dangling requests in your Generator/Env. Aborting all unfinished requests."
            )
            unfinished_request_ids = list(output_processor.request_states.keys())
            await engine.abort(unfinished_request_ids)

        # TODO(team): remove once vllm fixes this
        # otherwise waking it up will output gibberish: https://github.com/vllm-project/vllm/issues/17103
        await self.reset_prefix_cache()
        level = 1 if self._is_lora else kwargs.get("level", 2)
        await self.llm.sleep(level=level)

    async def init_weight_update_communicator(
        self, master_addr, master_port, rank_offset, world_size, group_name, backend, override_existing: bool = False
    ):
        engine = self._get_engine()
        return await engine.collective_rpc(
            "init_weight_update_communicator",
            args=(master_addr, master_port, rank_offset, world_size, group_name, backend, override_existing),
        )

    async def update_named_weights(self, request: NamedWeightsUpdateRequest):
        if "names" not in request:
            raise ValueError(f"Expected update weight request with 'names' entry, got keys: {request.keys()}")

        if not len(request["names"]):
            raise ValueError("Update weight request should have atleast one entry in 'names'")

        # Check for LoRA disk loading request
        if self._is_lora_disk_loading_request(request):
            lora_path = request["extras"][0]["lora_disk_path"]
            return await self._load_lora_from_disk(lora_path)

        # Use the weight loader to coordinate weight transfer
        return await self._weight_loader.load_weights(request)

    async def begin_weight_update(self):
        """Signal engines to start accumulating weights for batched loading."""
        engine = self._get_engine()
        return await engine.collective_rpc("begin_weight_update")

    async def end_weight_update(self):
        """Flush accumulated weights via model.load_weights()."""
        engine = self._get_engine()
        return await engine.collective_rpc("end_weight_update")

    async def read_engine_weights(self, hf_names, dump_inventory: bool = False):
        """TEST-ONLY (Stage 6 weight-equality gate): read engine-side weights back
        under the trainer's HF parameter names, gathered across all TP/EP workers.

        Returns ``List[Dict]`` (one dict per worker rank), each as produced by
        ``WorkerWrap.read_named_weights``. The caller assembles the per-rank
        contributions (TP/EP shards) into the full HF tensors to compare against
        the trainer's post-step weights.
        """
        engine = self._get_engine()
        return await engine.collective_rpc("read_named_weights", args=(list(hf_names), dump_inventory))

    async def teardown(self):
        await self._destroy_weights_update_group()

    async def reset_prefix_cache(self):
        engine = self._get_engine()
        await engine.reset_prefix_cache()

    async def _destroy_weights_update_group(self):
        engine = self._get_engine()
        return await engine.collective_rpc("destroy_weights_update_group")

    # ----------------------------------------
    # Methods for handling OpenAI API requests
    # ----------------------------------------

    async def _handle_openai_request(self, request_payload: Dict[str, Any], endpoint: str) -> Dict[str, Any]:
        """Handle OpenAI API request."""
        assert endpoint in ["/chat/completions", "/completions"]

        body = request_payload.get("json", {})
        headers = request_payload.get("headers", {})

        # Apply configured sampling params from generator config.
        # Harbor requests may include their own sampling params; we override
        # with the SkyRL generator config so rollout exploration is consistent.
        sp = getattr(self, "_openai_sampling_params", {})
        body.update({
            "temperature": sp.get("temperature", 1.0),
            "top_p": sp.get("top_p", 1.0),
            "top_k": sp.get("top_k", -1),
            "min_p": sp.get("min_p", 0.0),
        })

        # 1. Build request
        try:
            if endpoint == "/chat/completions":
                request = ChatCompletionRequest(**body)
            else:
                request = CompletionRequest(**body)
            assert request.stream is False, "Streaming is not supported in SkyRL yet, please set stream to False."
        except Exception as e:
            return _build_error_response(
                str(e), HTTPStatus.BAD_REQUEST.phrase, HTTPStatus.BAD_REQUEST.value
            )

        # 2. Call vllm engine
        try:
            # Create a minimal request-like object with attributes used by vLLM
            minimal_request = _MinimalRequest(headers)
            if endpoint == "/chat/completions":
                generator = await self.openai_serving_chat.create_chat_completion(request, minimal_request)
                assert isinstance(generator, (ChatCompletionResponse, ErrorResponse))
            else:
                generator = await self.openai_serving_completion.create_completion(request, minimal_request)
                assert isinstance(generator, (CompletionResponse, ErrorResponse))
            return generator.model_dump()

        except Exception as e:
            # Handle it here so we can surface the error from a ray worker.
            #
            # Input-overflow (VLLMValidationError raised at serving.py during
            # input validation, e.g. "You passed 32769 input tokens ... context
            # length is only 32768") is a *deterministic* client error: retrying
            # the identical over-budget prompt can never succeed. Classify it as
            # HTTP 400 (BAD_REQUEST) so that downstream LiteLLM/Harbor map it to
            # a ContextWindowExceededError (non-retryable) instead of treating a
            # generic 500 as a transient server error and retrying. Retrying the
            # doomed request across many concurrent trials is what exhausts the
            # entrypoint actor's file descriptors and aborts its uvloop event
            # loop (uv__epoll_ctl_prep SIGABRT) -> ray.WorkerCrashedError.
            # See project notes: nemotron-junit a3 #11 (chain 521442-448).
            is_input_overflow = False
            try:
                from vllm.exceptions import VLLMValidationError

                if isinstance(e, VLLMValidationError):
                    param = getattr(e, "parameter", None)
                    is_input_overflow = param == "input_tokens" or "input tokens" in str(e)
            except ImportError:
                # Older vLLM without VLLMValidationError: fall back to message match.
                is_input_overflow = "input tokens" in str(e) and "context length" in str(e)

            status = HTTPStatus.BAD_REQUEST if is_input_overflow else HTTPStatus.INTERNAL_SERVER_ERROR
            if is_input_overflow:
                logger.warning("Input-overflow rejected by vLLM serving (returning 400, non-retryable): %s", e)

            return _build_error_response(str(e), status.phrase, status.value)

    async def chat_completion(self, request_payload: Dict[str, Any]) -> Dict[str, Any]:
        """OpenAI-compatible HTTP endpoint for handling `/chat/completions` in Python vLLM engine.

        Accepts a JSON-serializable payload: {"json": <request-body>, "headers": <headers-dict>}.
        Constructs a minimal request-like object for vLLM's openai_serving_chat.
        Returns a plain dict, either a ChatCompletionResponse or an ErrorResponse, both defined
        in vllm.entrypoints.openai.protocol.
        """
        return await self._handle_openai_request(request_payload, endpoint="/chat/completions")

    async def completion(self, request_payload: Dict[str, Any]) -> Dict[str, Any]:
        """OpenAI-compatible HTTP endpoint for handling `/completions` in Python vLLM engine.

        Accepts a JSON-serializable payload: {"json": <request-body>, "headers": <headers-dict>}.
        Constructs a minimal request-like object for vLLM's openai_serving_completion.
        Returns a plain dict, either a CompletionResponse or an ErrorResponse, both defined
        in vllm.entrypoints.openai.protocol.
        """
        return await self._handle_openai_request(request_payload, endpoint="/completions")

    async def get_stats(self) -> Dict[str, Any]:
        """Get accumulated vLLM engine statistics for the current step.

        Returns a dict with the following keys:
        - peak_*: Peak values observed during the step
        - median_*: Median values across active samples
        - mean_*: Mean values across active samples
        - num_samples: Total number of stat samples collected
        - num_active_samples: Number of samples with active requests
        - timestamp: Unix timestamp of last sample
        - engine_id: Unique identifier for this engine instance

        Note: Stats are reset after reading to provide fresh stats per training step.

        Used by VLLMStatsCallback to collect and aggregate stats across engines.
        """
        # Reset=True ensures each training step gets fresh stats
        stats = V1LoggingStatLoggerFixed.get_stats_by_engine_id(self._stats_engine_id, reset=True)
        if stats is None:
            # Return empty stats if no data recorded yet
            stats = {
                # Peak values
                "peak_prompt_throughput": 0.0,
                "peak_generation_throughput": 0.0,
                "peak_running_reqs": 0,
                "peak_waiting_reqs": 0,
                "peak_gpu_cache_usage_perc": 0.0,
                "peak_prefix_cache_hit_rate": 0.0,
                # Median values
                "median_prompt_throughput": 0.0,
                "median_generation_throughput": 0.0,
                "median_running_reqs": 0.0,
                "median_waiting_reqs": 0.0,
                "median_gpu_cache_usage_perc": 0.0,
                "median_prefix_cache_hit_rate": 0.0,
                # Mean values
                "mean_prompt_throughput": 0.0,
                "mean_generation_throughput": 0.0,
                # Per-request latency stats
                "latency_prefill_mean": 0.0,
                "latency_prefill_median": 0.0,
                "latency_prefill_p90": 0.0,
                "latency_decode_mean": 0.0,
                "latency_decode_median": 0.0,
                "latency_decode_p90": 0.0,
                "latency_e2e_mean": 0.0,
                "latency_e2e_median": 0.0,
                "latency_e2e_p90": 0.0,
                "latency_queued_mean": 0.0,
                "latency_queued_median": 0.0,
                "latency_queued_p90": 0.0,
                "latency_ttft_mean": 0.0,
                "latency_ttft_median": 0.0,
                "latency_ttft_p90": 0.0,
                "latency_num_finished_requests": 0,
                "total_preempted_reqs": 0,
                # Legacy field names
                "avg_prompt_throughput": 0.0,
                "avg_generation_throughput": 0.0,
                "num_running_reqs": 0,
                "num_waiting_reqs": 0,
                "gpu_cache_usage_perc": 0.0,
                "prefix_cache_hit_rate": 0.0,
                # Metadata
                "num_samples": 0,
                "num_active_samples": 0,
                "timestamp": time.time(),
            }
        stats["engine_id"] = self._stats_engine_id
        return stats

    async def abort_generation(self) -> None:
        """
        Abort all running and waiting requests, which make the ongoing requests return the
        already-generated tokens with a stop_reason of "abort".
        """
        engine = self._get_engine()
        # Collect all request IDs currently tracked by the scheduler/output processor
        unfinished_request_ids = list(engine.output_processor.request_states.keys())
        if unfinished_request_ids:
            await engine.abort(unfinished_request_ids)
        await engine.reset_prefix_cache()  # avoid KV-cache pollution
        logger.info(f"abort_generation() finished, aborted {len(unfinished_request_ids)} requests")


class _MinimalRequest:
    """
    Minimal request-like object for vLLM's openai_serving_chat and openai_serving_completion.

    We cannot use the original user Request object because it cannot be serialized and hence
    cannot be a ray method argument. Instead we take the original request's headers and
    reconstruct an instance of _MinimalRequest to mimic the FastAPI Request object.

    The fields depend on what vLLM accesses internally.
    """

    def __init__(self, headers):
        self.headers = headers  # Expect a mapping with .get support
        self.state = SimpleNamespace()  # vLLM sets raw_request.state.request_metadata


class VLLMWeightTransferReceiver:
    """Receives weights via broadcast or CUDA IPC for vLLM.

    Handles both transfer strategies based on the request contents.
    Created locally in WorkerWrap with worker-specific state.
    """

    def __init__(self, model_update_group: Any, model_config: Any, device: torch.device) -> None:
        """Initialize the receiver with worker-local state.

        Args:
            model_update_group: Torch process group for weight updates.
            model_config: vLLM model configuration.
            device: CUDA device for this worker.
        """
        self.model_update_group = model_update_group
        self.model_config = model_config
        self.device = device

    def receive_weights(self, request: NamedWeightsUpdateRequest) -> Iterator[Tuple[str, torch.Tensor]]:
        """Receive weights and yield (name, tensor) tuples.

        Args:
            request: Weight update request with names, dtypes, shapes, and optionally IPC handles.
        """
        extras = request.get("extras")
        is_ipc = extras and len(extras) > 0 and "ipc_handles" in extras[0]

        if is_ipc:
            yield from self._receive_ipc(request)
        else:
            yield from self._receive_broadcast(request)

    def _receive_broadcast(self, request: NamedWeightsUpdateRequest) -> Iterator[Tuple[str, torch.Tensor]]:
        """Receive weights via torch.distributed.broadcast."""
        import os
        _fuse = os.environ.get("SKYRL_FUSE_WEIGHTS", "0") == "1"
        for name, dtype_str, shape in zip(request["names"], request["dtypes"], request["shapes"]):
            dtype = str_to_torch_dtype(dtype_str)
            if not _fuse:
                assert dtype == self.model_config.dtype, f"mismatch dtype: src {dtype}, dst {self.model_config.dtype}"
            # Always receive in sender's dtype, load_weights handles conversion
            weight = torch.empty(shape, dtype=dtype, device="cuda")
            torch.distributed.broadcast(weight, 0, group=self.model_update_group)
            yield name, weight

    def _receive_ipc(self, request: NamedWeightsUpdateRequest) -> Iterator[Tuple[str, torch.Tensor]]:
        """Receive weights via CUDA IPC handles."""
        names = request["names"]
        dtypes = request["dtypes"]
        shapes = request["shapes"]
        sizes = request.get("sizes", [])
        ipc_handles = [extra["ipc_handles"] for extra in request["extras"]]
        packed = request.get("packed", False)

        if packed:
            assert len(ipc_handles) == 1, "packed weight update should receive one ipc handle for all tensors"
            assert len(set(dtypes)) == 1, "packed weight update should have all tensors with the same dtype"
            assert (
                str_to_torch_dtype(dtypes[0]) == self.model_config.dtype
            ), f"mismatch dtype: src {dtypes[0]}, dst {self.model_config.dtype}"
            assert len(sizes) == len(names), "sizes must be provided for packed weight update"
            assert all(isinstance(size, int) for size in sizes), "sizes should be a list of integers"

            cuda_device = torch.cuda.current_device()
            props = torch.cuda.get_device_properties(cuda_device)
            physical_gpu_id = str(props.uuid)

            handle = ipc_handles[0][physical_gpu_id]
            device_id = self.device.index
            func, args = handle
            list_args = list(args)
            list_args[6] = device_id
            packed_tensor = func(*list_args)

            offset = 0
            for name, shape, size in zip(names, shapes, sizes):
                yield name, packed_tensor[offset : offset + size].view(*shape)
                offset += size
        else:
            cuda_device = torch.cuda.current_device()
            props = torch.cuda.get_device_properties(cuda_device)
            physical_gpu_id = str(props.uuid)
            for name, dtype_str, shape, ipc_handle in zip(names, dtypes, shapes, ipc_handles):
                dtype = str_to_torch_dtype(dtype_str)
                assert dtype == self.model_config.dtype, f"mismatch dtype: src {dtype}, dst {self.model_config.dtype}"

                handle = ipc_handle[physical_gpu_id]
                device_id = self.device.index
                func, args = handle
                list_args = list(args)
                list_args[6] = device_id
                weight = func(*list_args)
                yield name, weight


class VLLMWeightLoader(WeightLoader):
    """Loads weights into vLLM engine, managing RPC coordination.

    This loader encapsulates the collective_rpc calls to workers.
    Workers create VLLMWeightTransferReceiver locally for the actual weight transfer.
    """

    def __init__(self, engine: Any, is_async: bool = False) -> None:
        """Initialize the loader.

        Args:
            engine: The vLLM engine (LLM or AsyncLLMEngine).
            is_async: Whether this is for AsyncVLLMInferenceEngine.
        """
        self._engine = engine.engine if hasattr(engine, "engine") else engine
        self._is_async = is_async

    async def load_weights(self, request: NamedWeightsUpdateRequest) -> None:
        """Load weights by coordinating RPC to workers.

        Sends the request to workers via collective_rpc. Workers create
        the receiver locally and use it to receive and load weights.

        Args:
            request: Weight update request containing names, dtypes, shapes,
                    and optionally IPC handles.
        """
        if self._is_async:
            await self._engine.collective_rpc(
                "load_weights",
                args=(request,),
            )
        else:
            await asyncio.to_thread(
                self._engine.collective_rpc,
                "load_weights",
                args=(request,),
            )


VLLMRayActor = ray.remote(VLLMInferenceEngine)
AsyncVLLMRayActor = ray.remote(AsyncVLLMInferenceEngine)
