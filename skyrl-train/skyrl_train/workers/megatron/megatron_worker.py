import torch
import torch.nn as nn
import torch.distributed
import ray
from transformers import AutoTokenizer, AutoConfig
from huggingface_hub import snapshot_download

import asyncio
import importlib.util
import os
from enum import StrEnum
from functools import partial
from typing import List, Dict, Any, Optional
from collections import defaultdict
from loguru import logger
from skyrl_train.utils.progress import tqdm
from omegaconf import OmegaConf

from megatron.bridge import AutoBridge
import megatron.core.parallel_state as mpu
from megatron.core.optimizer import DistributedOptimizer
from megatron.core.optimizer_param_scheduler import OptimizerParamScheduler

from skyrl_train.distributed.megatron.optimizer import (
    init_megatron_optim_config,
    get_megatron_optimizer,
    get_megatron_optimizer_param_scheduler,
)
from skyrl_train.distributed.dispatch import MeshRank
from skyrl_train.distributed.utils import init_worker_process_group_with_device
from skyrl_train.distributed.megatron.megatron_strategy import MegatronStrategy
from skyrl_train.distributed.megatron.megatron_utils import print_model_size, broadcast_object_across_pp_ranks
from skyrl_train.utils.utils import update_model_config, str_to_torch_dtype, get_physical_gpu_id
from skyrl_train.utils.hf_load_retry import load_pretrained_with_retry
import skyrl_train.models.grug_megatron_bridge  # noqa: F401  # registers the Grug bridge with Megatron-Bridge
from skyrl_train.models.grug_moe import GRUG_MOE_MODEL_TYPE, is_grug_router_bias, validate_grug_training_strategy
from skyrl_train.training_batch import (
    GLOBAL_LOSS_DENOM_METADATA_KEY,
    TrainingBatchIterator,
    TrainingOutputBatch,
    gradient_accumulation_steps,
)
from skyrl_train.megatron_timing import (
    FINAL_BARRIER,
    OPTIMIZER_STEP,
    WORLD_METRIC_REDUCTION,
    MegatronTrainTimings,
    publish_megatron_train_timings,
)
from skyrl_train.learner_memory import LearnerMemory
from skyrl_train.optimizer_state_metrics import OptimizerStateObserver
from skyrl_train.utils.metrics import policy_progress_metrics, policy_training_metrics
from skyrl_train.workers.worker import (
    PolicyWorkerBase,
    RefWorkerBase,
    CriticWorkerBase,
)
from skyrl_train.workers.megatron.megatron_model_wrapper import MegatronModelWrapper, MegatronPolicyMicroBatch
from skyrl_train.utils.profiler import Profiler
from skyrl_train.weight_sync import WeightExtractor, WeightChunk
from skyrl_train.weight_sync.weight_extractor import validate_weight_sync_mode, weight_sync_dtype
from skyrl_train.workers.grug_validation import GrugValidationSnapshot
from skyrl_train.weight_change_probe import WirePublicationObserver


class _MegatronInitMode(StrEnum):
    TRAINING = "training"
    CHECKPOINT_EXPORT = "checkpoint-export"


class MegatronWeightExtractor(WeightExtractor):
    """Extracts weights from Megatron model-parallel models.

    Uses Megatron's bridge to export weights in HuggingFace format.

    Args:
        bridge: Megatron AutoBridge instance for weight conversion
        actor_module: The actor module to extract weights from
        model_type: HF ``model_type`` of the policy; selects per-tensor wire dtypes (Grug's router bias stays fp32)
        enable_bucketing: If True, group parameters into size-based buckets for packing
        bucket_size_threshold_GB: Size threshold in GB for bucketing (only used if enable_bucketing=True)
        training_dtype: Training dtype for size calculation (only used if enable_bucketing=True)
    """

    def __init__(
        self,
        bridge,
        actor_module,
        model_type: str,
        enable_bucketing: bool = False,
        bucket_size_threshold_GB: float = 1.0,
        training_dtype: torch.dtype = torch.bfloat16,
    ):
        self.bridge = bridge
        self.actor_module = actor_module
        self.model_type = model_type
        self.enable_bucketing = enable_bucketing
        self.bucket_size_threshold_GB = bucket_size_threshold_GB
        self.training_dtype = training_dtype

        # Initialize bucketing if enabled
        if enable_bucketing:
            self._init_param_buckets()
        else:
            self.param_buckets = None

    def _init_param_buckets(self):
        """Initialize parameter buckets for packing."""
        # Get conversion tasks from bridge
        weight_conversion_tasks = self.bridge.get_conversion_tasks(self.actor_module)

        # Calculate size for each parameter
        param_info = []

        def calculate_size_in_bytes(param, tp_size, ep_size):
            if param is None:
                # need to broadcast for other pp ranks
                size_in_bytes = None
            else:
                # Calculate size for this parameter
                prec_to_bytes = {
                    torch.bfloat16: 2,
                    torch.float32: 4,
                }
                scale = prec_to_bytes[self.training_dtype] / prec_to_bytes[param.dtype]
                size_in_bytes = param.element_size() * param.numel() * tp_size * ep_size * scale

            # Broadcast size_in_bytes across pipeline parallel ranks
            return broadcast_object_across_pp_ranks(size_in_bytes)

        for task in weight_conversion_tasks:
            param_info.append(
                (
                    task,
                    calculate_size_in_bytes(
                        task.param_weight,
                        task.mapping.tp_size,
                        task.mapping.ep_size if task.mapping.is_expert else 1,
                    ),
                )
            )

        # Group parameters into buckets based on size threshold. Each bucket is packed into one
        # buffer of a single dtype, so tensors with a non-default wire dtype get their own bucket.
        self.param_buckets = [[]]
        curr_size = 0
        for task, size in param_info:
            separate = self._has_special_wire_dtype(task)
            if separate or curr_size + size > self.bucket_size_threshold_GB * 1024**3:
                self.param_buckets.append([])
                curr_size = 0
            self.param_buckets[-1].append(task)
            curr_size += size
            if separate:
                self.param_buckets.append([])
                curr_size = 0
        self.param_buckets = [bucket for bucket in self.param_buckets if bucket]

    def _has_special_wire_dtype(self, task) -> bool:
        hf_names = task.mapping.hf_param
        if isinstance(hf_names, dict):
            hf_names = hf_names.values()
        else:
            hf_names = [hf_names]
        return any(is_grug_router_bias(self.model_type, name) for name in hf_names)

    def _wire_tensor(self, name: str, tensor: torch.Tensor, dtype: torch.dtype, device) -> torch.Tensor:
        return tensor.to(device=device, dtype=weight_sync_dtype(self.model_type, name, dtype), non_blocking=True)

    def extract_weights(self, dtype: torch.dtype):
        """Extract weights from Megatron model.

        Args:
            dtype: Target dtype for inference

        Yields:
            WeightChunk objects (one per parameter, or one per bucket if bucketing enabled)
        """
        device = torch.cuda.current_device()

        if not self.enable_bucketing:
            # No bucketing: yield one chunk per parameter
            hf_params_generator = self.bridge.export_hf_weights(
                self.actor_module,
                show_progress=False,
                conversion_tasks=None,
            )

            for name, tensor in hf_params_generator:
                tensor = self._wire_tensor(name, tensor, dtype, device)

                yield WeightChunk(
                    names=[name],
                    dtypes=[str(tensor.dtype)],
                    shapes=[list(tensor.shape)],
                    tensors=[tensor],
                )
        else:
            # Bucketing mode: iterate over buckets, yield one chunk per bucket
            for bucket in self.param_buckets:
                hf_params_generator = self.bridge.export_hf_weights(
                    self.actor_module,
                    show_progress=False,
                    conversion_tasks=bucket,
                )

                # Collect all parameters in this bucket into one chunk
                names = []
                dtypes_list = []
                shapes = []
                tensors = []

                for name, tensor in hf_params_generator:
                    tensor = self._wire_tensor(name, tensor, dtype, device)

                    names.append(name)
                    dtypes_list.append(str(tensor.dtype))
                    shapes.append(list(tensor.shape))
                    tensors.append(tensor)

                # Yield one chunk containing all parameters in this bucket
                if tensors:
                    yield WeightChunk(
                        names=names,
                        dtypes=dtypes_list,
                        shapes=shapes,
                        tensors=tensors,
                    )


class MegatronWorker:
    def init_configs(
        self, model_path, megatron_config, model_config_kwargs, transformer_config_kwargs, bf16=True, flash_attn=False
    ):
        """
        Initialize the Megatron-Bridge bridge and provider objects + hf_config and tokenizer
        """
        hf_config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
        validate_grug_training_strategy(getattr(hf_config, "model_type", None), "megatron")
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

        override_config_kwargs = {
            "bos_token_id": tokenizer.bos_token_id,
            "eos_token_id": tokenizer.eos_token_id,
            "pad_token_id": tokenizer.pad_token_id,
        }
        override_config_kwargs.update(model_config_kwargs.get("model_config", {}))
        update_model_config(hf_config, override_config_kwargs=override_config_kwargs)

        # if flash_attn is enabled, we use flash attention backend, otherwise fall back to fused attention backend
        transformer_config_kwargs = OmegaConf.to_container(transformer_config_kwargs, resolve=True)
        transformer_config_kwargs["attention_backend"] = "flash" if flash_attn else "fused"

        if not self.cfg.trainer.gradient_checkpointing:
            for key in ("recompute_granularity", "recompute_method", "recompute_num_layers"):
                transformer_config_kwargs[key] = None

        bridge = AutoBridge.from_hf_pretrained(model_path, trust_remote_code=True)
        provider = bridge.to_megatron_provider()
        provider.tensor_model_parallel_size = megatron_config.tensor_model_parallel_size
        provider.pipeline_model_parallel_size = megatron_config.pipeline_model_parallel_size
        provider.pipeline_dtype = torch.bfloat16 if bf16 else torch.float32
        provider.context_parallel_size = megatron_config.context_parallel_size
        provider.expert_model_parallel_size = megatron_config.expert_model_parallel_size
        provider.expert_tensor_parallel_size = megatron_config.expert_tensor_parallel_size
        provider.sequence_parallel = megatron_config.tensor_model_parallel_size > 1
        provider.attention_backend = "flash" if flash_attn else "fused"
        provider.variable_seq_lengths = True
        provider.masked_softmax_fusion = True
        provider.moe_token_dispatcher_type = "alltoall"
        # Megatron-Bridge enables wgrad fusion whenever Transformer Engine is present, but the
        # non-TE output layer still needs APEX's fused_weight_gradient_mlp_cuda extension.
        provider.gradient_accumulation_fusion = importlib.util.find_spec("fused_weight_gradient_mlp_cuda") is not None

        for k, v in transformer_config_kwargs.items():
            setattr(provider, k, v)
        provider.finalize()

        self.provider = provider
        self.bridge = bridge

        self.strategy.hf_config = hf_config
        self.tokenizer = tokenizer

    def make_megatron_module(
        self,
        wrap_with_ddp: bool = True,
        ddp_config: Optional[Dict[str, Any]] = None,
        bf16: bool = True,
    ) -> List[nn.Module]:
        """
        Creates a megatron GPTModel (optionally DDP wrapped) using the bridge.
        """
        from megatron.core.distributed.distributed_data_parallel_config import DistributedDataParallelConfig

        default_ddp_config = DistributedDataParallelConfig()
        if wrap_with_ddp:
            default_ddp_config.use_distributed_optimizer = True
        if ddp_config is not None:
            for k, v in ddp_config.items():
                setattr(default_ddp_config, k, v)
        model = self.provider.provide_distributed_model(
            ddp_config=default_ddp_config, wrap_with_ddp=wrap_with_ddp, bf16=bf16
        )
        return model

    def forward(self, data):
        """
        Override `Worker.forward` to support passing the full mini batch to the MegatronModelWrapper.forward method.
        """
        # Run in micro batches grouped into a single mini-batch
        micro_bsz = self.cfg.trainer.micro_forward_batch_size_per_gpu
        micro_batches = data.chunk(micro_bsz)

        # Build micro-batch dicts expected by policy.forward_mini_batch
        micro_dicts = []
        device = torch.cuda.current_device()
        for micro in micro_batches:
            micro.to(device)
            sequences = micro["sequences"]
            attention_mask = micro["attention_mask"]
            num_actions = micro.metadata["response_length"]
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 0)
            micro_dicts.append(
                {
                    "sequences": sequences,
                    "attention_mask": attention_mask,
                    "position_ids": position_ids,
                    "num_actions": num_actions,
                }
            )

        self.model.eval()
        seq_len = micro_dicts[0]["sequences"].shape[1]
        mbs = micro_dicts[0]["sequences"].shape[0]
        with torch.no_grad():
            log_probs = self.model.forward(
                micro_batches=micro_dicts,
                seq_len=seq_len,
                micro_batch_size=mbs,
                temperature=self.cfg.generator.sampling_params.temperature,
            )
        if self.cfg.trainer.policy.megatron_config.check_train_eval_parity:
            self._log_forward_fingerprint("forward", micro_dicts)
            with torch.no_grad():
                repeated = self.model.forward(
                    micro_batches=micro_dicts,
                    seq_len=seq_len,
                    micro_batch_size=mbs,
                    temperature=self.cfg.generator.sampling_params.temperature,
                )
            if mpu.is_pipeline_last_stage(ignore_virtual=True):
                diff = (repeated.float() - log_probs.float()).abs()
                logger.info(
                    f"parity probe forward() repeat dp_rank={mpu.get_data_parallel_rank()}: mean abs "
                    f"{diff.mean().item():.6f}, max abs {diff.max().item():.6f}"
                )

        log_probs = log_probs.to("cpu")
        output = TrainingOutputBatch({"output": log_probs})
        output.metadata = data.metadata
        return output

    def _log_forward_fingerprint(self, call: str, micro_dicts: List[dict]) -> None:
        """Log checksums of this rank's inputs and parameters so two calls can be compared."""
        token_sum = sum(int(micro["sequences"].long().sum().item()) for micro in micro_dicts)
        mask_sum = sum(int(micro["attention_mask"].long().sum().item()) for micro in micro_dicts)
        position_sum = sum(int(micro["position_ids"].long().sum().item()) for micro in micro_dicts)
        shapes = [tuple(micro["sequences"].shape) for micro in micro_dicts[:3]]
        with torch.no_grad():
            param_sum = 0.0
            param_count = 0
            for chunk in self.actor_module:
                for param in chunk.parameters():
                    param_sum += param.double().sum().item()
                    param_count += param.numel()
        logger.info(
            f"parity probe {call} fingerprint rank={torch.distributed.get_rank()} "
            f"dp_rank={mpu.get_data_parallel_rank()} pp_rank={mpu.get_pipeline_model_parallel_rank()} "
            f"micros={len(micro_dicts)} shapes={shapes} tokens={token_sum} mask={mask_sum} "
            f"positions={position_sum} num_actions={micro_dicts[0]['num_actions']} "
            f"params={param_count} param_sum={param_sum:.6f}"
        )

    def _log_train_eval_parity_probe(self, micro_buffer: List[MegatronPolicyMicroBatch]) -> None:
        """Re-run forward-only passes on the training micro-batches and compare them with the old log-probs.

        The eval-mode pass measures whether the log-prob forward is repeatable at all; the
        train-mode pass isolates module train/eval behaviour from the backward pass. The
        training metrics then report the remaining forward-backward drift.
        """
        micro_dicts = [
            {
                "sequences": micro.sequences,
                "attention_mask": micro.attention_mask,
                "position_ids": micro.position_ids,
                "num_actions": micro.num_actions,
            }
            for micro in micro_buffer
        ]
        self._log_forward_fingerprint("ppo_train", micro_dicts)
        seq_len = micro_buffer[0].sequences.shape[1]
        micro_bsz = micro_buffer[0].sequences.shape[0]
        old = torch.cat([micro.old_action_log_probs for micro in micro_buffer]).float()
        mask = torch.cat([micro.loss_mask for micro in micro_buffer]).bool()
        for mode in ("eval", "train"):
            self.model.eval() if mode == "eval" else self.model.train()
            with torch.no_grad():
                repeated = self.model.forward(
                    micro_batches=micro_dicts,
                    seq_len=seq_len,
                    micro_batch_size=micro_bsz,
                    temperature=self.cfg.generator.sampling_params.temperature,
                )
            if not mpu.is_pipeline_last_stage(ignore_virtual=True):
                continue
            diff = (repeated.float() - old.to(repeated.device)).abs()[mask.to(repeated.device)]
            logger.info(
                f"train/eval parity probe dp_rank={mpu.get_data_parallel_rank()} {mode}-mode forward vs old "
                f"log-probs: mean abs {diff.mean().item():.6f}, max abs {diff.max().item():.6f}, "
                f"exact fraction {(diff == 0).float().mean().item():.4f}"
            )

    def save_hf_model(self, export_dir: str, tokenizer):
        # Save model in HuggingFace safetensors format
        self.strategy.save_hf_model(
            self.bridge,
            self.model,
            export_dir,
            tokenizer=tokenizer,
        )


class MegatronPolicyWorkerBase(MegatronWorker, PolicyWorkerBase):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.model: MegatronModelWrapper = None
        self.actor_module: List[nn.Module] = None
        self.scheduler: OptimizerParamScheduler = None
        self.optimizer: DistributedOptimizer = None
        self.profiler: Profiler = None
        self._warned_exact_unit_policy_ratio = False
        self._memory = LearnerMemory(
            enabled=bool(OmegaConf.select(self.cfg, "trainer.policy_train_spans", default=False)), rank=self._rank
        )
        # A checkpoint can be restored after init_model, so the initial version is
        # unknown until this worker has completed an update with explicit metadata.
        self._completed_update: int | None = None
        self._optimizer_state_observer = OptimizerStateObserver(
            enabled=self.cfg.trainer.optimizer_state_metrics, rank=self._rank
        )

    def forward(self, data):
        with self._memory.span(
            "learner_logprob_forward", step=data.metadata.get("global_step"), step_kind="target_update"
        ):
            return super().forward(data)

    def offload_to_cpu(self, pin_memory=True, non_blocking=True, offload_optimizer=True, offload_model=True):
        self.strategy.offload_to_cpu(
            self.actor_module, self.optimizer, pin_memory, non_blocking, offload_optimizer, offload_model
        )

    def backload_to_gpu(self, non_blocking=True, backload_optimizer=True, backload_model=True):
        self.strategy.backload_to_gpu(
            self.actor_module, self.optimizer, non_blocking, backload_optimizer, backload_model
        )

    def init_worker_process_group(self):
        """
        Override DistributedTorchRayActor.init_worker_process_group to use megatron distributed setup to create the mesh.
        """
        # Device-pinned NCCL PG init via the shared helper — pins set_device(LOCAL_RANK) and
        # passes device_id so ProcessGroupNCCL never "guesses device ID based on global rank".
        # The guess deadlocks the first collective (weight-init barrier) on unmasked-CVD clusters
        # where every actor sees all GPUs (cw-rno2a); see init_worker_process_group_with_device.
        init_worker_process_group_with_device(
            timeout_seconds=int(self.cfg.trainer.distributed.worker_collective_timeout_seconds)
        )

        # Explicitly wrap torch.distributed.broadcast in torch.no_grad() to avoid a warning in Megatron training where the
        # autograd engine tries to track gradients through the default Torch kernel. This fixes a deprecated behaviour in
        # PyTorch, preventing potential silent errors in future versions.

        if not getattr(torch.distributed, "_skyrl_broadcast_no_grad_patched", False):
            _orig_broadcast = torch.distributed.broadcast

            def _broadcast_no_grad(*args, **kwargs):
                with torch.no_grad():
                    return _orig_broadcast(*args, **kwargs)

            torch.distributed.broadcast = _broadcast_no_grad
            torch.distributed._skyrl_broadcast_no_grad_patched = True

        self.strategy = MegatronStrategy(
            megatron_config=self.cfg.trainer.policy.megatron_config,
            optimizer_config=self.cfg.trainer.policy.optimizer_config,
            seed=self.cfg.trainer.seed,
        )
        self.strategy.setup_distributed()

        self.mesh_rank = MeshRank(
            dp=mpu.get_data_parallel_rank(),
            sp=mpu.get_context_parallel_rank(),
            tp=mpu.get_tensor_model_parallel_rank(),
            pp=mpu.get_pipeline_model_parallel_rank(),
            world_size=self._world_size,
            dp_size=mpu.get_data_parallel_world_size(),
            pp_size=mpu.get_pipeline_model_parallel_world_size(),
        )

    def _initialize_policy_modules(self, model_path: str, *, mode: _MegatronInitMode) -> None:
        """Construct the shared Megatron model graph at the checkpoint geometry."""
        for_training = mode is _MegatronInitMode.TRAINING
        self.init_configs(
            model_path,
            self.cfg.trainer.policy.megatron_config,
            self.cfg.trainer.policy.megatron_config.model_config_kwargs,
            self.cfg.trainer.policy.megatron_config.transformer_config_kwargs,
            bf16=self.cfg.trainer.bf16,
            flash_attn=self.cfg.trainer.flash_attn,
        )

        self.actor_module = self.make_megatron_module(
            wrap_with_ddp=for_training,
            ddp_config=self.cfg.trainer.policy.megatron_config.ddp_config if for_training else None,
            bf16=self.cfg.trainer.bf16,
        )

        if self._local_rank == 0 and not os.path.exists(
            model_path
        ):  # if not local path, try downloading model weights from huggingface
            # Retry transient HF weight-index/safetensors fetch flakes (EOF /
            # IncompleteRead / dropped connection / spurious "no .safetensors")
            # that otherwise kill the whole gang at scale; genuine missing/auth
            # failures still surface. no-op if already downloaded.
            retry = self.cfg.trainer.model_load_retry
            load_pretrained_with_retry(
                lambda: snapshot_download(model_path),
                model_id=model_path,
                max_retries=int(retry.max_retries),
                backoff_base=float(retry.backoff_base_seconds),
                backoff_cap=float(retry.backoff_cap_seconds),
            )
        torch.distributed.barrier()

        if self._rank == 0:
            print_model_size(self.actor_module[0])

    def init_model(self, model_path, num_training_steps: int = 1e9):
        """Initialize the model, optimizer, and scheduler for the policy worker."""
        self._initialize_policy_modules(model_path, mode=_MegatronInitMode.TRAINING)

        # create profiler
        if self.cfg.trainer.policy.megatron_config.torch_profiler_config.enable:
            self.profiler = Profiler(self.cfg.trainer.policy.megatron_config.torch_profiler_config)

        # create optimizer
        optim_config = init_megatron_optim_config(
            self.cfg.trainer.policy.optimizer_config, self.cfg.trainer.policy.megatron_config.optimizer_config_kwargs
        )
        self.optimizer = get_megatron_optimizer(self.actor_module, optim_config)

        self._normalize_mini_batch_size()

        # create scheduler
        self.scheduler = get_megatron_optimizer_param_scheduler(
            optimizer=self.optimizer,
            config=self.cfg.trainer.policy.optimizer_config,
            num_training_steps=num_training_steps,
        )

        # create worker model
        self.model = MegatronModelWrapper(
            config=self.cfg,
            actor_module=self.actor_module,
            actor_optimizer=self.optimizer,
            policy_loss_fn=self.policy_loss_fn,
            logprob_chunk_size=OmegaConf.select(
                self.cfg, "trainer.policy.megatron_config.logprob_chunk_size", default=None
            ),
        )

        # Initialize weight extractor
        self.use_cuda_ipc = self.cfg.generator.weight_sync_backend == "nccl" and self.cfg.trainer.placement.colocate_all
        # TODO(haochen): Now bucketing is only enabled for the CUDA IPC
        # transfer strategy, we can enable it for other strategies as well.
        model_type = self.strategy.hf_config.model_type
        validate_weight_sync_mode(model_type, fuse_weights=bool(self.cfg.generator.fuse_weights))
        self.weight_extractor = MegatronWeightExtractor(
            bridge=self.bridge,
            actor_module=self.actor_module,
            model_type=model_type,
            enable_bucketing=self.use_cuda_ipc,
            bucket_size_threshold_GB=self.cfg.generator.weight_transfer_threshold_cuda_ipc_GB,
            training_dtype=torch.bfloat16 if self.cfg.trainer.bf16 else torch.float32,
        )

        self.empty_cuda_cache = self.cfg.trainer.policy.megatron_config.empty_cuda_cache
        self._memory.snapshot("model_ready")

    def init_model_for_export(self, model_path: str) -> None:
        """Initialize Megatron model structure without optimizer or training state."""
        self._initialize_policy_modules(model_path, mode=_MegatronInitMode.CHECKPOINT_EXPORT)
        self.model = MegatronModelWrapper(
            config=self.cfg,
            actor_module=self.actor_module,
            logprob_chunk_size=OmegaConf.select(
                self.cfg, "trainer.policy.megatron_config.logprob_chunk_size", default=None
            ),
        )

    # This cannot inherit PolicyWorkerBase.ppo_train: Megatron Core must own
    # pipeline scheduling and gradient accumulation, so only policy semantics
    # are shared with the ordinary worker through backend-neutral utilities.
    def ppo_train(self, train_data) -> "TrainingOutputBatch":
        """Train through Megatron Core's pipeline scheduler."""
        timing = MegatronTrainTimings(
            enabled=bool(OmegaConf.select(self.cfg, "trainer.policy_train_spans", default=False))
        )
        outcome = "failure"
        try:
            with self._memory.span(
                "ppo_forward_backward_update", step=int(train_data.metadata["global_step"]), step_kind="target_update"
            ):
                output = self._ppo_train_with_timings(train_data, timing)
            self._completed_update = int(train_data.metadata["global_step"])
            outcome = "success"
            return output
        finally:
            try:
                observations = timing.finish()
                if observations:
                    publish_megatron_train_timings(
                        observations,
                        step=int(train_data.metadata["global_step"]),
                        rank=torch.distributed.get_rank(),
                        outcome=outcome,
                    )
            except Exception as error:
                logger.warning("Could not publish Megatron policy timings: {}", error)

    def _ppo_train_with_timings(self, train_data, timing: MegatronTrainTimings) -> "TrainingOutputBatch":
        dataloader = TrainingBatchIterator(train_data, self.cfg.trainer.micro_train_batch_size_per_gpu)

        micro_batches_per_mini_batch = gradient_accumulation_steps(
            self.policy_mini_batch_size_per_gpu,
            self.cfg.trainer.micro_train_batch_size_per_gpu,
        )

        status_list = []
        all_metrics = defaultdict(list)
        policy_update_steps = 0

        if self.profiler is not None:
            self.profiler.start()

        for epoch in range(self.cfg.trainer.update_epochs_per_batch):
            self.optimizer.zero_grad()
            pbar = tqdm(
                dataloader,
                desc=f"Policy Train epoch [{epoch + 1}/{self.cfg.trainer.update_epochs_per_batch}]",
                disable=not self.strategy.is_rank_0(),
            )

            micro_buffer = []
            for local_step, experience in enumerate(pbar):
                experience.to_device(torch.cuda.current_device())
                sequences = experience.sequences
                attention_mask = experience.attention_mask
                position_ids = attention_mask.long().cumsum(-1) - 1
                position_ids.masked_fill_(attention_mask == 0, 0)

                micro_buffer.append(
                    MegatronPolicyMicroBatch(
                        sequences=sequences,
                        attention_mask=attention_mask,
                        position_ids=position_ids,
                        num_actions=experience.num_actions,
                        old_action_log_probs=experience.action_log_probs,
                        base_action_log_probs=experience.base_action_log_probs,
                        advantages=experience.advantages,
                        loss_mask=experience.loss_mask,
                        rollout_action_logprobs=experience.rollout_logprobs,
                        response_span_tags=experience.response_span_tags,
                        global_loss_denom=(experience.metadata or {}).get(GLOBAL_LOSS_DENOM_METADATA_KEY),
                    )
                )

                if len(micro_buffer) == micro_batches_per_mini_batch:
                    if self.cfg.trainer.policy.megatron_config.check_train_eval_parity:
                        self._log_train_eval_parity_probe(micro_buffer)
                    # run mini-batch forward-backward and then one optimizer step
                    self.model.train()
                    for chunk in self.actor_module:
                        # if use distributed optimizer, zero grad buffer will be handled by optimizer
                        chunk.zero_grad_buffer()
                    seq_len = micro_buffer[0].sequences.shape[1]
                    micro_bsz = micro_buffer[0].sequences.shape[0]

                    metrics_list = self.model.forward_backward_mini_batch(
                        micro_batches=micro_buffer,
                        seq_len=seq_len,
                        micro_batch_size=micro_bsz,
                        temperature=self.cfg.generator.sampling_params.temperature,
                        timings=timing,
                    )

                    if self.empty_cuda_cache:
                        torch.cuda.empty_cache()

                    with timing.span(OPTIMIZER_STEP):
                        grad_norm = self.strategy.optimizer_step(
                            self.optimizer,
                            self.model,
                            self.scheduler,
                            name="actor",
                            after_step=(
                                partial(
                                    self._optimizer_state_observer.after_step,
                                    model_chunks=self.actor_module,
                                    optimizer=self.optimizer,
                                    step=int(train_data.metadata["global_step"]),
                                    minibatch=policy_update_steps + 1,
                                )
                                if self._optimizer_state_observer.enabled
                                else None
                            ),
                        )

                    # within a DP group, metrics are already the same across all workers - we then just all reduce across
                    # the whole world size to get the metrics for the global micro batch
                    for i, metrics in enumerate(metrics_list):
                        status = metrics.copy()
                        status["policy_lr"] = self.optimizer.param_groups[0]["lr"]
                        if not self.cfg.trainer.algorithm.use_kl_loss:
                            status.pop("policy_kl")

                        # Attach grad norm only for the last micro in the mini-batch
                        if i == len(metrics_list) - 1 and grad_norm is not None:
                            status["raw_grad_norm"] = grad_norm

                        # attach response_length
                        status["response_length"] = micro_buffer[i].num_actions

                        with timing.span(WORLD_METRIC_REDUCTION):
                            status = self.strategy.all_reduce(status)
                        status_list.append(status)
                        for k, v in status.items():
                            all_metrics[k].append(v)

                    pbar.set_postfix(policy_progress_metrics(status_list[-1]))

                    policy_update_steps += 1
                    micro_buffer = []

            # drop any trailing micros that don't fill a mini-batch (keep behavior consistent)
            micro_buffer = []

        with timing.span(FINAL_BARRIER):
            torch.distributed.barrier()
        if self.profiler is not None:
            self.profiler.stop_and_save()
            self.profiler.stop_trace()

        status_mean = policy_training_metrics(all_metrics, policy_update_steps)
        if status_mean.get("ppo_ratio_exact_unit_fraction") == 1.0 and not self._warned_exact_unit_policy_ratio:
            logger.warning(
                "Megatron's recomputed old log probabilities exactly match the training forward for every policy "
                "token. PPO clip bounds cannot activate until the mini-batch contains a forward after an optimizer "
                "update; clip-bound sweeps are inert with the current update geometry."
            )
            self._warned_exact_unit_policy_ratio = True

        output = TrainingOutputBatch()
        output.metadata = {"train_status": status_mean}
        return output

    async def broadcast_to_inference_engines(self, inference_engine_client, *, publication=None):
        # Enclose extraction as well as transfer: gathering/conversion can peak
        # before the first tensor reaches the publication communicator.
        probe = None
        if publication is not None:
            if self.use_cuda_ipc:
                raise ValueError("wire probe does not support CUDA IPC publication")
            if torch.distributed.get_rank() == 0:
                if not hasattr(self, "_wire_publication_observer"):
                    self._wire_publication_observer = WirePublicationObserver(seed=int(self.cfg.trainer.seed))
                probe = self._wire_publication_observer
                probe.begin(**publication)
        try:
            with self._memory.span("weight_publication", step=self._completed_update, step_kind="completed_update"):
                result = await self._broadcast_to_inference_engines(inference_engine_client, probe=probe)
        except BaseException:
            if probe is not None:
                probe.finish(publication_id=publication["publication_id"], success=False)
            raise
        if probe is not None:
            probe.staging_complete()
        return result

    def finish_weight_change_probe(self, publication_id: str):
        """The driver calls this only after every publication RPC succeeded."""
        if torch.distributed.get_rank() == 0:
            self._wire_publication_observer.finish(publication_id=publication_id, success=True)

    async def _broadcast_to_inference_engines(self, inference_engine_client, *, probe=None):
        from torch.multiprocessing.reductions import reduce_tensor

        use_prefix_cache = self.cfg.generator.enable_prefix_caching
        generator_dtype = str_to_torch_dtype(self.cfg.generator.model_dtype)
        cache_reset_task = None
        if use_prefix_cache and torch.distributed.get_rank() == 0:
            # clear prefix cache
            cache_reset_task = inference_engine_client.reset_prefix_cache()

        torch.cuda.empty_cache()

        # #1685 fix ported from fsdp_worker.broadcast_to_inference_engines (FlashInfer-CUTLASS
        # w13 gate/up swap skipped on the megatron RL update path -> MoE token-salad): bracket
        # the WHOLE multi-chunk sync with vLLM's layerwise reload so per-chunk model.load_weights
        # DEFER processing and a single finalize re-runs process_weights_after_loading
        # (re-applying swap_w13_to_w31) EXACTLY once. Without it the engine holds checkpoint
        # [gate;up] while the FlashInfer-CUTLASS kernel reads [up;gate]. Swap-inert on
        # triton/dense backends, so byte-identical there. Rank 0 drives
        # the engine RPC (same global-rank-0 semantics the broadcast/update loop uses below).
        _w13_bracket = not self.use_cuda_ipc and not bool(self.cfg.generator.fuse_weights)

        # Extract weights using the initialized extractor
        if not self.use_cuda_ipc:
            # Open the layerwise-reload bracket (rank 0 drives the engine RPC).
            if _w13_bracket and torch.distributed.get_rank() == 0:
                await inference_engine_client.begin_weight_reload()
            if _w13_bracket:
                torch.distributed.barrier()

            # Broadcast path: one chunk per parameter
            # NOTE: need to optimize this to use buckets for non-colocated weight sync as well
            for chunk in self.weight_extractor.extract_weights(generator_dtype):
                # Each chunk contains one parameter
                assert len(chunk) == 1
                name = chunk.names[0]
                tensor = chunk.tensors[0]

                if probe is not None:
                    probe.capture(name, tensor)

                if torch.distributed.get_rank() == 0:
                    update_weight_task = asyncio.create_task(
                        inference_engine_client.update_named_weights(
                            {
                                "names": [name],
                                "dtypes": [chunk.dtypes[0]],
                                "shapes": [list(tensor.shape)],
                            }
                        )
                    )

                # Broadcast weights from training rank 0 to inference engine ranks via the update group
                def broadcast_tensor(tensor):
                    if torch.distributed.get_rank() == 0:
                        torch.distributed.broadcast(tensor.data, 0, group=self._model_update_group)

                await asyncio.to_thread(broadcast_tensor, tensor)
                if torch.distributed.get_rank() == 0:
                    await update_weight_task
                torch.distributed.barrier()

            # Close the layerwise-reload bracket: finalize_layerwise_reload re-runs
            # process_weights_after_loading over every layer ONCE -> re-applies the
            # FlashInfer-CUTLASS w13 [gate;up]->[up;gate] swap the per-chunk loads skipped.
            if _w13_bracket:
                torch.distributed.barrier()
                if torch.distributed.get_rank() == 0:
                    await inference_engine_client.finish_weight_reload()
        else:
            # CUDA IPC path: one chunk per bucket (for packing)
            device = torch.cuda.current_device()
            weights_update_request = {"names": [], "dtypes": [], "shapes": [], "sizes": [], "extras": []}

            for chunk in self.weight_extractor.extract_weights(generator_dtype):
                # Each chunk contains all parameters in one bucket
                # Calculate total size for packing (in number of elements)
                total_numel = sum(t.numel() for t in chunk.tensors)
                chunk_dtypes = {t.dtype for t in chunk.tensors}
                assert len(chunk_dtypes) == 1, f"packed weight chunk mixes dtypes: {chunk_dtypes}"
                packed_tensor = torch.empty(
                    total_numel,
                    device=device,
                    dtype=chunk_dtypes.pop(),
                    requires_grad=False,
                )

                offset = 0
                # Copy tensors into consolidated buffers
                for name, tensor, shape, dtype_name in zip(chunk.names, chunk.tensors, chunk.shapes, chunk.dtypes):
                    size = tensor.numel()
                    packed_tensor[offset : offset + size].copy_(tensor.detach().view(-1))
                    offset += size
                    weights_update_request["names"].append(name)
                    weights_update_request["dtypes"].append(dtype_name)
                    weights_update_request["shapes"].append(shape)
                    weights_update_request["sizes"].append(size)

                ipc_handle = reduce_tensor(packed_tensor)
                ipc_handle = {get_physical_gpu_id(): ipc_handle}
                ipc_handle_list = [None] * torch.distributed.get_world_size()
                torch.distributed.all_gather_object(ipc_handle_list, ipc_handle)

                ipc_handles = {}
                for d in ipc_handle_list:
                    ipc_handles.update(d)

                weights_update_request["extras"].append({"ipc_handles": ipc_handles})
                weights_update_request["packed"] = True

                if torch.distributed.get_rank() == 0:
                    await inference_engine_client.update_named_weights(weights_update_request)
                    weights_update_request = {"names": [], "dtypes": [], "shapes": [], "sizes": [], "extras": []}

                # force collect any sent tensors if possible to be memory efficient
                torch.cuda.ipc_collect()

        torch.distributed.barrier()
        torch.cuda.synchronize()

        if cache_reset_task is not None:
            await cache_reset_task
        torch.cuda.empty_cache()
        torch.distributed.barrier()

    def grug_validation_snapshot(self, names=()):
        """Return the calling rank and requested Grug weights in HF layout, gathered on rank 0.

        Every rank must call this with the same names because the bridge export
        is collective. The weights mapping is empty on nonzero ranks.
        """
        if self.strategy.hf_config.model_type != GRUG_MOE_MODEL_TYPE:
            raise ValueError("grug_validation_snapshot is only valid for Grug models")
        wanted = set(names)
        is_rank0 = torch.distributed.get_rank() == 0
        weights = {}
        for name, tensor in self.bridge.export_hf_weights(self.actor_module, show_progress=False):
            if is_rank0 and name in wanted:
                weights[name] = tensor.detach().to("cpu", dtype=torch.float32).contiguous()
        missing = wanted.difference(weights) if is_rank0 else set()
        if missing:
            raise KeyError(f"missing Grug state entries: {sorted(missing)}")
        return GrugValidationSnapshot(
            rank=torch.distributed.get_rank(),
            attention_backend=str(self.provider.attention_backend),
            weights=weights,
        )

    def get_weight_statistics(self):
        """Compute lightweight statistics for model weights"""
        raise NotImplementedError()

    def _set_pad_token_id(self, pad_token_id):
        # this already gets set in the init_model method
        pass


class MegatronRefWorkerBase(MegatronWorker, RefWorkerBase):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.model: MegatronModelWrapper = None
        self.actor_module: List[nn.Module] = None

    def offload_to_cpu(self, pin_memory=True, non_blocking=True, **kwargs):
        self.strategy.offload_to_cpu(self.actor_module, None, pin_memory, non_blocking)

    def backload_to_gpu(self, non_blocking=True, **kwargs):
        self.strategy.backload_to_gpu(self.actor_module, None, non_blocking)

    def init_worker_process_group(self):
        """
        Override DistributedTorchRayActor.init_worker_process_group to use megatron distributed setup to create the mesh.
        """
        # Device-pinned NCCL PG init via the shared helper (see init_worker_process_group_with_device) —
        # avoids the ProcessGroupNCCL device-guess collective deadlock on unmasked-CVD clusters (cw-rno2a).
        init_worker_process_group_with_device(
            timeout_seconds=int(self.cfg.trainer.distributed.worker_collective_timeout_seconds)
        )

        self.strategy = MegatronStrategy(
            megatron_config=self.cfg.trainer.ref.megatron_config,
            optimizer_config=None,
            seed=self.cfg.trainer.seed,
        )
        self.strategy.setup_distributed()

        self.mesh_rank = MeshRank(
            dp=mpu.get_data_parallel_rank(),
            sp=mpu.get_context_parallel_rank(),
            tp=mpu.get_tensor_model_parallel_rank(),
            pp=mpu.get_pipeline_model_parallel_rank(),
            world_size=self._world_size,
            dp_size=mpu.get_data_parallel_world_size(),
            pp_size=mpu.get_pipeline_model_parallel_world_size(),
        )

    def init_model(self, model_path, num_training_steps: int = 1e9):
        """
        Initialize the model for the ref worker.
        """
        # initialize the bridge and provider objects
        self.init_configs(
            model_path,
            self.cfg.trainer.ref.megatron_config,
            self.cfg.trainer.ref.megatron_config.model_config_kwargs,
            self.cfg.trainer.ref.megatron_config.transformer_config_kwargs,
            bf16=self.cfg.trainer.bf16,
            flash_attn=self.cfg.trainer.flash_attn,
        )

        self.actor_module = self.make_megatron_module(
            wrap_with_ddp=False,
            ddp_config=None,
            bf16=self.cfg.trainer.bf16,
        )

        # download model weights from huggingface (need to be done for ref worker as well, else errors when colocate_all=False)
        if self._local_rank == 0 and not os.path.exists(
            model_path
        ):  # if not local path, try downloading model weights from huggingface
            # Retry transient HF weight-index/safetensors fetch flakes (EOF /
            # IncompleteRead / dropped connection / spurious "no .safetensors")
            # that otherwise kill the whole gang at scale; genuine missing/auth
            # failures still surface. no-op if already downloaded.
            retry = self.cfg.trainer.model_load_retry
            load_pretrained_with_retry(
                lambda: snapshot_download(model_path),
                model_id=model_path,
                max_retries=int(retry.max_retries),
                backoff_base=float(retry.backoff_base_seconds),
                backoff_cap=float(retry.backoff_cap_seconds),
            )
        torch.distributed.barrier()

        # load weights
        if self._rank == 0:
            print_model_size(self.actor_module[0])

        # create worker model — ref honors its OWN logprob_chunk_size key
        self.model = MegatronModelWrapper(
            config=self.cfg,
            actor_module=self.actor_module,
            logprob_chunk_size=OmegaConf.select(
                self.cfg, "trainer.ref.megatron_config.logprob_chunk_size", default=None
            ),
        )

    def get_weight_statistics(self):
        """Compute lightweight statistics for model weights"""
        raise NotImplementedError()

    def _set_pad_token_id(self, pad_token_id):
        # this already gets set in the init_model method
        pass


class MegatronCriticWorkerBase(MegatronWorker, CriticWorkerBase):
    def __init__(self, **kwargs):
        raise NotImplementedError()


PolicyWorker = ray.remote(num_gpus=1)(MegatronPolicyWorkerBase)
RefWorker = ray.remote(num_gpus=1)(MegatronRefWorkerBase)
CriticWorker = ray.remote(num_gpus=1)(MegatronCriticWorkerBase)
