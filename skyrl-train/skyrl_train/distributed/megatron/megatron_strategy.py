import os
import random
from datetime import timedelta
from typing import List, Union, Optional
from jaxtyping import Float

import numpy as np
import torch
import torch.nn as nn
from torch import optim
from torch import distributed as dist

from skyrl_train.distributed.strategy import DistributedStrategy
from skyrl_train.distributed.utils import ModelOrModelOptimPair
from skyrl_train.utils.io import io
from skyrl_train.workers.megatron.megatron_model_wrapper import MegatronModelWrapper
import megatron.core.parallel_state as mpu
from skyrl_train.distributed.megatron.megatron_utils import (
    offload_megatron_model_to_cpu,
    load_megatron_model_to_gpu,
    offload_megatron_optimizer,
    load_megatron_optimizer,
    offload_megatron_grads_to_cpu,
    load_megatron_grads_to_gpu,
)

from megatron.core.dist_checkpointing.strategies import base as ckpt_base
from megatron.core.dist_checkpointing.strategies.async_utils import AsyncCallsQueue
from megatron.core import dist_checkpointing
from megatron.core.dist_checkpointing.serialization import (
    get_default_load_sharded_strategy,
    get_default_save_sharded_strategy,
)
from megatron.core.dist_checkpointing.strategies.fully_parallel import (
    FullyParallelLoadStrategyWrapper,
    FullyParallelSaveStrategyWrapper,
)
from transformers import PreTrainedTokenizer
from megatron.core.optimizer import DistributedOptimizer
from megatron.core.optimizer_param_scheduler import OptimizerParamScheduler


# Optimizer checkpoint format and gather transport.
#
# Save and load the optimizer state with megatron's 'fully_reshardable' format in its
# memory-efficient mode (`distrib_optim_fully_reshardable_mem_efficient=True`). This mode does
# two things:
#   * It makes standard (non-flattened) ShardedTensors. The torch_dist save strategy accepts
#     these. Megatron's default format, 'fully_sharded_model_space', makes `flattened_range`
#     tensors instead. torch_dist rejects those (`ShardedTensor.flattened_range is not
#     supported`), which crashed the east18 run at the gs1 checkpoint.
#   * It gathers the DP-rank-0 optimizer state over gloo (`device="cpu"`), not CUDA.
#
# We used 'dp_zero_gather_scatter' before. torch_dist accepts its tensors too. But its
# `sharded_param_state_dp_zero` always calls `get_parameter_state_dp_zero(use_gloo_comm=False)`,
# which stages the gather on the GPU (`send_tensor.cuda()`). Right after a training step the model,
# the fp32 grad buffer, and the optimizer state all sit on the GPU. So that gather runs out of
# memory at save: a 30B-A3B run died at gs1 when it asked for 6.75 GiB with 5.95 GiB free. Upstream
# also marks that format "will be deprecated". 'fully_reshardable' + mem_efficient is megatron's own
# knob for this problem: the gloo gather needs no GPU memory, so we do not patch megatron internals.
#
# Pass this through `metadata=`. The positional `sharding_type=` argument is deprecated in
# megatron-core 0.18. Save and load must use the same format. Megatron writes the format into the
# checkpoint (`param_state_sharding_type`) and reads it back on load. We still pass it on load so the
# load template matches.
_MEGATRON_OPTIM_CKPT_METADATA = {
    "distrib_optim_sharding_type": "fully_reshardable",
    "distrib_optim_fully_reshardable_mem_efficient": True,
}


class MegatronStrategy(DistributedStrategy):
    """
    The strategy for training with Megatron.
    """

    def __init__(
        self,
        megatron_config,
        optimizer_config=None,
        seed: int = 42,
    ) -> None:
        super().__init__()
        self.megatron_config = megatron_config
        self.optimizer_config = optimizer_config
        self.seed = seed
        self.hf_config = None  # Set by the megatron worker once configs are initialized.

        # NOTE: Set Megatron dist checkpoint async backend to persistent to avoid `os.fork()`-ing
        # short-lived background workers, which does not work well with Ray.
        ckpt_base.async_calls = AsyncCallsQueue(persistent=True)

    def set_seed(self, seed: int) -> None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        if torch.cuda.device_count() > 0:
            from megatron.core import tensor_parallel

            tensor_parallel.model_parallel_cuda_manual_seed(seed)

    def setup_distributed(self, timeout=timedelta(minutes=30)) -> None:
        local_rank = int(os.environ.get("LOCAL_RANK", "-1"))
        if local_rank != -1:
            torch.cuda.set_device(local_rank)

        mpu.initialize_model_parallel(
            tensor_model_parallel_size=self.megatron_config.tensor_model_parallel_size,
            pipeline_model_parallel_size=self.megatron_config.pipeline_model_parallel_size,
            expert_model_parallel_size=self.megatron_config.expert_model_parallel_size,
            expert_tensor_parallel_size=self.megatron_config.expert_tensor_parallel_size,
            use_sharp=False,
            context_parallel_size=self.megatron_config.context_parallel_size,
            nccl_communicator_config_path=None,
        )
        self.set_seed(self.seed)
        self.world_size = dist.get_world_size()

    def offload_to_cpu(
        self, model, optimizer, pin_memory=True, non_blocking=True, offload_optimizer=True, offload_model=True
    ):
        """
        Offload model weights and optimizer to CPU memory.
        """
        if offload_model:
            offload_megatron_model_to_cpu(model)
        if optimizer and offload_optimizer:
            offload_megatron_grads_to_cpu(model)
            offload_megatron_optimizer(optimizer)
        torch.cuda.synchronize()
        torch.cuda.empty_cache()

    def backload_to_gpu(self, model, optimizer, non_blocking=True, backload_optimizer=True, backload_model=True):
        """Reload model weights back to GPU."""
        if backload_model:
            load_megatron_model_to_gpu(model)
        if optimizer and backload_optimizer:
            load_megatron_grads_to_gpu(model)
            load_megatron_optimizer(optimizer)
        torch.cuda.synchronize()

    def backward(self, loss: torch.Tensor, model, optimizer: optim.Optimizer, **kwargs) -> None:
        raise NotImplementedError()

    def optimizer_step(
        self,
        optimizer: optim.Optimizer,
        model,
        scheduler,
        name="model",
        **kwargs,
    ) -> Optional[Float[torch.Tensor, "1"]]:
        """Perform optimizer step"""
        _, grad_norm, _ = optimizer.step()
        scheduler.step(1)
        optimizer.zero_grad()
        return grad_norm

    def prepare(
        self, *models_or_model_optim_pairs: ModelOrModelOptimPair
    ) -> Union[List[ModelOrModelOptimPair], ModelOrModelOptimPair]:
        raise NotImplementedError()

    def save_checkpoint(
        self,
        model: MegatronModelWrapper,
        ckpt_dir: str,
        node_local_rank: int,
        optimizer: Optional[DistributedOptimizer] = None,
        scheduler: Optional[OptimizerParamScheduler] = None,
        client_state: dict = {},
        tag=None,
        tokenizer: Optional[PreTrainedTokenizer] = None,
    ):
        # Extract base model.
        model: List[nn.Module] = model.actor_module
        assert len(model) == 1, "Megatron virtual pipeline parallel is not yet supported"
        model = model[0]
        if hasattr(model, "module"):
            model = model.module

        # Create checkpoint directory if it doesn't exist.
        if node_local_rank == 0:
            io.makedirs(ckpt_dir, exist_ok=True)

        # All ranks wait for the checkpoint directory to be created before saving.
        dist.barrier()

        # Collect the sharded state dicts for model and optimizer, and full state dict for the scheduler.
        sharded_state_dict = {}
        model_sharded_state_dict = model.sharded_state_dict()
        sharded_state_dict["model"] = model_sharded_state_dict
        if optimizer:
            sharded_state_dict["optimizer"] = optimizer.sharded_state_dict(
                model_sharded_state_dict, metadata=_MEGATRON_OPTIM_CKPT_METADATA
            )
        if scheduler:
            sharded_state_dict["lr_scheduler"] = scheduler.state_dict()

        # Save RNG state.
        sharded_state_dict["rng"] = self.get_rng_state()

        # Save the checkpoint across ranks in parallel.
        save_strategy = get_default_save_sharded_strategy("torch_dist")
        save_strategy = FullyParallelSaveStrategyWrapper(
            save_strategy, mpu.get_data_parallel_group(with_context_parallel=True)
        )

        with io.local_work_dir(ckpt_dir) as work_dir:
            # TODO(tgriggs): Support configurable async saves.
            async_save_request = dist_checkpointing.save(
                sharded_state_dict=sharded_state_dict,
                checkpoint_dir=work_dir,
                sharded_strategy=save_strategy,
                async_sharded_save=False,
                validate_access_integrity=True,
            )
            assert async_save_request is None, "Async save is not yet supported for Megatron"

            # Only global rank 0 saves the Huggingface config and tokenizer.
            if self.is_rank_0():
                hf_dir = os.path.join(work_dir, "huggingface")
                self.save_hf_configs(self.hf_config, hf_dir, tokenizer)

                # Persist replicated client state (e.g. ZClip / StaleClip warmup
                # counters + EMA stats) so they survive chain-restarts, matching
                # the FSDP2 strategy. client_state is global (not sharded), so a
                # single rank-0 file suffices; every rank reads it back on load.
                extra_state_path = os.path.join(work_dir, "extra_state.pt")
                with io.open_file(extra_state_path, "wb") as f:
                    torch.save({"client_state": client_state, "tag": tag}, f)

        dist.barrier()
        ckpt_base.async_calls.close()
        ckpt_base.async_calls = AsyncCallsQueue(persistent=True)
        self.print(f"Checkpoint successfully saved to {ckpt_dir}")

    def load_checkpoint(
        self,
        model: MegatronModelWrapper,
        ckpt_dir: str,
        optimizer: Optional[DistributedOptimizer] = None,
        scheduler: Optional[OptimizerParamScheduler] = None,
        load_module_strict: bool = True,
        load_optimizer_states: bool = True,
        load_lr_scheduler_states: bool = True,
        load_runtime_state: bool = True,
    ):
        if not ckpt_dir or not io.exists(ckpt_dir):
            raise FileNotFoundError(f"Checkpoint directory not found: {ckpt_dir}")

        # Extract base model.
        model: List[nn.Module] = model.actor_module
        assert len(model) == 1, "Megatron virtual pipeline parallel is not yet supported"
        unwrapped_model = model[0]
        if hasattr(unwrapped_model, "module"):
            unwrapped_model = unwrapped_model.module

        # Extract sharded state dicts.
        sharded_state_dict = {}
        model_sharded_state_dict = unwrapped_model.sharded_state_dict()
        sharded_state_dict["model"] = model_sharded_state_dict
        if optimizer and load_optimizer_states:
            sharded_state_dict["optimizer"] = optimizer.sharded_state_dict(
                model_sharded_state_dict, is_loading=True, metadata=_MEGATRON_OPTIM_CKPT_METADATA
            )
        if scheduler and load_lr_scheduler_states:
            sharded_state_dict["lr_scheduler"] = scheduler.state_dict()

        with io.local_read_dir(ckpt_dir) as read_dir:
            # Load the checkpoint in parallel.
            load_strategy = get_default_load_sharded_strategy(read_dir)
            load_strategy = FullyParallelLoadStrategyWrapper(
                load_strategy, mpu.get_data_parallel_group(with_context_parallel=True)
            )
            state_dict = dist_checkpointing.load(
                sharded_state_dict=sharded_state_dict, checkpoint_dir=read_dir, sharded_strategy=load_strategy
            )

        # Load the model, optimizer, and scheduler state dicts.
        assert "model" in state_dict, (
            f"Model state dict not found in checkpoint loaded from {ckpt_dir}. Available keys: {state_dict.keys()}"
        )
        model[0].load_state_dict(state_dict["model"], strict=load_module_strict)
        self.print("Loaded model state dict.")

        if optimizer and load_optimizer_states:
            assert "optimizer" in state_dict, (
                f"Optimizer state dict not found in checkpoint loaded from {ckpt_dir}. Available keys: {state_dict.keys()}"
            )
            optimizer.load_state_dict(state_dict["optimizer"])
            self.print("Loaded optimizer state dict.")

        if scheduler and load_lr_scheduler_states:
            assert "lr_scheduler" in state_dict, (
                f"LR scheduler state dict not found in checkpoint loaded from {ckpt_dir}. Available keys: {state_dict.keys()}"
            )
            scheduler.load_state_dict(state_dict["lr_scheduler"])
            self.print("Loaded LR scheduler state dict.")

        # Load RNG state, if present.
        if load_runtime_state and "rng" in state_dict:
            self.load_rng_state(state_dict["rng"])

        # Restore replicated client state (ZClip / StaleClip), if present. Guarded
        # for backward-compat with checkpoints written before this file existed.
        states = {}
        extra_state_path = os.path.join(ckpt_dir, "extra_state.pt")
        if load_runtime_state and io.exists(extra_state_path):
            with io.open_file(extra_state_path, "rb") as f:
                extra_state = torch.load(f, weights_only=False)
            states = extra_state.get("client_state", {}) or {}
            self.print("Loaded client state (ZClip / StaleClip) from checkpoint.")

        return ckpt_dir, states

    def save_hf_model(self, bridge, model: MegatronModelWrapper, output_dir: str, tokenizer=None, **kwargs) -> None:
        # Create checkpoint directory if it doesn't exist.
        if self.is_rank_0():
            io.makedirs(output_dir, exist_ok=True)
        dist.barrier()

        # All ranks call into bridge.
        with io.local_work_dir(output_dir) as work_dir:
            bridge.save_hf_weights(model.actor_module, work_dir)
            self.print(f"Successfully saved HF safetensors model to {output_dir}")

            # Only rank 0 saves the Huggingface config and tokenizer.
            if self.is_rank_0():
                self.save_hf_configs(self.hf_config, work_dir, tokenizer)
                self.print(f"Successfully saved HF config and tokenizer to {output_dir}")

        # The Ray caller waits for every rank result; no collective needs to span artifact publication.
