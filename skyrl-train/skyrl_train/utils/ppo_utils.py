# This code is adapted from VERL
# https://github.com/volcengine/verl/blob/main/verl/trainer/ppo/core_algos.py
# The original copyright is reproduced below:
# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2022 The HuggingFace Team. All rights reserved.
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

from collections import defaultdict
from enum import StrEnum
from functools import wraps
from typing import Callable, List, Literal, Optional, Tuple, Union

import numpy as np
import ray
import torch
from jaxtyping import Float
from loguru import logger
from omegaconf import DictConfig

from skyrl_train.training_batch import TrainingInputBatch

# Import cloudpickle for function serialization
try:
    import cloudpickle
except ImportError:
    # Fallback to pickle if cloudpickle is not available
    import pickle as cloudpickle


# TODO (erictang000): unused right now, but will be useful as we add more algorithm support
class AdaptiveKLController:
    """
    Adaptive KL controller described in the paper:
    https://arxiv.org/pdf/1909.08593.pdf
    """

    def __init__(self, init_kl_coef, target, horizon):
        self.value = init_kl_coef
        self.target = target
        self.horizon = horizon

    def update(self, current, n_steps):
        target = self.target
        proportional_error = np.clip(current / target - 1, -0.2, 0.2)
        mult = 1 + proportional_error * n_steps / self.horizon
        self.value *= mult


class FixedKLController:
    """Fixed KL controller."""

    def __init__(self, kl_coef):
        self.value = kl_coef

    def update(self, current, n_steps):
        pass


def get_kl_controller(algorithm_cfg: DictConfig):
    if algorithm_cfg.kl_ctrl.type == "fixed":
        return FixedKLController(kl_coef=algorithm_cfg.kl_loss_coef)
    elif algorithm_cfg.kl_ctrl.type == "adaptive":
        if algorithm_cfg.kl_ctrl.horizon <= 0:
            raise ValueError(f"horizon must be larger than 0. Got {algorithm_cfg.kl_ctrl.horizon}")
        return AdaptiveKLController(
            init_kl_coef=algorithm_cfg.kl_loss_coef,
            target=algorithm_cfg.kl_ctrl.kl_target,
            horizon=algorithm_cfg.kl_ctrl.horizon,
        )
    else:
        raise ValueError(f"Invalid KL controller type: {algorithm_cfg.kl_ctrl.type}")


def masked_mean(tensor: torch.Tensor, mask: Optional[torch.Tensor], dim: Optional[int] = None) -> torch.Tensor:
    if mask is None:
        return tensor.mean(axis=dim)
    return (tensor * mask).sum(axis=dim) / mask.sum(axis=dim).clamp(min=1.0)


@torch.no_grad()
def compute_approx_kl(
    log_probs: torch.Tensor,
    log_probs_base: torch.Tensor,
    loss_mask: Optional[torch.Tensor] = None,
    kl_estimator_type: str = "k3",
) -> torch.Tensor:
    """
    Compute the approximate KL divergence between two distributions.
    Schulman blog: http://joschu.net/blog/kl-approx.html

    Args:
        log_probs: Log probabilities of the new distribution.
        log_probs_base: Log probabilities of the base distribution.
        action_mask: Mask for actions.
    """
    if kl_estimator_type == "k1":
        kld = log_probs - log_probs_base
    elif kl_estimator_type == "abs":
        kld = (log_probs - log_probs_base).abs()
    elif kl_estimator_type == "k2":
        kld = 0.5 * (log_probs - log_probs_base).square()
    # J. Schulman. Approximating kl divergence, 2020.
    # URL http://joschu.net/blog/kl-approx.html.
    elif kl_estimator_type == "k3":
        kl = log_probs_base - log_probs
        # For numerical stability
        kl = torch.clamp(kl, min=-20, max=20)
        ratio = torch.exp(kl)
        kld = (ratio - kl - 1).contiguous()
        kld = torch.clamp(kld, min=-10, max=10)
    else:
        raise ValueError(f"Invalid KL estimator type: {kl_estimator_type}")

    if loss_mask is not None:
        kld = kld * loss_mask
    return kld


@torch.no_grad()
def normalize_advantages_dict(data: TrainingInputBatch) -> TrainingInputBatch:
    """Normalizes the advantages in the data batch.

    Expects:
        - `["advantages"]`: Float[torch.Tensor, "batch_size seqlen"]
        - `["response_mask"]`: Float[torch.Tensor, "batch_size seqlen"]
    """
    advantages: Float[torch.Tensor, "batch_size seqlen"] = data["advantages"]
    response_masks: Float[torch.Tensor, "batch_size seqlen"] = data["response_mask"]
    num_actions: float = response_masks.sum()
    # mean
    mean: float = advantages.mean()
    # std
    std: float = ((advantages - mean).pow(2) * response_masks).sum()
    rstd: float = (std / num_actions).clamp(min=1e-8).rsqrt()

    data["advantages"] = (advantages - mean) * rstd
    return data


def masked_var(values, mask, unbiased=True):
    """Compute variance of tensor with masked values."""
    mean = masked_mean(values, mask)
    centered_values = values - mean
    variance = masked_mean(centered_values**2, mask)
    if unbiased:
        mask_sum = mask.sum()
        if mask_sum == 0:
            raise ValueError("At least one element in the mask has to be 1.")
        # note that if mask_sum == 1, then there is a division by zero issue
        # to avoid it you just need to use a larger minibatch_size
        if mask_sum == 1:
            raise ValueError("The sum of the mask is one, which can cause a division by zero.")
        bessel_correction = mask_sum / (mask_sum - 1)
        variance = variance * bessel_correction
    return variance


def masked_whiten(values, mask, shift_mean=True):
    """Whiten values with masked values."""
    mean, var = masked_mean(values, mask), masked_var(values, mask)
    whitened = (values - mean) * torch.rsqrt(var + 1e-8)
    if not shift_mean:
        whitened += mean
    return whitened


def ppo_critic_loss(
    values: torch.Tensor,
    old_values: torch.Tensor,
    returns: torch.Tensor,
    config: DictConfig,
    loss_mask: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, Optional[float]]:
    if config.value_clip is not None:
        values_clipped = old_values + (values - old_values).clamp(-config.value_clip, config.value_clip)
        surr1 = (values_clipped - returns) ** 2
        surr2 = (values - returns) ** 2
        loss = torch.max(surr1, surr2)
        clipfrac = masked_mean((surr1 > surr2).float(), loss_mask).mean().detach().item()
    else:
        clipfrac = None
        loss = (values - returns) ** 2

    loss = masked_mean(loss, loss_mask, dim=-1).mean()
    return 0.5 * loss, clipfrac


# Shared registry actor class for both policy loss and advantage estimator registries
@ray.remote
class RegistryActor:
    """Shared Ray actor for managing function registries across processes."""

    def __init__(self):
        self.registry = {}

    def register(self, name: str, func_serialized: bytes):
        """Register a serialized function."""
        self.registry[name] = func_serialized

    def get(self, name: str):
        """Get a serialized function by name."""
        return self.registry.get(name)

    def list_available(self):
        """List all available function names."""
        return list(self.registry.keys())

    def unregister(self, name: str):
        """Unregister a function by name."""
        return self.registry.pop(name, None)


class BaseFunctionRegistry:
    """Base class for function registries with Ray actor synchronization."""

    # Subclasses should override these class attributes
    _actor_name = None
    _function_type = "Function"

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        cls._functions = {}
        cls._ray_actor = None
        cls._synced_to_actor = False

    @classmethod
    def _get_or_create_actor(cls):
        """Get or create the Ray actor for managing the registry using get_if_exists."""
        if not ray.is_initialized():
            raise Exception("Ray is not initialized, cannot create registry actor")

        if cls._ray_actor is None:
            # Use get_if_exists to create actor only if it doesn't exist
            cls._ray_actor = RegistryActor.options(name=cls._actor_name, get_if_exists=True).remote()
        return cls._ray_actor

    @classmethod
    def _sync_local_to_actor(cls):
        """Sync all local functions to Ray actor."""
        if cls._synced_to_actor:
            return
        if not ray.is_initialized():
            raise Exception("Ray is not initialized, cannot sync with actor")

        try:
            actor = cls._get_or_create_actor()
            if actor is not None:
                for name, func in cls._functions.items():
                    func_serialized = cloudpickle.dumps(func)
                    ray.get(actor.register.remote(name, func_serialized))
                cls._synced_to_actor = True
        except Exception as e:
            logger.error(f"Error syncing {cls._function_type} to actor: {e}")
            raise e

    @classmethod
    def sync_with_actor(cls):
        """Sync local registry with Ray actor if Ray is available."""
        # Only try if Ray is initialized
        if not ray.is_initialized():
            raise Exception("Ray is not initialized, cannot sync with actor")

        # First check if the actor is still alive
        # NOTE(Charlie): This is mainly for unit tests, where we run multiple unit tests in the
        # same Python process, and each unit test has ray init/shutdown. This makes cls's attributes
        # outdated (e.g. the _ray_actor points to a stale actor in the previous ray session).
        try:
            _ = ray.get_actor(cls._actor_name)  # this raises exception if the actor is stale
        except ValueError:
            cls._ray_actor = None
            cls._synced_to_actor = False

        # First, sync our local functions to the actor
        cls._sync_local_to_actor()

        actor = cls._get_or_create_actor()
        if actor is None:
            return

        available = ray.get(actor.list_available.remote())

        # Sync any new functions from actor to local registry
        for name in available:
            if name not in cls._functions:
                func_serialized = ray.get(actor.get.remote(name))
                if func_serialized is not None:
                    # Deserialize the function
                    try:
                        func = cloudpickle.loads(func_serialized)
                        cls._functions[name] = func
                    except Exception as e:
                        # If deserialization fails, skip this function
                        logger.error(f"Error deserializing {name} from actor: {e}")
                        raise e

    @classmethod
    def register(cls, name: Union[str, StrEnum], func: Callable):
        """Register a function.

        If ray is initialized, this function will get or create a named ray actor (RegistryActor)
        for the registry, and sync the registry to the actor.

        If ray is not initalized, the function will be stored in the local registry only.

        To make sure all locally registered functions are available to all ray processes,
        call sync_with_actor() after ray.init().

        Args:
            name: Name of the function to register. Can be a string or a StrEnum.
            func: Function to register.

        Raises:
            ValueError: If the function is already registered.
        """
        # Convert enum to string if needed
        # note: StrEnum is not cloudpickleable: https://github.com/cloudpipe/cloudpickle/issues/558
        if isinstance(name, StrEnum):
            name = name.value

        if name in cls._functions:
            raise ValueError(f"{cls._function_type} '{name}' already registered")

        # Always store in local registry first
        cls._functions[name] = func

        # Try to sync with Ray actor if Ray is initialized
        if ray.is_initialized():
            actor = cls._get_or_create_actor()
            if actor is not None:
                # Serialize the function using cloudpickle
                func_serialized = cloudpickle.dumps(func)
                ray.get(actor.register.remote(name, func_serialized))

    @classmethod
    def get(cls, name: str) -> Callable:
        """Get a function by name.

        If ray is initialized, this function will first sync the local registry with the RegistryActor.
        Then it will return the function if it is found in the registry.

        Args:
            name: Name of the function to get. Can be a string or a StrEnum.

        Returns:
            The function if it is found in the registry.
        """
        # Try to sync with actor first if Ray is available
        if ray.is_initialized():
            cls.sync_with_actor()

        if name not in cls._functions:
            available = list(cls._functions.keys())
            raise ValueError(f"Unknown {cls._function_type.lower()} '{name}'. Available: {available}")
        return cls._functions[name]

    @classmethod
    def list_available(cls) -> List[str]:
        """List all registered functions."""
        # Try to sync with actor first if Ray is available
        if ray.is_initialized():
            cls.sync_with_actor()
        return list(cls._functions.keys())

    @classmethod
    def unregister(cls, name: Union[str, StrEnum]):
        """Unregister a function. Useful for testing."""
        # Convert enum to string if needed
        if isinstance(name, StrEnum):
            name = name.value

        # Try to sync with actor first to get any functions that might be in the actor but not local
        if ray.is_initialized():
            cls.sync_with_actor()

        # Track if we found the function anywhere
        found_locally = name in cls._functions
        found_in_actor = False

        # Remove from local registry if it exists
        if found_locally:
            del cls._functions[name]

        # Try to remove from Ray actor if Ray is available
        if ray.is_initialized():
            actor = cls._get_or_create_actor()
            if actor is not None:
                # Check if it exists in actor first
                available_in_actor = ray.get(actor.list_available.remote())
                if name in available_in_actor:
                    found_in_actor = True
                    ray.get(actor.unregister.remote(name))

        # Only raise error if the function wasn't found anywhere
        if not found_locally and not found_in_actor:
            raise ValueError(f"{cls._function_type} '{name}' not registered")

    @classmethod
    def reset(cls):
        """Resets the registry (useful for testing purposes)."""
        if ray.is_initialized() and cls._ray_actor is not None:
            try:
                actor = ray.get_actor(cls._actor_name)  # this raises exception if the actor is stale
                ray.kill(actor)
            except ValueError:
                pass  # Actor may already be gone
        cls._functions.clear()
        cls._ray_actor = None
        cls._synced_to_actor = False

    @classmethod
    def repopulate(cls):
        """Repopulate the registry with the default functions."""
        cls.reset()
        cls.register(cls._function_type, cls._function_type)


class AdvantageEstimator(StrEnum):
    GAE = "gae"
    GRPO = "grpo"
    RLOO = "rloo"
    RLOO_N = "rloo_n"  # RLOO-Neutral: excludes masked samples from baseline
    REINFORCE_PP = "reinforce++"


class AdvantageEstimatorRegistry(BaseFunctionRegistry):
    """
    Registry for advantage estimator functions.

    This registry allows users to register custom advantage estimators without modifying
    the skyrl_train package. Custom estimators can be registered by calling
    AdvantageEstimatorRegistry.register() directly or by using the @register_advantage_estimator
    decorator.

    See examples/algorithms/custom_advantage_estimator for a simple example of how to
    register and use custom advantage estimators.
    """

    _actor_name = "advantage_estimator_registry"
    _function_type = "advantage estimator"

    @classmethod
    def repopulate_registry(cls):
        ae_avail = set(cls.list_available())
        ae_types = {
            "grpo": [AdvantageEstimator.GRPO, compute_grpo_outcome_advantage],
            "gae": [AdvantageEstimator.GAE, compute_gae_advantage_return],
            "rloo": [AdvantageEstimator.RLOO, compute_rloo_outcome_advantage],
            "rloo_n": [AdvantageEstimator.RLOO_N, compute_rloo_n_outcome_advantage],
            "reinforce++": [AdvantageEstimator.REINFORCE_PP, compute_reinforce_plus_plus_outcome_advantage],
        }

        for ae_name, (ae_type, ae_func) in ae_types.items():
            if ae_name not in ae_avail:
                cls.register(ae_type, ae_func)


class PolicyLossType(StrEnum):
    REGULAR = "regular"
    DUAL_CLIP = "dual_clip"
    GSPO = "gspo"
    CISPO = "cispo"
    CLIP_COV = "clip_cov"
    KL_COV = "kl_cov"
    SAPO = "sapo"


class PolicyLossRegistry(BaseFunctionRegistry):
    """
    Registry for policy loss functions.

    This registry allows users to register custom policy loss functions without modifying
    the skyrl_train package. Custom functions can be registered by calling
    PolicyLossRegistry.register() directly or by using the @register_policy_loss
    decorator.

    See examples/algorithms/custom_policy_loss for a simple example of how to
    register and use custom policy loss functions.
    """

    _actor_name = "policy_loss_registry"
    _function_type = "policy loss"

    @classmethod
    def repopulate_registry(cls):
        """Repopulate the registry with default policy loss functions."""
        pl_avail = set(cls.list_available())
        pl_types = {
            "regular": [PolicyLossType.REGULAR, ppo_policy_loss],
            "dual_clip": [PolicyLossType.DUAL_CLIP, ppo_policy_loss],
            "gspo": [PolicyLossType.GSPO, gspo_policy_loss],
            "clip_cov": [PolicyLossType.CLIP_COV, compute_policy_loss_clip_cov],
            "kl_cov": [PolicyLossType.KL_COV, compute_policy_loss_kl_cov],
            "sapo": [PolicyLossType.SAPO, sapo_policy_loss],
        }

        for pl_name, (pl_type, pl_func) in pl_types.items():
            if pl_name not in pl_avail:
                cls.register(pl_type, pl_func)


def register_advantage_estimator(name: Union[str, AdvantageEstimator]):
    """Decorator to register an advantage estimator function."""

    def decorator(func: Callable):
        @wraps(func)
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)

        AdvantageEstimatorRegistry.register(name, wrapper)
        return wrapper

    return decorator


def register_policy_loss(name: Union[str, PolicyLossType]):
    """Decorator to register a policy loss function."""

    def decorator(func: Callable):
        @wraps(func)
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)

        PolicyLossRegistry.register(name, wrapper)
        return wrapper

    return decorator


def sync_registries():
    """Sync the registries with the ray actor once ray is initialized"""
    if not ray.is_initialized():
        raise ValueError("Ray is not initialized, cannot sync registries")
    PolicyLossRegistry.sync_with_actor()
    AdvantageEstimatorRegistry.sync_with_actor()
    logger.info("Synced registries to ray actor")


def _safe_exp_delta(delta: torch.Tensor, clip: float = 20.0, out_dtype=None) -> torch.Tensor:
    """
    Clamp the delta before exponentiating to avoid potential overflow.
    """
    y = torch.clamp(delta.to(torch.float32), -clip, clip).exp()
    return y.to(out_dtype or delta.dtype)


def _log_ratio_diag_zero_metrics(n_position_buckets: int = 10) -> dict:
    """The full key set the diagnostic emits, with all values zero.

    Used as a fallback so every rank contributes identical keys to
    `strategy.all_reduce(status)` even if a rank's input is empty/all-padded
    or the helper raises. Mismatched keysets across ranks deadlock the per-key
    NCCL all_reduce (this killed v2 and v3 of the diagnostic).
    """
    keys_base = ["log_ratio_abs_mean", "log_ratio_abs_max",
                 "n_tokens_dp_gt_1pct", "n_tokens_dp_gt_10pct", "n_tokens_dp_gt_50pct",
                 "log_ratio_abs_p99"]
    keys_pos = [f"log_ratio_abs_pos{i * (100 // n_position_buckets):02d}" for i in range(n_position_buckets)]
    return {k: 0.0 for k in keys_base + keys_pos}


def compute_log_ratio_diagnostics(
    log_probs: torch.Tensor,
    old_log_probs: torch.Tensor,
    loss_mask: torch.Tensor,
    n_position_buckets: int = 10,
) -> dict:
    """Per-token probability-change diagnostics — v4 (all-ranks, full key set always).

    v1 (reverted in 741bc3f8) crashed Perlmutter 52563046 with NCCL timeouts:
    ran inside training_step (64×/global_step), used .quantile() and boolean
    indexing, issued 17 sequential .item() syncs → per-rank latency variance.

    v2 (fixed v1's latency, but introduced a new bug) gated to rank 0 only so
    other ranks skipped the call. Their `status` dict was missing the diagnostic
    keys, and the downstream `strategy.all_reduce(status)` iterates per-key →
    keys mismatched across ranks → NCCL hang. Killed Perlmutter 52593758 at
    global_step 1.

    v3: drop the rank-0 gate. Every rank runs the diagnostic on the same shapes
    (each rank's micro-batch), so all_reduce sees identical keysets. The mean-
    reduced values are slightly different from rank-0-only (averaged across
    ranks rather than one rank's view) but more statistically robust. Counts
    (n_tokens_dp_gt_*) are mean-reduced into per-rank averages — multiply by
    world_size at read-time for global totals if needed.

    Caller must still gate this to:
      - last micro-batch of a global_step (to avoid 64× redundant work)
      - AFTER optimizer step (so the GPU is idle in the gap, not contending
        with a pending NCCL collective)

    Per-call cost (CPU-measured, see /tmp/diagnostic_scaling_results.json):
        B=512 T=4096:   ~30 ms (vs v1: ~112 ms)
        B=512 T=16384: ~117 ms (vs v1: ~441 ms)
    On CUDA, expected to be 5-10× faster than CPU because the bottleneck is
    GPU compute, not stream sync (only 2 syncs total in v2 vs 17 in v1).

    Op-level swaps:
      - .quantile(0.99) → torch.topk(k=N/100).values.min() (~10× faster, no sort)
      - x[mask.bool()] → (x * mask) sum/max ops (no boolean indexing copy)
      - 10 Python-loop bucket means → 1 scatter_add call
      - 17 .item() calls → 2 syncs (one for k, one for final stack→tolist)

    Returns the same metric set as v1 (so wandb keys are unchanged):

        log_ratio_abs_mean       — mean of |log r_t| over masked tokens
        log_ratio_abs_p99        — ~p99 of |log r_t| (topk approximation)
        log_ratio_abs_max        — max of |log r_t| in this batch
        n_tokens_dp_gt_1pct      — count of tokens whose probability changed by >1%
        n_tokens_dp_gt_10pct     — count of tokens whose probability changed by >10%
        n_tokens_dp_gt_50pct     — count of tokens whose probability changed by >50%
        log_ratio_abs_pos00..90  — mean |log r_t| per relative-position bucket

    (log_ratio_abs_std dropped — std of all valid is more honestly captured by
    looking at p99 vs mean, and computing it cleanly without boolean indexing
    requires another sync.)

    Threshold rationale: |log r_t| > log(1+x) means the probability ratio
    moved by more than x. log(1.01) ≈ 0.01, log(1.10) ≈ 0.095, log(1.50) ≈ 0.405.
    We use 0.01 / 0.10 / 0.50 as approximate thresholds.

    Always returns the full key set (zeros where input is empty/all-padded), so
    downstream `strategy.all_reduce(status)` sees identical keys on every rank.
    Returning a partial/empty dict on some ranks would deadlock the per-key
    NCCL all_reduce — that bug killed v3 (Perlmutter 52616953, watchdog timeout
    on a NumelIn=1 ALLREDUCE).
    """
    zero_metrics = _log_ratio_diag_zero_metrics(n_position_buckets)

    if log_probs.numel() == 0:
        return zero_metrics

    abs_log_ratio = (log_probs - old_log_probs).detach().abs().clamp(max=20.0).float()
    mask_f = loss_mask.float()
    masked = abs_log_ratio * mask_f

    # SYNC #1 (early, unavoidable): need int n_valid to determine topk's k.
    n_valid_int = int(mask_f.sum().item())
    if n_valid_int == 0:
        return zero_metrics
    n_valid_t = torch.as_tensor(float(n_valid_int), device=log_probs.device)

    # Build all GPU-resident scalar tensors first; no sync until the final stack.
    out_tensors = {
        "log_ratio_abs_mean":   masked.sum() / n_valid_t,
        "log_ratio_abs_max":    masked.max(),
        "n_tokens_dp_gt_1pct":  ((abs_log_ratio > 0.01) * mask_f).sum(),
        "n_tokens_dp_gt_10pct": ((abs_log_ratio > 0.10) * mask_f).sum(),
        "n_tokens_dp_gt_50pct": ((abs_log_ratio > 0.50) * mask_f).sum(),
    }

    # p99 via topk, no sort. Sentinel masks out invalid positions.
    sentinel = torch.tensor(-1.0e9, device=log_probs.device, dtype=abs_log_ratio.dtype)
    flat = torch.where(mask_f.bool(), abs_log_ratio, sentinel).flatten()
    k = max(1, n_valid_int // 100)
    out_tensors["log_ratio_abs_p99"] = torch.topk(flat, k=min(k, flat.numel()), largest=True).values.min()

    # Per-position bucket means via single scatter_add (no Python-loop allocations).
    B, T = log_probs.shape
    seq_lens = mask_f.sum(dim=-1, keepdim=True).clamp(min=1)
    positions = torch.arange(T, device=log_probs.device, dtype=torch.float32).unsqueeze(0).expand(B, T)
    buckets = (positions / seq_lens * n_position_buckets).clamp(0, n_position_buckets - 1).long()
    bsums = torch.zeros(n_position_buckets, device=log_probs.device, dtype=torch.float32)
    bcounts = torch.zeros(n_position_buckets, device=log_probs.device, dtype=torch.float32)
    bsums.scatter_add_(0, buckets.flatten(), masked.flatten())
    bcounts.scatter_add_(0, buckets.flatten(), mask_f.flatten())
    bucket_means = bsums / bcounts.clamp(min=1)

    # SYNC #2 (final): stack everything, transfer once.
    keys = list(out_tensors.keys())
    base_vals = torch.stack([out_tensors[k].float() for k in keys]).cpu().tolist()
    metrics = dict(zip(keys, base_vals))
    bucket_vals = bucket_means.cpu().tolist()
    for i in range(n_position_buckets):
        metrics[f"log_ratio_abs_pos{i * (100 // n_position_buckets):02d}"] = bucket_vals[i]

    return metrics


# ---------------------------------------------------------------------------
# v5: cross-micro-batch accumulation
# ---------------------------------------------------------------------------
# v4 ran the full diagnostic only on the LAST micro-batch of each global_step
# (gated by `(local_step + 1) % accumulation_steps == 0`). With
# `update_epochs_per_batch=1`, the forward pass that produces `action_log_probs`
# happens BEFORE optimizer_step for every micro-batch, so a single micro-batch's
# log-ratio reflects only the vLLM↔FSDP precision delta — and observing that
# delta on just one of 16 micro-batches yields a noisy sample.
#
# v5 changes the gating: every micro-batch contributes its partial stats to a
# per-rank accumulator; the last micro-batch finalizes (divides sums by counts,
# extracts p99 from the concatenated topk samples). The final scalar metric set
# is identical to v4's (same wandb keys), so downstream all_reduce(status) stays
# keyset-compatible across ranks.
def _empty_log_ratio_accumulator(device, n_position_buckets: int = 10) -> dict:
    """Zero-initialized per-rank accumulator for log-ratio diagnostics.

    All tensors live on `device`. `topk_abs` is an empty 1-D tensor that grows
    via torch.cat in `merge_log_ratio_partial`.
    """
    return {
        "abs_sum":    torch.zeros((), device=device, dtype=torch.float32),
        "n_valid":    torch.zeros((), device=device, dtype=torch.float32),
        "abs_max":    torch.zeros((), device=device, dtype=torch.float32),
        "n_gt_1pct":  torch.zeros((), device=device, dtype=torch.float32),
        "n_gt_10pct": torch.zeros((), device=device, dtype=torch.float32),
        "n_gt_50pct": torch.zeros((), device=device, dtype=torch.float32),
        "topk_abs":   torch.zeros((0,), device=device, dtype=torch.float32),
        "bsums":      torch.zeros(n_position_buckets, device=device, dtype=torch.float32),
        "bcounts":    torch.zeros(n_position_buckets, device=device, dtype=torch.float32),
    }


def compute_log_ratio_partial(
    log_probs: torch.Tensor,
    old_log_probs: torch.Tensor,
    loss_mask: torch.Tensor,
    n_position_buckets: int = 10,
) -> dict:
    """Compute per-micro-batch partial stats for cross-batch log-ratio
    accumulation. Returns a dict of GPU tensors matching the accumulator schema
    in `_empty_log_ratio_accumulator`.

    One sync per call (`int(mask_f.sum().item())`) — used to set `k_local` for
    the topk so it never reaches into padded positions. With 16 micro-batches
    per global_step, total per-step sync overhead is ~16 ms; finalize adds 1
    more for the GPU→CPU stack transfer.

    `topk_abs` stores this micro-batch's top-1%-of-valid values. At finalize,
    the concatenated tensor across micro-batches is the global "top of top"
    sample; its min approximates p99.
    """
    device = log_probs.device

    if log_probs.numel() == 0:
        return _empty_log_ratio_accumulator(device, n_position_buckets)

    abs_log_ratio = (log_probs - old_log_probs).detach().abs().clamp(max=20.0).float()
    mask_f = loss_mask.float()
    masked = abs_log_ratio * mask_f

    # SYNC #1 (per micro-batch): need n_valid_int to bound k_local. Without it,
    # using flat.numel()//100 contaminates the topk with masked zeros when the
    # response is short relative to context length (common at high max_seq_len).
    n_valid_int = int(mask_f.sum().item())
    if n_valid_int == 0:
        return _empty_log_ratio_accumulator(device, n_position_buckets)

    # Topk over masked-aware flat. Sentinel ensures topk never picks a padded
    # position when k_local <= n_valid (always true by construction).
    sentinel = torch.tensor(-1.0e9, device=device, dtype=abs_log_ratio.dtype)
    flat = torch.where(mask_f.bool(), abs_log_ratio, sentinel).flatten()
    k_local = max(1, n_valid_int // 100)
    topk_abs = torch.topk(flat, k=min(k_local, flat.numel()), largest=True).values.float()

    # Per-position bucket sums via single scatter_add (no Python loop allocs).
    B, T = log_probs.shape
    seq_lens = mask_f.sum(dim=-1, keepdim=True).clamp(min=1)
    positions = torch.arange(T, device=device, dtype=torch.float32).unsqueeze(0).expand(B, T)
    buckets = (positions / seq_lens * n_position_buckets).clamp(0, n_position_buckets - 1).long()
    bsums = torch.zeros(n_position_buckets, device=device, dtype=torch.float32)
    bcounts = torch.zeros(n_position_buckets, device=device, dtype=torch.float32)
    bsums.scatter_add_(0, buckets.flatten(), masked.flatten())
    bcounts.scatter_add_(0, buckets.flatten(), mask_f.flatten())

    return {
        "abs_sum":    masked.sum().float(),
        "n_valid":    torch.as_tensor(float(n_valid_int), device=device, dtype=torch.float32),
        "abs_max":    masked.max().float(),
        "n_gt_1pct":  ((abs_log_ratio > 0.01) * mask_f).sum().float(),
        "n_gt_10pct": ((abs_log_ratio > 0.10) * mask_f).sum().float(),
        "n_gt_50pct": ((abs_log_ratio > 0.50) * mask_f).sum().float(),
        "topk_abs":   topk_abs,
        "bsums":      bsums,
        "bcounts":    bcounts,
    }


def merge_log_ratio_partial(acc: dict, partial: dict) -> None:
    """In-place merge: additive for sums/counts, max for abs_max, concat for
    topk_abs. Mutates `acc`.
    """
    acc["abs_sum"]    = acc["abs_sum"]    + partial["abs_sum"]
    acc["n_valid"]    = acc["n_valid"]    + partial["n_valid"]
    acc["abs_max"]    = torch.maximum(acc["abs_max"], partial["abs_max"])
    acc["n_gt_1pct"]  = acc["n_gt_1pct"]  + partial["n_gt_1pct"]
    acc["n_gt_10pct"] = acc["n_gt_10pct"] + partial["n_gt_10pct"]
    acc["n_gt_50pct"] = acc["n_gt_50pct"] + partial["n_gt_50pct"]
    acc["topk_abs"]   = torch.cat([acc["topk_abs"], partial["topk_abs"]])
    acc["bsums"]      = acc["bsums"]      + partial["bsums"]
    acc["bcounts"]    = acc["bcounts"]    + partial["bcounts"]


def finalize_log_ratio_metrics(acc: dict, n_position_buckets: int = 10) -> dict:
    """Reduce the accumulator to the same scalar metric dict that v4 returned.

    One sync (final stack→CPU transfer). Returns the full keyset always
    (zeros where input was empty), so downstream per-key `all_reduce(status)`
    stays keyset-compatible across ranks — the bug that killed v2/v3.
    """
    device = acc["abs_sum"].device

    n_valid_safe = acc["n_valid"].clamp(min=1.0)
    abs_mean = acc["abs_sum"] / n_valid_safe
    abs_max = acc["abs_max"]

    # Global p99 from concatenated per-mb topks. Each micro-batch contributed
    # its top-1%-of-valid; the concatenation is ≈ top 1% of the global token
    # set, and its min is ≈ p99. Slight bias when micro-batches are heavily
    # heterogeneous in size, but monitoring-grade accuracy is sufficient.
    if acc["topk_abs"].numel() > 0:
        abs_p99 = acc["topk_abs"].min()
    else:
        abs_p99 = torch.zeros((), device=device, dtype=torch.float32)

    bucket_means = acc["bsums"] / acc["bcounts"].clamp(min=1.0)

    base_keys = [
        "log_ratio_abs_mean", "log_ratio_abs_max",
        "n_tokens_dp_gt_1pct", "n_tokens_dp_gt_10pct", "n_tokens_dp_gt_50pct",
        "log_ratio_abs_p99",
    ]
    base_vals = torch.stack([
        abs_mean.float(),
        abs_max.float(),
        acc["n_gt_1pct"].float(),
        acc["n_gt_10pct"].float(),
        acc["n_gt_50pct"].float(),
        abs_p99.float(),
    ]).cpu().tolist()
    metrics = dict(zip(base_keys, base_vals))

    bucket_vals = bucket_means.cpu().tolist()
    for i in range(n_position_buckets):
        metrics[f"log_ratio_abs_pos{i * (100 // n_position_buckets):02d}"] = bucket_vals[i]

    return metrics


@register_policy_loss(PolicyLossType.REGULAR)
@register_policy_loss(PolicyLossType.DUAL_CLIP)
def ppo_policy_loss(
    log_probs: torch.Tensor,
    old_log_probs: torch.Tensor,
    advantages: torch.Tensor,
    config: DictConfig,
    loss_mask: Optional[torch.Tensor] = None,
    rollout_logprobs: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, float]:
    assert config.policy_loss_type in ["regular", "dual_clip"], "loss_type must be either 'regular' or 'dual_clip'"
    loss_reduction = config.loss_reduction
    assert loss_reduction in [
        "token_mean",
        "sequence_mean",
        "seq_mean_token_sum_norm",
        "seq_mean_token_sum_norm_global",
    ], (
        "loss_reduction must be 'token_mean', 'sequence_mean', 'seq_mean_token_sum_norm', "
        "or 'seq_mean_token_sum_norm_global'"
    )

    ratio = _safe_exp_delta(log_probs - old_log_probs, clip=20.0, out_dtype=log_probs.dtype)
    surr1 = ratio * advantages
    surr2 = ratio.clamp(1 - config.eps_clip_low, 1 + config.eps_clip_high) * advantages
    loss = -torch.min(surr1, surr2)
    clip_ratio = masked_mean((-surr2 > -surr1).float(), loss_mask).mean().detach().item()
    clip_pg_losses1 = loss
    if config.policy_loss_type == "dual_clip":
        pg_losses3 = -advantages * config.clip_ratio_c
        clip_pg_losses2 = torch.min(pg_losses3, clip_pg_losses1)
        loss = torch.where(advantages < 0, clip_pg_losses2, clip_pg_losses1)

    # TIS graceful degrade (Fix A): apply the truncated-importance-sampling
    # ratio only when use_tis is on AND this batch actually carries rollout
    # logprobs. When the whole training batch came back with no logprobs the
    # trainer leaves rollout_logprobs as None (instead of hard-asserting); we
    # then fall back to the standard policy loss for THIS batch only. The
    # tis/batch_skipped_no_logprobs metric is emitted driver-side by the trainer.
    if config.use_tis and rollout_logprobs is not None:
        from loguru import logger as logger_

        logger_.debug(f"Using TIS with dtype: {rollout_logprobs.dtype}")
        # Apply truncated importance sampling -> https://fengyao.notion.site/off-policy-rl
        tis_imp_ratio = _safe_exp_delta(old_log_probs - rollout_logprobs, clip=20.0, out_dtype=log_probs.dtype)
        tis_imp_ratio = torch.clamp(tis_imp_ratio, max=config.tis_imp_ratio_cap)
        loss = loss * tis_imp_ratio

    loss = reduce_loss(
        loss,
        loss_mask,
        loss_reduction,
        config.max_seq_len,
        global_denom=getattr(config, "global_loss_denom", None),
    )
    return loss, clip_ratio


@register_policy_loss(PolicyLossType.SAPO)
def sapo_policy_loss(
    log_probs: torch.Tensor,
    old_log_probs: torch.Tensor,
    advantages: torch.Tensor,
    config: DictConfig,
    loss_mask: Optional[torch.Tensor] = None,
    rollout_logprobs: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, float]:
    """
    SAPO (Soft Adaptive Policy Optimization) policy loss function.

    Compute the smoothed policy objective and related metrics for SAPO.

    See https://arxiv.org/pdf/2511.20347 for more details.

    """
    # SAPO must use sequence_mean reduction
    loss_reduction = config.loss_reduction
    if loss_reduction != "sequence_mean":
        # The SAPO paper uses sequence_mean reduction; there's no reason
        # why a user couldn't use token_mean reduction, but
        # it's not clear whether it would be stable or not.
        from loguru import logger as logger_  # have to do lazy import to avoid pickling error

        logger_.warning(f"With SAPO it's recommended to use 'sequence_mean' loss reduction; got {loss_reduction}")

    # temperature for positive and negative token updates
    tau_pos = torch.as_tensor(config.sapo.tau_pos, dtype=advantages.dtype, device=advantages.device)
    tau_neg = torch.as_tensor(config.sapo.tau_neg, dtype=advantages.dtype, device=advantages.device)

    def gate_function(x, tau):
        """The gating function used in SAPO"""
        return torch.sigmoid(tau * (x - 1.0)) * (4.0 / tau)

    # compute IS at token level:
    # r_{i,t}(θ) = π_θ(y_{i,t}|x, y_{i,<t}) / π_θold(y_{i,t}|x, y_{i,<t})]
    # In log space: log(r_{i,t}(θ)) = log_probs - old_log_probs
    log_ratio = log_probs - old_log_probs

    # Clamp log_ratio for stability -> avoid overflow in exp()
    log_ratio = torch.clamp(log_ratio, min=-20.0, max=20.0)

    # finally exp() to remove log and get r_{i,t}(θ)
    ratio = torch.exp(log_ratio)

    # tau_{i,t} is tau_pos if adv > 0 else tau_neg
    taus = torch.where(
        condition=advantages > 0,
        input=tau_pos,  # if A_{i,t} > 0 we set to tau_pos
        other=tau_neg,  # if A_{i,t} <= 0 we set to tau_neg
    )

    # compute the gates f_{i,t}(r_{i,t}(θ)) at token level
    gates = gate_function(ratio, taus)

    # compute policy gradient loss
    loss = -gates * advantages

    # for SAPO, we need to aggregate the loss at the sequence level (seq-mean-token-mean)
    loss = reduce_loss(loss, loss_mask, loss_reduction, config.max_seq_len)

    # SAPO does not use clipping, so we set clip_ratio to 0.0 for compatibility
    clip_ratio = 0.0

    return loss, clip_ratio


@register_policy_loss(PolicyLossType.GSPO)
def gspo_policy_loss(
    log_probs: torch.Tensor,
    old_log_probs: torch.Tensor,
    advantages: torch.Tensor,
    config: DictConfig,
    loss_mask: Optional[torch.Tensor] = None,
    rollout_logprobs: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, float]:
    """
    GSPO (Group Sequence Policy Optimization) policy loss function,
    as proposed in https://arxiv.org/abs/2507.18071.

    This implements sequence-level importance sampling instead of token-level importance sampling.
    The key difference is that importance weights are computed at the sequence level and then
    applied uniformly across all tokens in the sequence. This can lead to more stable training
    dynamics by reducing the variance in clipping behavior within sequences.

    The variant of GSPO used here is GSPO-token, a generalization which allows for token-level
    advantages [equations 14 and 15 in the paper].
    """
    # GSPO must use sequence_mean reduction
    loss_reduction = config.loss_reduction
    if loss_reduction != "sequence_mean":
        # The GSPO paper uses sequence_mean reduction; there's no reason
        # why a user couldn't use token_mean reduction, but
        # it's not clear whether it would be stable or not.
        from loguru import logger as logger_  # have to do lazy import to avoid pickling error

        logger_.warning(f"With GSPO it's recommended to use 'sequence_mean' loss reduction; got {loss_reduction}")

    # Compute log ratios
    log_ratio = log_probs - old_log_probs

    # Key GSPO innovation: sequence-level importance sampling
    # Instead of using per-token ratios, compute sequence-averaged ratios
    log_importance_weights = masked_mean(log_ratio, loss_mask, dim=-1).unsqueeze(-1)

    # s_i,t(θ) = sg[s_i(θ)] · π_θ(y_i,t|x, y_i,<t) / sg[π_θ(y_i,t|x, y_i,<t)]
    # In log space: log(s_i,t(θ)) = sg[log(s_i(θ))] + log_probs - sg[log_probs]
    # note: we put the addition at the end to avoid precision issues,
    # per https://github.com/volcengine/verl/pull/2775#discussion_r2241500280
    log_token_importance_weights = log_probs - log_probs.detach() + log_importance_weights.detach()
    # clip to avoid overflow
    log_token_importance_weights = torch.clamp(log_token_importance_weights, max=10)
    ratio = torch.exp(log_token_importance_weights)

    # Standard PPO surrogate objective with sequence-level importance weights
    surr1 = ratio * advantages
    surr2 = ratio.clamp(1 - config.eps_clip_low, 1 + config.eps_clip_high) * advantages
    loss = -torch.min(surr1, surr2)

    # Compute clipping ratio for monitoring
    clip_ratio = masked_mean((-surr2 > -surr1).float(), loss_mask).mean().detach().item()

    loss = reduce_loss(loss, loss_mask, loss_reduction, config.max_seq_len)

    return loss, clip_ratio


@register_policy_loss(PolicyLossType.CISPO)
def compute_policy_loss_cispo(
    log_probs: torch.Tensor,
    old_log_probs: torch.Tensor,
    advantages: torch.Tensor,
    config: DictConfig,
    loss_mask: Optional[torch.Tensor] = None,
    rollout_logprobs: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, float]:
    """Implementation of CISPO (Clipped IS-weight Policy Optimization) loss function,
    as proposed in https://arxiv.org/abs/2506.13585.

    Instead of clipping the importance sampling ratio in the loss directly, as done
    in PPO loss, CISPO clips the importance sampling ratio in the policy gradient
    update. This means the model can still learn from samples whose importance sampling
    ratio is clipped in CISPO, as opposed to PPO where these samples have zero
    gradient and are essentially ignored.
    """
    ratio = _safe_exp_delta(log_probs - old_log_probs, clip=20.0, out_dtype=log_probs.dtype)
    clamped_ratio = torch.clamp(ratio, 1 - config.cispo.cispo_eps_clip_low, 1 + config.cispo.cispo_eps_clip_high)
    loss = -advantages * clamped_ratio.detach() * log_probs

    is_clipped = (ratio < 1 - config.cispo.cispo_eps_clip_low) | (ratio > 1 + config.cispo.cispo_eps_clip_high)
    clip_ratio = masked_mean(is_clipped.float(), loss_mask).mean().detach().item()

    loss = reduce_loss(loss, loss_mask, config.loss_reduction, config.max_seq_len)
    return loss, clip_ratio


@register_policy_loss(PolicyLossType.CLIP_COV)
def compute_policy_loss_clip_cov(
    log_probs: torch.Tensor,
    old_log_probs: torch.Tensor,
    advantages: torch.Tensor,
    config: DictConfig,
    loss_mask: Optional[torch.Tensor] = None,
    rollout_logprobs: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, float]:
    """Clip-Cov policy loss function implementation.

    Adapted from https://github.com/PRIME-RL/Entropy-Mechanism-of-RL/blob/main/verl/trainer/ppo/core_algos.py

    This method combines standard PPO clipping with covariance-based clipping
    to provide more stable training dynamics.
    """
    # Extract config parameters with defaults
    clip_cov_ratio = config.clip_cov.clip_ratio
    clip_cov_lb = config.clip_cov.clip_cov_lb
    clip_cov_ub = config.clip_cov.clip_cov_ub

    negative_approx_kl = log_probs - old_log_probs
    ratio = torch.exp(negative_approx_kl)

    pg_losses1 = -advantages * ratio

    pg_losses2 = -advantages * torch.clamp(ratio, 1 - config.eps_clip_low, 1 + config.eps_clip_high)
    clip_by_origin = (pg_losses2 > pg_losses1) & (loss_mask > 0)

    # Compute covariance for clipping decision
    cov_all = (advantages - masked_mean(advantages, loss_mask)) * (
        log_probs - masked_mean(log_probs.detach(), loss_mask)
    )
    cov_all[loss_mask == 0] = -torch.inf
    cov_all[clip_by_origin] = -torch.inf

    # Determine number of tokens to clip based on clip_ratio
    clip_num = max(int(clip_cov_ratio * loss_mask.sum().item()), 1)
    top_k_idx = (cov_all < clip_cov_ub) & (cov_all > clip_cov_lb) & (loss_mask > 0)
    top_k_idx = torch.nonzero(top_k_idx)

    if len(top_k_idx) > 0:
        perm = torch.randperm(len(top_k_idx))
        top_k_idx = top_k_idx[perm[: min(clip_num, len(top_k_idx))]]
    else:
        top_k_idx = torch.empty((0, 2), device=cov_all.device, dtype=torch.long)

    # Create correction mask
    corr = torch.ones_like(advantages)
    if len(top_k_idx) > 0:
        corr[top_k_idx[:, 0], top_k_idx[:, 1]] = 0

    # Compute clip fraction for monitoring
    clip_frac = masked_mean((corr == 0).float(), loss_mask)

    # Apply correction mask to losses
    pg_losses = torch.maximum(pg_losses1, pg_losses2) * corr
    pg_loss = reduce_loss(
        loss=pg_losses,
        loss_mask=loss_mask,
        loss_reduction=config.loss_reduction,
        max_seq_len=config.max_seq_len,
    )

    return pg_loss, clip_frac.item()


@register_policy_loss(PolicyLossType.KL_COV)
def compute_policy_loss_kl_cov(
    log_probs: torch.Tensor,
    old_log_probs: torch.Tensor,
    advantages: torch.Tensor,
    config: DictConfig,
    loss_mask: Optional[torch.Tensor] = None,
    rollout_logprobs: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, float]:
    """KL-Cov policy loss function implementation.

    Adapted from https://github.com/PRIME-RL/Entropy-Mechanism-of-RL/blob/main/verl/trainer/ppo/core_algos.py

    Uses covariance-based selection to apply KL regularization to a subset of tokens.
    """
    kl_cov_frac = config.kl_cov.kl_cov_frac  # This should be a percentage (e.g., 0.2 for 20%)
    ppo_kl_coef = config.kl_cov.ppo_kl_coef

    negative_approx_kl = log_probs - old_log_probs
    abs_kl = negative_approx_kl.abs()
    ratio = torch.exp(negative_approx_kl)

    pg_losses1 = -advantages * ratio
    pg_losses_kl = -advantages * ratio + ppo_kl_coef * abs_kl
    pg_losses = pg_losses1.clone()

    all_valid = loss_mask > 0
    all_valid_idx = torch.nonzero(all_valid.reshape(-1), as_tuple=True)[0]
    all_valid_adv = advantages[all_valid].detach().reshape(-1).cpu()
    all_valid_logp = log_probs[all_valid].detach().reshape(-1).cpu()

    if len(all_valid_adv) > 0:
        cov_lst_all = (all_valid_adv - all_valid_adv.mean()) * (all_valid_logp - all_valid_logp.mean())
        # Use percentage-based selection like the reference implementation
        k_percent_nums = max(1, int(len(cov_lst_all) * kl_cov_frac))

        if k_percent_nums > 0:
            large_cov_idxs = torch.topk(cov_lst_all, min(k_percent_nums, len(cov_lst_all)), largest=True).indices

            if len(large_cov_idxs) > 0:
                large_cov_idxs = all_valid_idx[large_cov_idxs]
                pg_losses[
                    large_cov_idxs // advantages.shape[1],
                    large_cov_idxs % advantages.shape[1],
                ] = pg_losses_kl[
                    large_cov_idxs // advantages.shape[1],
                    large_cov_idxs % advantages.shape[1],
                ]

    pg_loss = reduce_loss(
        loss=pg_losses,
        loss_mask=loss_mask,
        loss_reduction=config.loss_reduction,
        max_seq_len=config.max_seq_len,
    )

    # NOTE (sumanthrh): Since the pg clip ratio is not applicable for KL-COV so we just use 0.0
    return pg_loss, 0.0


def reduce_loss(
    loss: torch.Tensor,
    loss_mask: Optional[torch.Tensor],
    loss_reduction: Literal[
        "token_mean", "sequence_mean", "seq_mean_token_sum_norm", "seq_mean_token_sum_norm_global"
    ],
    max_seq_len: Optional[int] = None,
    global_denom: Optional[float] = None,
) -> torch.Tensor:
    if loss_reduction == "token_mean":
        # sum over *all* valid tokens, divide by total valid-token count
        loss = masked_mean(loss, loss_mask)
    elif loss_reduction == "sequence_mean":
        # per-sequence token-mean (dim=-1), then batch-mean
        loss = masked_mean(loss, loss_mask, dim=-1).mean()
    elif loss_reduction == "seq_mean_token_sum_norm":
        # per-sequence token-sum, normalized by the max sequence length, then batch mean
        # this is the Dr. GRPO loss reduction to avoid length bias by normalizing by a constant
        assert max_seq_len is not None, "max_seq_len must be provided for seq_mean_token_sum_norm loss reduction"
        # NOTE: max_seq_len is computed as cfg.generator.max_input_length + cfg.generator.sampling_params.max_generate_length by default
        if loss_mask is not None:
            seq_losses = torch.sum(loss * loss_mask, dim=-1) / max_seq_len
        else:
            # If no mask, assume all tokens are valid
            seq_losses = torch.sum(loss, dim=-1) / max_seq_len
        loss = torch.mean(seq_losses)
    elif loss_reduction == "seq_mean_token_sum_norm_global":
        # GLOBAL length-unbiased normalizer (Dr.GRPO fixed-const promoted to global).
        # Sum (NOT mean) the masked per-token loss for this micro-batch, divide by a
        # single global denominator Z = global_num_seqs * max_seq_len that is computed
        # once on the driver via a single all_reduce. Summing here (rather than meaning)
        # is what lets each micro-batch contribute its raw numerator to the one global
        # denominator -> the realized objective is (1/Z) * sum over the whole batch,
        # sidestepping the mean-of-per-microbatch-means size bias under grad-accum + async.
        # The caller MUST drop the /accumulation_steps on this (already-globally-normed)
        # policy term (see worker.py training_step).
        assert global_denom is not None, "global_denom must be provided for seq_mean_token_sum_norm_global"
        if loss_mask is not None:
            loss = torch.sum(loss * loss_mask) / global_denom
        else:
            # If no mask, assume all tokens are valid
            loss = torch.sum(loss) / global_denom
    else:
        raise ValueError(f"Invalid loss reduction type: {loss_reduction}")
    return loss


# NOTE (erictang000): below ported from verl
@register_advantage_estimator(AdvantageEstimator.REINFORCE_PP)
def compute_reinforce_plus_plus_outcome_advantage(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    gamma: float,
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute advantage for REINFORCE++.
    This implementation is based on the paper: https://arxiv.org/abs/2501.03262

    Args:
        - token_level_rewards: Float[torch.Tensor, "batch_size seqlen"]
        - response_mask: Float[torch.Tensor, "batch_size seqlen"]

    Returns:
        - advantages: Float[torch.Tensor, "batch_size seqlen"]
        - returns: Float[torch.Tensor, "batch_size seqlen"]
    """
    with torch.no_grad():
        returns = torch.zeros_like(token_level_rewards)
        running_return = 0

        for t in reversed(range(token_level_rewards.shape[1])):
            running_return = token_level_rewards[:, t] + gamma * running_return
            returns[:, t] = running_return
            # Reset after EOS
            running_return = running_return * response_mask[:, t]

        advantages = masked_whiten(returns, response_mask)
        advantages = advantages * response_mask

    return advantages, returns


@register_advantage_estimator(AdvantageEstimator.RLOO)
def compute_rloo_outcome_advantage(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    index: np.ndarray,
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute advantage for RLOO based on https://arxiv.org/abs/2402.14740

    This advantage estimator is also used in LOOP (https://arxiv.org/pdf/2502.01600),
    and was originally introduced in "Buy 4 REINFORCE Samples, Get a Baseline for Free!"
    (https://openreview.net/pdf?id=r1lgTGL5DE).

    Args:
        - token_level_rewards: Float[torch.Tensor, "batch_size seqlen"]
        - response_mask: Float[torch.Tensor, "batch_size seqlen"]
        - index: np.ndarray (batch_size)

    Returns:
        - advantages: Float[torch.Tensor, "batch_size seqlen"]
        - returns: Float[torch.Tensor, "batch_size seqlen"]
    """
    from loguru import logger as logger_

    scores = token_level_rewards.sum(dim=-1)

    id2score = defaultdict(list)
    id2mean = {}

    with torch.no_grad():
        bsz = scores.shape[0]
        for i in range(bsz):
            id2score[index[i]].append(scores[i])
        for idx in id2score:
            if len(id2score[idx]) == 1:
                id2mean[idx] = torch.tensor(0.0, device=scores.device)
            elif len(id2score[idx]) > 1:
                id2mean[idx] = torch.mean(torch.stack(id2score[idx]))
        for i in range(bsz):
            response_num = len(id2score[index[i]])
            if response_num > 1:
                factor = response_num / (response_num - 1)
                scores[i] = (scores[i] - id2mean[index[i]]) * factor
            else:
                # if there's only one response, set the advantage to 0
                logger_.warning(f"Only one response for prompt index {index[i]}, setting advantage to 0")
                scores[i] = 0.0
        scores = scores.unsqueeze(-1) * response_mask

    return scores, scores


@register_advantage_estimator(AdvantageEstimator.RLOO_N)
def compute_rloo_n_outcome_advantage(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    index: np.ndarray,
    exclude_from_baseline: Optional[np.ndarray] = None,
    config=None,
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    RLOO-N (RLOO-Neutral): RLOO variant that excludes masked samples from baseline computation.

    This addresses a key limitation in standard RLOO when handling failed samples:
    - Infrastructure failures (DaytonaError, NetworkError) should be treated as "neutral" -
      they don't reflect agent quality and shouldn't affect the baseline.
    - Agent failures (timeout, context overflow) should be included with zero reward.

    When exclude_from_baseline[i] is True:
    1. The sample is excluded from the group baseline calculation
    2. The sample receives advantage=0 (no gradient contribution)
    3. Other samples in the group have their baselines computed WITHOUT this sample

    This is different from just setting reward=0, which would still pollute the baseline
    by dragging down the mean for the entire group.

    Args:
        - token_level_rewards: Float[torch.Tensor, "batch_size seqlen"]
        - response_mask: Float[torch.Tensor, "batch_size seqlen"]
        - index: np.ndarray (batch_size) - group IDs for each sample
        - exclude_from_baseline: Optional[np.ndarray] (batch_size) - bool array, True = exclude

    Returns:
        - advantages: Float[torch.Tensor, "batch_size seqlen"]
        - returns: Float[torch.Tensor, "batch_size seqlen"]
    """
    from loguru import logger as logger_

    scores = token_level_rewards.sum(dim=-1)
    bsz = scores.shape[0]

    # Minimum included samples per group for a reliable leave-one-out baseline.
    # Groups below this threshold get advantage=0 for all samples.
    min_group_size = 2  # backwards-compatible default
    filter_zero_reward_groups = True  # skip all-zero-reward groups by default
    if config is not None:
        min_group_size = getattr(config, 'rloo_n_min_group_size', 2)
        filter_zero_reward_groups = getattr(config, 'rloo_n_filter_zero_reward_groups', True)

    # Default: include all samples in baseline
    if exclude_from_baseline is None:
        exclude_from_baseline = np.zeros(bsz, dtype=bool)

    # Build per-group score lists, separating included vs excluded
    id2included_scores = defaultdict(list)  # scores to include in baseline
    id2included_indices = defaultdict(list)  # indices of included samples
    id2excluded_indices = defaultdict(list)  # indices of excluded samples

    with torch.no_grad():
        # First pass: categorize samples
        for i in range(bsz):
            group_id = index[i]
            if exclude_from_baseline[i]:
                id2excluded_indices[group_id].append(i)
            else:
                id2included_scores[group_id].append(scores[i])
                id2included_indices[group_id].append(i)

        # Detect zero-variance reward groups: when all included samples have
        # the same reward, RLOO advantage is mathematically zero. Training on
        # these groups contributes noise that can push the policy toward
        # entropy collapse. Filter them out.
        # This handles binary reward (all 0 or all 1), shaped reward where
        # all samples tie, and any other constant-reward scenario.
        id2no_variance = {}
        n_no_variance_groups = 0
        n_no_variance_samples = 0
        for group_id in set(index):
            included = id2included_scores[group_id]
            if filter_zero_reward_groups and len(included) > 1:
                stacked = torch.stack(included)
                has_no_variance = (stacked.max() - stacked.min()).item() == 0.0
                id2no_variance[group_id] = has_no_variance
                if has_no_variance:
                    n_no_variance_groups += 1
                    n_no_variance_samples += len(included) + len(id2excluded_indices[group_id])
            else:
                id2no_variance[group_id] = False

        # Second pass: compute baselines using only included samples
        id2mean = {}
        for group_id in set(index):
            included = id2included_scores[group_id]
            if id2no_variance.get(group_id, False):
                # Zero-variance group — skip entirely
                id2mean[group_id] = torch.tensor(0.0, device=scores.device)
            elif len(included) < min_group_size:
                # Below minimum group size — can't compute reliable baseline
                id2mean[group_id] = torch.tensor(0.0, device=scores.device)
            else:
                id2mean[group_id] = torch.mean(torch.stack(included))

        # Third pass: compute advantages
        for i in range(bsz):
            group_id = index[i]

            if exclude_from_baseline[i]:
                # Excluded samples get zero advantage (no gradient contribution)
                scores[i] = 0.0
                continue

            if id2no_variance.get(group_id, False):
                # Zero-variance reward group — zero advantage, no gradient
                scores[i] = 0.0
                continue

            # For included samples: use leave-one-out baseline from OTHER included samples
            included_scores = id2included_scores[group_id]
            n_included = len(included_scores)

            if n_included < min_group_size:
                # Below minimum group size — zero advantage for all included samples
                logger_.warning(
                    f"RLOO-N: Group {group_id} has {n_included} included sample(s) "
                    f"(min_group_size={min_group_size}), setting advantage to 0"
                )
                scores[i] = 0.0
            else:
                # Standard RLOO leave-one-out: baseline = mean of OTHER samples
                # With correction factor: (n / (n-1)) * (score - group_mean)
                factor = n_included / (n_included - 1)
                scores[i] = (scores[i] - id2mean[group_id]) * factor

        # Log summary statistics
        n_excluded = sum(len(v) for v in id2excluded_indices.values())
        n_groups_all_excluded = sum(
            1 for group_id in set(index)
            if len(id2included_scores[group_id]) == 0
        )
        n_groups_below_min = sum(
            1 for group_id in set(index)
            if 0 < len(id2included_scores[group_id]) < min_group_size
        )
        n_total_groups = len(set(index))
        if n_excluded > 0 or n_groups_below_min > 0 or n_no_variance_groups > 0:
            logger_.info(
                f"RLOO-N: {n_excluded}/{bsz} samples excluded from baseline, "
                f"{n_groups_all_excluded} groups had all samples excluded, "
                f"{n_groups_below_min} groups below min_group_size={min_group_size}"
                + (f", {n_no_variance_groups}/{n_total_groups} groups filtered "
                   f"(zero reward variance, {n_no_variance_samples} samples)" if n_no_variance_groups > 0 else "")
            )

        scores = scores.unsqueeze(-1) * response_mask

    return scores, scores


@register_advantage_estimator("rloo_n_pbs")
def compute_rloo_n_pbs_advantage(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    index: np.ndarray,
    exclude_from_baseline: Optional[np.ndarray] = None,
    config=None,
    token_level_shaping: Optional[torch.Tensor] = None,
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor]:
    """RLOO-N outcome advantage + potential-based shaping (Stage C / F6).

    Combines RLOO-N's per-trajectory outcome advantage (computed exactly as
    ``compute_rloo_n_outcome_advantage`` — the outcome term reads ``rewards``
    ONLY and is left bit-for-bit intact) with the per-token potential-based
    shaping channel ``token_level_shaping`` (the PBS test-delta credit scattered
    onto the EDIT-token span by ``pbs_shaping.compute_pbs_token_shaping``).

    The combination is the additive + separate seam proven in Stage B:

        advantage = rloo_n_outcome_advantage + token_level_shaping * response_mask

    Properties:
      * ``token_level_shaping is None`` or all-zeros ⇒ this returns EXACTLY the
        RLOO-N advantage (pure RLOO-N; the flag-off / no-signal path).
      * PBS is policy-invariant (Ng 1999): ``token_level_shaping`` is a true
        potential difference ``γ·Φ(s') − Φ(s)`` built upstream, so adding it
        cannot change the optimal policy.
      * The shaping is masked by ``response_mask`` and only applied to response
        tokens (the same support as the outcome advantage), so the
        advantage/loss denominator (``response_mask.sum()``) is unchanged — no
        seqnorm-style denominator break.

    Returns ``(advantages, returns)`` with the same shape/semantics as RLOO-N
    (advantages == returns; critic-free).
    """
    # Outcome term: unchanged RLOO-N (reads `rewards` only).
    outcome_adv, _ = compute_rloo_n_outcome_advantage(
        token_level_rewards=token_level_rewards,
        response_mask=response_mask,
        index=index,
        exclude_from_baseline=exclude_from_baseline,
        config=config,
        **kwargs,
    )

    if token_level_shaping is None:
        return outcome_adv, outcome_adv

    with torch.no_grad():
        shaping = token_level_shaping.to(device=outcome_adv.device, dtype=outcome_adv.dtype)
        # Defensive shape-align (right-padded to response_mask width).
        if shaping.shape != response_mask.shape:
            sl = min(shaping.shape[-1], response_mask.shape[-1])
            aligned = torch.zeros_like(response_mask, dtype=outcome_adv.dtype)
            aligned[..., :sl] = shaping[..., :sl]
            shaping = aligned
        combined = outcome_adv + shaping * response_mask

    return combined, combined


@register_advantage_estimator(AdvantageEstimator.GAE)
def compute_gae_advantage_return(
    token_level_rewards: Float[torch.Tensor, "batch_size seqlen"],
    values: Float[torch.Tensor, "batch_size seqlen"],
    response_mask: Float[torch.Tensor, "batch_size seqlen"],
    gamma: float,
    lambd: float,
    **kwargs,
) -> Tuple[Float[torch.Tensor, "batch_size seqlen"], Float[torch.Tensor, "batch_size seqlen"]]:
    """
    Compute advantage and return for GAE.

    Adapted from https://github.com/huggingface/trl/blob/main/trl/trainer/ppo_trainer.py
    """
    with torch.no_grad():
        lastgaelam = 0
        advantages_reversed = []
        gen_len = token_level_rewards.shape[-1]

        for t in reversed(range(gen_len)):
            nextvalues = values[:, t + 1] if t < gen_len - 1 else 0.0
            delta = token_level_rewards[:, t] + gamma * nextvalues - values[:, t]
            lastgaelam = delta + gamma * lambd * lastgaelam
            advantages_reversed.append(lastgaelam)
        advantages = torch.stack(advantages_reversed[::-1], dim=1)

        returns = advantages + values
        advantages = masked_whiten(advantages, response_mask)
    return advantages, returns


@register_advantage_estimator(AdvantageEstimator.GRPO)
def compute_grpo_outcome_advantage(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    index: np.ndarray,
    epsilon: float = 1e-6,
    grpo_norm_by_std: bool = True,
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute advantage for GRPO, operating only on Outcome reward (with only one scalar reward for each response).

    Expects:
        - token_level_rewards: Float[torch.Tensor, "batch_size seqlen"]
        - response_mask: Float[torch.Tensor, "batch_size seqlen"]
        - index: np.ndarray (batch_size)
        - epsilon: float
        - grpo_norm_by_std: bool

    Returns:
        - advantages: Float[torch.Tensor, "batch_size seqlen"]
        - returns: Float[torch.Tensor, "batch_size seqlen"]
    """
    # this assumes response-level rewards
    scores = token_level_rewards.sum(dim=-1)

    id2score = defaultdict(list)
    id2mean = {}
    id2std = {}

    with torch.no_grad():
        bsz = scores.shape[0]
        for i in range(bsz):
            id2score[index[i]].append(scores[i])
        for idx in id2score:
            if len(id2score[idx]) == 1:
                id2mean[idx] = torch.tensor(0.0)
                id2std[idx] = torch.tensor(1.0)
            elif len(id2score[idx]) > 1:
                id2mean[idx] = torch.mean(torch.tensor(id2score[idx]))
                id2std[idx] = torch.std(torch.tensor([id2score[idx]]))
            else:
                raise ValueError(f"no score in prompt index: {idx}")
        for i in range(bsz):
            if grpo_norm_by_std:
                scores[i] = (scores[i] - id2mean[index[i]]) / (id2std[index[i]] + epsilon)
            else:
                scores[i] = scores[i] - id2mean[index[i]]
        scores = scores.unsqueeze(-1) * response_mask

    return scores, scores


def repopulate_all_registries():
    PolicyLossRegistry.repopulate_registry()
    AdvantageEstimatorRegistry.repopulate_registry()


def compute_advantages_and_returns(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    index: np.ndarray,
    adv_estimator: AdvantageEstimator,
    config: DictConfig,
    values: Optional[torch.Tensor] = None,
    grpo_norm_by_std: bool = True,
    gamma=1.0,
    lambd=1.0,
    exclude_from_baseline: Optional[np.ndarray] = None,
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor]:
    estimator_func = AdvantageEstimatorRegistry.get(adv_estimator)

    return estimator_func(
        token_level_rewards=token_level_rewards,
        response_mask=response_mask,
        index=index,
        values=values,
        grpo_norm_by_std=grpo_norm_by_std,
        gamma=gamma,
        lambd=lambd,
        config=config,
        exclude_from_baseline=exclude_from_baseline,
        **kwargs,
    )
