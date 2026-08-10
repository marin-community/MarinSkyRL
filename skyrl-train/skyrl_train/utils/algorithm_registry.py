"""Registries for policy losses and advantage estimators.

Adapted from VERL's ``trainer/ppo/core_algos.py`` (ByteDance and Hugging Face),
licensed under Apache 2.0.
"""

from __future__ import annotations

from enum import StrEnum
from functools import wraps
from typing import Callable, Union

import ray
from loguru import logger

from skyrl_train.utils.function_registry import BaseFunctionRegistry


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


def register_advantage_estimator(name: Union[str, AdvantageEstimator]):
    """Decorator to register an advantage estimator function."""
    registry_name = name.value if isinstance(name, AdvantageEstimator) else name

    def decorator(func: Callable):
        @wraps(func)
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)

        AdvantageEstimatorRegistry.register(registry_name, wrapper)
        return wrapper

    return decorator


def register_policy_loss(name: Union[str, PolicyLossType]):
    """Decorator to register a policy loss function."""
    registry_name = name.value if isinstance(name, PolicyLossType) else name

    def decorator(func: Callable):
        @wraps(func)
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)

        PolicyLossRegistry.register(registry_name, wrapper)
        return wrapper

    return decorator


def sync_registries():
    """Sync the registries with the ray actor once ray is initialized"""
    if not ray.is_initialized():
        raise ValueError("Ray is not initialized, cannot sync registries")
    PolicyLossRegistry.sync_with_actor()
    AdvantageEstimatorRegistry.sync_with_actor()
    logger.info("Synced registries to ray actor")
