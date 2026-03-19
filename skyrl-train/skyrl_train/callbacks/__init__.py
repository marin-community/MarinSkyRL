"""
Callback system for SkyRL trainers.

This module provides a HuggingFace-style callback system for extending trainer behavior
with periodic actions like checkpointing, evaluation, logging, and custom hooks.
"""

from .base import (
    TrainerCallback,
    TrainerState,
    TrainerControl,
    CallbackHandler,
    AtomicStepCounter,
)
from .builtin import (
    CheckpointCallback,
    EvaluationCallback,
    HFModelSaveCallback,
    RefModelUpdateCallback,
    ProgressCallback,
    LoggingCallback,
    DefaultCallbackHandler,
    DataTrackingCallback,
    # YAML configuration support
    CALLBACK_REGISTRY,
    register_callback,
    create_callback_from_config,
    create_callbacks_from_config,
    get_available_callback_types,
)

__all__ = [
    # Core classes
    "TrainerCallback",
    "TrainerState",
    "TrainerControl",
    "CallbackHandler",
    "AtomicStepCounter",
    # Built-in callbacks
    "CheckpointCallback",
    "EvaluationCallback",
    "HFModelSaveCallback",
    "RefModelUpdateCallback",
    "ProgressCallback",
    "LoggingCallback",
    "DefaultCallbackHandler",
    "DataTrackingCallback",
    # YAML configuration support
    "CALLBACK_REGISTRY",
    "register_callback",
    "create_callback_from_config",
    "create_callbacks_from_config",
    "get_available_callback_types",
]
