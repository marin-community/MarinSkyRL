"""
Database utilities for SkyRL model registration.

This module provides Supabase-backed registration for trained RL models,
enabling automatic tracking of training runs in a centralized database.

Copied from LLaMA-Factory's unified_db module to avoid the dependency.
"""

from .utils import (
    load_supabase_keys,
    register_trained_model,
    register_agent,
    register_hf_dataset,
    get_dataset_by_name,
    get_model_by_name,
    get_agent_by_name,
)

__all__ = [
    "load_supabase_keys",
    "register_trained_model",
    "register_agent",
    "register_hf_dataset",
    "get_dataset_by_name",
    "get_model_by_name",
    "get_agent_by_name",
]
