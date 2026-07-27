# utility functions used for CPU tests

import importlib.machinery
import importlib.util
import sys
import types

from skyrl_train.config.utils import get_default_config
from omegaconf import OmegaConf


def example_dummy_config():
    cfg = get_default_config()
    # TODO (sumanthrh): Some of these overrides are no longer needed after reading from the config file
    trainer_overrides = {
        "project_name": "unit-test",
        "run_name": "test-run",
        "logger": "tensorboard",
        "micro_train_batch_size_per_gpu": 2,
        "train_batch_size": 2,
        "eval_batch_size": 2,
        "update_epochs_per_batch": 1,
        "epochs": 1,
        "max_prompt_length": 20,
        "use_sample_packing": False,
        "seed": 42,
        "resume_mode": "none",
        "algorithm": {
            "advantage_estimator": "grpo",
            "use_kl_estimator_k3": False,
            "use_abs_kl": False,
            "kl_estimator_type": "k1",
            "use_kl_loss": True,
            "kl_loss_coef": 0.0,
            "loss_reduction": "token_mean",
            "grpo_norm_by_std": True,
        },
    }
    generator_overrides = {
        "sampling_params": {"max_generate_length": 20},
        "n_samples_per_prompt": 1,
        "batched": False,
        "max_turns": 1,
        "enable_http_endpoint": False,
        "http_endpoint_host": "127.0.0.1",
        "http_endpoint_port": 8000,
    }
    OmegaConf.update(cfg, "trainer", trainer_overrides)
    OmegaConf.update(cfg, "generator", generator_overrides)

    return cfg


def stub_megatron_modules() -> None:
    """Install stub `megatron` submodules when megatron is not installed.

    Several skyrl_train megatron modules import `megatron.core` names at module
    load, but the functions the CPU tests exercise never call into them. The CPU
    CI environment has no megatron, so the stubs let those modules import. No-op
    when the real megatron is importable, so a real-megatron env is untouched.
    Idempotent: missing attributes are added to already-installed stubs.
    """
    try:
        if importlib.util.find_spec("megatron") is not None:
            return
    except ValueError:
        # A spec-less stub from an earlier call/test is already installed.
        pass

    stub_attrs = {
        "megatron": {},
        "megatron.core": {},
        "megatron.core.parallel_state": {},
        "megatron.core.pipeline_parallel": {"get_forward_backward_func": lambda: None},
        "megatron.core.distributed": {
            "DistributedDataParallel": type("DistributedDataParallel", (), {}),
            "finalize_model_grads": lambda *args, **kwargs: None,
        },
        "megatron.core.transformer": {},
        "megatron.core.transformer.module": {"Float16Module": type("Float16Module", (), {})},
        "megatron.core.optimizer": {"ChainedOptimizer": type("ChainedOptimizer", (), {})},
        "megatron.core.utils": {"get_attr_wrapped_model": lambda *args, **kwargs: None},
        "megatron.core.packed_seq_params": {"PackedSeqParams": type("PackedSeqParams", (), {})},
    }
    for name, members in stub_attrs.items():
        module = sys.modules.setdefault(name, types.ModuleType(name))
        # Give stubs a spec so later find_spec("megatron") calls see a normal
        # module instead of raising ValueError on a spec-less one.
        if getattr(module, "__spec__", None) is None:
            module.__spec__ = importlib.machinery.ModuleSpec(name, loader=None)
        for attr, value in members.items():
            setattr(module, attr, value)
