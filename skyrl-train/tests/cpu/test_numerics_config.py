import sys
import types

import pytest
import torch
from omegaconf import OmegaConf
from transformers import LlamaConfig

import skyrl_train.inference_engines.ray_wrapped_inference_engine as rwie
import skyrl_train.objective.losses  # noqa: F401
from skyrl_train.config.decode_invariant import DECODE_INVARIANT_ATTENTION_BACKEND
from skyrl_train.config.grug_vllm_shapes import ROTARY_POSITIONS, VOCAB
from skyrl_train.config.numerics import TRITON_MOE_BACKEND, Numerics
from skyrl_train.config.trajectory_runner_capabilities import TrajectoryRunnerMode
from skyrl_train.config.utils import get_default_config
from skyrl_train.entrypoints import main_base
from skyrl_train.inference_engines.utils import get_vllm_sampling_params
from skyrl_train.models.grug_moe import GrugMoeConfig
from skyrl_train.utils.utils import validate_cfg

GYM = TrajectoryRunnerMode.SKYRL_GYM


def _snowball_cfg(model_path: str):
    """A run of a Snowball-shaped Grug policy on the base config, with unpacked sequences and one GPU per engine."""
    cfg = get_default_config()
    cfg.trainer.logger = "console"
    cfg.trainer.use_sample_packing = False
    cfg.trainer.policy.model.path = model_path
    cfg.generator.inference_engine_tensor_parallel_size = 1
    cfg.generator.num_inference_engines = 4
    return cfg


@pytest.fixture
def snowball_model(tmp_path) -> str:
    path = tmp_path / "snowball"
    GrugMoeConfig().save_pretrained(path)
    return str(path)


def _engine_options(cfg, monkeypatch) -> dict:
    """The options the engine factory receives for ``cfg``."""
    options = {}

    def capture(**kwargs):
        options.update(kwargs)
        return []

    monkeypatch.setattr(rwie, "create_ray_wrapped_inference_engines", capture)
    main_base.create_ray_wrapped_inference_engines_from_config(cfg, colocate_pg=None, tokenizer=None)
    return options


@pytest.mark.parametrize(
    ("clear_cache", "numerics", "decode_invariant"),
    [(True, Numerics.EXACT, True), (False, Numerics.NATIVE, False)],
    ids=["supported", "fallback"],
)
def test_a_snowball_runs_engines_follow_the_resolved_numerics(
    snowball_model, monkeypatch, clear_cache, numerics, decode_invariant
):
    cfg = _snowball_cfg(snowball_model)
    cfg.generator.weight_sync_pause.clear_cache = clear_cache

    validate_cfg(cfg)

    assert cfg.trainer.algorithm.resolved_numerics == numerics
    options = _engine_options(cfg, monkeypatch)
    assert options["decode_invariant"] is decode_invariant
    if decode_invariant:
        # Exact sets the engine kernels it computes, and samples with the program whose log-probabilities the
        # trainer computes.
        assert options["vllm_attention_backend"] == DECODE_INVARIANT_ATTENTION_BACKEND
        assert options["engine_init_kwargs"]["moe_backend"] == TRITON_MOE_BACKEND
        assert options["engine_init_kwargs"]["logprobs_mode"] == "processed_logprobs"
        assert get_vllm_sampling_params(cfg.generator.sampling_params)["min_tokens"] == 0
    else:
        assert options["vllm_attention_backend"] is None
        assert "moe_backend" not in options["engine_init_kwargs"]
        assert "logprobs_mode" not in options["engine_init_kwargs"]


def _settings(values: dict):
    def change(cfg, tmp_path, monkeypatch):
        for path, value in values.items():
            OmegaConf.update(cfg, path, value, force_add=True)

    return change


def _policy(model_config):
    def change(cfg, tmp_path, monkeypatch):
        path = tmp_path / "policy"
        model_config.save_pretrained(path)
        cfg.trainer.policy.model.path = str(path)

    return change


def _unreadable_policy(cfg, tmp_path, monkeypatch):
    cfg.trainer.policy.model.path = str(tmp_path / "missing")


def _trainer_flash_attention(cfg, tmp_path, monkeypatch):
    # validate_cfg imports flash_attn, a GPU package, for trainer.flash_attn=true.
    monkeypatch.setitem(sys.modules, "flash_attn", types.ModuleType("flash_attn"))
    cfg.trainer.flash_attn = True


def _blackwell_gpu(cfg, tmp_path, monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device=None: (10, 0))
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda device=None: "NVIDIA GB200")


def _unchanged(cfg, tmp_path, monkeypatch):
    pass


MEGATRON = "trainer.policy.megatron_config"
FALLBACKS = {
    "another model family": (_policy(LlamaConfig(num_hidden_layers=1)), GYM, "not Grug"),
    "another Grug shape": (_policy(GrugMoeConfig(num_key_value_heads=4)), GYM, "num_key_value_heads"),
    "a larger vocabulary": (_policy(GrugMoeConfig(vocab_size=VOCAB + 64)), GYM, "vocab_size"),
    "Hero parts": (_policy(GrugMoeConfig(sconv=True)), GYM, "Hero"),
    "unreadable model config": (_unreadable_policy, GYM, "could not be read"),
    "trainer tensor parallelism": (
        _settings({f"{MEGATRON}.tensor_model_parallel_size": 2}),
        GYM,
        "tensor_model_parallel_size",
    ),
    "expert tensor parallelism": (
        _settings({f"{MEGATRON}.expert_tensor_parallel_size": 2}),
        GYM,
        "expert_tensor_parallel_size",
    ),
    "four pipeline stages": (
        _settings({f"{MEGATRON}.pipeline_model_parallel_size": 4}),
        GYM,
        "pipeline_model_parallel_size",
    ),
    "overlapped parameter all-gather": (
        _settings({f"{MEGATRON}.ddp_config.overlap_param_gather": True}),
        GYM,
        "overlap_param_gather",
    ),
    "another token dispatcher": (
        _settings({f"{MEGATRON}.transformer_config_kwargs.moe_token_dispatcher_type": "allgather"}),
        GYM,
        "moe_token_dispatcher_type",
    ),
    "multi-layer recompute units": (
        _settings({f"{MEGATRON}.transformer_config_kwargs.recompute_num_layers": 2}),
        GYM,
        "recompute",
    ),
    "an overridden model setting": (
        _settings({f"{MEGATRON}.transformer_config_kwargs.layernorm_epsilon": 1e-6}),
        GYM,
        "layernorm_epsilon",
    ),
    "sequences past the rotary table": (
        _settings({"generator.sampling_params.max_generate_length": ROTARY_POSITIONS}),
        GYM,
        "max_generate_length",
    ),
    "an engine rotary override": (
        _settings({"generator.rope_scaling": {"rope_type": "yarn", "factor": 2.0}}),
        GYM,
        "rope_scaling",
    ),
    "FlashAttention 2 engines": (
        _settings({"generator.engine_init_kwargs.attention_config": {"flash_attn_version": 2}}),
        GYM,
        "flash_attn_version",
    ),
    "packed sequences": (_settings({"trainer.use_sample_packing": True}), GYM, "use_sample_packing"),
    "trainer flash attention": (_trainer_flash_attention, GYM, "flash_attn"),
    "engine tensor parallelism": (
        _settings({"generator.inference_engine_tensor_parallel_size": 2, "generator.num_inference_engines": 2}),
        GYM,
        "inference_engine_tensor_parallel_size",
    ),
    "remote engines": (
        _settings(
            {
                "generator.run_engines_locally": False,
                "generator.remote_inference_engine_urls": [f"127.0.0.1:{8000 + index}" for index in range(4)],
            }
        ),
        GYM,
        "run_engines_locally",
    ),
    "eager engines": (_settings({"generator.enforce_eager": True}), GYM, "enforce_eager"),
    "another attention backend": (
        _settings({"generator.vllm_attention_backend": "FLASHINFER"}),
        GYM,
        "vllm_attention_backend",
    ),
    "another MoE backend": (
        _settings({"generator.engine_init_kwargs.moe_backend": "flashinfer_cutlass"}),
        GYM,
        "moe_backend",
    ),
    "another expert-parallel combine": (
        _settings({"generator.engine_init_kwargs.all2all_backend": "deepep_low_latency"}),
        GYM,
        "all2all_backend",
    ),
    "an engine compilation config": (
        _settings({"generator.engine_init_kwargs.compilation_config": {"mode": 0}}),
        GYM,
        "compilation_config",
    ),
    "speculative decoding": (
        _settings({"generator.speculative_decoding": {"method": "eagle3", "num_speculative_tokens": 3}}),
        GYM,
        "speculative_decoding",
    ),
    "temperature": (_settings({"generator.sampling_params.temperature": 0.7}), GYM, "temperature"),
    "truncated sampling": (_settings({"generator.sampling_params.top_p": 0.9}), GYM, "top_p"),
    "a runner without serving ranks": (_unchanged, TrajectoryRunnerMode.HARBOR, "serving engine rank"),
    "a GPU other than Hopper": (_blackwell_gpu, GYM, "compute capability 10.0"),
}


@pytest.mark.parametrize(
    ("change", "runner_mode", "reason"), FALLBACKS.values(), ids=[name.replace(" ", "-") for name in FALLBACKS]
)
def test_exact_falls_back_to_native_and_records_why(snowball_model, tmp_path, monkeypatch, change, runner_mode, reason):
    cfg = _snowball_cfg(snowball_model)
    change(cfg, tmp_path, monkeypatch)

    validate_cfg(cfg, runner_mode)

    assert cfg.trainer.algorithm.resolved_numerics == Numerics.NATIVE
    assert any(reason in recorded for recorded in cfg.trainer.algorithm.numerics_fallback_reasons)
    # A native run samples with the program it was configured with.
    assert "logprobs_mode" not in cfg.generator.engine_init_kwargs


def test_configured_native_numerics_is_kept_where_exact_is_supported(snowball_model):
    cfg = _snowball_cfg(snowball_model)
    cfg.trainer.algorithm.numerics = str(Numerics.NATIVE)

    validate_cfg(cfg)

    assert cfg.trainer.algorithm.resolved_numerics == Numerics.NATIVE
    assert cfg.trainer.algorithm.numerics_fallback_reasons == []
    assert cfg.generator.vllm_attention_backend is None
    assert "logprobs_mode" not in cfg.generator.engine_init_kwargs
