"""The launch config the trainer gives each vendored vLLM kernel, from the kernels a vLLM worker loaded."""

from skyrl_train.models.grug_inductor_kernels import (
    VENDORED_KERNELS,
    KernelConfigs,
    Launch,
    engine_kernel_choices,
    source_digest,
)

NORM_2048 = {"kwargs": {"XBLOCK": 1, "R0_BLOCK": 2048}, "num_warps": 16, "num_stages": 1}
NORM_4096 = {"kwargs": {"XBLOCK": 1, "R0_BLOCK": 4096}, "num_warps": 16, "num_stages": 1}
# An Inductor ``.best_config`` file naming R0_BLOCK 4,096, with the fields Inductor stores beside the config.
BEST_4096 = {
    "XBLOCK": 1,
    "R0_BLOCK": 4096,
    "num_warps": 16,
    "num_stages": 1,
    "configs_hash": "c0ffee",
    "found_by_coordesc": False,
    "time_taken_ms": 3,
    "triton_cache_hash": "beef",
}


def _loaded(role: str, launches: list[dict], best_config: dict | None) -> dict:
    return {"sha256": source_digest(VENDORED_KERNELS[role][1]), "launches": launches, "best_config": best_config}


def test_engine_kernel_choices_take_the_config_each_vendored_kernel_launched():
    choices = engine_kernel_choices(
        [
            # Autotuned in this process: the kernel holds the one config it launched, while the cache file, which
            # ranks sharing one Inductor cache overwrite, names another.
            _loaded("rms_norm", [NORM_2048], BEST_4096),
            # Not launched in this process: the kernel holds every candidate, so its ``.best_config`` decides.
            _loaded("final_norm", [NORM_2048, NORM_4096], BEST_4096),
            # A kernel the trainer does not vendor.
            {"sha256": "0" * 64, "launches": [NORM_2048], "best_config": None},
        ]
    )
    configs = KernelConfigs.from_records({role: choice["launch"] for role, choice in choices.items()})

    assert configs.launch("rms_norm") == Launch((("R0_BLOCK", 2048), ("XBLOCK", 1)), 16, 1)
    assert configs.launch("final_norm") == Launch((("R0_BLOCK", 4096), ("XBLOCK", 1)), 16, 1)
    assert (choices["rms_norm"]["source"], choices["rms_norm"]["best_config_agrees"]) == ("autotuner", False)
    assert sorted(choices) == ["final_norm", "rms_norm"]
    # Kernels the worker did not load keep the trainer's defaults.
    assert configs.launch("residual_norm") == KernelConfigs().launch("residual_norm")
