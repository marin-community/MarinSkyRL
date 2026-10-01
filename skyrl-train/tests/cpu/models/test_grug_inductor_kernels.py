"""The trainer's record of how a vLLM worker's compiled forward ran the kernels it vendors."""

import pytest

from skyrl_train.models.grug_inductor_kernels import (
    VENDORED_KERNELS,
    KernelConfigs,
    Launch,
    engine_kernels,
    kernel_configs,
    recorded_gate_columns,
)

NORM_2048 = {"kwargs": {"XBLOCK": 1, "R0_BLOCK": 2048}, "num_warps": 16, "num_stages": 1}
XSA = {"kwargs": {"XBLOCK": 2, "R0_BLOCK": 128}, "num_warps": 2, "num_stages": 1}


def _launched(role: str, name: str, launches: list[dict], edit: tuple[str, str] = ("", "")) -> dict:
    """A kernel a worker launched: a vendored function as a graph module numbers it, optionally with one line edited."""
    vendored_name, text = VENDORED_KERNELS[role]
    return {"name": name, "source": text.replace(vendored_name, name).replace(*edit), "launches": launches}


# The XSA kernel of a graph whose head-gate GEMM was left at 20 columns reads the gate at that row stride.
UNPADDED_GATE = ("in_ptr2 + (x0 + 24*x1)", "in_ptr2 + (x3)")


def test_engine_kernels_identify_renumbered_kernels_and_each_layers_head_gate_width():
    kernels = [
        # The first graph's kernels keep the vendored numbering; the second graph, compiled without the padded
        # head-gate weight's copy, numbers every kernel one lower.
        _launched("xsa_gate", "triton_red_fused_xsa_1", [XSA]),
        _launched("rms_norm", "triton_red_fused_rms_norm_2", [NORM_2048]),
        _launched("xsa_gate", "triton_red_fused_xsa_0", [XSA], UNPADDED_GATE),
        _launched("rms_norm", "triton_red_fused_rms_norm_1", [NORM_2048]),
        # A kernel the trainer does not vendor.
        {
            "name": "triton_poi_fused_mm_t_0",
            "source": "def triton_poi_fused_mm_t_0(in_ptr0):\n    pass\n",
            "launches": [XSA],
        },
    ]
    # Two eager steps of a three-layer model: layers 0 and 1 in the first graph, layer 2 in the second.
    sequence = [4, 0, 1, 4, 0, 1, 2, 3]
    launched = engine_kernels(kernels, [sequence, sequence])

    assert launched.gate_columns == (24, 24, 20)
    assert launched.launches == {"xsa_gate": XSA, "rms_norm": NORM_2048}
    configs = KernelConfigs.from_engine({"launches": launched.launches, "gate_columns": launched.gate_columns})
    assert configs.launch("rms_norm") == Launch((("R0_BLOCK", 2048), ("XBLOCK", 1)), 16, 1)
    # Kernels the worker did not launch keep the trainer's defaults.
    assert configs.launch("residual_norm") == KernelConfigs().launch("residual_norm")
    with kernel_configs(configs):
        assert [recorded_gate_columns(layer) for layer in range(3)] == [24, 24, 20]
    assert recorded_gate_columns(2) is None


def test_engine_kernels_refuse_steps_that_ran_different_head_gate_widths():
    kernels = [
        _launched("xsa_gate", "triton_red_fused_xsa_1", [XSA]),
        _launched("xsa_gate", "triton_red_fused_xsa_0", [XSA], UNPADDED_GATE),
    ]
    with pytest.raises(ValueError, match="different head-gate widths"):
        engine_kernels(kernels, [[0], [1]])
