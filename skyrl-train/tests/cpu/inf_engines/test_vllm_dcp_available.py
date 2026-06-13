"""Stage-1 fail-fast: the vLLM in this environment must expose Decode Context
Parallel (DCP).

DCP is the one hard external dependency of the vLLM-DCP rollout port: the
production SIF / RL venv's vLLM must surface `decode_context_parallel_size` as a
native `EngineArgs` / `ParallelConfig` field and build a DCP process group
(`get_dcp_group`). If a SIF lacks these, DCP cannot ship on it — and that is a
SIF-rebuild blocker that must surface at build/CI time, not deep in a multi-node
vLLM init.

These tests are skipped where vllm is not importable (e.g. the Mac dev box, which
cannot import the full GPU stack). On a runtime where vllm imports (the Jupiter RL
venv / production SIF) they are the gate.

See notes/RL/skyrl/vllm_dcp_rollout_stages/stage1_vllm_support_and_plumbing_scope.md.

Run:
    uv run --isolated --extra dev pytest tests/cpu/inf_engines/test_vllm_dcp_available.py -v
"""

import pytest

vllm = pytest.importorskip("vllm", reason="vllm not importable in this environment")


def test_engine_args_exposes_dcp_field():
    """`decode_context_parallel_size` is a native vLLM EngineArgs field.

    This is the exact kwarg Stage 1 threads through and Stages 2/3 rely on.
    """
    from vllm.engine.arg_utils import EngineArgs

    assert (
        "decode_context_parallel_size" in EngineArgs.__dataclass_fields__
        or hasattr(EngineArgs, "decode_context_parallel_size")
    ), "vLLM EngineArgs lacks decode_context_parallel_size — DCP unsupported on this vLLM (SIF-rebuild blocker)"


def test_parallel_config_exposes_dcp_field():
    """`ParallelConfig.decode_context_parallel_size` — the exact symbol Stage 2/3 rely on."""
    from vllm.config.parallel import ParallelConfig

    assert hasattr(
        ParallelConfig, "decode_context_parallel_size"
    ), "vLLM ParallelConfig lacks decode_context_parallel_size — DCP unsupported (SIF-rebuild blocker)"


def test_get_dcp_group_importable():
    """The DCP process-group plumbing is present."""
    from vllm.distributed.parallel_state import get_dcp_group  # noqa: F401


def test_async_engine_args_accepts_dcp_kwarg():
    """`AsyncEngineArgs(decode_context_parallel_size=1)` constructs without error.

    Proves the kwarg is accepted by the async path SkyRL uses
    (vllm.AsyncEngineArgs(**kwargs) in vllm_engine.py). dcp=1 is the disabled default,
    so this construction must mirror today's behavior exactly.
    """
    args = vllm.AsyncEngineArgs(decode_context_parallel_size=1)
    assert args.decode_context_parallel_size == 1
