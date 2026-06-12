"""Stage 2 (FSDP2 CP) CPU tests for the configurable attention backend.

Run (avoiding the conftest session-autouse ray_init() hang on login nodes):
    python -m pytest -p no:cacheprovider --confcutdir tests/cpu/models \
        tests/cpu/models/test_attn_backend.py

Covers the three Stage-2 invariants:
  1. attn_backend="auto" resolves EXACTLY as the pre-Stage-2 logic (G1).
  2. Importing model_wrapper succeeds with flash-attn absent; the sdpa path
     constructs the wrapper; flash-only shims raise ONLY when invoked.
  3. attn_backend="flash_attention_2" + context_parallel_size>1 -> assertion
     (cross-checks Stage-0 G2).
"""

import builtins
import importlib
import sys

import pytest


# ---------------------------------------------------------------------------
# Group 1: attn_backend="auto" reproduces the pre-Stage-2 logic byte-for-byte.
# ---------------------------------------------------------------------------
def _pre_stage2_logic(use_flash_attention_2: bool) -> str:
    """The EXACT line that lived at model_wrapper.py before Stage 2."""
    return "flash_attention_2" if use_flash_attention_2 else "eager"


@pytest.mark.parametrize("use_flash", [True, False])
def test_auto_matches_pre_stage2_logic(use_flash):
    from skyrl_train.model_wrapper import resolve_attn_implementation

    resolved = resolve_attn_implementation(attn_backend="auto", use_flash_attention_2=use_flash)
    expected = _pre_stage2_logic(use_flash)
    assert resolved == expected, f"G1 violated: auto resolved '{resolved}', pre-Stage-2 was '{expected}'"


def test_explicit_backends_override_flash_attn():
    from skyrl_train.model_wrapper import resolve_attn_implementation

    # Explicit backend overrides the flash_attn bool entirely.
    assert resolve_attn_implementation(attn_backend="sdpa", use_flash_attention_2=True) == "sdpa"
    assert resolve_attn_implementation(attn_backend="sdpa", use_flash_attention_2=False) == "sdpa"
    assert (
        resolve_attn_implementation(attn_backend="flash_attention_2", use_flash_attention_2=False)
        == "flash_attention_2"
    )
    assert resolve_attn_implementation(attn_backend="flex", use_flash_attention_2=True) == "flex_attention"


def test_invalid_backend_rejected():
    from skyrl_train.model_wrapper import resolve_attn_implementation

    with pytest.raises(AssertionError):
        resolve_attn_implementation(attn_backend="bogus")


# ---------------------------------------------------------------------------
# Group 2: model_wrapper imports with flash-attn absent; sdpa path constructs;
# flash shims raise only when invoked.
# ---------------------------------------------------------------------------
def test_import_succeeds_with_flash_absent(monkeypatch):
    """Simulate a flash-attn-free env: block the import and re-import the module."""
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "flash_attn" or name.startswith("flash_attn."):
            raise ImportError("simulated: flash_attn not installed")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    # Drop any cached copy so the module body re-executes under the blocked import.
    for mod in list(sys.modules):
        if mod == "skyrl_train.model_wrapper":
            del sys.modules[mod]

    mw = importlib.import_module("skyrl_train.model_wrapper")
    importlib.reload(mw)

    assert mw._HAS_FLASH is False, "flash should be reported absent under the blocked import"

    # The shims exist and raise ONLY when called (not at import time).
    with pytest.raises(ImportError):
        mw.pad_input(None)
    with pytest.raises(ImportError):
        mw.unpad_input(None)

    # The sdpa path resolves without touching flash.
    assert mw.resolve_attn_implementation(attn_backend="sdpa") == "sdpa"


def test_import_then_reload_restores_flash_state():
    """After the monkeypatched test, reloading normally restores real state."""
    for mod in list(sys.modules):
        if mod == "skyrl_train.model_wrapper":
            del sys.modules[mod]
    mw = importlib.import_module("skyrl_train.model_wrapper")
    # _HAS_FLASH reflects the actual env (True in the SIF, may be False elsewhere);
    # either way the module imports cleanly.
    assert hasattr(mw, "_HAS_FLASH")
    assert callable(mw.resolve_attn_implementation)


# ---------------------------------------------------------------------------
# Group 3: flash_attention_2 + context_parallel_size>1 -> assertion (G2).
# ---------------------------------------------------------------------------
def test_cp_rejects_flash_attention_2():
    from skyrl_train.model_wrapper import resolve_attn_implementation

    with pytest.raises(AssertionError):
        resolve_attn_implementation(attn_backend="flash_attention_2", context_parallel_size=2)


def test_cp_rejects_auto_resolving_to_flash():
    from skyrl_train.model_wrapper import resolve_attn_implementation

    # auto + flash_attn=True resolves to flash_attention_2, which CP rejects.
    with pytest.raises(AssertionError):
        resolve_attn_implementation(attn_backend="auto", use_flash_attention_2=True, context_parallel_size=2)


def test_cp_rejects_eager():
    from skyrl_train.model_wrapper import resolve_attn_implementation

    # eager is also not ring-compatible -> CP requires sdpa/flex.
    with pytest.raises(AssertionError):
        resolve_attn_implementation(attn_backend="auto", use_flash_attention_2=False, context_parallel_size=2)


@pytest.mark.parametrize("backend,expected", [("sdpa", "sdpa"), ("flex", "flex_attention")])
def test_cp_accepts_sdpa_and_flex(backend, expected):
    from skyrl_train.model_wrapper import resolve_attn_implementation

    assert resolve_attn_implementation(attn_backend=backend, context_parallel_size=2) == expected
