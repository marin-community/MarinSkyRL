"""Stage-1 environment gate for the FSDP2 torch-native Context-Parallel (CP) port.

Stage 1 pins ``torch>=2.10`` for the trainer + actively-used vLLM path so the
torch-native CP API (``torch.distributed.tensor.experimental.context_parallel``,
the ring-SDPA load balancer, and the private offset/unshard helpers that
Stages 4/5 import) is guaranteed present. This module is the gate: it FAILS on
any torch < 2.10 (e.g. the legacy torch-2.9 RL venv) and FAILS LOUDLY if any of
the exact CP symbols later stages depend on has moved/disappeared.

Asserts (per stage1_torch_pin_scope.md "Invariants tests must assert"):
  1. ``torch.__version__ >= 2.10`` — compared with ``packaging.version`` parsing,
     NOT a string compare (the repo has a known lexicographic version-compare
     bug, see commit 829ae2f; "2.9" > "2.10" lexicographically would silently
     pass a string compare).
  2. ``from torch.distributed.tensor.experimental import context_parallel`` ok.
  3. ``from torch.distributed.tensor.experimental._attention import
     set_rotate_method, context_parallel_unshard`` ok — these are the EXACT
     symbols Stages 4/5 import; a private-path move in 2.11+ is a CRITICAL find.
  4. ``torch.nn.functional.scaled_dot_product_attention`` present (ring-SDPA
     backend dependency).

Run INSIDE the torch-2.11 SIF (NOT the torch-2.9 RL venv, which will correctly
fail assertion 1 — that failure is the whole point of Stage 1):

    apptainer exec --nv <skyrl_megatron_vllm0202rc0_r3.sif> \
        python -m pytest tests/cpu/distributed/test_torch_cp_available.py -v

See notes/RL/skyrl/fsdp2_context_parallel_stages/{README,stage1_torch_pin_scope}.md.
"""

import torch
from packaging.version import Version


def _torch_base_version() -> Version:
    """Parse torch's version using only its release tuple (strip +cu130 local tag)."""
    # torch.__version__ looks like "2.11.0+cu130"; Version() parses the local
    # segment fine, but we compare on the public release to keep intent obvious.
    return Version(torch.__version__.split("+", 1)[0])


def test_torch_version_at_least_2_10():
    """torch >= 2.10 (semantic compare; lexicographic string compare would lie)."""
    ver = _torch_base_version()
    assert ver >= Version("2.10"), (
        f"torch.__version__={torch.__version__!r} (parsed {ver}) is < 2.10; "
        "Stage-1 requires torch>=2.10 for torch-native context_parallel. "
        "If this fires in the RL venv, run inside the torch-2.11 SIF instead."
    )


def test_context_parallel_importable():
    """The public CP entrypoint Stage 4 wraps the forward with."""
    from torch.distributed.tensor.experimental import context_parallel  # noqa: F401


def test_cp_private_attention_symbols_importable():
    """The EXACT private helpers Stages 4/5 import — fail loudly if the path moved."""
    from torch.distributed.tensor.experimental._attention import (  # noqa: F401
        context_parallel_unshard,
        set_rotate_method,
    )


def test_scaled_dot_product_attention_present():
    """Ring-SDPA backend dependency (CP uses SDPA, not flash-attn varlen)."""
    assert hasattr(torch.nn.functional, "scaled_dot_product_attention")
