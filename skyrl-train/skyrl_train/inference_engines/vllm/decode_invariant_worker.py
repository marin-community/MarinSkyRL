"""Worker extension of a decode-invariant vLLM engine (``decode_invariant``).

vLLM resolves an engine's worker extension class in each worker process before it builds the worker, so importing this
module installs every decode-invariant patch before the model loads, compiles and captures CUDA graphs.
"""

from typing import Any

from skyrl_train.inference_engines.vllm import decode_invariant
from skyrl_train.inference_engines.vllm.vllm_engine import WorkerWrap


class DecodeInvariantWorkerWrap(WorkerWrap):
    """``WorkerWrap`` for an engine whose workers run the decode-invariant patches."""

    def probe_numerics_provenance(self) -> dict[str, Any]:
        """``WorkerWrap``'s provenance with the decode-invariant parts this worker installed."""
        provenance = super().probe_numerics_provenance()
        provenance["versions"]["decode_invariant"] = ",".join(decode_invariant.installed_parts())
        return provenance


decode_invariant.install()
