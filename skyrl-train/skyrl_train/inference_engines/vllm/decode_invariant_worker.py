"""Worker extension of a decode-invariant vLLM engine (``decode_invariant``).

vLLM resolves an engine's worker extension class in each worker process before it builds the worker, so importing this
module installs every decode-invariant patch before the model loads, compiles and captures CUDA graphs.
"""

from skyrl_train.inference_engines.vllm import decode_invariant
from skyrl_train.inference_engines.vllm.vllm_engine import WorkerWrap

# The expert-block receiver method that installs a weight sync's weights.
EXPERT_BLOCK_INSTALL = "receive_weights"


class DecodeInvariantWorkerWrap(WorkerWrap):
    """``WorkerWrap`` that checks the router weights hold bf16 values after every broadcast or expert-block sync."""

    def skyrl_finish_weight_reload(self) -> None:
        super().skyrl_finish_weight_reload()
        decode_invariant.check_router_weights(self.model_runner.model)

    def expert_block_rpc(self, method: str, *args):
        result = super().expert_block_rpc(method, *args)
        if method == EXPERT_BLOCK_INSTALL:
            decode_invariant.check_router_weights(self.model_runner.model)
        return result


decode_invariant.install()
