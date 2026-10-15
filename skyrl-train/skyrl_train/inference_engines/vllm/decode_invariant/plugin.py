"""The vLLM general plugin (``vllm.general_plugins`` entry point) that points vLLM's CUSTOM attention backend at
``attention_backend.DecodeInvariantAttentionBackend``.

vLLM loads general plugins in its API, engine-core and worker processes. The registration names the class path, so a
process imports the backend only when its engine selects ``attention_backend=CUSTOM``.
"""

from vllm.v1.attention.backends.registry import AttentionBackendEnum, register_backend

DECODE_INVARIANT_ATTENTION_BACKEND_CLASS = (
    "skyrl_train.inference_engines.vllm.decode_invariant.attention_backend.DecodeInvariantAttentionBackend"
)


def register_attention_backend() -> None:
    register_backend(AttentionBackendEnum.CUSTOM, DECODE_INVARIANT_ATTENTION_BACKEND_CLASS)
