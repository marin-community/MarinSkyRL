"""A decode-invariant vLLM engine for Grug: each token's log-probabilities are the same floating-point values
whichever step and batch compute them.

``engine`` validates the engine settings and builds the vLLM engine kwargs. ``attention_backend`` is the CUSTOM
attention backend (FA3 with fixed splits, a per-request plan and the one-row window path), ``model`` the Grug model
class whose router logits and head gates come from the fixed-tile GEMM and whose expert kernels use one tile
configuration, ``plugin`` the vLLM general plugin that registers the backend. ``fa3`` and ``gemm`` hold the FA3
planning logic, the raw FA3 calls and the fixed-tile GEMM.
"""
