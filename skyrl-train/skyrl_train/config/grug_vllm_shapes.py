"""The Snowball shapes that compiled vLLM's kernels, copied into ``models/grug_inductor_kernels.py``, hard-code."""

HIDDEN = 2560
HEADS = 20
KV_HEADS = 5
HEAD_DIM = 128
# The shared expert's intermediate width (its gate and up projections' outputs).
SHARED_WIDTH = 2560
# The query's two scale factors (``qk_mult`` and the long-layer factor), compiled into the q/k kernels as constants.
QUERY_FACTORS = (1.5703274004183787, 1.0)
