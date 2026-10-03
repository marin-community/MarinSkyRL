"""The Snowball shapes that compiled vLLM's kernels, copied into ``models/grug_inductor_kernels.py``, hard-code."""

HIDDEN = 2560
HEADS = 20
KV_HEADS = 5
HEAD_DIM = 128
# The shared expert's intermediate width (its gate and up projections' outputs).
SHARED_WIDTH = 2560
# The query's two scale factors (``qk_mult`` and the long-layer factor), compiled into the q/k kernels as constants.
QUERY_FACTORS = (1.5703274004183787, 1.0)
# The layer, final and embedding norms' epsilon (``rms_norm_eps``), compiled into the norm kernels.
RMS_NORM_EPS = 1e-5
# Rows of vLLM's rotary table (``max_position_embeddings``): the RoPE kernel reads positions below this.
ROTARY_POSITIONS = 65536
# Snowball's vocabulary: the embedding kernel reads token ids below this.
VOCAB = 128256
