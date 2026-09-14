"""Exercise Snowball's H100 FlashAttention forward and backward at the real shape."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from levanter.grug.attention import AttentionMask, attention


def main() -> None:
    if jax.default_backend() != "gpu":
        raise RuntimeError(f"expected GPU backend, got {jax.default_backend()}")

    batch, sequence, query_heads, kv_heads, head_dim = 1, 8192, 20, 5, 128
    q_key, k_key, v_key = jax.random.split(jax.random.PRNGKey(0), 3)
    q = jax.random.normal(q_key, (batch, sequence, query_heads, head_dim), dtype=jnp.bfloat16)
    k = jax.random.normal(k_key, (batch, sequence, kv_heads, head_dim), dtype=jnp.bfloat16)
    v = jax.random.normal(v_key, (batch, sequence, kv_heads, head_dim), dtype=jnp.bfloat16)

    for sliding_window in (2048, None):
        mask = AttentionMask.causal(sliding_window=sliding_window)

        @jax.jit
        @jax.value_and_grad
        def objective(q, k, v):
            out = attention(q, k, v, mask, implementation="gpu_fa4_cute")
            return jnp.mean(jnp.square(out.astype(jnp.float32)))

        value, gradients = objective(q, k, v)
        jax.block_until_ready((value, gradients))
        if not bool(jnp.isfinite(value)):
            raise RuntimeError(f"non-finite FA4 loss for sliding_window={sliding_window}")
        if not all(bool(jnp.all(jnp.isfinite(gradient))) for gradient in gradients):
            raise RuntimeError(f"non-finite FA4 gradient for sliding_window={sliding_window}")
        print(
            "FA4 Snowball attention passed:",
            f"sliding_window={sliding_window}",
            f"loss={float(value):.8f}",
            f"gradient_shapes={[gradient.shape for gradient in gradients]}",
            flush=True,
        )


if __name__ == "__main__":
    main()
