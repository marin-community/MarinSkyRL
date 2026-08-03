"""Generate the Grug MuonH oracle from Marin's pinned JAX implementation."""

from __future__ import annotations

import hashlib
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax

from experiments.grug.moe.optimizer import GrugMoeMuonHConfig


OUTPUT = Path(__file__).with_name("grug_muonh_jax_golden.npz")
STEPS = 3


def values(shape, start, stop):
    return jnp.linspace(start, stop, int(np.prod(shape)), dtype=jnp.float32).reshape(shape)


params = {
    "model": {
        "token_embed": values((5, 4), -0.7, 0.8),
        "layers": {
            "0": {
                "self_attn": {
                    "q_proj": values((4, 6), -0.6, 0.9),
                    "attn_gate": values((4, 2), -0.3, 0.5),
                },
                "mlp": {
                    "router": values((4, 3), -0.4, 0.7),
                    "experts": {"gate_proj": values((2, 4, 6), -0.8, 0.6)},
                    "shared": {"up_proj": values((4, 6), -0.5, 0.75)},
                },
                "gated_norm": {"down_proj": values((4, 4), -0.45, 0.55)},
                "input_layernorm": values((4,), 0.8, 1.2),
                "bias": values((6,), -0.2, 0.3),
            }
        },
        "output_proj": values((4, 5), -0.65, 0.85),
    }
}

paths = {
    "embed": ("model", "token_embed"),
    "q_proj": ("model", "layers", "0", "self_attn", "q_proj"),
    "attn_gate": ("model", "layers", "0", "self_attn", "attn_gate"),
    "router": ("model", "layers", "0", "mlp", "router"),
    "expert": ("model", "layers", "0", "mlp", "experts", "gate_proj"),
    "shared": ("model", "layers", "0", "mlp", "shared", "up_proj"),
    "gated_norm": ("model", "layers", "0", "gated_norm", "down_proj"),
    "norm": ("model", "layers", "0", "input_layernorm"),
    "bias": ("model", "layers", "0", "bias"),
    "output": ("model", "output_proj"),
}
routes = {
    "embed": "adam",
    "q_proj": "muonh",
    "attn_gate": "adam",
    "router": "adam",
    "expert": "muonh",
    "shared": "muonh",
    "gated_norm": "muonh",
    "norm": "adam",
    "bias": "adam",
    "output": "adamh",
}
transpose = {"q_proj", "attn_gate", "router", "expert", "shared", "gated_norm", "output"}


def get(tree, path):
    for part in path:
        tree = tree[part]
    return tree


def torch_layout(identifier, array):
    result = np.asarray(array)
    if identifier in transpose:
        result = np.swapaxes(result, -1, -2)
    return result.copy()


fixture = {
    "metadata_seed": np.asarray(20260730, dtype=np.int64),
    "metadata_steps": np.asarray(STEPS, dtype=np.int64),
    "metadata_shared_lr": np.asarray(0.03, dtype=np.float64),
    "metadata_adam_lr": np.asarray(0.004, dtype=np.float64),
    "metadata_names": np.asarray(list(paths)),
    "metadata_routes": np.asarray([routes[name] for name in paths]),
}
for identifier, path in paths.items():
    fixture[f"initial__{identifier}"] = torch_layout(identifier, get(params, path))

config = GrugMoeMuonHConfig(
    learning_rate=0.03,
    adam_lr=0.004,
    momentum=0.95,
    nesterov=True,
    backend_steps=5,
    beta1=0.9,
    beta2=0.95,
    epsilon=1e-8,
    muon_epsilon=1e-8,
    max_grad_norm=None,
    coefficient_type="quintic",
    lr_schedule="constant",
    warmup=0,
    weight_decay=123.0,
)
optimizer = config.build(STEPS)
state = optimizer.init(params)

rng = np.random.default_rng(20260730)
for step_index in range(STEPS):
    gradients = jax.tree.map(
        lambda value: jnp.asarray(
            rng.standard_normal(value.shape, dtype=np.float32) + np.float32(0.05 * (step_index + 1))
        ),
        params,
    )
    for identifier, path in paths.items():
        fixture[f"gradient_{step_index + 1}__{identifier}"] = torch_layout(identifier, get(gradients, path))

    updates, state = optimizer.update(gradients, state, params)
    params = optax.apply_updates(params, updates)

    inner_states = state.inner_state.inner_states
    muon_momentum = inner_states["muonh"].inner_state[0].momentum_buffer
    adamh_state = inner_states["adamh"].inner_state[0]
    adam_state = inner_states["adam"].inner_state[0]
    for identifier, path in paths.items():
        fixture[f"parameter_{step_index + 1}__{identifier}"] = torch_layout(identifier, get(params, path))
        if routes[identifier] == "muonh":
            fixture[f"momentum_{step_index + 1}__{identifier}"] = torch_layout(identifier, get(muon_momentum, path))
        elif routes[identifier] == "adamh":
            fixture[f"adamh_mu_{step_index + 1}__{identifier}"] = torch_layout(identifier, get(adamh_state.mu, path))
            fixture[f"adamh_nu_{step_index + 1}__{identifier}"] = torch_layout(identifier, get(adamh_state.nu, path))
        else:
            fixture[f"adam_mu_{step_index + 1}__{identifier}"] = torch_layout(identifier, get(adam_state.mu, path))
            fixture[f"adam_nu_{step_index + 1}__{identifier}"] = torch_layout(identifier, get(adam_state.nu, path))

    expert = fixture[f"parameter_{step_index + 1}__expert"]
    fixture[f"expert_norm_{step_index + 1}"] = np.linalg.norm(expert, axis=(-2, -1))

np.savez_compressed(OUTPUT, **fixture)
print(OUTPUT)
print(hashlib.sha256(OUTPUT.read_bytes()).hexdigest())
