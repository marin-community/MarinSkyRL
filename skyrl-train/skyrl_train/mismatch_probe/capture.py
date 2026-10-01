"""Capture the trainer's per-region Megatron activations for the one-GPU parity harness.

The capture runs inside a native-routing probe forward. On data-parallel rank 0 it records, for
the first micro-batch and each configured global layer that this pipeline stage owns, the tensors
at every region boundary of the Grug decoder layer, in the router's sequence-major token order:

- ``input``: residual stream entering the layer;
- ``attention_norm``: gated-RMSNorm output fed to attention;
- ``attention``: attention output after XSA, the head gate and the output projection;
- ``residual_after_attention``: residual stream entering the MLP norm;
- ``mlp_norm``: gated-RMSNorm output fed to the router, experts and shared expert;
- ``router_probs`` and ``router_map``: combine weights and selected experts;
- ``shared_expert``: shared-expert output;
- ``mlp``: MoE block output (routed plus shared);
- ``output``: residual stream leaving the layer;
- ``attention_query``, ``attention_key``, ``attention_value`` and ``attention_core``: the core
  attention's inputs and output (FA3's, under ``fa3_attention`` or ``vllm_steps``), which a vLLM
  re-read's engine capture holds too.

The last stage also records ``lm_head``: the output layer's input rows and each row's maximum and
log-sum-exp of the logits. The file also holds the micro-batch token IDs and attention mask, so the
harness can rebuild positions after left-padding removal.
"""

from __future__ import annotations

import io
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from typing import Any

import torch

from skyrl_train.io.io import write_bytes_atomic


def _first_tensor(value: Any) -> torch.Tensor:
    return value[0] if isinstance(value, (tuple, list)) else value


def _gpt_models(actor_module):
    for chunk in actor_module:
        # Unwrap DDP and Float16Module to the GPT model that owns the decoder.
        while hasattr(chunk, "module"):
            chunk = chunk.module
        yield chunk


def _grug_layers(actor_module, layer_numbers: set[int]):
    for model in _gpt_models(actor_module):
        for layer in model.decoder.layers:
            if layer.layer_number in layer_numbers:
                yield layer


@contextmanager
def capture_layer_regions(
    actor_module: Sequence[torch.nn.Module],
    layers: Sequence[int],
    destination: str,
    *,
    enabled: bool,
) -> Iterator[dict[int | str, dict[str, torch.Tensor]]]:
    """Record region tensors of the first forward through each listed 0-based layer, then write them."""
    captured: dict[int | str, dict[str, torch.Tensor]] = {}
    if not enabled or not layers:
        yield captured
        return
    handles = []

    def store(layer_index: int | str, name: str, tensor: torch.Tensor) -> None:
        regions = captured.setdefault(layer_index, {})
        if name not in regions:
            regions[name] = tensor.detach().to("cpu", copy=True)

    for layer in _grug_layers(actor_module, {index + 1 for index in layers}):
        index = layer.layer_number - 1

        def pre_layer(module, args, kwargs, index=index):
            store(index, "input", _first_tensor(args) if args else kwargs["hidden_states"])

        handles.append(layer.register_forward_pre_hook(pre_layer, with_kwargs=True))
        handles.append(
            layer.register_forward_hook(lambda m, a, out, index=index: store(index, "output", _first_tensor(out)))
        )
        handles.append(
            layer.input_layernorm.register_forward_hook(
                lambda m, a, out, index=index: store(index, "attention_norm", _first_tensor(out))
            )
        )
        handles.append(
            layer.self_attention.register_forward_hook(
                lambda m, a, out, index=index: store(index, "attention", _first_tensor(out))
            )
        )

        def core_attention(module, args, kwargs, out, index=index):
            for name, tensor in zip(("attention_query", "attention_key", "attention_value"), args[:3], strict=True):
                store(index, name, tensor)
            store(index, "attention_core", _first_tensor(out))

        handles.append(layer.self_attention.core_attention.register_forward_hook(core_attention, with_kwargs=True))
        handles.append(
            layer.pre_mlp_layernorm.register_forward_pre_hook(
                lambda m, args, index=index: store(index, "residual_after_attention", _first_tensor(args))
            )
        )
        handles.append(
            layer.pre_mlp_layernorm.register_forward_hook(
                lambda m, a, out, index=index: store(index, "mlp_norm", _first_tensor(out))
            )
        )
        handles.append(
            layer.mlp.register_forward_hook(lambda m, a, out, index=index: store(index, "mlp", _first_tensor(out)))
        )
        router = getattr(layer.mlp, "router", None)
        if router is not None:

            def router_hook(module, args, out, index=index):
                probs, routing_map = out
                store(index, "router_probs", probs)
                store(index, "router_map", routing_map)

            handles.append(router.register_forward_hook(router_hook))
        shared = getattr(layer.mlp, "shared_experts", None)
        if shared is not None:
            handles.append(
                shared.register_forward_hook(
                    lambda m, a, out, index=index: store(index, "shared_expert", _first_tensor(out))
                )
            )
    for model in _gpt_models(actor_module):
        if not model.post_process:
            continue

        def lm_head(module, args, out):
            logits = _first_tensor(out).detach().float()
            store("lm_head", "hidden", _first_tensor(args))
            store("lm_head", "max", logits.max(dim=-1).values)
            store("lm_head", "logsumexp", logits.logsumexp(dim=-1))

        handles.append(model.output_layer.register_forward_hook(lm_head))
    try:
        yield captured
    finally:
        for handle in handles:
            handle.remove()


def write_capture(
    destination: str, captured: Mapping[int | str, Mapping[str, torch.Tensor]], batch: Mapping[str, Any]
) -> None:
    """Write one pipeline stage's captured regions with the micro-batch token layout."""
    payload = io.BytesIO()
    torch.save({"layers": dict(captured), **batch}, payload)
    write_bytes_atomic(destination, payload.getvalue())
