"""Architecture-neutral observations of MoE router forwards.

Observers run synchronously inside the router forward. A callback that retains
an observation beyond the callback must detach or clone its tensors to avoid
retaining the autograd graph.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from weakref import WeakSet

import torch
import torch.nn.functional as F
from torch import nn


@dataclass(frozen=True)
class RouterObservation:
    """One router decision, expressed independently of the model architecture."""

    router: nn.Module
    router_inputs: torch.Tensor
    selection_logits: torch.Tensor
    selection_log_probs: torch.Tensor
    natural_selected_experts: torch.Tensor
    selected_experts: torch.Tensor
    combine_weights: torch.Tensor


RouterObserver = Callable[[RouterObservation], None]
_active_observer: ContextVar[RouterObserver | None] = ContextVar("router_observer", default=None)
_instrumented_qwen_routers: WeakSet[nn.Module] = WeakSet()


@contextmanager
def observe_router_forwards(observer: RouterObserver) -> Iterator[None]:
    """Send router observations to ``observer`` for the current execution context."""

    token = _active_observer.set(observer)
    try:
        yield
    finally:
        _active_observer.reset(token)


def emit_router_forward(
    *,
    router: nn.Module,
    router_inputs: torch.Tensor,
    selection_logits: torch.Tensor,
    natural_selected_experts: torch.Tensor,
    selected_experts: torch.Tensor,
    combine_weights: torch.Tensor,
) -> None:
    """Emit an observation when the current execution context has an observer."""

    observer = _active_observer.get()
    if observer is None:
        return
    observer(
        RouterObservation(
            router=router,
            router_inputs=router_inputs,
            selection_logits=selection_logits,
            selection_log_probs=F.log_softmax(selection_logits.float(), dim=-1),
            natural_selected_experts=natural_selected_experts,
            selected_experts=selected_experts,
            combine_weights=combine_weights,
        )
    )


def _observe_qwen_router(
    router: nn.Module,
    inputs: tuple[torch.Tensor, ...],
    output: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
) -> None:
    if _active_observer.get() is None:
        return
    if len(inputs) != 1 or not isinstance(output, tuple) or len(output) != 3:
        raise RuntimeError("instrumented Qwen router returned an unsupported interface")
    router_logits, combine_weights, selected_experts = output
    router_inputs = inputs[0].reshape(-1, inputs[0].shape[-1])
    emit_router_forward(
        router=router,
        router_inputs=router_inputs,
        selection_logits=router_logits,
        natural_selected_experts=selected_experts,
        selected_experts=selected_experts,
        combine_weights=combine_weights,
    )


def instrument_moe_routers(model: nn.Module) -> int:
    """Enable observations for every supported MoE router under ``model``.

    Native MarinSkyRL routers emit observations directly. Hugging Face Qwen
    routers need an idempotent forward hook. The return value is the number of
    supported routers found, including routers enabled by an earlier call.
    """

    router_count = 0
    for module in model.modules():
        if getattr(module, "_emits_router_observations", False):
            router_count += 1
            continue
        if not (type(module).__name__.startswith("Qwen") and type(module).__name__.endswith("TopKRouter")):
            continue
        router_count += 1
        if module not in _instrumented_qwen_routers:
            module.register_forward_hook(_observe_qwen_router)
            _instrumented_qwen_routers.add(module)
    return router_count
