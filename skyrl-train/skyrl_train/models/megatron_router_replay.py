"""Megatron router replay (R3) — layout math and controller, megatron-free.

Everything here is CPU-testable without the ``megatron`` extra: the capture
layer mapping, the sequence-major flatten that mirrors mcore's router view
(``[S, B, E].view(-1, E)`` ⇒ token index ``s*B + b``), the TP sequence-parallel
slice, the per-layer replay controller, and the geometry validation. The
install step that touches real ``TopKRouter`` objects lives in
``skyrl_train.workers.megatron.router_replay_install``.

Controller lifecycle (per micro-batch forward through mcore's pipeline
scheduler)::

    controller.begin_forward(per_layer_targets, mask, response_mask)
    out = model(...)                      # each local router fires once
    controller.end_forward()
    ...
    controller.assert_drained()           # after every mini-batch / forward-only pass

Under activation recompute the router fires again during backward. Those calls
arrive in ``IDLE`` phase and are served from a per-layer FIFO recorded during
``FORWARD``, mirroring mcore's own ``RouterReplay`` design but keyed on the
global ``layer_number`` instead of a global instantiation-order list.
"""

from __future__ import annotations

from collections import deque
from enum import Enum
from typing import Callable, Mapping, Optional, Sequence, Tuple

import torch

from skyrl_train.models.router_replay import SENTINEL_EXPERT_ID, require_scalar_num_actions

__all__ = [
    "MegatronRouterReplay",
    "LayerReplayHandle",
    "MIN_ROUTER_TOPK",
    "capture_layer_indices",
    "expand_moe_layer_freq",
    "num_moe_layers",
    "sequence_major_flatten",
    "slice_sequence_parallel",
    "validate_replay_geometry",
]

# The all-K-sentinel capture convention is only unambiguous when native top-k
# indices are distinct, which requires top-k >= 2.
MIN_ROUTER_TOPK = 2


def capture_layer_indices(moe_layer_pattern: Sequence[int]) -> dict[int, int]:
    """Map global 1-indexed ``layer_number`` → capture index for MoE layers.

    ``moe_layer_pattern[i] == 1`` means the layer at 0-based position ``i``
    (global ``layer_number`` ``i + 1``) is a MoE layer. The capture index is
    the layer's position among MoE layers — the index into the ``L`` axis of
    the rollout's ``rollout_routed_experts`` tensor.
    """
    mapping: dict[int, int] = {}
    capture_idx = 0
    for position, is_moe in enumerate(moe_layer_pattern):
        if is_moe == 1:
            mapping[position + 1] = capture_idx
            capture_idx += 1
    return mapping


def num_moe_layers(moe_layer_pattern: Sequence[int]) -> int:
    """Count MoE entries in a 0/1 layer pattern."""
    return sum(1 for is_moe in moe_layer_pattern if is_moe == 1)


def expand_moe_layer_freq(moe_layer_freq, num_layers: int) -> list[int]:
    """Resolve mcore's ``config.moe_layer_freq`` to a 0/1 pattern list.

    Mirrors ``get_gpt_decoder_layer_specs`` in megatron-core: an integer N
    means one expert layer every N layers (``i % N == 0`` for 0-based ``i``);
    a list is taken as-is and must have one entry per layer. 0 = dense, 1 = MoE.
    """
    if isinstance(moe_layer_freq, int):
        return [1 if i % moe_layer_freq == 0 else 0 for i in range(num_layers)]
    if isinstance(moe_layer_freq, list):
        if len(moe_layer_freq) != num_layers:
            raise ValueError(
                f"Invalid length of moe_layer_freq: {len(moe_layer_freq)}, expected {num_layers}, "
                f"current moe layer pattern: {moe_layer_freq}"
            )
        return list(moe_layer_freq)
    raise ValueError(f"Invalid moe_layer_freq: {type(moe_layer_freq)}, {moe_layer_freq}")


def sequence_major_flatten(x: torch.Tensor) -> torch.Tensor:
    """Flatten ``[B, S', ...]`` to ``[S'*B, ...]`` in sequence-major order.

    Mirrors the router-side flatten mcore applies to ``[S, B, E]`` activations
    (``view(-1, E)``): flat index ``s*B + b``. Any tensor pushed through the
    same sequence transform as the model input lands, after this flatten, in
    exactly the router's token order.
    """
    if x.ndim < 2:
        raise ValueError(f"sequence_major_flatten expects [B, S', ...], got shape {tuple(x.shape)}")
    batch_size, seq_len = x.shape[0], x.shape[1]
    return x.transpose(0, 1).reshape(seq_len * batch_size, *x.shape[2:])


def slice_sequence_parallel(
    x: torch.Tensor, *, seq_len: int, batch_size: int, tp_rank: int, tp_size: int
) -> torch.Tensor:
    """Slice a sequence-major flat tensor to one TP rank's contiguous chunk.

    Under TP sequence parallelism each rank's router sees its contiguous
    ``seq_len // tp_size`` rows of the sequence dim, i.e. flat rows
    ``[tp_rank * local * batch_size, (tp_rank + 1) * local * batch_size)``.
    The padding helpers upstream guarantee ``seq_len % tp_size == 0``.
    """
    if seq_len % tp_size != 0:
        raise ValueError(
            f"router replay: seq_len {seq_len} not divisible by tp_size {tp_size}; "
            "the Megatron padding helpers must run before the slice"
        )
    local = seq_len // tp_size
    start = tp_rank * local * batch_size
    return x[start : start + local * batch_size]


class _Phase(Enum):
    IDLE = "idle"
    FORWARD = "forward"


class MegatronRouterReplay:
    """Per-layer replay controller for megatron-core ``TopKRouter.router_replay``.

    Duck-types the single method mcore calls (``get_replay_topk``, through
    :class:`LayerReplayHandle`). One instance per rank, installed on every
    local ``TopKRouter``; ``local_layer_indices`` are the capture indices this
    rank's model chunks own.
    """

    def __init__(self, local_layer_indices: Sequence[int], *, recompute_enabled: bool) -> None:
        self.local_layer_indices: tuple[int, ...] = tuple(sorted(local_layer_indices))
        # Filled by the installer from the resolved model config; the target
        # builder validates every batch against them before arming.
        self.num_moe_layers_total: Optional[int] = None
        self.topk: Optional[int] = None
        # id(model chunk) -> capture indices owned by that chunk. Empty for a
        # single chunk; with virtual pipelining each forward bracket arms only
        # the layers of the chunk mcore is about to call.
        self.local_indices_for_module: dict[int, tuple[int, ...]] = {}
        self._recompute_enabled = recompute_enabled
        self._phase = _Phase.IDLE
        # layer capture idx -> (targets [N, K], replay mask [N]) for the current forward
        self._current: dict[int, Tuple[torch.Tensor, torch.Tensor]] = {}
        self._expected: tuple[int, ...] = ()
        self._consumed: set[int] = set()
        self._response_mask: Optional[torch.Tensor] = None
        self._record_recompute = False
        self._fifo: dict[int, deque] = {idx: deque() for idx in self.local_layer_indices}
        self._masked_rows = 0
        self._hit_rows = 0
        self._response_rows = 0
        self._sentinel_rows = 0
        self._native_mismatch_rows = 0
        self._native_set_mismatch_rows = 0
        self._executed_rows = 0
        self._forward_masked_rows = 0
        self._valid_mask: Optional[torch.Tensor] = None
        self._loss_free_loads: Optional[dict[int, torch.Tensor]] = None

    def begin_loss_free_bias_window(self) -> None:
        """Count actual forward routes once, excluding checkpoint recomputes."""
        if self._loss_free_loads is not None:
            raise RuntimeError("router replay: loss-free bias window is already open")
        self._loss_free_loads = {}

    def take_loss_free_bias_loads(self) -> dict[int, torch.Tensor]:
        """Consume local per-layer assignment counts after an optimizer window."""
        self.assert_drained()
        if self._loss_free_loads is None:
            raise RuntimeError("router replay: no loss-free bias window is open")
        loads = self._loss_free_loads
        self._loss_free_loads = None
        missing = set(self.local_layer_indices) - set(loads)
        if missing:
            raise RuntimeError(f"router replay: loss-free bias missing layers {sorted(missing)}")
        return loads

    # ------------------------------------------------------------- drivers

    def begin_forward(
        self,
        per_layer_targets: Mapping[int, torch.Tensor],
        mask: torch.Tensor,
        response_mask: Optional[torch.Tensor] = None,
        *,
        record_recompute: bool = True,
        valid_mask: Optional[torch.Tensor] = None,
    ) -> None:
        """Arm the controller for one forward over the model's local layers.

        ``per_layer_targets`` maps capture index → ``[N, K]`` long targets in
        the router's token order; ``mask`` is the ``[N]`` replay gate.
        ``response_mask`` (optional, ``[N]``) marks response-window rows before
        sentinel exclusion and feeds the ``sentinel_fraction`` metric.
        ``record_recompute`` is true for training forwards whose backward will
        replay activation-checkpointed layers, even when forward runs under no_grad.
        """
        if self._phase is not _Phase.IDLE:
            raise RuntimeError("router replay: begin_forward while a forward is already armed (phase must be IDLE)")
        self._current = {idx: (targets, mask) for idx, targets in per_layer_targets.items()}
        self._expected = tuple(sorted(per_layer_targets))
        self._consumed = set()
        self._response_mask = response_mask
        self._valid_mask = valid_mask
        self._record_recompute = record_recompute
        self._phase = _Phase.FORWARD

    def end_forward(self) -> None:
        """Close the forward bracket and verify every armed layer fired once."""
        if self._phase is not _Phase.FORWARD:
            raise RuntimeError("router replay: end_forward without an armed forward")
        missing = [idx for idx in self._expected if idx not in self._consumed]
        if missing:
            raise ValueError(
                f"router replay: armed layer(s) {missing} never fired during the forward — "
                "the capture layer mapping does not match the model"
            )
        self._current = {}
        self._expected = ()
        self._response_mask = None
        self._valid_mask = None
        self._record_recompute = False
        self._phase = _Phase.IDLE

    def abort_forward(self) -> None:
        """Force-reset after a mid-forward failure; never raises.

        Clears the armed targets and any partial recompute FIFO entries so the
        controller is usable again; the caller re-raises the original error.
        """
        self._current = {}
        self._expected = ()
        self._consumed = set()
        self._response_mask = None
        self._valid_mask = None
        self._record_recompute = False
        self._phase = _Phase.IDLE
        for fifo in self._fifo.values():
            fifo.clear()

    def assert_drained(self) -> None:
        """Assert the recompute FIFO is empty and no forward is armed.

        Called at the end of every mini-batch and every forward-only pass; a
        violation means router calls and forward brackets went out of sync.
        """
        if self._phase is not _Phase.IDLE:
            raise RuntimeError("router replay: assert_drained while a forward is still armed")
        outstanding = ", ".join(f"layer {idx} has {len(fifo)} outstanding" for idx, fifo in self._fifo.items() if fifo)
        if outstanding:
            raise RuntimeError(f"router replay: recompute FIFO not drained: {outstanding}")

    def pop_metrics(self) -> dict[str, float]:
        """Return ``hit_fraction`` / ``sentinel_fraction`` since the last pop.

        ``hit_fraction`` is replayed rows over masked rows (1.0 unless a masked
        row carried an all-sentinel target — a layout bug). ``sentinel_fraction``
        is sentinel response rows over response rows (rollout capture loss).
        """
        hit_fraction = self._hit_rows / self._masked_rows if self._masked_rows else 1.0
        sentinel_fraction = self._sentinel_rows / self._response_rows if self._response_rows else 0.0
        native_mismatch_fraction = self._native_mismatch_rows / self._masked_rows if self._masked_rows else 0.0
        native_set_mismatch_fraction = self._native_set_mismatch_rows / self._masked_rows if self._masked_rows else 0.0
        executed_route_match_fraction = (
            self._executed_rows / self._forward_masked_rows if self._forward_masked_rows else 1.0
        )
        self._masked_rows = 0
        self._hit_rows = 0
        self._response_rows = 0
        self._sentinel_rows = 0
        self._native_mismatch_rows = 0
        self._native_set_mismatch_rows = 0
        self._executed_rows = 0
        self._forward_masked_rows = 0
        return {
            "hit_fraction": hit_fraction,
            "sentinel_fraction": sentinel_fraction,
            "native_mismatch_fraction": native_mismatch_fraction,
            "native_set_mismatch_fraction": native_set_mismatch_fraction,
            "executed_route_match_fraction": executed_route_match_fraction,
        }

    def observe_executed_routing_map(self, layer_idx: int, selected: torch.Tensor, routing_map: torch.Tensor) -> None:
        """Check the actual Grug dispatch map against the captured expert set."""
        expected_map = torch.zeros_like(routing_map).scatter(1, selected, True)
        if not torch.equal(routing_map, expected_map):
            raise RuntimeError(f"router replay: layer {layer_idx} dispatch map differs from selected experts")
        if self._phase is not _Phase.FORWARD:
            return
        targets, mask = self._current[layer_idx]
        mask = mask.to(device=routing_map.device, dtype=torch.bool)
        target_map = torch.zeros_like(routing_map).scatter(1, targets.to(routing_map.device).clamp_min(0), True)
        if not torch.equal(routing_map[mask], target_map[mask]):
            raise RuntimeError(f"router replay: layer {layer_idx} did not execute captured expert set")
        self._executed_rows += mask.sum().item()

    # ------------------------------------------------------ router-side entry

    def get_replay_topk(
        self,
        layer_idx: int,
        scores: torch.Tensor,
        topk: int,
        num_groups: Optional[int] = None,
        group_topk: Optional[int] = None,
        default_compute_topk: Optional[
            Callable[[torch.Tensor, int, Optional[int], Optional[int]], Tuple[torch.Tensor, torch.Tensor]]
        ] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return ``(probs, top_indices)`` with rollout choices on masked rows.

        Masked rows return the captured target experts; every other row returns
        the native top-k. ``probs`` are always the live routing scores gathered
        at the returned indices, so gradients flow through the gate on both
        replayed and native rows. Runs mcore's ``default_compute_topk``
        unconditionally to keep the autograd graph identical to a flag-off
        forward.
        """
        probs, native_idx = default_compute_topk(scores, topk, num_groups=num_groups, group_topk=group_topk)
        targets, mask = self._targets_for_call(layer_idx, scores)

        if targets.shape[0] != scores.shape[0]:
            raise ValueError(
                f"router replay: layer {layer_idx} targets have {targets.shape[0]} rows "
                f"but the router scored {scores.shape[0]} tokens — the route tensor and the "
                "model input took different layout transforms"
            )
        if targets.shape[1] != topk:
            raise ValueError(
                f"router replay: layer {layer_idx} targets carry K={targets.shape[1]} but the router uses topk={topk}"
            )

        mask = mask.to(device=scores.device, dtype=torch.bool)
        targets = targets.to(device=scores.device)
        idx = torch.where(mask.unsqueeze(-1), targets, native_idx)
        probs = scores.gather(1, idx)

        replayed = mask.sum().item()
        self._native_mismatch_rows += (mask & (targets != native_idx).any(dim=-1)).sum().item()
        self._native_set_mismatch_rows += (
            (mask & (targets.sort(dim=-1).values != native_idx.sort(dim=-1).values).any(dim=-1)).sum().item()
        )
        if self._phase is _Phase.FORWARD:
            self._forward_masked_rows += replayed
            if self._loss_free_loads is not None:
                if self._valid_mask is None or self._valid_mask.shape != mask.shape:
                    raise RuntimeError("router replay: loss-free bias requires an aligned valid-token mask")
                valid = self._valid_mask.to(device=idx.device, dtype=torch.bool)
                loads = torch.bincount(idx[valid].reshape(-1), minlength=scores.shape[1]).float()
                previous = self._loss_free_loads.get(layer_idx)
                self._loss_free_loads[layer_idx] = loads if previous is None else previous + loads
        # A masked row whose target is all-sentinel means the mask and the
        # target tensor disagree (layout bug): count it so hit_fraction < 1.0
        # surfaces it as a hard error at mini-batch end.
        lost = (mask & (targets == SENTINEL_EXPERT_ID).all(dim=-1)).sum().item()
        self._masked_rows += replayed
        self._hit_rows += replayed - lost
        if self._response_mask is not None:
            response = self._response_mask.to(device=scores.device)
            self._response_rows += response.sum().item()
            self._sentinel_rows += (response & ~mask).sum().item()
        return probs, idx

    def _targets_for_call(self, layer_idx: int, scores: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if self._phase is _Phase.FORWARD:
            if layer_idx not in self._current:
                raise ValueError(
                    f"router replay: layer {layer_idx} fired with no targets armed for this forward "
                    f"(armed: {list(self._current)}); the capture layer mapping does not match the model"
                )
            if layer_idx in self._consumed:
                raise ValueError(f"router replay: layer {layer_idx} consumed twice in one forward")
            self._consumed.add(layer_idx)
            targets, mask = self._current[layer_idx]
            # Activation-checkpointed training forwards run under no_grad;
            # the caller, not grad mode, identifies whether backward follows.
            if self._recompute_enabled and self._record_recompute:
                if layer_idx not in self._fifo:
                    raise ValueError(f"router replay: layer {layer_idx} has no FIFO; not a local layer")
                self._fifo[layer_idx].append((targets, mask))
            return targets, mask
        if not self._recompute_enabled or layer_idx not in self._fifo or not self._fifo[layer_idx]:
            raise RuntimeError(
                f"router replay: recompute without a recorded forward (layer {layer_idx}, "
                f"recompute_enabled={self._recompute_enabled})"
            )
        return self._fifo[layer_idx].popleft()


class LayerReplayHandle:
    """Per-router duck-type of mcore's ``RouterReplay`` bound to one capture index.

    Assigned to ``TopKRouter.router_replay``; mcore calls
    ``get_replay_topk(scores, topk, num_groups, group_topk, _compute_topk)``
    positionally.
    """

    def __init__(self, controller: MegatronRouterReplay, layer_idx: int) -> None:
        self._controller = controller
        self.layer_idx = layer_idx

    def get_replay_topk(
        self,
        scores: torch.Tensor,
        topk: int,
        num_groups: Optional[int] = None,
        group_topk: Optional[int] = None,
        default_compute_topk: Optional[Callable] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return self._controller.get_replay_topk(
            self.layer_idx, scores, topk, num_groups, group_topk, default_compute_topk
        )

    def observe_executed_routing_map(self, selected: torch.Tensor, routing_map: torch.Tensor) -> None:
        self._controller.observe_executed_routing_map(self.layer_idx, selected, routing_map)


def validate_replay_geometry(
    *,
    num_layers_captured: int,
    expected_moe_layers: int,
    topk_captured: int,
    expected_topk: int,
    num_experts: int,
    targets: torch.Tensor,
    response_len: int,
    num_actions,
) -> None:
    """Validate a ``rollout_routed_experts`` tensor against the model geometry.

    Raises ``ValueError`` naming the offending quantity on any mismatch;
    raises ``NotImplementedError`` for a per-sample ``num_actions`` list
    (mirrors the FSDP path).
    """
    require_scalar_num_actions(num_actions)
    if num_layers_captured != expected_moe_layers:
        raise ValueError(
            f"router replay: rollout_routed_experts carries L={num_layers_captured} layers but the model has "
            f"expected_moe_layers={expected_moe_layers} MoE layers"
        )
    if expected_topk < MIN_ROUTER_TOPK:
        raise ValueError(
            f"router replay: expected_topk={expected_topk} < {MIN_ROUTER_TOPK}; the all-K sentinel convention is "
            "ambiguous when native top-k indices need not be distinct"
        )
    if topk_captured != expected_topk:
        raise ValueError(
            f"router replay: rollout_routed_experts carries K={topk_captured} but the model uses "
            f"expected_topk={expected_topk}"
        )
    if targets.numel() and (targets.min().item() < 0 or targets.max().item() >= num_experts):
        raise ValueError(
            f"router replay: rollout_routed_experts carries an expert id outside [0, num_experts={num_experts})"
        )
    if response_len != num_actions:
        raise ValueError(f"router replay: response_len={response_len} != num_actions={num_actions}")
