"""Hybrid Muon + AdamW optimizer for FSDP2 policy training.

Muon (Momentum Orthogonalized by Newton-Schulz) is only correct for the 2-D
weight matrices of a transformer's hidden layers. The embedding table, the final
LM head, every LayerNorm/RMSNorm gain, all biases, and any 0-/1-D scalar must be
trained by a standard adaptive method (AdamW here). This module wires the
*standard hybrid recipe*: Muon on the 2-D matmul weights, AdamW on everything
else, exposed as a single ``torch.optim.Optimizer``-compatible object so the rest
of SkyRL (scheduler, grad-clip, CPU offload, checkpoint save/load, StaleClip
lr-scaling) keeps working unchanged.

Implementation: a thin composite over two real child optimizers —
``torch.optim.Muon`` (2-D hidden weights) and ``torch.optim.AdamW`` (everything
else). The composite proxies ``param_groups`` and ``state`` to the children so
that code which iterates ``optimizer.param_groups`` (scheduler lr, StaleClip lr
scaling) and ``optimizer.state[param]`` (CPU offload/backload) operates on the
union transparently. We deliberately reuse the shipped optimizers rather than
re-implement their kernels.

FSDP2 note: under FSDP2 the 2-D weights arrive as row-sharded ``DTensor``s. The
Newton-Schulz iteration (``G @ G.T`` plus a global ``.norm()``) runs correctly on
those DTensors because DTensor sharding-propagation inserts the right collectives
(all-gather for the matmul, all-reduce for the norm) — verified to match the
full-tensor NS result within bf16 tolerance.
"""

from collections.abc import Mapping
from itertools import chain
from typing import Any, Iterable, List, Optional

import torch
from torch import Tensor
from torch.optim import AdamW, Muon
from torch.optim.optimizer import Optimizer


class _MergedState(Mapping):
    """Read/write view over the children optimizers' per-param state dicts.

    Each parameter lives in exactly one child, so lookups/assignments route to
    the owning child. ``torch.optim.lr_scheduler`` never touches ``.state``; the
    only consumers are SkyRL's CPU offload/backload utils which do
    ``optimizer.state[param][key] = tensor.to(...)`` — supported here because the
    returned per-param dict is the child's live dict (mutated in place).
    """

    def __init__(self, children):
        self._children = children

    def _owner(self, key):
        for c in self._children:
            if key in c.state:
                return c
        return None

    def __getitem__(self, key):
        c = self._owner(key)
        if c is None:
            # Mirror defaultdict(dict) semantics torch optimizers rely on.
            for child in self._children:
                if any(key is p for g in child.param_groups for p in g["params"]):
                    return child.state[key]
            raise KeyError(key)
        return c.state[key]

    def __iter__(self):
        for c in self._children:
            yield from c.state
        return

    def __len__(self):
        return sum(len(c.state) for c in self._children)

    def __bool__(self):
        return any(len(c.state) for c in self._children)


def is_muon_param(name: str, param: Tensor) -> bool:
    """Return True iff ``param`` should be optimized by Muon.

    The standard recipe: Muon only on 2-D hidden-layer weight matrices. Anything
    that is not exactly 2-D (biases, norms, scalars, 1-D vectors) goes to AdamW.
    The embedding table and the final LM head are 2-D but must still use AdamW
    (per the Muon paper), so they are excluded by name.
    """
    if param.ndim != 2:
        return False
    lname = name.lower()
    adamw_name_markers = (
        "embed",   # embed_tokens, word_embeddings, wte, position_embeddings
        "lm_head",  # output projection
        "wte",
        "wpe",
        "shared",  # tied embedding
    )
    if any(marker in lname for marker in adamw_name_markers):
        return False
    return True


class HybridMuon(Optimizer):
    """Composite optimizer: Muon on 2-D hidden weights, AdamW on the rest.

    Subclasses ``torch.optim.Optimizer`` (required: ``LRScheduler`` does an
    ``isinstance(optimizer, Optimizer)`` check) but delegates ``step``,
    ``param_groups`` and ``state`` to two child optimizers (``torch.optim.Muon``
    + ``torch.optim.AdamW``) so we reuse the shipped kernels rather than
    re-implementing them. The call-sites SkyRL uses are: ``step``, ``zero_grad``,
    ``param_groups`` (read + per-group ``lr`` mutation by scheduler/StaleClip),
    ``state`` (per-param tensor offload), ``state_dict``/``load_state_dict``.
    """

    def __init__(
        self,
        muon_params: Iterable[Tensor],
        adamw_params: Iterable[Tensor],
        *,
        muon_lr: float = 0.02,
        muon_weight_decay: float = 0.0,
        muon_momentum: float = 0.95,
        muon_nesterov: bool = True,
        ns_steps: int = 5,
        muon_eps: float = 1e-7,
        adjust_lr_fn: Optional[str] = None,
        adamw_lr: float = 8e-6,
        adamw_betas: tuple = (0.9, 0.999),
        adamw_eps: float = 1e-8,
        adamw_weight_decay: float = 0.0,
    ) -> None:
        muon_params = [p for p in muon_params]
        adamw_params = [p for p in adamw_params]
        if len(muon_params) == 0:
            raise ValueError(
                "HybridMuon constructed with zero Muon params — classification found "
                "no 2-D hidden weights. Refusing to silently degrade to AdamW-only."
            )

        self.muon = Muon(
            muon_params,
            lr=muon_lr,
            weight_decay=muon_weight_decay,
            momentum=muon_momentum,
            nesterov=muon_nesterov,
            ns_steps=ns_steps,
            eps=muon_eps,
            adjust_lr_fn=adjust_lr_fn,
        )
        # AdamW group may legitimately be empty in some toy models; only build it
        # if there are params, but keep a stable child list either way.
        self.adamw = (
            AdamW(
                adamw_params,
                lr=adamw_lr,
                betas=adamw_betas,
                eps=adamw_eps,
                weight_decay=adamw_weight_decay,
            )
            if adamw_params
            else None
        )

        # Satisfy Optimizer's bookkeeping with the union of params, then delegate
        # param_groups/state to the children so per-group lr mutation + per-param
        # offload route to the real child optimizers.
        all_params = list(chain(muon_params, adamw_params))
        super().__init__(all_params, defaults={"lr": muon_lr})
        # ``param_groups`` is a plain list whose *elements* are the children's
        # live group dicts → pg["lr"] = x and group.setdefault("initial_lr", ..)
        # mutate the child group in place.
        self.param_groups = list(
            chain.from_iterable(c.param_groups for c in self._children)
        )
        self.state = _MergedState(self._children)

    # --- children iteration helper ----------------------------------------
    @property
    def _children(self) -> List[torch.optim.Optimizer]:
        return [c for c in (self.muon, self.adamw) if c is not None]

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for c in self._children:
            c.step()
        return loss

    def zero_grad(self, set_to_none: bool = True):
        for c in self._children:
            c.zero_grad(set_to_none=set_to_none)

    def state_dict(self) -> dict[str, Any]:
        return {
            "muon": self.muon.state_dict(),
            "adamw": self.adamw.state_dict() if self.adamw is not None else None,
        }

    def load_state_dict(self, sd: dict[str, Any]):
        self.muon.load_state_dict(sd["muon"])
        if self.adamw is not None and sd.get("adamw") is not None:
            self.adamw.load_state_dict(sd["adamw"])


def build_hybrid_muon(named_parameters, optim_config) -> HybridMuon:
    """Split ``named_parameters`` into Muon (2-D hidden weights) vs AdamW (rest)
    groups and construct a :class:`HybridMuon` from ``optim_config``.

    ``optim_config`` is the policy ``optimizer_config`` mapping. Recognized keys:
      - ``lr``: AdamW-group LR (kept identical to the AdamW baseline).
      - ``adam_betas``, ``weight_decay``: AdamW-group hyperparameters.
      - ``optimizer_kwargs`` (Muon-group overrides):
          ``muon_lr`` (default 0.02), ``muon_momentum`` (default 0.95),
          ``ns_steps`` (default 5), ``muon_weight_decay`` (default = ``weight_decay``),
          ``muon_nesterov`` (default True), ``adjust_lr_fn`` (default None).
    """
    muon_params, adamw_params = [], []
    muon_names, adamw_names = [], []
    for name, p in named_parameters:
        if not p.requires_grad:
            continue
        if is_muon_param(name, p):
            muon_params.append(p)
            muon_names.append(name)
        else:
            adamw_params.append(p)
            adamw_names.append(name)

    extra = dict(optim_config.get("optimizer_kwargs", {}) or {})
    adamw_lr = float(optim_config.lr)
    adamw_wd = float(optim_config.get("weight_decay", 0.0))
    adamw_betas = tuple(optim_config.get("adam_betas", (0.9, 0.999)))

    opt = HybridMuon(
        muon_params,
        adamw_params,
        muon_lr=float(extra.get("muon_lr", 0.02)),
        muon_weight_decay=float(extra.get("muon_weight_decay", adamw_wd)),
        muon_momentum=float(extra.get("muon_momentum", 0.95)),
        muon_nesterov=bool(extra.get("muon_nesterov", True)),
        ns_steps=int(extra.get("ns_steps", 5)),
        adjust_lr_fn=extra.get("adjust_lr_fn", None),
        adamw_lr=adamw_lr,
        adamw_betas=adamw_betas,
        adamw_eps=float(extra.get("adamw_eps", 1e-8)),
        adamw_weight_decay=adamw_wd,
    )
    opt._muon_param_names = muon_names  # type: ignore[attr-defined]
    opt._adamw_param_names = adamw_names  # type: ignore[attr-defined]
    return opt
