"""ZClip — reactive grad-norm spike mitigation via EMA z-score.

Port of https://github.com/bluorion-com/ZClip adapted to SkyRL's
strategy-managed gradient handling. SkyRL's strategy already handles
the cross-shard reduction for FSDP1/FSDP2/DDP via ``clip_grad_norm_``,
so this module consumes a scalar pre-clip ``grad_norm`` and returns an
effective ``max_norm`` to clip to. The actual rescaling is done by the
caller (typically by additional in-place gradient scaling after the
standard clip).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class ZClipState:
    """Persistent EMA state for :class:`ZClip`."""

    mean: float = 0.0
    var: float = 0.0
    initialized: bool = False
    warmup_buffer: List[float] = field(default_factory=list)


class ZClip:
    """Adaptive grad-norm clipping via EMA z-score.

    Args:
        alpha: EMA smoothing for mean and variance of grad_norm.
            Default 0.97 (paper default).
        z_thresh: Z-score above which a grad_norm is treated as a spike.
            Default 2.5 (paper default).
        warmup_steps: Steps to collect grad_norms before initializing
            the EMA. During warmup, no adaptive clipping is applied
            (only the static ``max_grad_norm`` floor, if set).
            NOTE: tuned to 3 for short (60-80 step) RL ablations. On
            Perlmutter 52905139 the prior 25-step warmup left ZClip
            still in warmup_remaining=3 when a collapse-onset grad
            spike landed (1.30 then 1.26 at steps 22-23), so the
            mechanism never got the chance to engage. Trade-off: a
            very short warmup means the EMA mean/var is noisy for the
            first ~10 steps post-warmup, which can cause false
            triggers if the first 3 samples happen to be unusually
            low. For longer runs (>200 steps) consider raising back
            toward 25.
        max_grad_norm: Hard ceiling for the effective clip — the
            returned value is always min(adaptive_clip, max_grad_norm).
            Set to None to let ZClip return arbitrarily large clips,
            but typically you want this as a safety net.
        clip_option: How to scale the threshold once a spike is
            detected (only used in ``mode="zscore"``):
              - ``"adaptive_scaling"``: threshold = mean + (z_thresh * std) / (z / z_thresh).
                Larger outliers get tighter clips.
              - ``"mean"``: threshold = mean. Hard clip back to baseline.
        clip_factor: Multiplier applied to the adaptive threshold.
            Lower values (0.5-0.9) are more aggressive.
        mode: ``"zscore"`` (default) or ``"percentile"`` (always clip
            to mean + z_thresh*std regardless of z-score).
        skip_update_on_spike: If True, the EMA is not updated when a
            spike is detected — protects EMA stats from contamination.
        eps: Numerical stability epsilon.
        enabled: Master toggle. When False, ``compute_max_norm`` always
            returns ``max_grad_norm`` (caller falls back to its static
            max_grad_norm).
    """

    def __init__(
        self,
        alpha: float = 0.97,
        z_thresh: float = 2.5,
        warmup_steps: int = 3,
        max_grad_norm: Optional[float] = None,
        clip_option: str = "adaptive_scaling",
        clip_factor: float = 1.0,
        mode: str = "zscore",
        skip_update_on_spike: bool = False,
        eps: float = 1e-6,
        enabled: bool = True,
    ):
        self.alpha = alpha
        self.z_thresh = z_thresh
        self.warmup_steps = warmup_steps
        self.max_grad_norm = max_grad_norm
        self.mode = mode.lower()
        self.clip_factor = clip_factor
        self.skip_update_on_spike = skip_update_on_spike
        self.eps = eps
        self.enabled = enabled

        if self.mode == "zscore":
            if clip_option.lower() not in ("mean", "adaptive_scaling"):
                raise ValueError(
                    f"ZClip clip_option must be 'mean' or 'adaptive_scaling', got {clip_option!r}"
                )
            self.clip_option = clip_option.lower()
        elif self.mode == "percentile":
            self.clip_option = None
        else:
            raise ValueError(f"ZClip mode must be 'zscore' or 'percentile', got {mode!r}")

        self.state = ZClipState()
        self.last_decision: Dict[str, float] = {}

    def compute_max_norm(self, grad_norm: float) -> Optional[float]:
        """Return the max_norm to clip to for this step, or None for no override.

        Always returns ``min(adaptive_threshold, self.max_grad_norm)`` when
        triggered. During warmup, returns ``self.max_grad_norm`` (or None).
        """
        if not self.enabled:
            self.last_decision = {"triggered": 0.0, "effective_max": float(self.max_grad_norm or 0.0)}
            return self.max_grad_norm

        if not self.state.initialized:
            self.state.warmup_buffer.append(grad_norm)
            if len(self.state.warmup_buffer) >= self.warmup_steps:
                self._initialize_ema()
            self.last_decision = {
                "triggered": 0.0,
                "warmup": 1.0,
                "warmup_remaining": float(max(0, self.warmup_steps - len(self.state.warmup_buffer))),
                "effective_max": float(self.max_grad_norm or 0.0),
            }
            return self.max_grad_norm

        clip_val = self._compute_clip_val(grad_norm)
        triggered = clip_val is not None

        if triggered:
            adaptive = clip_val if self.max_grad_norm is None else min(clip_val, self.max_grad_norm)
        else:
            adaptive = self.max_grad_norm

        if not (triggered and self.skip_update_on_spike):
            update_with = clip_val if clip_val is not None else grad_norm
            self._update_ema(update_with)

        std = self.state.var**0.5
        z = (grad_norm - self.state.mean) / (std + self.eps) if std > 0 else 0.0
        self.last_decision = {
            "triggered": float(triggered),
            "z_score": float(z),
            "ema_mean": float(self.state.mean),
            "ema_std": float(std),
            "raw_grad_norm": float(grad_norm),
            "effective_max": float(adaptive if adaptive is not None else 0.0),
        }
        return adaptive

    def _initialize_ema(self) -> None:
        buf = self.state.warmup_buffer
        self.state.mean = sum(buf) / len(buf)
        self.state.var = sum((x - self.state.mean) ** 2 for x in buf) / len(buf)
        self.state.initialized = True
        self.state.warmup_buffer = []

    def _update_ema(self, grad_norm: float) -> None:
        s = self.state
        s.mean = self.alpha * s.mean + (1 - self.alpha) * grad_norm
        s.var = self.alpha * s.var + (1 - self.alpha) * (grad_norm - s.mean) ** 2

    def _compute_clip_val(self, grad_norm: float) -> Optional[float]:
        std = self.state.var**0.5
        if self.mode == "percentile":
            threshold = self.state.mean + self.z_thresh * std
            return threshold if grad_norm > threshold else None

        if std <= self.eps:
            return None
        z = (grad_norm - self.state.mean) / (std + self.eps)
        if z <= self.z_thresh:
            return None
        if self.clip_option == "adaptive_scaling":
            eta = z / self.z_thresh
            threshold = self.state.mean + (self.z_thresh * std) / eta
            return threshold * self.clip_factor
        return self.state.mean

    def state_dict(self) -> Dict:
        return {
            "mean": self.state.mean,
            "var": self.state.var,
            "initialized": self.state.initialized,
            "warmup_buffer": list(self.state.warmup_buffer),
        }

    def load_state_dict(self, sd: Dict) -> None:
        self.state.mean = sd.get("mean", 0.0)
        self.state.var = sd.get("var", 0.0)
        self.state.initialized = sd.get("initialized", False)
        self.state.warmup_buffer = list(sd.get("warmup_buffer", []))
