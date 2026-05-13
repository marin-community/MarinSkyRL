"""StaleClip — predictive LR damping based on batch staleness × policy entropy.

Hypothesis (formed from observed grad-spike + entropy-jump correlated with
stale_min increments in late-training, low-entropy regimes on the
OpenThoughts-Agent fully-async RL stack):

  When the batch lacks an on-policy anchor (``stale_min > 0``) AND the
  policy has concentrated (rolling-window mean entropy below a threshold),
  IS-corrected gradients of stale rollouts can correlate across the batch
  and Adam compounds the correlated bias into a grad spike. Pre-emptively
  damping the LR for the offending step bounds the update without
  requiring after-the-fact spike detection.

The signal arrives BEFORE the optimizer step (stale_min is known at batch
composition time, rolling entropy is known from prior steps), so the
damping is predictive rather than reactive.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Optional


@dataclass
class StaleClipState:
    """Persistent state for :class:`StaleClip` (rolling entropy window)."""

    entropy_history: Deque[float] = field(default_factory=lambda: deque(maxlen=10))


class StaleClip:
    """Predictive LR damping when an all-stale batch lands in a concentrated regime.

    Args:
        alpha: Per-unit-of-stale_min damping rate. With alpha=0.3:
            stale_min=1 → 0.7× LR, stale_min=2 → 0.4×, stale_min=3 → 0.1× (clipped at min_lr_scale).
        entropy_threshold: Below this rolling-mean entropy, regime is
            "concentrated" and damping engages. Above, no damping.
        entropy_window: Number of recent entropy values for the rolling mean.
        min_lr_scale: Lower bound on the damping factor. Default 0.1 (10× cut).
        enabled: Master toggle.
    """

    def __init__(
        self,
        alpha: float = 0.3,
        entropy_threshold: float = 0.15,
        entropy_window: int = 10,
        min_lr_scale: float = 0.1,
        enabled: bool = True,
    ):
        self.alpha = alpha
        self.entropy_threshold = entropy_threshold
        self.entropy_window = entropy_window
        self.min_lr_scale = min_lr_scale
        self.enabled = enabled

        self.state = StaleClipState(entropy_history=deque(maxlen=entropy_window))
        self.last_decision: Dict[str, float] = {}

    def update_entropy(self, entropy: float) -> None:
        """Push a fresh entropy reading. Call once per step before
        ``compute_lr_scale``."""
        if entropy is None:
            return
        self.state.entropy_history.append(float(entropy))

    def compute_lr_scale(self, stale_min: Optional[int]) -> float:
        """Return the multiplicative LR scale for the upcoming optimizer step.

        Returns 1.0 (no damping) if any of:
          - ``enabled`` is False
          - ``stale_min`` is None or 0 (on-policy anchor present)
          - rolling entropy is unavailable or above ``entropy_threshold``
        """
        if not self.enabled:
            self.last_decision = {"triggered": 0.0, "scale": 1.0}
            return 1.0

        if stale_min is None or stale_min <= 0:
            self.last_decision = {
                "triggered": 0.0,
                "scale": 1.0,
                "stale_min": float(stale_min) if stale_min is not None else -1.0,
            }
            return 1.0

        if not self.state.entropy_history:
            self.last_decision = {"triggered": 0.0, "scale": 1.0}
            return 1.0

        rolling = sum(self.state.entropy_history) / len(self.state.entropy_history)
        if rolling >= self.entropy_threshold:
            self.last_decision = {
                "triggered": 0.0,
                "scale": 1.0,
                "stale_min": float(stale_min),
                "rolling_entropy": float(rolling),
            }
            return 1.0

        scale = max(self.min_lr_scale, 1.0 - self.alpha * float(stale_min))
        self.last_decision = {
            "triggered": 1.0,
            "scale": float(scale),
            "stale_min": float(stale_min),
            "rolling_entropy": float(rolling),
        }
        return scale

    @staticmethod
    def apply_scale(optimizer, scale: float) -> List[float]:
        """Multiply each param_group's lr by ``scale``. Returns the original
        lrs so they can be restored after ``optimizer.step()``."""
        original = []
        for pg in optimizer.param_groups:
            original.append(pg["lr"])
            pg["lr"] = pg["lr"] * scale
        return original

    @staticmethod
    def restore_lrs(optimizer, original_lrs: List[float]) -> None:
        """Restore lrs captured by :meth:`apply_scale`."""
        for pg, lr in zip(optimizer.param_groups, original_lrs):
            pg["lr"] = lr

    def state_dict(self) -> Dict:
        return {"entropy_history": list(self.state.entropy_history)}

    def load_state_dict(self, sd: Dict) -> None:
        history = sd.get("entropy_history", [])
        self.state.entropy_history = deque(history, maxlen=self.entropy_window)
