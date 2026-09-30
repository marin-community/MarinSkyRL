import math
from collections.abc import Mapping


def validate_replay_keep_fraction(fraction: float | None, field: str) -> None:
    """Require a finite fraction of the native router's selection threshold."""
    if (
        isinstance(fraction, bool)
        or not isinstance(fraction, (int, float))
        or not math.isfinite(fraction)
        or not 0 <= fraction <= 1
    ):
        raise ValueError(f"{field} must be explicit and in [0, 1]")


def validate_router_replay_config(skyrl: Mapping) -> None:
    """Require an enabled Megatron replay controller for filtered training."""
    trainer = skyrl.get("trainer", {})
    for role in ("policy", "ref"):
        megatron = (trainer.get(role) or {}).get("megatron_config") or {}
        fraction = megatron.get("moe_router_replay_keep_fraction")
        if fraction is None:
            continue
        field = f"trainer.{role}.megatron_config.moe_router_replay_keep_fraction"
        if trainer.get("strategy") != "megatron" or not megatron.get("moe_router_replay"):
            raise ValueError(f"{field} requires enabled Megatron router replay")
        validate_replay_keep_fraction(fraction, field)
