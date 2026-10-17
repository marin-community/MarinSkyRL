from enum import StrEnum

from omegaconf import DictConfig


class RemoteReadMode(StrEnum):
    PER_KEY = "per_key"
    PREFETCH = "prefetch"


def validate_remote_read_config(cfg: DictConfig) -> None:
    for role in ("policy", "ref"):
        mode = cfg.trainer[role].model.remote_read_mode
        try:
            RemoteReadMode(mode)
        except ValueError as error:
            raise ValueError(
                f"trainer.{role}.model.remote_read_mode must be per_key or prefetch; got {mode!r}"
            ) from error
