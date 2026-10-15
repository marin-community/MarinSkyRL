"""How the trainer and the rollout engines compute, ``trainer.numerics``."""

from dataclasses import dataclass
from enum import StrEnum

from omegaconf import DictConfig

NUMERICS_KEY = "trainer.numerics"
CHECK_KEY = f"{NUMERICS_KEY}.check"


class NumericsMode(StrEnum):
    """The kernels the trainer and the rollout engines run: each side's own (``native``) or vLLM's batch-invariant
    kernels on both sides (``batch_invariant``)."""

    NATIVE = "native"
    BATCH_INVARIANT = "batch_invariant"

    @property
    def bit_identical(self) -> bool:
        """True for a mode that promises the trainer the engines' log-probabilities bit for bit; no mode does yet."""
        return False


class CheckFailure(StrEnum):
    """What the exactness check does when a token's log-probability differs: log it and continue, or stop the run."""

    LOG = "log"
    STOP = "stop"


@dataclass(frozen=True)
class ExactnessCheckConfig:
    """When the online exactness check runs (``every_weight_syncs`` 0 = never) and what a mismatch does."""

    every_weight_syncs: int
    on_failure: CheckFailure

    @property
    def enabled(self) -> bool:
        return self.every_weight_syncs > 0


@dataclass(frozen=True)
class NumericsResolution:
    """The numerics mode a run uses."""

    mode: NumericsMode


def resolve_numerics(cfg: DictConfig) -> NumericsResolution:
    """The numerics mode a run of ``cfg`` uses. Reads the config and never writes to it."""
    return NumericsResolution(NumericsMode(cfg.trainer.numerics.mode))


def exactness_check_config(cfg: DictConfig) -> ExactnessCheckConfig:
    """The exactness check settings of ``cfg``."""
    check = cfg.trainer.numerics.check
    return ExactnessCheckConfig(int(check.every_weight_syncs), CheckFailure(check.on_failure))


def validate_numerics_config(cfg: DictConfig) -> None:
    """Accept a known mode, ``batch_invariant`` only with local vLLM engines, and check settings the mode supports."""
    numerics = cfg.trainer.numerics
    try:
        mode = resolve_numerics(cfg).mode
    except ValueError as error:
        choices = ", ".join(mode.value for mode in NumericsMode)
        raise ValueError(f"{NUMERICS_KEY}.mode must be one of {choices}; got {numerics.mode!r}") from error
    if mode is NumericsMode.BATCH_INVARIANT:
        if cfg.generator.backend != "vllm":
            raise ValueError(f"{NUMERICS_KEY}.mode=batch_invariant requires generator.backend='vllm'")
        if not cfg.generator.run_engines_locally:
            raise ValueError(
                f"{NUMERICS_KEY}.mode=batch_invariant cannot configure a remote inference server; run the vLLM "
                "engines locally so both rollout and trainer activation is guaranteed"
            )
    if int(numerics.check.every_weight_syncs) < 0:
        raise ValueError(f"{CHECK_KEY}.every_weight_syncs must be at least 0; got {numerics.check.every_weight_syncs}")
    try:
        check = exactness_check_config(cfg)
    except ValueError as error:
        choices = ", ".join(failure.value for failure in CheckFailure)
        raise ValueError(
            f"{CHECK_KEY}.on_failure must be one of {choices}; got {numerics.check.on_failure!r}"
        ) from error
    if check.enabled and cfg.generator.backend != "vllm":
        raise ValueError(f"the exactness check requires generator.backend='vllm'; got {cfg.generator.backend!r}")
    if check.on_failure is CheckFailure.STOP and not mode.bit_identical:
        raise ValueError(
            f"{CHECK_KEY}.on_failure=stop requires a numerics mode that promises bit-identical log-probabilities; "
            f"{NUMERICS_KEY}.mode={mode} does not"
        )
