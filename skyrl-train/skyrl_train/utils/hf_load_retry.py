# --- Rank-0 HF weight-index resolution retry (transient EOF flake) -----------
# At scale (e.g. 32 FSDP workers on CoreWeave) a single rank's HF weight load —
# `AutoModelForCausalLM.from_pretrained` (FSDP path) or `snapshot_download`
# (Megatron path), really the huggingface_hub weight-INDEX / safetensors
# resolution it triggers — intermittently flakes with a network/EOF-class error
# (an `IncompleteRead`, a dropped connection, or a transient "no
# model.safetensors"-style fetch failure even though the repo is fine and vLLM
# already loaded it). With no retry, that one transient miss kills the whole
# gang. We wrap the load in an exponential-backoff retry that catches ONLY the
# transient classes and re-raises everything else immediately, so a genuinely
# missing repo/file still surfaces.
#
# This module is dependency-light on purpose (only `os` + `loguru`) so it is
# safe to import from BOTH the FSDP model wrapper and the Megatron worker
# without pulling in heavy deps (flash-attn/peft) or risking a circular import.
#
# Tunables (match the repo's os.environ.get style, e.g. SKYRL_GDN_MASK_FLA):
#   SKYRL_HF_LOAD_MAX_RETRIES   (default 5)  number of retries after attempt 1
#   SKYRL_HF_LOAD_BACKOFF_BASE  (default 2.0) base seconds for 2**n backoff
#   SKYRL_HF_LOAD_BACKOFF_CAP   (default 32.0) per-sleep cap in seconds
#
# Guarded imports: huggingface_hub errors / requests / urllib3 may be absent in
# some envs; we degrade to whatever is importable rather than hard-failing the
# module import.
import os
import time
from typing import Optional, Tuple

from loguru import logger

_HF_TRANSIENT_EXC: Tuple[type, ...] = (OSError,)  # OSError covers EOFError-ish IO + connection resets
# Genuine "not there"/auth/gated failures that must NEVER be retried — they are
# checked FIRST in is_transient_hf_load_error and short-circuit to non-transient
# even though some subclass HfHubHTTPError (which IS in the transient tuple).
_HF_FATAL_EXC: Tuple[type, ...] = ()
try:
    from huggingface_hub.errors import (
        HfHubHTTPError,
        EntryNotFoundError,
        LocalEntryNotFoundError,
        RepositoryNotFoundError,
        RevisionNotFoundError,
        GatedRepoError,
    )

    # RepositoryNotFoundError / RevisionNotFoundError / GatedRepoError are
    # deliberately EXCLUDED from the transient set (they subclass HfHubHTTPError,
    # so we must list them as fatal to override that). snapshot_download raises
    # these same hf_hub classes for its transient failures, so the Megatron path
    # is covered by the exact same classifier with no change.
    _HF_TRANSIENT_EXC = _HF_TRANSIENT_EXC + (HfHubHTTPError, EntryNotFoundError, LocalEntryNotFoundError)
    _HF_FATAL_EXC = _HF_FATAL_EXC + (RepositoryNotFoundError, RevisionNotFoundError, GatedRepoError)
except Exception:  # noqa: BLE001 — huggingface_hub absent or API moved
    pass
try:
    import requests as _requests

    _HF_TRANSIENT_EXC = _HF_TRANSIENT_EXC + (_requests.exceptions.RequestException,)
except Exception:  # noqa: BLE001
    pass
try:
    import urllib3 as _urllib3

    _HF_TRANSIENT_EXC = _HF_TRANSIENT_EXC + (_urllib3.exceptions.HTTPError,)
except Exception:  # noqa: BLE001
    pass


def is_transient_hf_load_error(exc: BaseException) -> bool:
    """True if exc (or any cause in its chain) is a transient HF fetch flake.

    We treat as transient: the guarded hf_hub HTTP/entry-not-found errors,
    requests/urllib3 connection errors, OSError-class IO (EOF / IncompleteRead /
    connection reset), and the safetensors-index "no/cannot find ... .safetensors"
    fetch failure that is really a dropped weight-index download (NOT a genuinely
    missing file). A RepositoryNotFoundError / gated/auth failure is NOT matched
    here (it is excluded from `_HF_TRANSIENT_EXC` / listed in `_HF_FATAL_EXC`), so
    it propagates and surfaces.
    """
    seen = set()
    cur: Optional[BaseException] = exc
    while cur is not None and id(cur) not in seen:
        seen.add(id(cur))
        # Fatal classes (missing repo / bad revision / gated) override transient
        # even when they subclass HfHubHTTPError — they must surface, not retry.
        if _HF_FATAL_EXC and isinstance(cur, _HF_FATAL_EXC):
            return False
        if isinstance(cur, _HF_TRANSIENT_EXC):
            return True
        msg = str(cur).lower()
        # EOF / partial-read / connection-drop signatures that may arrive as a
        # bare RuntimeError/ValueError from deep in the safetensors-index path.
        if any(
            s in msg
            for s in (
                "incompleteread",
                "incomplete read",
                "eof",
                "connection reset",
                "connection aborted",
                "connection broken",
                "remote end closed",
                "broken pipe",
            )
        ):
            return True
        # The reported flake: a "no/can't find safetensors" message that is
        # actually a failed weight-index fetch (repo is fine; vLLM loaded it).
        if "safetensors" in msg and any(
            s in msg for s in ("no ", "not find", "cannot find", "couldn't find", "could not find", "does not")
        ):
            return True
        cur = cur.__cause__ or cur.__context__
    return False


def load_pretrained_with_retry(load_fn, *, model_id: str):
    """Call `load_fn()` (a no-arg HF-load closure — `from_pretrained` or
    `snapshot_download`) with exponential backoff on transient HF
    weight-index/safetensors fetch flakes only.

    Mirrors the vLLM engine-init retry (commit 6f945cdf): a local helper + a
    bounded attempt loop, per-retry `logger.warning(rank, attempt, exc)`, and a
    final re-raise of the last exception if every attempt fails. Non-transient
    exceptions (genuine missing repo/file, auth) are re-raised immediately.
    """
    max_retries = int(os.environ.get("SKYRL_HF_LOAD_MAX_RETRIES", "5"))
    backoff_base = float(os.environ.get("SKYRL_HF_LOAD_BACKOFF_BASE", "2.0"))
    backoff_cap = float(os.environ.get("SKYRL_HF_LOAD_BACKOFF_CAP", "32.0"))
    rank = os.environ.get("RANK", os.environ.get("LOCAL_RANK", "?"))
    total_attempts = max_retries + 1  # initial try + retries

    last_exc: Optional[BaseException] = None
    for attempt in range(total_attempts):
        try:
            return load_fn()
        except Exception as e:  # noqa: BLE001 — narrowed by is_transient_hf_load_error below
            if not is_transient_hf_load_error(e):
                raise
            last_exc = e
            if attempt == total_attempts - 1:
                logger.error(
                    f"[rank {rank}] HF weight load for '{model_id}' still failing after "
                    f"{total_attempts} attempts with a transient fetch error; re-raising: "
                    f"{str(e).splitlines()[0] if str(e) else type(e).__name__}"
                )
                raise
            backoff = min(backoff_base * (2 ** attempt), backoff_cap)
            logger.warning(
                f"[rank {rank}] transient HF weight-index/safetensors fetch error loading "
                f"'{model_id}' on attempt {attempt + 1}/{total_attempts}; retrying in "
                f"{backoff:.0f}s: {type(e).__name__}: {str(e).splitlines()[0] if str(e) else ''}"
            )
            time.sleep(backoff)
    # Unreachable (loop either returns or raises), but keep the type-checker happy.
    assert last_exc is not None
    raise last_exc
