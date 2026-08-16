#!/usr/bin/env python3
"""Supervise one node of a multi-node MarinSkyRL Iris task.

iris gang-schedules a multi-node job as N coscheduled tasks (one per node) and
runs THIS SAME entrypoint on every node, injecting ``IRIS_TASK_ID`` /
``IRIS_NUM_TASKS``. This script bootstraps one cross-node Ray cluster, then
runs the driver on rank 0:

- **rank 0 (head):** ``ray start --head``; publish the head IP to a shared
  rendezvous file; wait until ``ray.nodes()`` shows all ``IRIS_NUM_TASKS``
  nodes joined; then ``exec`` the MarinSkyRL training command with
  ``RAY_ADDRESS`` pointing at the head so skyrl-train's bare ``ray.init()``
  attaches to the existing cluster.
- **ranks 1..N-1 (workers):** read the head IP from the rendezvous, run
  ``ray start --address=<head_ip>:<port>``, then BLOCK until the head's
  ``done`` marker or SIGTERM. They contribute their 8 H100s to the Ray
  cluster; they do NOT run the training driver.

Head-IP discovery: rank 0 uses iris's ``IRIS_ADVERTISE_HOST`` (routable IP
under ``host_network: true``) and publishes it to a rendezvous file on a shared
object store (``--rendezvous-dir``). The URI may be ``gs://``, ``s3://``, or a
local path — opened via ``fsspec``.
"""

from __future__ import annotations

import argparse
import contextlib
from dataclasses import asdict, dataclass, field, fields
from enum import StrEnum
import glob
import json
import os
import queue
import signal
import socket
import subprocess
import sys
import tempfile
import threading
import time
from typing import Protocol
from cloud.iris.artifacts import ArtifactSource, fs_and_path, materialize
from marinskyrl.hf_model import validate_portable_hf_model_files
from cloud.iris.env_vars import (
    DEBUG_ARTIFACT_DIR_ENV,
    FR_DUMP_TEMP_FILE_ENV,
    NCCL_DEBUG_INFO_TEMP_FILE_ENV,
    ensure_debug_artifact_directories,
    iris_ray_cluster_owner_environment,
)
from cloud.iris.model_paths import unsupported_model_path_message
from marinskyrl.resource_locator import is_cloud_uri, join_resource_path
from cloud.iris.paths import resolve_repo_path
from cloud.iris.ray_storage import (
    DEFAULT_RAY_SPILL_DIR,
    RaySpillBackend,
    RaySpillTarget,
    resolve_ray_spill_target,
    validate_ray_spill_dir,
)
from cloud.iris.runtime_bundle import validate_bundled_runtime

try:
    from skyrl_train.ray_metrics import ray_metrics_telemetry
except ImportError as error:
    # An installed rigging without the telemetry submodule raises ImportError, not
    # ModuleNotFoundError; the name check still keeps a failure inside these packages visible.
    if error.name not in {"prometheus_client", "rigging", "skyrl_train"}:
        raise
    _RAY_METRICS_UNAVAILABLE_REASON = str(error)

    def ray_metrics_telemetry(node_ip: str, metrics_port: int):
        _log(f"Ray metric forwarding is unavailable: {_RAY_METRICS_UNAVAILABLE_REASON}")
        return contextlib.nullcontext()


RENDEZVOUS_FILENAME = "ray_head.json"
DONE_FILENAME = "ray_head.done"


@dataclass(frozen=True)
class RendezvousPayload:
    head_ip: str
    head_node: str
    port: int
    num_tasks: int
    python_version: str
    ray_version: str
    written_at: float

    @classmethod
    def from_dict(cls, value: object) -> "RendezvousPayload":
        if not isinstance(value, dict):
            raise RuntimeError("Ray head rendezvous must be a JSON object")
        required_fields = tuple(field.name for field in fields(cls))
        missing_fields = [field for field in required_fields if value.get(field) in (None, "")]
        if missing_fields:
            raise RuntimeError(f"Ray head rendezvous is missing fields: {', '.join(missing_fields)}")
        try:
            return cls(
                head_ip=str(value["head_ip"]),
                head_node=str(value["head_node"]),
                port=int(value["port"]),
                num_tasks=int(value["num_tasks"]),
                python_version=str(value["python_version"]),
                ray_version=str(value["ray_version"]),
                written_at=float(value["written_at"]),
            )
        except (TypeError, ValueError) as error:
            raise RuntimeError("Ray head rendezvous contains invalid field values") from error


def _ray_bin() -> str:
    """Resolve the ``ray`` CLI from the SAME venv as the running interpreter.

    iris's uv-sync activates /app/.venv on $PATH, which has no ``ray``; use the
    binary next to this interpreter, falling back to PATH.
    """
    import shutil

    candidate = os.path.join(os.path.dirname(sys.executable), "ray")
    if os.path.exists(candidate):
        return candidate
    found = shutil.which("ray")
    return found or "ray"


# Generous: cold GPU nodes (image pull + setup) take minutes to reach rendezvous.
DEFAULT_RENDEZVOUS_TIMEOUT = 1800
DEFAULT_CLUSTER_JOIN_TIMEOUT = 1800
POLL_INTERVAL = 5
# Tolerates clock skew between nodes and the time rank-0 needs to start Ray.
RENDEZVOUS_FRESHNESS_SLACK = 60
# Bound the rank-0 rendezvous PutObject: unbounded fsspec/s3fs put can wedge the
# head forever. Hard timeout per attempt + bounded retries; final failure raises.
RENDEZVOUS_WRITE_ATTEMPTS = 5
RENDEZVOUS_WRITE_TIMEOUT = 30  # seconds per attempt
# Bound `ray start --head` so a hung Ray CLI fails loud (TimeoutExpired).
RAY_START_HEAD_TIMEOUT = 300  # seconds


def _log(msg: str) -> None:
    print(f"[task-runtime] {msg}", flush=True)


def stage_train_data(train_data_json: str) -> None:
    """Extract the HF task dataset(s) to this NODE's local task dir on EVERY node.

    The controller runs on every node before Ray bootstrap; pods don't share a
    filesystem, so each pod fetches+extracts the parquet repo to the identical
    node-local path the rollout workers read. Idempotent (``on_exist=skip``).
    """
    import json as _json

    try:
        train_data = _json.loads(train_data_json)
    except (ValueError, TypeError):
        train_data = [train_data_json] if train_data_json else []
    if not train_data:
        return

    # PYTHONPATH includes /app (launcher bootstrap), so cloud.iris is importable.
    from cloud.iris.rl_data import resolve_rl_train_data

    _log(f"Staging train_data on this node (rank {_rank()}/{_num_tasks()}): {train_data}")
    # This HF *parquet* dataset is NOT pre-cached, so force-clear HF_HUB_OFFLINE /
    # TRANSFORMERS_OFFLINE for the extraction only (the ranks' env is untouched).
    # Without this the snapshot_download inside resolve_rl_train_data dies
    # OfflineModeIsEnabled.
    saved = {k: os.environ.get(k) for k in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE")}
    for k in saved:
        os.environ.pop(k, None)
    try:
        resolved = resolve_rl_train_data(train_data, on_exist="skip", verbose=True)
    finally:
        for k, v in saved.items():
            if v is not None:
                os.environ[k] = v
    _log(f"train_data staged to node-local paths: {resolved}")


def _warm_model_snapshot_hash(model_path: str) -> str:
    """Deterministic synthetic 40-hex 'commit' for the warm S3-synced snapshot dir.

    Offline ``from_pretrained`` / ``snapshot_download(local_files_only=True)`` resolve a
    repo via ``<cache>/models--<org>--<name>/refs/main`` -> a ``snapshots/<hash>/`` dir.
    Under HF_HUB_OFFLINE=1 there is no hash validation, so a STABLE synthetic hash
    keyed on the repo id gives an idempotent, collision-free (``otwarm:`` prefixed)
    snapshot dir that re-syncs skip-in-place on a ``--max-retries`` re-bring.
    """
    import hashlib

    return hashlib.sha1(("otwarm:" + model_path).encode()).hexdigest()


def _warm_sync_model_from_s3(model_path: str, warm_source: str) -> bool:
    """In-region warm path for :func:`stage_model`.

    Sync the FLAT weight/config/tokenizer files seeded at ``warm_source`` (an ``s3://``
    CW-object-store prefix, convention ``s3://marin-us-east-02a/models/<org>--<name>/``)
    INTO this node's HF hub cache under a synthetic-revision snapshot, so the offline FSDP
    ranks + vLLM engines resolve ``from_pretrained(<repo-id>)`` from the warm node-local
    cache — ``model.path`` stays the repo-id, the ranks are untouched.

    Returns True if the warm source existed + synced cleanly; False if it is missing /
    empty / incomplete so the caller falls back to the HF ``snapshot_download`` prestage.
    Idempotent + resumable: size-skips files already present.

    Reuses the SAME boto3 + ``AWS_ENDPOINT_URL`` creds path the rendezvous / spill /
    term-artifact writers use; ``_pin_boto3_s3_addressing_style`` (called before this in
    main()) already pinned virtual-hosted addressing for CW-R2, and we ALSO pass an
    explicit botocore ``Config`` here so this call is correct even if invoked out of
    order.
    """
    if not warm_source or not warm_source.startswith("s3://"):
        _log(f"warm sync: warm_source {warm_source!r} is not an s3:// URI; skipping warm path")
        return False
    import boto3
    from botocore.config import Config
    from huggingface_hub import constants as _hf_constants

    # Parse s3://bucket/prefix... -> (bucket, prefix without trailing slash).
    without_scheme = warm_source[len("s3://") :]
    bucket, _, prefix = without_scheme.partition("/")
    prefix = prefix.rstrip("/")
    if not bucket or not prefix:
        _log(f"warm sync: malformed warm_source {warm_source!r} (need s3://bucket/prefix); skipping")
        return False

    endpoint = os.environ.get("AWS_ENDPOINT_URL")
    style = os.environ.get("OT_AGENT_S3_ADDRESSING_STYLE", "virtual")
    client = boto3.client("s3", endpoint_url=endpoint, config=Config(s3={"addressing_style": style}))

    # List every seeded object under the prefix (paginated).
    list_prefix = prefix + "/"
    objects: list[tuple[str, int]] = []
    try:
        paginator = client.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=bucket, Prefix=list_prefix):
            for obj in page.get("Contents", []):
                key = obj["Key"]
                if key.endswith("/"):
                    continue  # pseudo-dir marker
                objects.append((key, int(obj.get("Size", 0))))
    except Exception as exc:  # noqa: BLE001 - any S3 error -> clean HF fallback
        _log(f"warm sync: list_objects_v2 on s3://{bucket}/{list_prefix} FAILED ({exc!r}); HF fallback")
        return False

    if not objects:
        _log(f"warm sync: no objects under {warm_source} (not seeded yet); HF fallback")
        return False
    # Completeness guard: a config.json is mandatory for from_pretrained. Its absence means
    # a half-seeded / wrong prefix -> do NOT build a broken cache; fall back to HF.
    if not any(os.path.basename(k) == "config.json" for k, _ in objects):
        _log(f"warm sync: {warm_source} has {len(objects)} objects but NO config.json (incomplete seed); HF fallback")
        return False

    cache = _hf_constants.HF_HUB_CACHE
    folder = "models--" + model_path.replace("/", "--")
    snap_hash = _warm_model_snapshot_hash(model_path)
    snap_dir = os.path.join(cache, folder, "snapshots", snap_hash)
    refs_dir = os.path.join(cache, folder, "refs")
    os.makedirs(snap_dir, exist_ok=True)
    os.makedirs(refs_dir, exist_ok=True)

    total_files = total_bytes = skipped = 0
    _log(f"warm sync: {len(objects)} object(s) from {warm_source} -> {snap_dir} (rank {_rank()}/{_num_tasks()})")
    for key, size in objects:
        rel = key[len(list_prefix) :]
        if not rel:
            continue
        dest = os.path.join(snap_dir, rel)
        if os.path.exists(dest) and os.path.getsize(dest) == size:
            skipped += 1
            continue
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        # Download to a temp sibling then rename, so an interrupted pull never leaves a
        # truncated file that the size-skip would later treat as complete.
        tmp_dest = dest + ".otwarm.partial"
        last_exc: BaseException | None = None
        for attempt in range(1, 4):
            try:
                client.download_file(bucket, key, tmp_dest)
                os.replace(tmp_dest, dest)
                total_files += 1
                total_bytes += size
                last_exc = None
                break
            except Exception as exc:  # noqa: BLE001 - retry a few times, then fail loud
                last_exc = exc
                _log(f"warm sync: download attempt {attempt}/3 for {key} failed ({exc!r})")
                try:
                    os.path.exists(tmp_dest) and os.remove(tmp_dest)
                except OSError:
                    pass
                time.sleep(min(10, 2**attempt))
        if last_exc is not None:
            # A partial in-region sync would leave a broken cache; fail this warm attempt so
            # the caller falls back to the HF prestage rather than shipping a corrupt cache.
            _log(f"warm sync: giving up on {key} after 3 attempts ({last_exc!r}); HF fallback")
            return False

    # Point refs/main at the synthetic snapshot so offline resolution finds it.
    with open(os.path.join(refs_dir, "main"), "w") as f:
        f.write(snap_hash)
    _log(
        f"warm sync: DONE — {total_files} downloaded ({total_bytes / 1073741824.0:.2f} GiB), "
        f"{skipped} already-present; refs/main -> {snap_hash}. Offline from_pretrained "
        f"({model_path}) resolves from {snap_dir}."
    )
    return True


def stage_model(model_path: str, warm_source: str | None = None) -> None:
    """Pre-download the policy model into this NODE's local HF cache on EVERY node.

    The controller runs on every node before Ray bootstrap, so pre-download the
    weights ONCE PER NODE here (N pulls, not N*8) into the node-local HF cache.
    The download runs in the controller's pre-Ray phase where no collective
    barrier is active. With ``HF_HUB_OFFLINE=1`` set for the training ranks
    (config extra_env), all ranks on a node read the pre-populated cache.
    Idempotent: ``snapshot_download`` skips already-complete cached files.

    The pod env carries ``HF_HUB_OFFLINE=1`` / ``TRANSFORMERS_OFFLINE=1`` for the
    ranks; this download MUST reach the Hub, so it runs ``snapshot_download`` in a
    SUBPROCESS whose env has both flags stripped (an in-process env-pop is too late —
    huggingface_hub caches HF_HUB_OFFLINE into a module constant at import). The
    ranks' env is untouched.
    """
    if is_cloud_uri(model_path):
        raise ValueError(unsupported_model_path_message(model_path))
    if not model_path or os.path.isdir(model_path):
        _log(f"stage_model: skip (model_path={model_path!r} is empty or a local directory)")
        return

    # WARM PATH: if a seeded in-region CW-S3 source exists, sync the weights from
    # there (in-datacenter, reliable) into the node-local HF cache INSTEAD of pulling
    # from HF Hub. On success the offline ranks + vLLM load from the warm cache as
    # after the HF prestage. On a missing/empty/incomplete source or ANY error we fall
    # through to the HF snapshot_download prestage below (byte-identical). warm_source
    # is None unless the launcher forwarded --model-warm-source.
    if warm_source:
        try:
            if _warm_sync_model_from_s3(model_path, warm_source):
                return
            _log(
                f"stage_model: warm source {warm_source} missing/empty/incomplete "
                f"-> HF snapshot_download prestage fallback"
            )
        except Exception as exc:  # noqa: BLE001 - never let the warm path block bring-up
            _log(
                f"stage_model: warm sync from {warm_source} FAILED ({exc!r}) -> HF snapshot_download prestage fallback"
            )

    # Weights + config + tokenizer + any trust_remote_code modeling files. Mirrors
    # mirror_hf_to_gcs.INCLUDE_PATTERNS so from_pretrained resolves fully offline.
    allow_patterns = ["*.safetensors", "*.json", "*.txt", "*.model", "*.py"]

    # Download in a SUBPROCESS with the offline flags stripped from ITS env. An
    # in-process os.environ.pop does NOT work here: huggingface_hub snapshots
    # HF_HUB_OFFLINE into a module CONSTANT at IMPORT time, so clearing the env
    # var afterward leaves the cached constant True -> snapshot_download raises
    # OfflineModeIsEnabled. A fresh child process re-reads the (cleaned) env at
    # its own import. The child inherits HF_HOME/HF_HUB_CACHE, so it populates
    # the SAME node-local cache the offline ranks then read.
    child_env = {k: v for k, v in os.environ.items() if k not in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE")}
    child_env["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"  # keep captured stderr bounded
    code = (
        "import sys\n"
        "from huggingface_hub import snapshot_download\n"
        "p = snapshot_download(sys.argv[1], allow_patterns=sys.argv[2].split(','))\n"
        "print('PRESTAGE_LOCAL_DIR=' + p)\n"
    )
    _log(f"Pre-staging model on this node (rank {_rank()}/{_num_tasks()}): {model_path}")
    last_err = ""
    # A stalled snapshot_download (mid-download socket hang) blocks subprocess.run
    # forever without a per-attempt timeout. HF resumes the partial `.incomplete`
    # shard on the next attempt, so a killed-mid-download attempt loses nothing.
    # 600s comfortably covers a clean ~160 GB pull yet fits several retries inside
    # the 1800s gang-join budget.
    PRESTAGE_ATTEMPT_TIMEOUT_S = 600
    for attempt in range(1, 7):
        try:
            proc = subprocess.run(
                [sys.executable, "-c", code, model_path, ",".join(allow_patterns)],
                env=child_env,
                capture_output=True,
                text=True,
                timeout=PRESTAGE_ATTEMPT_TIMEOUT_S,
            )
        except subprocess.TimeoutExpired:
            last_err = (
                f"snapshot_download stalled > {PRESTAGE_ATTEMPT_TIMEOUT_S}s "
                "(mid-download socket hang); killed, retrying (HF resumes the partial shard)"
            )
            _log(f"model prestage attempt {attempt}/6 TIMED OUT: {last_err}")
            time.sleep(min(30, 2**attempt))
            continue
        if proc.returncode == 0:
            local_dir = ""
            for line in proc.stdout.splitlines():
                if line.startswith("PRESTAGE_LOCAL_DIR="):
                    local_dir = line.split("=", 1)[1]
            _log(f"model pre-staged to node-local HF cache: {local_dir}")
            return
        last_err = (proc.stderr or proc.stdout or "")[-800:]
        _log(f"model prestage attempt {attempt}/6 failed (rc={proc.returncode}): {last_err}")
        time.sleep(min(30, 2**attempt))
    raise RuntimeError(f"model prestage failed after 6 attempts for {model_path}: {last_err}")


def materialize_model_export(source_uri: str, local_path: str, source_identity: str) -> None:
    """Copy and validate an object-store HF export on this allocated node."""
    source = ArtifactSource(uri=source_uri, local_path=local_path, identity=source_identity)
    artifact = materialize(source, validate=validate_portable_hf_model_files)
    _log(
        f"Model export staged on rank {_rank()}/{_num_tasks()}: {source.uri} -> {source.local_path} "
        f"({len(artifact.files)} files, identity={source.identity})"
    )


def materialize_data_sources(data_sources_json: str) -> None:
    """Copy immutable train and validation data locators onto this allocated node."""
    sources = json.loads(data_sources_json)
    if not isinstance(sources, list):
        raise ValueError("--data-sources-json must contain a JSON list")
    for value in sources:
        source = ArtifactSource(uri=value["uri"], local_path=value["local_path"], identity=value["identity"])
        artifact = materialize(source)
        _log(
            f"Dataset staged on rank {_rank()}/{_num_tasks()}: {source.uri} -> {source.local_path} "
            f"({len(artifact.files)} files, identity={source.identity})"
        )


# Special tokens the delphi_v0 reasoning protocol depends on; asserted present in the
# tokenizer's vocab after the override so a lossy SFT export (fragmented-to-bytes tokens)
# fails loud here instead of silently collapsing reward at the first rollout.
_REQUIRED_CHAT_TEMPLATE_TOKENS = ("<|start_think|>", "<|end_think|>")


def apply_policy_chat_template(model_path: str, template_repo_rel: str) -> None:
    """Force the policy tokenizer's cached chat template to ``template_repo_rel`` on THIS node.

    The delphi SFT repos ship no chat_template, so the policy tokenizer must be forced to the
    delphi_v0 template the rollouts + held-out eval use, else reward silently collapses on a
    train/rollout/eval mismatch. Must run on EVERY node (the ``skyrl_entrypoint`` actor may
    load its tokenizer on any node), after the model is staged into the node-local cache and
    before Ray bootstrap.

    Args:
        model_path: HF repo ID staged in the node-local cache or an object-store model's
            materialized local directory.
        template_repo_rel: chat-template jinja path, resolved against the in-pod repo root.

    Raises:
        RuntimeError: if the override does not take, or a required think-protocol token
            (``<|start_think|>`` / ``<|end_think|>``) is not a single registered token.
    """
    template_path = resolve_repo_path(template_repo_rel)
    delphi = template_path.read_text()

    # Import transformers/hf lazily (matches wait_for_nodes' local `import ray` and
    # stage_model): the controller bootstraps Ray on every node and must not pull these
    # heavy ML deps into the fast bootstrap path for configs that set no chat-template.
    from huggingface_hub import snapshot_download
    from transformers import AutoTokenizer

    snap = model_path if os.path.isdir(model_path) else snapshot_download(model_path)
    tc_path = os.path.join(snap, "tokenizer_config.json")
    jinja_path = os.path.join(snap, "chat_template.jinja")

    # Back up originals once (copy resolved content, not the cache symlink).
    for p in (tc_path, jinja_path):
        if os.path.exists(p):
            bak = p + ".plainbak"
            if not os.path.exists(bak):
                with open(p, "rb") as s, open(bak, "wb") as d:
                    d.write(s.read())

    # Write delphi_v0 as a real chat_template.jinja (break the cache symlink; keep the blob).
    if os.path.islink(jinja_path):
        os.remove(jinja_path)
    with open(jinja_path, "w") as f:
        f.write(delphi)

    # And as the chat_template key in tokenizer_config.json (break symlink).
    tc = {}
    if os.path.exists(tc_path):
        with open(tc_path) as f:
            tc = json.load(f)
    tc["chat_template"] = delphi
    if os.path.islink(tc_path):
        os.remove(tc_path)
    with open(tc_path, "w") as f:
        json.dump(tc, f, ensure_ascii=False, indent=2)

    # Verify the loaded tokenizer now renders delphi_v0 and keeps the think protocol tokens.
    tok = AutoTokenizer.from_pretrained(snap, trust_remote_code=True)
    ct = tok.chat_template
    if not ct or ct.strip() != delphi.strip():
        raise RuntimeError(
            f"chat_template override did NOT take for {model_path} "
            f"(loaded len={len(ct) if ct else None}, expected {len(delphi)})"
        )
    vocab = tok.get_vocab()
    missing = [t for t in _REQUIRED_CHAT_TEMPLATE_TOKENS if t not in vocab]
    if missing:
        raise RuntimeError(
            f"delphi_v0 think-protocol tokens {missing} are NOT single registered tokens in "
            f"{model_path}'s tokenizer (lossy SFT export?) — they would fragment to bytes and "
            f"break the reward/parse contract. Aborting before a silent reward-zero run."
        )
    _log(
        f"apply_policy_chat_template: delphi_v0 applied + verified for {model_path} "
        f"(chat_template len={len(ct)}, tokens OK) on rank {_rank()}/{_num_tasks()} (snapshot={snap})"
    )


def policy_chat_template_model(prestage_model: str, model_local_path: str) -> str:
    """Return the model directory or Hub ID whose tokenizer should be rewritten."""
    model_path = prestage_model or model_local_path
    if not model_path:
        raise ValueError(
            "--policy-chat-template requires --prestage-model or --model-local-path "
            "(the template override rewrites the materialized model tokenizer)"
        )
    return model_path


def _rank() -> int:
    # IRIS_TASK_ID is the full task path (e.g. "/user/job/0"); on retried tasks
    # iris appends a ":N" retry suffix. The rank is the trailing path segment
    # with any retry suffix stripped.
    return int(os.environ.get("IRIS_TASK_ID", "0").rsplit("/", 1)[-1].split(":", 1)[0])


def _num_tasks() -> int:
    return int(os.environ.get("IRIS_NUM_TASKS", "1"))


def _own_ip() -> str:
    """Routable IP of this node.

    Prefers iris's ``IRIS_ADVERTISE_HOST`` (the routable IP iris computed for
    this task under ``host_network: true``); falls back to a UDP-socket probe.
    """
    advertised = os.environ.get("IRIS_ADVERTISE_HOST")
    if advertised and advertised != "127.0.0.1":
        return advertised
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("8.8.8.8", 80))
        return s.getsockname()[0]
    finally:
        s.close()


def _iface_for_ip(ip: str) -> str | None:
    """Name of the interface holding ``ip``, or None if it cannot be determined.

    Reads /proc/net/fib_trie-free: uses the socket interface table directly so no
    external binary (`ip`, `ifconfig`) and no third-party import is required.
    """
    try:
        import fcntl
        import struct

        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            for _, name in socket.if_nameindex():
                try:
                    packed = fcntl.ioctl(
                        sock.fileno(),
                        0x8915,  # SIOCGIFADDR
                        struct.pack("256s", name[:15].encode()),
                    )
                    if socket.inet_ntoa(packed[20:24]) == ip:
                        return name
                except OSError:
                    continue  # interface has no IPv4 address
        finally:
            sock.close()
    except Exception as exc:  # noqa: BLE001
        _log(f"[fabric] interface lookup failed: {exc}")
    return None


def pin_socket_ifname() -> str | None:
    """Derive GLOO_SOCKET_IFNAME from the address iris advertises for this task.

    Returns the interface name derived on THIS node, or None when nothing was
    derived (an operator already set the variable, or no interface holds the
    node's IP).

    Gloo carries the DP-rank-0 optimizer gather at checkpoint save (the megatron
    strategy picks ``fully_reshardable`` + mem_efficient precisely so that gather
    runs on CPU and needs no GPU memory). With no interface pin gloo applies its
    own heuristic; on this cluster it picked loopback, every rank advertised
    127.0.0.1, and the mesh never formed.

    NCCL's ``^a,b,c`` exclude syntax cannot be reused here — gloo reads the value
    as a literal interface name and dies with ``Unable to find address for: ^ibs``.
    A static name cannot be used either: the PF differs per node and region.

    So derive it. ``IRIS_ADVERTISE_HOST`` is the pod IP, injected per pod by iris
    through the Kubernetes downward API (``status.podIP``), which is exactly the
    address whose interface the collectives must bind. Whatever interface holds
    it is correct on this node by construction.

    iris runs this same entrypoint on every node of the gang, so every node
    derives its own name before its own ``ray start`` and Ray's workers inherit
    it. The name is only ever correct for the node that derived it.

    Fail-safe by design: an existing value is never overridden, and any failure to
    resolve leaves the environment untouched and the previous behaviour intact.
    """
    if os.environ.get("GLOO_SOCKET_IFNAME"):
        _log("[fabric] GLOO_SOCKET_IFNAME already set; leaving it alone")
        return None
    ip = _own_ip()
    if not ip or ip == "127.0.0.1":
        _log("[fabric] no routable IP; not pinning GLOO_SOCKET_IFNAME")
        return None
    iface = _iface_for_ip(ip)
    if not iface:
        _log(f"[fabric] no interface holds {ip}; not pinning GLOO_SOCKET_IFNAME")
        return None
    os.environ["GLOO_SOCKET_IFNAME"] = iface
    _log(f"[fabric] GLOO_SOCKET_IFNAME={iface} (derived from {ip})")
    return iface


def training_driver_env(derived_gloo_ifname: str | None) -> dict[str, str]:
    """Environment for the rank-0 training driver subprocess.

    Strips a GLOO_SOCKET_IFNAME that ``pin_socket_ifname`` derived on this node.
    skyrl-train forwards GLOO_SOCKET_IFNAME from the driver's environment into
    ``ray.init``'s job-level ``runtime_env``, which pushes ONE value to the actors
    on EVERY node. NIC names are not uniform across this gang: job
    20260729-102429-52af30 had the head derive ``enp90s0np0`` while 10.168.206.93
    names its NIC ``enp90s0f0np0``, so megatron's gloo group creation on that node
    died with ``Unable to find address for: enp90s0np0``. Each node's own
    controller already derived the right name before its ``ray start``.

    A value an operator set explicitly (via the launcher or the config's
    ``extra_env``) reaches every pod already and is passed through untouched.
    """
    env = os.environ.copy()
    env.update(iris_ray_cluster_owner_environment())
    if derived_gloo_ifname is not None and env.get("GLOO_SOCKET_IFNAME") == derived_gloo_ifname:
        del env["GLOO_SOCKET_IFNAME"]
        _log(f"[fabric] withholding node-derived GLOO_SOCKET_IFNAME={derived_gloo_ifname} from the training driver")
    return env


# ---------------------------------------------------------------------------
# Rendezvous — head publishes its IP, workers poll for it. Backend-agnostic via
# fsspec so the URI scheme (gs://, s3://, file://, plain path) selects storage.
# ---------------------------------------------------------------------------


def _pin_boto3_s3_addressing_style() -> None:
    """Pin virtual-hosted (hostname-based) S3 addressing for the boto3 code path,
    cluster-wide, via an AWS config file + ``AWS_CONFIG_FILE``.

    Companion to the fsspec/s3fs pin in ``fs_and_path``: the CoreWeave object
    store (marin-us-east-02a / R2) REJECTS path-style S3 requests with
    ``PathStyleRequestNotAllowed``. Ray's object-spill IO workers call a bare
    ``boto3.resource("s3")`` with NO botocore ``Config`` in child processes we
    never construct, and botocore has NO env var for ``addressing_style`` (it is
    resolved ONLY from the shared-config file's ``s3`` section). So we WRITE a
    minimal AWS config file that sets ``s3.addressing_style`` and point
    ``AWS_CONFIG_FILE`` at it. This runs before ``ray start`` (head + every
    worker) so every Ray subprocess / raylet IO worker / the training driver
    inherits it.

    R2 credentials/region/endpoint come from env vars independently of the config
    file. Env-overridable via ``OT_AGENT_S3_ADDRESSING_STYLE`` (default
    ``virtual``; set ``path``/``auto`` for a path-style store). If an operator
    already exported ``AWS_CONFIG_FILE``, we RESPECT it and skip.
    """
    style = os.environ.get("OT_AGENT_S3_ADDRESSING_STYLE", "virtual")
    existing = os.environ.get("AWS_CONFIG_FILE")
    if existing and os.path.exists(existing):
        _log(
            f"AWS_CONFIG_FILE already set ({existing}); NOT overriding for S3 "
            f"addressing_style. If Ray spill hits PathStyleRequestNotAllowed, add "
            f"'s3 =\\n    addressing_style = {style}' to that file's [default] profile."
        )
        return
    cfg_path = os.path.join(tempfile.gettempdir(), "ot_agent_aws_config")
    try:
        with open(cfg_path, "w") as f:
            f.write(f"[default]\ns3 =\n    addressing_style = {style}\n")
        os.environ["AWS_CONFIG_FILE"] = cfg_path
        _log(
            f"S3 addressing_style={style} pinned via AWS_CONFIG_FILE={cfg_path} "
            f"(boto3/Ray-object-spill virtual-hosted addressing for CoreWeave R2)."
        )
    except OSError as exc:  # noqa: BLE001 - best-effort; falls back to boto3 default
        _log(
            f"WARNING: could not write AWS config for S3 addressing_style ({exc}); "
            f"Ray object-spill may hit PathStyleRequestNotAllowed on CoreWeave R2."
        )


def ensure_fr_dump_dir() -> None:
    """``mkdir -p`` the NCCL flight-recorder dump directory on THIS node before torch init.

    The FR dump path is set via ``TORCH_NCCL_DUMP_ON_TIMEOUT=1`` +
    ``TORCH_NCCL_DEBUG_INFO_TEMP_FILE`` / ``TORCH_FR_DUMP_TEMP_FILE`` (a per-rank
    FILENAME PREFIX; torch appends the global rank). torch's ``DebugInfoWriter``
    is a plain ``std::ofstream`` that does NOT ``mkdir -p`` its parent, so on a
    collective timeout every rank's dump silently fails to write.

    This controller runs on EVERY node in the pre-Ray phase (before any
    torch/NCCL init), so creating the dir here once per node guarantees it exists.
    The dir is DERIVED from the cvar (``dirname`` of the dump-path prefix), so it
    is robust to the path changing per-job. Best-effort: a failure here must never
    block bring-up.
    """
    # TORCH_FR_DUMP_TEMP_FILE is the newer alias torch checks first;
    # TORCH_NCCL_DEBUG_INFO_TEMP_FILE is the canonical name. The value is a per-rank
    # FILENAME PREFIX (torch appends the global rank), so the dir to create is its dirname.
    dump_prefix = os.environ.get(FR_DUMP_TEMP_FILE_ENV) or os.environ.get(NCCL_DEBUG_INFO_TEMP_FILE_ENV)
    if not dump_prefix:
        _log("FR dump dir: no TORCH_(FR_DUMP|NCCL_DEBUG_INFO)_TEMP_FILE set; nothing to create.")
        return
    dump_dir = os.path.dirname(dump_prefix)
    if not dump_dir:
        # Bare filename with no directory component (e.g. the generic "/tmp/nccl_fr_rank"
        # has dirname "/tmp", which exists; a relative bare name has no dir to make).
        return
    try:
        os.makedirs(dump_dir, exist_ok=True)
        _log(f"FR dump dir ensured (mkdir -p): {dump_dir} (from cvar prefix {dump_prefix!r})")
    except OSError as exc:  # noqa: BLE001 - best-effort; never block bring-up on instrumentation
        _log(
            f"WARNING: could not create FR dump dir {dump_dir} ({exc}); "
            f"NCCL flight-recorder dumps may fail to write on a collective timeout."
        )


def _rendezvous_uri(rendezvous_dir: str) -> str:
    return f"{rendezvous_dir.rstrip('/')}/{RENDEZVOUS_FILENAME}"


def _done_uri(rendezvous_dir: str) -> str:
    return f"{rendezvous_dir.rstrip('/')}/{DONE_FILENAME}"


def _runtime_versions() -> tuple[str, str]:
    # Preserve this module's bootstrap fast path; Ray is only needed once the
    # head or worker reaches cluster rendezvous.
    import ray  # noqa: PLC0415

    python_version = ".".join(str(component) for component in sys.version_info[:3])
    return python_version, ray.__version__


def validate_rendezvous_runtime(
    payload: RendezvousPayload,
    *,
    worker_node: str,
    python_version: str,
    ray_version: str,
) -> RendezvousPayload:
    """Reject a worker whose Python or Ray version differs from the head."""
    head_python_version = payload.python_version
    head_ray_version = payload.ray_version
    if head_python_version == python_version and head_ray_version == ray_version:
        return payload

    raise RuntimeError(
        "Iris runtime version mismatch before Ray join: "
        f"head {payload.head_node} uses Python {head_python_version} and Ray {head_ray_version}; "
        f"worker {worker_node} uses Python {python_version} and Ray {ray_version}"
    )


def _write_rendezvous_once(fs, path: str, payload: dict[str, object]) -> None:
    with fs.open(path, "w") as f:
        json.dump(payload, f)


def write_rendezvous(rendezvous_dir: str, head_ip: str, ray_port: int) -> None:
    uri = _rendezvous_uri(rendezvous_dir)
    python_version, ray_version = _runtime_versions()
    payload = RendezvousPayload(
        head_ip=head_ip,
        head_node=socket.gethostname(),
        port=ray_port,
        num_tasks=_num_tasks(),
        python_version=python_version,
        ray_version=ray_version,
        written_at=time.time(),
    )
    serialized_payload = asdict(payload)
    fs, path = fs_and_path(uri)
    # Bound the object-store PutObject with a hard per-attempt timeout via a DAEMON
    # thread + join(timeout) + bounded retries/backoff. An unbounded s3fs/fsspec put
    # has no connect/read timeout and hangs the head invisibly. A daemon thread (NOT a
    # ThreadPoolExecutor) is used because a non-daemon executor worker thread is JOINED
    # by Python's atexit `_python_exit` forever at interpreter shutdown if abandoned,
    # leaving a zombie process; a daemon thread is not joined, so a wedged write cannot
    # block process exit.
    last_exc: BaseException | None = None
    for attempt in range(1, RENDEZVOUS_WRITE_ATTEMPTS + 1):
        t0 = time.time()
        _log(
            f"Writing rendezvous {uri} (attempt {attempt}/{RENDEZVOUS_WRITE_ATTEMPTS}, "
            f"per-attempt timeout {RENDEZVOUS_WRITE_TIMEOUT}s)..."
        )
        result_box: dict = {}

        def _target() -> None:
            try:
                _write_rendezvous_once(fs, path, serialized_payload)
                result_box["ok"] = True
            except BaseException as exc:  # noqa: BLE001 - surface to the joiner
                result_box["exc"] = exc

        writer = threading.Thread(target=_target, name=f"rendezvous-write-{attempt}", daemon=True)
        writer.start()
        writer.join(timeout=RENDEZVOUS_WRITE_TIMEOUT)
        if writer.is_alive():
            # STALLED: the put is still running past the timeout. Abandon the daemon
            # thread (never joined at exit, so it cannot wedge process teardown) and
            # fall through to retry / fail-fast.
            last_exc = TimeoutError(f"rendezvous PutObject did not complete within {RENDEZVOUS_WRITE_TIMEOUT}s")
            _log(
                f"Rendezvous write STALLED — timed out after {time.time() - t0:.1f}s "
                f"(attempt {attempt}/{RENDEZVOUS_WRITE_ATTEMPTS}); object-store PutObject "
                f"to {uri} is not completing."
            )
        elif "exc" in result_box:
            last_exc = result_box["exc"]
            _log(
                f"Rendezvous write FAILED after {time.time() - t0:.1f}s "
                f"(attempt {attempt}/{RENDEZVOUS_WRITE_ATTEMPTS}): {last_exc!r}"
            )
        else:
            _log(
                f"Wrote rendezvous {uri}: head_ip={head_ip} port={ray_port} "
                f"(attempt {attempt}, {time.time() - t0:.1f}s)"
            )
            return
        if attempt < RENDEZVOUS_WRITE_ATTEMPTS:
            backoff = min(2 ** (attempt - 1), 10)
            _log(f"Retrying rendezvous write in {backoff}s...")
            time.sleep(backoff)
    raise RuntimeError(
        f"Rank-0 failed to publish the rendezvous to {uri} after "
        f"{RENDEZVOUS_WRITE_ATTEMPTS} attempts (last error: {last_exc!r}). Failing "
        f"fast so the gang aborts with a clear cause instead of hanging to the worker "
        f"rendezvous deadline."
    ) from last_exc


def poll_rendezvous(
    rendezvous_dir: str,
    timeout: int,
    min_written_at: float | None = None,
) -> RendezvousPayload:
    """Poll for the head's rendezvous file. Returns its parsed payload.

    Payloads with ``written_at`` older than ``min_written_at`` (minus slack) are
    treated as stale (from a prior iris task attempt) and ignored.
    """
    uri = _rendezvous_uri(rendezvous_dir)
    fs, path = fs_and_path(uri)
    deadline = time.time() + timeout
    threshold = (min_written_at - RENDEZVOUS_FRESHNESS_SLACK) if min_written_at else None
    _log(f"Polling for rendezvous {uri} (timeout {timeout}s)...")
    while time.time() < deadline:
        try:
            if fs.exists(path):
                with fs.open(path, "r") as f:
                    payload = RendezvousPayload.from_dict(json.load(f))
                if threshold is not None and payload.written_at < threshold:
                    _log(
                        f"Ignoring stale rendezvous (written_at={payload.written_at:.0f} "
                        f"< threshold={threshold:.0f}); waiting for rank-0 rewrite."
                    )
                else:
                    _log(f"Found rendezvous: {asdict(payload)}")
                    return payload
        except RuntimeError:
            raise
        except Exception as exc:  # transient object-store hiccup
            _log(f"rendezvous poll error (will retry): {exc}")
        time.sleep(POLL_INTERVAL)
    raise TimeoutError(
        f"Worker rank {_rank()} timed out after {timeout}s waiting for "
        f"rank-0 rendezvous at {uri}. Did the head task fail to start?"
    )


def _set_marker(rendezvous_dir: str, name: str) -> None:
    uri = f"{rendezvous_dir.rstrip('/')}/{name}"
    try:
        fs, path = fs_and_path(uri)
        with fs.open(path, "w") as f:
            f.write(str(time.time()))
    except Exception as exc:
        _log(f"Warning: could not write marker {uri}: {exc}")


def _marker_exists(rendezvous_dir: str, name: str, min_written_at: float | None = None) -> bool:
    uri = f"{rendezvous_dir.rstrip('/')}/{name}"
    try:
        fs, path = fs_and_path(uri)
        if not fs.exists(path):
            return False
        if min_written_at is None:
            return True
        with fs.open(path, "r") as f:
            written_at = float(f.read().strip() or 0)
        return written_at >= (min_written_at - RENDEZVOUS_FRESHNESS_SLACK)
    except Exception:
        return False


def clear_rendezvous(rendezvous_dir: str) -> None:
    """Best-effort delete of the rendezvous + done markers (rank 0, on entry/exit)."""
    for name in (RENDEZVOUS_FILENAME, DONE_FILENAME):
        uri = f"{rendezvous_dir.rstrip('/')}/{name}"
        try:
            fs, path = fs_and_path(uri)
            if fs.exists(path):
                fs.rm(path)
                _log(f"Removed {uri}")
        except Exception as exc:
            _log(f"Warning: could not remove {uri}: {exc}")


# ---------------------------------------------------------------------------
# Ray cluster bootstrap (mirrors start_vllm_iris_controller).
# ---------------------------------------------------------------------------


# Ray port allocation on cw-us-east-02a — PIN every named system port OUTSIDE the
# worker_ports range so Ray's own randomized agent ports can never collide with it.
#
# Ray assigns several system components (metrics_export, runtime_env_agent,
# dashboard_agent_grpc, dashboard_agent_listen, node/object_manager) by picking a
# RANDOM free port from the same ephemeral zone as the default worker_ports range
# (10002–19999), and Ray's pre-start validation aborts the node when a random agent
# port lands inside worker_ports:
#   ValueError: Ray component worker_ports is trying to use a port number <N>
#   that is used by other components.
# Shifting worker_ports does NOT help — the random agent ports follow. The fix:
# keep worker_ports at Ray's DEFAULT 10002–19999 and PIN every agent port Ray would
# otherwise randomize to a fixed value in the low 8xxx band — OUTSIDE 10002–19999 and
# distinct from gcs(6379)/dashboard(8265)/client_server(10001). Applied on BOTH head
# and worker.
RAY_METRICS_EXPORT_PORT = 8090
RAY_RUNTIME_ENV_AGENT_PORT = 8092
RAY_DASHBOARD_AGENT_GRPC_PORT = 8093
RAY_DASHBOARD_AGENT_LISTEN_PORT = 8094
RAY_NODE_MANAGER_PORT = 8076
RAY_OBJECT_MANAGER_PORT = 8077


def _ray_port_flags() -> list[str]:
    """Ray port flags shared by head + worker: pin EVERY named system port that Ray
    would otherwise randomize to a fixed value OUTSIDE the default worker_ports range
    (10002–19999), so no random agent port can ever collide with worker_ports (see the
    collision note above). worker_ports is left at Ray's default."""
    return [
        f"--metrics-export-port={RAY_METRICS_EXPORT_PORT}",
        f"--runtime-env-agent-port={RAY_RUNTIME_ENV_AGENT_PORT}",
        f"--dashboard-agent-grpc-port={RAY_DASHBOARD_AGENT_GRPC_PORT}",
        f"--dashboard-agent-listen-port={RAY_DASHBOARD_AGENT_LISTEN_PORT}",
        f"--node-manager-port={RAY_NODE_MANAGER_PORT}",
        f"--object-manager-port={RAY_OBJECT_MANAGER_PORT}",
    ]


# --- Ray cgroup-aware memory ----------------------------------------------------
# In a memory-cgroup-limited pod, Ray can read the HOST's physical RAM (~2 TB via
# /proc/meminfo) instead of the --memory cgroup limit, and size its plasma object
# store at the default ~30% of that (~600 GB). On top of FSDP `cpu_offload`'s
# params+optimizer (also host RAM, OUTSIDE Ray's accounting) + the first training
# step's activations, that overran the container limit -> the OOM killer SIGKILLed
# an FSDP worker. Fix: read the container's cgroup limit and pass it to `ray start`
# as --memory, and BOUND the plasma store so it can't balloon off a misread host
# figure — leaving the bulk of host RAM for cpu_offload.
RAY_OBJECT_STORE_CAP_GIB = 96  # bounded plasma; default can be ~30% of *detected* RAM (huge if host is misread)
# Per-job override (env) for the plasma cap, in GiB. Default = RAY_OBJECT_STORE_CAP_GIB
# (96). The min(., cgroup//8) OOM guard below still applies as the HARD ceiling.
_RAY_STORE_CAP_ENV = "OT_AGENT_RAY_OBJECT_STORE_CAP_GIB"


def _cgroup_mem_limit_bytes() -> int | None:
    """The container's memory cgroup limit in bytes (cgroup v2 then v1), or None if
    unreadable / unlimited (so callers fall back to Ray's own detection for --memory)."""
    for path in (
        "/sys/fs/cgroup/memory.max",  # cgroup v2
        "/sys/fs/cgroup/memory/memory.limit_in_bytes",
    ):  # cgroup v1
        try:
            raw = open(path).read().strip()
        except OSError:
            continue
        if raw in ("max", ""):  # v2 unlimited
            return None
        try:
            v = int(raw)
        except ValueError:
            continue
        if v <= 0 or v > (1 << 62):  # v1 "unlimited" sentinel is a near-2^63 value
            return None
        return v
    return None


def _ray_mem_flags() -> list[str]:
    """Make `ray start` cgroup-aware (see the block comment above). Always bounds the
    plasma object store; additionally pins --memory to the cgroup limit when readable."""
    try:
        cap_gib = float(os.environ.get(_RAY_STORE_CAP_ENV, RAY_OBJECT_STORE_CAP_GIB))
    except ValueError:
        _log(
            f"Ray cgroup-aware: {_RAY_STORE_CAP_ENV}={os.environ.get(_RAY_STORE_CAP_ENV)!r} "
            f"not a number; falling back to default {RAY_OBJECT_STORE_CAP_GIB}GiB"
        )
        cap_gib = float(RAY_OBJECT_STORE_CAP_GIB)
    store_cap = int(cap_gib * (1 << 30))
    if cap_gib != RAY_OBJECT_STORE_CAP_GIB:
        _log(f"Ray cgroup-aware: plasma cap overridden via {_RAY_STORE_CAP_ENV} -> ~{cap_gib:.0f}GiB (pre-guard)")
    limit = _cgroup_mem_limit_bytes()
    flags: list[str] = []
    if limit:
        store_cap = min(store_cap, limit // 8)  # never let the store exceed ~1/8 of the container
        flags.append(f"--memory={limit}")
        _log(
            f"Ray cgroup-aware: --memory={limit} (~{limit / (1 << 30):.0f}GiB cgroup limit), "
            f"--object-store-memory={store_cap} (~{store_cap / (1 << 30):.0f}GiB plasma cap)"
        )
    else:
        _log(
            f"Ray cgroup-aware: no cgroup mem limit readable; bounding "
            f"--object-store-memory={store_cap} (~{store_cap / (1 << 30):.0f}GiB) only"
        )
    flags.append(f"--object-store-memory={store_cap}")
    return flags


def ray_start_head(
    head_ip: str,
    ray_port: int,
    spill_target: RaySpillTarget,
) -> None:
    spill_target.prepare_node()
    cmd = [
        _ray_bin(),
        "start",
        "--head",
        f"--node-ip-address={head_ip}",
        f"--port={ray_port}",
        "--dashboard-host=0.0.0.0",
        *_ray_port_flags(),
        *_ray_mem_flags(),
        *spill_target.head_flags(),
    ]
    _log(spill_target.description())
    _log(f"Starting Ray HEAD: {' '.join(cmd)}")
    t0 = time.time()
    subprocess.run(cmd, check=True, timeout=RAY_START_HEAD_TIMEOUT)
    _log(f"Ray HEAD subprocess returned (exit 0) in {time.time() - t0:.1f}s")


def ray_start_worker(
    head_ip: str,
    ray_port: int,
    node_ip: str,
    spill_target: RaySpillTarget,
) -> None:
    spill_target.prepare_node()
    cmd = [
        _ray_bin(),
        "start",
        f"--address={head_ip}:{ray_port}",
        f"--node-ip-address={node_ip}",
        *_ray_port_flags(),
        *_ray_mem_flags(),
        *spill_target.worker_flags(),
    ]
    _log(spill_target.description())
    _log(f"Starting Ray WORKER: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


def ray_stop() -> None:
    try:
        subprocess.run([_ray_bin(), "stop", "--force"], check=False, timeout=60)
    except subprocess.TimeoutExpired:
        _log("Warning: 'ray stop' timed out")


def wait_for_nodes(ray_address: str, expected_nodes: int, timeout: int, rewrite_cb=None) -> None:
    """Block until the Ray cluster reports ``expected_nodes`` alive nodes.

    ``rewrite_cb`` (head only): a no-arg callable invoked on every poll to RE-PUBLISH
    the rendezvous so its ``written_at`` stays fresh. A worker pod on a cold node can
    start >RENDEZVOUS_FRESHNESS_SLACK (60s) after the head wrote the rendezvous; its
    ``poll_rendezvous(min_written_at=worker_start)`` would then reject the head's
    one-shot rendezvous as "stale" and wait forever while the head waits for that node —
    a mutual deadlock. Rewriting each poll keeps the timestamp ahead of any late worker's
    freshness threshold without weakening the prior-ATTEMPT protection (a stale file from
    a dead PRIOR attempt is still never refreshed).
    """
    import ray

    deadline = time.time() + timeout
    _log(f"Waiting for {expected_nodes} Ray node(s) at {ray_address} (timeout {timeout}s)...")
    ray.init(address=ray_address, ignore_reinit_error=True)
    try:
        last_count = -1
        while time.time() < deadline:
            if rewrite_cb is not None:
                try:
                    rewrite_cb()
                except Exception as exc:
                    _log(f"Warning: rendezvous rewrite failed (will retry): {exc}")
            alive = [n for n in ray.nodes() if n.get("Alive")]
            count = len(alive)
            if count != last_count:
                _log(f"Ray nodes alive: {count}/{expected_nodes}")
                last_count = count
            if count >= expected_nodes:
                _log(f"All {expected_nodes} Ray node(s) joined. Resources: {ray.cluster_resources()}")
                return
            time.sleep(POLL_INTERVAL)
        raise TimeoutError(f"Only {last_count}/{expected_nodes} Ray nodes joined within {timeout}s.")
    finally:
        ray.shutdown()


# ---------------------------------------------------------------------------
# Roles.
# ---------------------------------------------------------------------------


def capture_termination_artifacts(rendezvous_dir: str | None, reason: str) -> None:
    """On teardown, snapshot a FAST diagnostic summary to the rendezvous store
    BEFORE the pod is reaped. Best-effort; never raises; bounded to finish inside
    the k8s grace period.

    An iris/k8s-level termination (ephemeral-storage EVICTION, cgroup OOM, VRAM
    OOM) sends the controller a plain SIGTERM and leaves NOTHING in the iris
    finelog — and the per-node Ray logs are deleted with the pod, so the real
    cause is unrecoverable post-mortem. This persists disk-hogs + GPU mem + df +
    dmesg-OOM, keyed by task id, to ``<rendezvous_dir>/term_artifacts/`` so the
    next probe reads the true cause (disk vs VRAM-OOM vs RAM-OOM).
    """
    if not rendezvous_dir:
        return
    import subprocess as _sp

    task_id = os.environ.get("IRIS_TASK_ID", "unknown").replace("/", "_")
    ts = int(time.time())

    def _run(cmd: str, timeout: int = 7) -> str:
        try:
            return _sp.run(cmd, shell=True, capture_output=True, text=True, timeout=timeout).stdout
        except Exception as exc:  # noqa: BLE001 - best-effort
            return f"<{cmd!r} failed: {exc}>"

    summary = "\n".join(
        [
            f"=== TERMINATION ARTIFACT task={task_id} ts={ts} reason={reason} ===",
            "--- df -h /tmp /dev/shm ---",
            _run("df -h /tmp /dev/shm 2>&1"),
            "--- top /tmp disk hogs (ephemeral-storage eviction cause) ---",
            _run("du -sh /tmp/* /tmp/ray/session*/logs /tmp/ray/session*/*spill* 2>/dev/null | sort -rh | head -25", 9),
            "--- nvidia-smi (VRAM OOM cause) ---",
            _run("nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv 2>&1"),
            "--- top RSS procs (host-RAM OOM cause) ---",
            _run("ps -eo pid,rss,comm --sort=-rss 2>/dev/null | head -12"),
            "--- dmesg OOM/kill tail ---",
            _run("dmesg 2>/dev/null | grep -iE 'oom|killed process|out of memory|Xid' | tail -10"),
        ]
    )
    try:
        uri = f"{rendezvous_dir.rstrip('/')}/term_artifacts/{task_id}_{ts}.txt"
        fs, path = fs_and_path(uri)
        with fs.open(path, "w") as f:
            f.write(summary)
        _log(f"[term-capture] wrote termination artifact -> {uri}")
    except Exception as exc:  # noqa: BLE001 - still emit to finelog as fallback
        _log(f"[term-capture] upload FAILED ({exc}); emitting inline:\n{summary}")


# --- Ray session-log -> object-store sync -----------------------------------------
# The per-actor Ray WORKER logs (/tmp/ray/session_*/logs/worker-*.{out,err},
# raylet.out, ...) are the only place the FSDP policy / rollout actor stdout+tracebacks
# land — the iris finelog aggregates only what reaches the head, and a pod GC / eviction
# DELETES these node-local logs with the pod. This periodically (+ on SIGTERM) uploads
# THIS node's session logs to the object store under the job's rendezvous prefix, keyed
# by node id, reusing the SAME fsspec/boto3 + AWS_ENDPOINT_URL creds path the rendezvous
# / spill / term-artifact writers already use fs_and_path. Per-node: each pod writes
# its own logs under <rendezvous_dir>/ray_session_logs/<node_id>/. Gate:
# OT_AGENT_RAY_LOG_SYNC (default "1"); interval OT_AGENT_RAY_LOG_SYNC_INTERVAL_S (300s).
RAY_LOG_SYNC_MAX_FILE_BYTES = 2 * 1024 * 1024 * 1024  # skip a single >2 GiB log (pathological)
# Ray emits many small log files, so upload time is dominated by per-object latency.
# Keep enough concurrency for a cold short-lived job to finish within the 60-second
# teardown budget while staying far below object-store request-rate limits.
RAY_LOG_SYNC_MAX_WORKERS = 16
RAY_LOG_SYNC_MAX_FAILURE_LOGS = 3
DEBUG_SYNC_MAX_FILE_BYTES = 512 * 1024 * 1024
DEBUG_SYNC_MAX_TOTAL_BYTES = 2 * 1024 * 1024 * 1024


@dataclass(frozen=True)
class RayLogSyncResult:
    uploaded_files: int = 0
    uploaded_bytes: int = 0
    unchanged_files: int = 0
    failed_files: int = 0


class RayLogSyncWaitStatus(StrEnum):
    DISABLED = "disabled"
    SKIPPED = "skipped"
    COMPLETED = "completed"
    TIMED_OUT = "timed_out"


class _RayLogFilesystem(Protocol):
    def put(self, local_path: str, remote_path: str) -> object: ...


@dataclass(frozen=True)
class _RayLogVersion:
    size_bytes: int
    modified_ns: int


@dataclass(frozen=True)
class _RayLogUpload:
    local_path: str
    remote_path: str
    version: _RayLogVersion


@dataclass(frozen=True)
class _RayLogUploadResult:
    upload: _RayLogUpload
    current_version: _RayLogVersion | None
    error: Exception | None


@dataclass(frozen=True)
class _RayLogScanFailure:
    local_path: str
    error: OSError


@dataclass(frozen=True)
class _RayLogUploadPlan:
    uploads: tuple[_RayLogUpload, ...]
    active_remote_paths: frozenset[str]
    unchanged_files: int
    scan_failures: tuple[_RayLogScanFailure, ...]


@dataclass(frozen=True)
class _RayLogSyncSettings:
    enabled: bool
    interval_seconds: int
    final_timeout_seconds: float

    @classmethod
    def from_environment(cls) -> "_RayLogSyncSettings":
        return cls(
            enabled=os.environ.get("OT_AGENT_RAY_LOG_SYNC", "1") == "1",
            interval_seconds=int(os.environ.get("OT_AGENT_RAY_LOG_SYNC_INTERVAL_S", "300")),
            final_timeout_seconds=float(os.environ.get("OT_AGENT_RAY_LOG_FINAL_SYNC_TIMEOUT_S", "60")),
        )


def _ray_log_version(path: str) -> _RayLogVersion:
    """Return the byte size and nanosecond modification time for one local log."""
    stat = os.stat(path)
    return _RayLogVersion(size_bytes=stat.st_size, modified_ns=stat.st_mtime_ns)


def _upload_ray_logs(
    filesystem: _RayLogFilesystem,
    uploads: tuple[_RayLogUpload, ...],
) -> tuple[_RayLogUploadResult, ...]:
    """Return one success or captured exception for each requested upload."""
    if not uploads:
        return ()
    pending: queue.Queue[_RayLogUpload] = queue.Queue()
    completed: queue.Queue[_RayLogUploadResult] = queue.Queue()
    for upload in uploads:
        pending.put(upload)

    def _worker() -> None:
        while True:
            try:
                upload = pending.get_nowait()
            except queue.Empty:
                return
            try:
                filesystem.put(upload.local_path, upload.remote_path)
                current_version = _ray_log_version(upload.local_path)
            except Exception as error:  # noqa: BLE001 - capture per-file I/O failures for the result and log
                completed.put(_RayLogUploadResult(upload=upload, current_version=None, error=error))
            else:
                completed.put(_RayLogUploadResult(upload=upload, current_version=current_version, error=None))

    workers = [
        threading.Thread(target=_worker, daemon=True, name=f"ray-log-upload-{index}")
        for index in range(min(RAY_LOG_SYNC_MAX_WORKERS, len(uploads)))
    ]
    for worker in workers:
        worker.start()
    for worker in workers:
        worker.join()
    return tuple(completed.get_nowait() for _ in uploads)


def _plan_ray_log_uploads(
    log_dirs: list[str],
    destination_path: str,
    uploaded_versions: dict[str, _RayLogVersion],
) -> _RayLogUploadPlan:
    uploads: list[_RayLogUpload] = []
    active_remote_paths: set[str] = set()
    scan_failures: list[_RayLogScanFailure] = []
    unchanged_files = 0
    for log_dir in log_dirs:
        session = os.path.basename(os.path.dirname(log_dir))
        for root, directories, files in os.walk(log_dir):
            directories.sort()
            for filename in sorted(files):
                local_path = os.path.join(root, filename)
                try:
                    version = _ray_log_version(local_path)
                except OSError as error:
                    scan_failures.append(_RayLogScanFailure(local_path=local_path, error=error))
                    continue
                if version.size_bytes > RAY_LOG_SYNC_MAX_FILE_BYTES:
                    continue
                relative_path = os.path.relpath(local_path, log_dir)
                remote_path = f"{destination_path}/{session}/{relative_path}"
                active_remote_paths.add(remote_path)
                if uploaded_versions.get(remote_path) == version:
                    unchanged_files += 1
                    continue
                uploads.append(_RayLogUpload(local_path=local_path, remote_path=remote_path, version=version))
    return _RayLogUploadPlan(
        uploads=tuple(uploads),
        active_remote_paths=frozenset(active_remote_paths),
        unchanged_files=unchanged_files,
        scan_failures=tuple(scan_failures),
    )


def _apply_ray_log_upload_results(
    uploaded_versions: dict[str, _RayLogVersion],
    plan: _RayLogUploadPlan,
    upload_results: tuple[_RayLogUploadResult, ...],
) -> tuple[RayLogSyncResult, tuple[_RayLogUploadResult, ...]]:
    """Prune stale versions, record stable uploads, and return the pass result and failures."""
    for remote_path in tuple(uploaded_versions):
        if remote_path not in plan.active_remote_paths:
            del uploaded_versions[remote_path]
    uploaded_files = uploaded_bytes = 0
    failures: list[_RayLogUploadResult] = []
    for upload_result in upload_results:
        if upload_result.error is not None or upload_result.current_version is None:
            failures.append(upload_result)
            continue
        upload = upload_result.upload
        uploaded_files += 1
        uploaded_bytes += upload.version.size_bytes
        if upload_result.current_version == upload.version:
            uploaded_versions[upload.remote_path] = upload.version
    return (
        RayLogSyncResult(
            uploaded_files=uploaded_files,
            uploaded_bytes=uploaded_bytes,
            unchanged_files=plan.unchanged_files,
            failed_files=len(plan.scan_failures) + len(failures),
        ),
        tuple(failures),
    )


def sync_debug_artifacts(rendezvous_dir: str | None, node_id: str, reason: str) -> None:
    """Boundedly persist this node's managed debug directory beside rendezvous data."""
    source_root = os.environ.get(DEBUG_ARTIFACT_DIR_ENV)
    if not rendezvous_dir or not source_root or not os.path.isdir(source_root):
        return
    destination = f"{rendezvous_dir.rstrip('/')}/debug_artifacts/{node_id}"
    try:
        filesystem, destination_path = fs_and_path(destination)
    except Exception as exc:  # noqa: BLE001 - teardown evidence is best-effort
        _log(f"[debug-sync] cannot resolve {destination} ({exc}) [{reason}]")
        return

    copied: list[dict[str, object]] = []
    skipped: list[dict[str, object]] = []
    total_bytes = 0
    for root, directories, files in os.walk(source_root):
        directories.sort()
        for filename in sorted(files):
            local_path = os.path.join(root, filename)
            relative_path = os.path.relpath(local_path, source_root)
            try:
                size = os.path.getsize(local_path)
                if size > DEBUG_SYNC_MAX_FILE_BYTES or total_bytes + size > DEBUG_SYNC_MAX_TOTAL_BYTES:
                    skipped.append({"path": relative_path, "bytes": size, "reason": "budget"})
                    continue
                filesystem.put(local_path, f"{destination_path}/{relative_path}")
                copied.append({"path": relative_path, "bytes": size})
                total_bytes += size
            except Exception as exc:  # noqa: BLE001 - retain a complete sync receipt
                skipped.append({"path": relative_path, "reason": str(exc)})

    receipt = {
        "schema_version": 1,
        "node_id": node_id,
        "reason": reason,
        "source_root": source_root,
        "copied": copied,
        "skipped": skipped,
        "copied_bytes": total_bytes,
    }
    try:
        with filesystem.open(f"{destination_path}/sync-manifest.json", "w") as manifest:
            json.dump(receipt, manifest, sort_keys=True)
        _log(f"[debug-sync] uploaded {len(copied)} file(s) / {total_bytes} bytes -> {destination} [{reason}]")
    except Exception as exc:  # noqa: BLE001 - teardown evidence is best-effort
        _log(f"[debug-sync] manifest upload failed ({exc}) [{reason}]")


@dataclass
class RayLogSyncSession:
    """Incrementally upload one node's Ray logs without overlapping sync passes."""

    rendezvous_dir: str | None
    node_id: str
    _settings: _RayLogSyncSettings = field(default_factory=_RayLogSyncSettings.from_environment, repr=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)
    _uploaded_versions: dict[str, _RayLogVersion] = field(default_factory=dict, repr=False)

    @property
    def destination(self) -> str | None:
        if not self.rendezvous_dir:
            return None
        return join_resource_path(self.rendezvous_dir, "ray_session_logs", self.node_id)

    def sync(self, reason: str) -> RayLogSyncResult:
        """Upload new or changed files and return per-pass file and byte counts."""
        if not self.destination or not self._settings.enabled:
            return RayLogSyncResult()

        log_dirs = sorted(glob.glob("/tmp/ray/session_*/logs"))
        if not log_dirs:
            return RayLogSyncResult()
        try:
            filesystem, destination_path = fs_and_path(self.destination)
        except Exception as error:  # noqa: BLE001 - best-effort teardown evidence
            _log(f"[ray-log-sync] cannot resolve dest {self.destination} ({error}) [{reason}]")
            return RayLogSyncResult()

        with self._lock:
            plan = _plan_ray_log_uploads(log_dirs, destination_path, self._uploaded_versions)
            upload_results = _upload_ray_logs(filesystem, plan.uploads)
            result, upload_failures = _apply_ray_log_upload_results(self._uploaded_versions, plan, upload_results)

        failure_messages = [
            f"stat failed for {failure.local_path}: {type(failure.error).__name__}: {failure.error}"
            for failure in plan.scan_failures
        ]
        failure_messages.extend(
            f"upload failed for {failure.upload.local_path}: {type(failure.error).__name__}: {failure.error}"
            for failure in upload_failures
        )
        for failure_message in failure_messages[:RAY_LOG_SYNC_MAX_FAILURE_LOGS]:
            _log(f"[ray-log-sync] {failure_message} [{reason}]")
        if len(failure_messages) > RAY_LOG_SYNC_MAX_FAILURE_LOGS:
            additional_failures = len(failure_messages) - RAY_LOG_SYNC_MAX_FAILURE_LOGS
            _log(f"[ray-log-sync] {additional_failures} additional sync failure(s) [{reason}]")
        _log(
            f"[ray-log-sync] uploaded {result.uploaded_files} file(s) / "
            f"{result.uploaded_bytes / 1073741824.0:.2f} GiB "
            f"(unchanged={result.unchanged_files}, failed={result.failed_files}) -> {self.destination} [{reason}]"
        )
        return result

    def sync_bounded(self, reason: str) -> RayLogSyncWaitStatus:
        """Wait at most the configured teardown budget and report the terminal wait status."""
        if not self.destination or not self._settings.enabled:
            return RayLogSyncWaitStatus.DISABLED
        timeout = self._settings.final_timeout_seconds
        if timeout <= 0:
            _log(f"[ray-log-sync] skipping final upload because timeout is {timeout}s [{reason}]")
            return RayLogSyncWaitStatus.SKIPPED

        sync_thread = threading.Thread(
            target=self.sync,
            args=(reason,),
            daemon=True,
            name="ray-log-final-sync",
        )
        sync_thread.start()
        sync_thread.join(timeout)
        if sync_thread.is_alive():
            _log(
                f"[ray-log-sync] final upload exceeded {timeout}s; continuing teardown with a partial upload [{reason}]"
            )
            return RayLogSyncWaitStatus.TIMED_OUT
        return RayLogSyncWaitStatus.COMPLETED

    def start_periodic(self) -> threading.Event:
        """Start periodic incremental uploads and return the event that stops new passes."""
        stop = threading.Event()
        if not self.destination or not self._settings.enabled:
            return stop
        interval = self._settings.interval_seconds
        if interval <= 0:
            return stop

        def _loop() -> None:
            if stop.wait(min(60, interval)):
                return
            self.sync("periodic")
            sync_debug_artifacts(self.rendezvous_dir, self.node_id, "periodic")
            while not stop.wait(interval):
                self.sync("periodic")
                sync_debug_artifacts(self.rendezvous_dir, self.node_id, "periodic")

        threading.Thread(target=_loop, daemon=True, name="ray-log-sync").start()
        _log(f"[ray-log-sync] started (every {interval}s, first ~{min(60, interval)}s) -> {self.destination}")
        return stop


def launch_training_driver(train_argv: list[str], env: dict[str, str]) -> subprocess.Popen:
    """Start the driver from the immutable runtime checkout, not the bootstrap bundle."""
    runtime_checkout = env.get("SKYRL_HOME")
    if not runtime_checkout:
        raise RuntimeError("SKYRL_HOME must identify the immutable MarinSkyRL runtime checkout")
    return subprocess.Popen(train_argv, env=env, cwd=runtime_checkout, start_new_session=True)


def run_head(args: argparse.Namespace, train_argv: list[str], derived_gloo_ifname: str | None = None) -> int:
    num_tasks = _num_tasks()
    head_ip = _own_ip()
    ray_port = args.ray_port
    ray_address = f"{head_ip}:{ray_port}"
    node_id = f"rank0-{socket.gethostname()}"
    _log(f"ROLE=head rank=0/{num_tasks} head_ip={head_ip} ray_port={ray_port}")
    ray_log_sync_stop: threading.Event | None = None
    ray_log_sync = RayLogSyncSession(args.rendezvous_dir, node_id)

    # Install the SIGTERM/SIGINT handler + termination-artifact capture at the TOP of
    # bring-up (BEFORE clear_rendezvous / ray_start_head / rendezvous write), so a reap
    # ANYWHERE in bring-up produces the term artifact instead of dying silently.
    # `process` is None during bring-up; the handler skips the driver kill until the
    # training driver is launched below (closure reads the current value).
    process = None

    def _shutdown(signum, _frame) -> None:
        _log(f"Received signal {signum}; terminating training driver and stopping Ray...")
        # Capture FIRST (before teardown mutates disk/GPU state) — a SIGTERM here is
        # often a k8s eviction / OOM whose cause survives nowhere else.
        capture_termination_artifacts(args.rendezvous_dir, f"signal {signum} (head rank 0)")
        # Flush this node's Ray session logs (per-actor worker stdout/tracebacks) before
        # the pod is reaped — Ray's node-local logs are deleted with the pod.
        ray_log_sync.sync_bounded(f"signal {signum} (head)")
        sync_debug_artifacts(args.rendezvous_dir, node_id, f"signal {signum} (head)")
        if process is not None:
            try:
                os.killpg(os.getpgid(process.pid), signal.SIGTERM)
                process.wait(timeout=60)
            except (subprocess.TimeoutExpired, ProcessLookupError):
                try:
                    os.killpg(os.getpgid(process.pid), signal.SIGKILL)
                except ProcessLookupError:
                    pass
        if args.rendezvous_dir and num_tasks > 1:
            _set_marker(args.rendezvous_dir, DONE_FILENAME)
        ray_stop()
        sys.exit(0)

    signal.signal(signal.SIGINT, _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    # On iris task retry, a rendezvous file from a previous attempt still points
    # at a now-dead head. Purge before starting the new head.
    if num_tasks > 1 and args.rendezvous_dir:
        clear_rendezvous(args.rendezvous_dir)

    ray_start_head(
        head_ip,
        ray_port,
        resolve_ray_spill_target(args.rendezvous_dir, args.ray_spill_backend, args.ray_spill_dir),
    )
    _log("Ray head bootstrap complete; entering rendezvous / cluster-join phase.")

    # Start the periodic Ray session-log -> object-store sync now that the session dir
    # exists (per-node, keyed by node id, under the job's rendezvous prefix).
    ray_log_sync_stop = ray_log_sync.start_periodic()

    if num_tasks > 1:
        if not args.rendezvous_dir:
            raise ValueError(
                "Multi-node iris slice (IRIS_NUM_TASKS>1) requires --rendezvous-dir "
                "(or OT_AGENT_IRIS_RENDEZVOUS_DIR) so worker ranks can find the head IP."
            )
        _log(
            f"[task-runtime] Ray head subprocess returned; writing rendezvous -> {_rendezvous_uri(args.rendezvous_dir)}"
        )
        write_rendezvous(args.rendezvous_dir, head_ip, ray_port)
        # Re-publish the rendezvous each poll so a late cold-node worker never sees it
        # as "stale" (see wait_for_nodes docstring — prevents the freshness deadlock).
        wait_for_nodes(
            ray_address,
            num_tasks,
            args.cluster_join_timeout,
            rewrite_cb=lambda: write_rendezvous(args.rendezvous_dir, head_ip, ray_port),
        )
    else:
        _log("Single-node slice: skipping rendezvous and multi-node wait.")

    with ray_metrics_telemetry(head_ip, RAY_METRICS_EXPORT_PORT):
        env = training_driver_env(derived_gloo_ifname)
        env["RAY_ADDRESS"] = ray_address  # skyrl-train's bare ray.init() attaches here
        env["PYTHONUNBUFFERED"] = "1"

        _log("Launching MarinSkyRL training driver:")
        _log("  " + " ".join(train_argv))
        sys.stdout.flush()
        sys.stderr.flush()

        # The SIGTERM/SIGINT handler is already installed at the top of run_head; assigning
        # `process` here arms its driver-teardown path (the closure reads this value).
        process = launch_training_driver(train_argv, env)

        exit_code = process.wait()
        if exit_code != 0:
            capture_termination_artifacts(args.rendezvous_dir, f"driver exit_code={exit_code} (head rank 0)")
        # Final flush of this node's Ray session logs before teardown reaps them.
        if ray_log_sync_stop is not None:
            ray_log_sync_stop.set()
        ray_log_sync.sync_bounded(f"driver exit_code={exit_code} (head)")
        sync_debug_artifacts(args.rendezvous_dir, node_id, f"driver exit_code={exit_code} (head)")
    # Signal workers to unpark, then tear down.
    if args.rendezvous_dir and num_tasks > 1:
        _set_marker(args.rendezvous_dir, DONE_FILENAME)
    ray_stop()
    if args.rendezvous_dir and num_tasks > 1:
        clear_rendezvous(args.rendezvous_dir)
    return exit_code


def run_worker(args: argparse.Namespace) -> int:
    worker_start = time.time()
    rank = _rank()
    num_tasks = _num_tasks()
    node_ip = _own_ip()
    node_id = f"rank{rank}-{socket.gethostname()}"
    _log(f"ROLE=worker rank={rank}/{num_tasks} node_ip={node_ip}")
    ray_log_sync = RayLogSyncSession(args.rendezvous_dir, node_id)

    if not args.rendezvous_dir:
        raise ValueError(
            "Worker rank requires --rendezvous-dir (or OT_AGENT_IRIS_RENDEZVOUS_DIR) to discover the head IP."
        )

    payload = poll_rendezvous(args.rendezvous_dir, args.rendezvous_timeout, min_written_at=worker_start)
    python_version, ray_version = _runtime_versions()
    payload = validate_rendezvous_runtime(
        payload,
        worker_node=node_id,
        python_version=python_version,
        ray_version=ray_version,
    )
    head_ip = payload.head_ip
    ray_port = payload.port
    ray_address = f"{head_ip}:{ray_port}"

    ray_start_worker(
        head_ip,
        ray_port,
        node_ip,
        resolve_ray_spill_target(args.rendezvous_dir, args.ray_spill_backend, args.ray_spill_dir),
    )
    wait_for_nodes(ray_address, num_tasks, args.cluster_join_timeout)
    _log(f"Worker rank {rank} joined Ray cluster at {ray_address}; parking until the head finishes.")

    # Periodic Ray session-log -> object-store sync for THIS worker node (the FSDP/rollout
    # actors on this node log to its local /tmp/ray session, deleted with the pod on GC).
    ray_log_sync_stop = ray_log_sync.start_periodic()

    stop = threading.Event()

    def _shutdown(signum, _frame) -> None:
        _log(f"Worker rank {rank} received signal {signum}; stopping Ray.")
        # A SIGTERM on a worker node is often a k8s eviction/OOM of that node (it
        # hosts the training actors' GPUs); capture its disk/GPU state before reap.
        capture_termination_artifacts(args.rendezvous_dir, f"signal {signum} (worker rank {rank})")
        # Flush this node's per-actor Ray worker logs before the pod is reaped.
        ray_log_sync.sync_bounded(f"signal {signum} (worker rank {rank})")
        sync_debug_artifacts(args.rendezvous_dir, node_id, f"signal {signum} (worker rank {rank})")
        stop.set()

    signal.signal(signal.SIGINT, _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    # Block until the head publishes the done marker (training finished) or we
    # are signalled. The training driver on rank 0 schedules actors onto this
    # node's GPUs; this process just keeps the Ray node alive.
    with ray_metrics_telemetry(node_ip, RAY_METRICS_EXPORT_PORT):
        while not stop.is_set():
            if _marker_exists(args.rendezvous_dir, DONE_FILENAME, min_written_at=worker_start):
                _log(f"Worker rank {rank} saw head done-marker; shutting down.")
                break
            time.sleep(POLL_INTERVAL)
        # Final flush of this worker node's Ray session logs before Ray teardown.
        ray_log_sync_stop.set()
        ray_log_sync.sync_bounded(f"worker rank {rank} teardown")
        sync_debug_artifacts(args.rendezvous_dir, node_id, f"worker rank {rank} teardown")
    ray_stop()
    return 0


def parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(
        description="Bootstrap one cross-node Ray cluster on an iris GPU slice and run "
        "the MarinSkyRL training driver on rank 0. Everything after `--` is the "
        "training command (e.g. `python -m skyrl_train.entrypoints.main_base <hydra args>`).",
    )
    parser.add_argument(
        "--ray-port",
        type=int,
        default=int(os.environ.get("OT_AGENT_IRIS_RAY_PORT", "6379")),
        help="Port the Ray head binds (default 6379).",
    )
    parser.add_argument(
        "--ray-spill-dir",
        type=validate_ray_spill_dir,
        default=DEFAULT_RAY_SPILL_DIR,
        help=f"Node-local Ray object-spill directory for the local backend (default {DEFAULT_RAY_SPILL_DIR}).",
    )
    parser.add_argument(
        "--ray-spill-backend",
        type=RaySpillBackend,
        choices=list(RaySpillBackend),
        default=RaySpillBackend.LOCAL,
        help="Ray object-spill backend (default local; r2 requires an s3:// rendezvous directory).",
    )
    parser.add_argument(
        "--rendezvous-dir",
        default=os.environ.get("OT_AGENT_IRIS_RENDEZVOUS_DIR"),
        help="Shared object-store/dir for the head/worker rendezvous (gs://, s3://, "
        "or a shared path). Defaults to $OT_AGENT_IRIS_RENDEZVOUS_DIR.",
    )
    parser.add_argument(
        "--rendezvous-timeout",
        type=int,
        default=DEFAULT_RENDEZVOUS_TIMEOUT,
        help=f"Seconds workers poll for the head rendezvous (default {DEFAULT_RENDEZVOUS_TIMEOUT}).",
    )
    parser.add_argument(
        "--cluster-join-timeout",
        type=int,
        default=DEFAULT_CLUSTER_JOIN_TIMEOUT,
        help=f"Seconds to wait for all nodes to join the Ray cluster (default {DEFAULT_CLUSTER_JOIN_TIMEOUT}).",
    )
    parser.add_argument(
        "--train-data",
        default=os.environ.get("OT_AGENT_IRIS_TRAIN_DATA", ""),
        help="JSON list of train_data HF dataset(s) to stage (extract to the node-local "
        "task dir) on EVERY node before Ray starts. Required for agentic terminal_bench "
        "rollouts on a multi-node slice with no shared filesystem.",
    )
    parser.add_argument(
        "--data-sources-json",
        default="",
        help="Immutable object-store data locators to materialize before Ray starts.",
    )
    parser.add_argument(
        "--prestage-model",
        default=os.environ.get("OT_AGENT_IRIS_PRESTAGE_MODEL", ""),
        help="HF repo ID of the policy model to pre-download into the node-local HF "
        "cache on EVERY node before Ray starts. Object-store URIs are unsupported; "
        "use --model-warm-source for an S3 mirror. Set by the launcher when the config "
        "runs HF_HUB_OFFLINE=1, so the FSDP ranks load from a warm node-local cache "
        "instead of each racing HF Hub at init.",
    )
    parser.add_argument(
        "--model-warm-source",
        default=os.environ.get("OT_AGENT_MODEL_WARM_SOURCE", ""),
        help="Optional in-region CW-object-store prefix (s3://marin-us-east-02a/models/"
        "<org>--<name>) that a one-time seed job (scripts/iris/mirror_hf_to_s3.py) has "
        "populated with the model weights. When set AND --prestage-model is set, the "
        "controller SYNCS the weights from here into the node-local HF cache (fast, "
        "in-datacenter) instead of pulling them from HF Hub. Missing/empty/incomplete "
        "source -> clean fallback to the HF snapshot_download prestage. Set by the "
        "launcher (auto-derived from the repo id).",
    )
    parser.add_argument(
        "--model-source-uri",
        default="",
        help="Object-store HF export to materialize on every node before Ray starts.",
    )
    parser.add_argument(
        "--model-local-path",
        default="",
        help="Task-local model directory, either pre-existing or populated from --model-source-uri.",
    )
    parser.add_argument(
        "--model-source-identity",
        default="",
        help="Immutable producer identity recorded beside the staged export.",
    )
    parser.add_argument(
        "--policy-chat-template",
        default=os.environ.get("OT_AGENT_IRIS_POLICY_CHAT_TEMPLATE", ""),
        help="Repo-relative path to a chat-template jinja to FORCE onto the policy "
        "tokenizer's cached tokenizer_config.json + chat_template.jinja on EVERY node "
        "before Ray (delphi single-turn RLVR: the SFT repo ships no template). Requires "
        "--prestage-model or --model-local-path. Empty disables the override.",
    )
    args, train_argv = parser.parse_known_args()
    # argparse leaves the `--` separator out of train_argv; strip a leading one
    # if the shell passed it through.
    if train_argv and train_argv[0] == "--":
        train_argv = train_argv[1:]
    if not train_argv:
        parser.error("No training command given. Pass it after `--`.")
    return args, train_argv


def _print_env_snapshot() -> None:
    _log("environment snapshot:")
    for key in (
        "IRIS_TASK_ID",
        "IRIS_NUM_TASKS",
        "IRIS_ADVERTISE_HOST",
        "RAY_ADDRESS",
        "SKYRL_HOME",
        "PYTHONPATH",
        "HF_HOME",
        "NUM_INFERENCE_ENGINES",
        "POLICY_NUM_NODES",
        "TENSOR_PARALLEL_SIZE",
    ):
        print(f"  {key}={os.environ.get(key, '<unset>')}", flush=True)


def main() -> None:
    validate_bundled_runtime()
    args, train_argv = parse_args()
    _print_env_snapshot()
    # Pin virtual-hosted S3 addressing for the boto3 path (Ray object-spill IO workers)
    # BEFORE any `ray start`, on head + every worker — CoreWeave R2 rejects path-style.
    _pin_boto3_s3_addressing_style()
    # Derive GLOO_SOCKET_IFNAME from the pod IP BEFORE any torch/gloo init, on head
    # and every worker. Must precede `ray start`: Ray actors inherit this env, and
    # the gloo mesh is built long after, at the first checkpoint save. The derived
    # name is node-local; training_driver_env keeps it out of the driver, which
    # would otherwise broadcast the head's name to every node.
    derived_gloo_ifname = pin_socket_ifname()
    # Ensure the NCCL flight-recorder dump dir exists on THIS node BEFORE any torch/NCCL
    # init, so a collective-timeout FR dump actually writes. See ensure_fr_dump_dir.
    ensure_fr_dump_dir()
    debug_artifact_root = os.environ.get(DEBUG_ARTIFACT_DIR_ENV)
    if debug_artifact_root:
        ensure_debug_artifact_directories(debug_artifact_root)
    # Stage the task dataset on THIS node before Ray bootstrap (head + every worker).
    # Without this, only rank-0 has the extracted tasks and the rollout workers die
    # with FileNotFoundError on task.toml. See stage_train_data docstring.
    if args.train_data:
        stage_train_data(args.train_data)
    if args.data_sources_json:
        materialize_data_sources(args.data_sources_json)
    if args.model_source_uri:
        if not args.model_local_path or not args.model_source_identity:
            raise ValueError("--model-source-uri requires --model-local-path and --model-source-identity")
        materialize_model_export(args.model_source_uri, args.model_local_path, args.model_source_identity)
    # Pre-download the policy weights into the node-local HF cache BEFORE Ray, so the
    # FSDP ranks load from a warm cache under HF_HUB_OFFLINE=1. See stage_model.
    if args.prestage_model:
        stage_model(args.prestage_model, warm_source=(args.model_warm_source or None))
    # Force the policy chat template onto the staged Hub snapshot or materialized local
    # model on every node before Ray; the training driver's tokenizer may load anywhere.
    if args.policy_chat_template:
        model_path = policy_chat_template_model(args.prestage_model, args.model_local_path)
        apply_policy_chat_template(model_path, args.policy_chat_template)
    rank = _rank()
    if rank == 0:
        exit_code = run_head(args, train_argv, derived_gloo_ifname)
    else:
        exit_code = run_worker(args)
    if exit_code != 0:
        sys.exit(exit_code)


if __name__ == "__main__":
    main()
