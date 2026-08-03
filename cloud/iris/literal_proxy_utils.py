"""literal_proxy_utils.py — co-located RecordProxy wiring for literal-token capture.

Drives harbor's LITERAL-TOKEN trace machinery (``harbor.literal.proxy.RecordProxy``)
from the OT-Agent launch path. When ``--record_literal`` is set, the RL / datagen
launchers co-locate a :class:`~harbor.literal.proxy.RecordProxy` alongside the
on-cluster vLLM server and route the agent's inference endpoint THROUGH the proxy.
The proxy transparently injects ``return_token_ids=True`` / ``logprobs=True`` into
each forwarded completion and appends the returned prompt/completion token IDs +
logprobs to a ``literal.jsonl`` log, which the ``opencode`` agent (and any agent
with ``SUPPORTS_LITERAL_TRACES``) merges back into its trajectory steps.

Default OFF: :func:`maybe_serve_literal_proxy` is a NULL context manager that
yields the upstream vLLM endpoint UNCHANGED and starts no server, so the launch
path is byte-identical to today (parity gate G0).

Contract (from ``harbor/src/harbor/literal/proxy.py``, read 2026-07-03):
  * ``RecordProxy(upstream_base_url, log_path, timeout=600.0)`` — forwards to
    ``upstream_base_url`` by RE-APPENDING the FULL incoming request path
    (``url = upstream_base_url + request.url.path``). So ``upstream_base_url``
    MUST be the vLLM ORIGIN (``http://host:port``), NOT the agent-facing
    ``.../v1`` base: passing the ``/v1`` base doubles the segment
    (``/v1/v1/chat/completions``) and vLLM 404s the request. We strip the
    launcher's ``.../v1`` endpoint to its origin before handing it to RecordProxy.
  * ``RecordProxy.app()`` returns a FastAPI app serving ``/v1/chat/completions``
    and ``/v1/completions`` (the surface harbor's installed agents call). We serve
    that app with uvicorn on a co-located port; the agent's api_base points at it.

Nothing here binds a socket unless ``enabled`` is True — the pure helpers and the
disabled path are fully unit-testable without a server (see tests/hpc/test_literal_proxy.py).
"""

from __future__ import annotations

import os
import re
import tempfile
import threading
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Iterator, Optional
from urllib.parse import urlsplit, urlunsplit

# NOTE (MarinSkyRL port): ``upath`` (universal_pathlib) is imported LAZILY inside the two
# remote-URI helpers below rather than at module scope. Those paths only run on iris where
# ``experiments_dir`` is a ``gs://…`` URI; there ``upath`` is provided by the locked environment
# (a harbor/fsspec dependency), exactly as ``harbor`` itself is. Keeping it lazy leaves the
# module import-clean + unit-testable on a plain checkout (local experiments_dir needs no
# upload) without adding an unresolvable dep to the MarinSkyRL lock. The registration/mint
# mechanism is otherwise byte-identical to the datagen ground truth (hpc/literal_proxy_utils.py).

# The co-located proxy binds this port by default. vLLM/datagen use 8000, so 8010
# avoids a collision while staying on the loopback interface (the proxy only ever
# fronts the LOCAL vLLM; external reach is via the existing pinggy/controller path).
DEFAULT_LITERAL_PROXY_PORT = 8010
DEFAULT_LITERAL_PROXY_HOST = "127.0.0.1"

# Filename harbor's opencode agent reads back from its logs dir
# (``OpenCode._LITERAL_LOG_FILENAME``); we name the co-located log the same for
# symmetry, though the co-located log lives on the CLUSTER (beside the job logs).
LITERAL_LOG_FILENAME = "literal.jsonl"


def _slug(job_name: str) -> str:
    """Filesystem-safe slug for a job name (``.`` preserved for readability)."""
    return re.sub(r"[^A-Za-z0-9._-]+", "-", (job_name or "job")).strip("-.") or "job"


def serve_token() -> str:
    """A per-serve token that distinguishes one serve attempt's literal log from another.

    ``--job_name`` is STABLE across a job's preempt-retries (resume fix a25a6125), so
    every serve attempt derives the SAME ``<slug>`` — and thus, without a per-serve
    qualifier, the SAME remote literal-log path. A low/zero-traffic RESUME serve (e.g.
    one that only re-runs export-push) would then overwrite attempt-0's populated log
    with an empty one. The historical design gave each serve a UNIQUE filename (that is
    why pre-resume-fix jobs left 2-3 ``logs/*_literal.jsonl`` files), and the correlator
    already unions every matched file — so restoring per-serve uniqueness is the fix.

    The token folds in the iris task attempt id when present (``IRIS_TASK_ID`` gains a
    ``:N`` retry suffix on a preempt-retried task) plus a serve-start timestamp, so a
    retry, a fresh relaunch, and a local run each get a DISTINCT
    ``logs/<slug>__<token>_literal.jsonl``. Computed ONCE at serve start and threaded
    through both the local staging path and the remote URI so they stay in sync.
    """
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    task_id = os.environ.get("IRIS_TASK_ID")
    if task_id:
        # "/user/job/0:2" -> "0-2": the rank + retry-attempt leaf, the per-serve id.
        leaf = _slug(task_id.rsplit("/", 1)[-1].replace(":", "-"))
        return f"{stamp}-{leaf}"
    return stamp


def _literal_log_name(job_name: str, token: str) -> str:
    """``<slug>__<token>_literal.jsonl`` — keeps the ``_literal.jsonl`` discovery suffix."""
    return f"{_slug(job_name)}__{_slug(token)}_{LITERAL_LOG_FILENAME}"


def literal_log_path(experiments_dir: str | Path, job_name: str, token: str) -> Path:
    """Deterministic LOCAL path for the co-located RecordProxy's ``literal.jsonl`` log.

    Lives under ``<experiments_dir>/logs/`` beside the other per-job logs. This is
    the path the proxy APPENDS to (harbor's ``RecordProxy`` uses a plain
    ``open(..., "a")``, which only works on a local filesystem). ``token`` (a
    :func:`serve_token`) makes each serve attempt write a DISTINCT file so a resume
    cannot clobber attempt-0's log. When ``experiments_dir`` is itself a remote URI
    (``gs://…``), do NOT collapse it to a junk local path (``Path("gs://…")`` →
    CWD-relative ``gs:/…``): the launch path stages the log locally under the system
    temp dir instead and uploads to the durable remote location via
    :func:`literal_log_remote_uri`. See :func:`maybe_serve_literal_proxy`.
    """
    return Path(experiments_dir) / "logs" / _literal_log_name(job_name, token)


# schemes we treat as durable object stores (upload target) rather than a local FS.
_REMOTE_PROTOCOLS = frozenset({"gs", "gcs", "s3", "az", "abfs", "abfss", "http", "https"})

# A remote URI survives a ``str()`` round-trip, but ``pathlib.Path`` collapses the
# ``//`` after the scheme (``Path("gs://x")`` -> ``"gs:/x"``), which hides the scheme
# from UPath (protocol -> ""). Re-expand a single-slash scheme so a Path-wrapped
# remote dir is still detected as remote. NB: a fully ``resolve()``-d local form
# (``"/app/gs:/x"``) keeps its leading ``/`` and is CORRECTLY treated as local — the
# scheme is unrecoverable once resolve() prepends the cwd, so callers must pass the
# ORIGINAL remote URI in that case (see local_runner_utils._serving_endpoint_meta,
# which passes the raw ``--experiments_dir`` arg rather than get_experiments_dir()'s
# resolved Path).
_COLLAPSED_SCHEME_RE = re.compile(r"^(gs|gcs|s3|az|abfs|abfss|https?):/(?=[^/])")


def _as_remote_uri_str(experiments_dir: str | Path) -> str:
    """Coerce ``experiments_dir`` to a string, re-expanding a Path-collapsed scheme."""
    return _COLLAPSED_SCHEME_RE.sub(lambda m: m.group(1) + "://", str(experiments_dir))


def literal_log_remote_uri(experiments_dir: str | Path, job_name: str, token: str) -> Optional[str]:
    """Durable remote URI for the literal log, or ``None`` for local ``experiments_dir``.

    On iris ``experiments_dir`` is a ``gs://…`` remote_output_dir. The proxy cannot
    append to object storage (no append semantics), so it writes locally and the
    launch path uploads the whole file here on a periodic + on-exit flush. Returns
    ``<experiments_dir>/logs/<slug>__<token>_literal.jsonl`` as a ``gs://…`` string
    when ``experiments_dir`` is remote; ``None`` when it is an ordinary local path
    (local runs and unit tests need no upload). ``token`` (a :func:`serve_token`) makes
    each serve attempt target a DISTINCT object so a resume cannot clobber attempt-0's
    log; the correlator globs ``*_literal.jsonl`` and unions every attempt.
    """
    from upath import UPath  # lazy: only the iris gs:// path needs it (image-provided)

    uri = _as_remote_uri_str(experiments_dir)
    protocol = UPath(uri).protocol
    if protocol not in _REMOTE_PROTOCOLS:
        return None
    return str(UPath(uri) / "logs" / _literal_log_name(job_name, token))


def _local_staging_path(job_name: str, token: str) -> Path:
    """A stable, absolute LOCAL staging path for the proxy to append to.

    Used when ``experiments_dir`` is remote (``gs://…``): the proxy appends here and
    the launch path uploads the file to :func:`literal_log_remote_uri`. Absolute (not
    CWD-relative) so it is robust regardless of the worker's working directory. Carries
    the same per-serve ``token`` as the remote URI so staging + upload stay in sync.
    """
    return Path(tempfile.gettempdir()) / "ot-agent-literal" / _literal_log_name(job_name, token)


def literal_proxy_endpoint(host: str = DEFAULT_LITERAL_PROXY_HOST, port: int = DEFAULT_LITERAL_PROXY_PORT) -> str:
    """The OpenAI-compatible base URL the agent should target to hit the proxy."""
    return f"http://{host}:{port}/v1"


def upstream_origin(endpoint: str) -> str:
    """Origin (``scheme://host:port``) of a vLLM OpenAI-compatible base URL.

    Harbor's ``RecordProxy`` forwards by re-appending the FULL incoming request
    path (``/v1/chat/completions``) onto its ``upstream_base_url``. So the base it
    receives MUST be the origin only — handing it the agent-facing ``.../v1`` base
    doubles the segment (``/v1/v1/chat/completions``) and vLLM 404s the request
    (``{"detail":"Not Found"}``). This strips any path/query so exactly one ``/v1``
    reaches vLLM.
    """
    parts = urlsplit(endpoint)
    if not parts.scheme or not parts.netloc:
        raise ValueError(f"upstream endpoint must be an absolute http(s) URL: {endpoint!r}")
    return urlunsplit((parts.scheme, parts.netloc, "", "", ""))


def _upload_literal_log(local_path: str | Path, remote_uri: str) -> bool:
    """Copy the whole local literal log to ``remote_uri`` (full-object overwrite).

    Object stores have no append, so each flush rewrites the entire object. The log
    is a JSONL of token IDs — small enough to rewrite periodically for a smoke and
    for a preempt-durable snapshot. Returns ``True`` iff a NON-EMPTY local file was
    actually uploaded; a missing OR empty local file (the proxy saw no traffic this
    serve) uploads nothing and returns ``False`` — so callers never claim a successful
    upload, and an empty resume serve never writes a 0-byte object.
    """
    from upath import UPath  # lazy: only the iris gs:// path needs it (image-provided)

    try:
        data = Path(local_path).read_bytes()
    except FileNotFoundError:
        return False
    if not data:
        return False
    remote = UPath(remote_uri)
    remote.parent.mkdir(parents=True, exist_ok=True)
    remote.write_bytes(data)
    return True


@contextmanager
def serve_record_proxy(
    upstream_endpoint: str,
    log_path: str | Path,
    *,
    host: str = DEFAULT_LITERAL_PROXY_HOST,
    port: int = DEFAULT_LITERAL_PROXY_PORT,
    timeout: float = 600.0,
    startup_timeout: float = 30.0,
    remote_uri: Optional[str] = None,
    flush_interval: float = 30.0,
) -> Iterator[str]:
    """Serve harbor's RecordProxy in front of ``upstream_endpoint`` on ``host:port``.

    Yields the proxy's OpenAI-compatible base URL (``http://host:port/v1``). The
    uvicorn server runs on a daemon thread and is shut down on context exit. The
    upstream need not be live at start time — RecordProxy forwards lazily per
    request, so it is safe to start the proxy BEFORE the vLLM engines are up.

    When ``remote_uri`` is set, a daemon thread uploads the (local) ``log_path`` to
    that durable object-store URI every ``flush_interval`` seconds and once more on
    exit, so the literal log survives worker teardown / preemption. This is the
    durability fix for iris, where ``experiments_dir`` is a ``gs://…`` URI that a
    plain local ``open(..., "a")`` can never reach.

    This binds a real socket and is used ONLY on the flag-ON launch path; unit
    tests exercise the app in-process via ``RecordProxy.app()`` + ASGITransport
    instead of calling this.
    """
    import uvicorn
    from harbor.literal.proxy import RecordProxy

    Path(log_path).parent.mkdir(parents=True, exist_ok=True)
    # RecordProxy re-appends the full request path (/v1/chat/completions), so it
    # must receive the vLLM ORIGIN, not the agent-facing .../v1 base (else /v1
    # doubles -> vLLM 404). See upstream_origin.
    origin = upstream_origin(upstream_endpoint)
    proxy = RecordProxy(origin, log_path, timeout=timeout)
    app = proxy.app()

    config = uvicorn.Config(app, host=host, port=port, log_level="warning")
    server = uvicorn.Server(config)

    thread = threading.Thread(target=server.run, name="record-proxy", daemon=True)
    thread.start()

    # Wait for uvicorn to flip started=True so the endpoint is live before we
    # hand it to the agent. Fail loud if it never comes up.
    waited = 0.0
    step = 0.1
    while not server.started and thread.is_alive() and waited < startup_timeout:
        thread.join(timeout=step)
        waited += step
    if not server.started:
        server.should_exit = True
        raise RuntimeError(f"RecordProxy failed to start on {host}:{port} within {startup_timeout}s")

    # Periodic uploader: snapshot the growing local log to durable storage so a
    # preemption between flushes loses at most ``flush_interval`` of records.
    stop_flushing = threading.Event()
    uploader: Optional[threading.Thread] = None
    if remote_uri:

        def _flush_loop() -> None:
            while not stop_flushing.wait(flush_interval):
                try:
                    _upload_literal_log(log_path, remote_uri)
                except Exception as exc:  # durability is best-effort; never crash the run
                    print(f"[literal-proxy] periodic literal-log upload failed: {exc}", flush=True)

        uploader = threading.Thread(target=_flush_loop, name="literal-uploader", daemon=True)
        uploader.start()

    print(
        f"[literal-proxy] RecordProxy serving {literal_proxy_endpoint(host, port)} "
        f"-> {origin} (log: {log_path}" + (f" -> {remote_uri}" if remote_uri else "") + ")",
        flush=True,
    )
    try:
        yield literal_proxy_endpoint(host, port)
    finally:
        server.should_exit = True
        thread.join(timeout=10.0)
        stop_flushing.set()
        if uploader is not None:
            uploader.join(timeout=5.0)
        if remote_uri:
            try:
                if _upload_literal_log(log_path, remote_uri):
                    print(
                        f"[literal-proxy] uploaded final literal log -> {remote_uri}",
                        flush=True,
                    )
                else:
                    print(
                        "[literal-proxy] no literal records captured this serve "
                        f"(nothing to upload; {remote_uri} left untouched)",
                        flush=True,
                    )
            except Exception as exc:
                print(f"[literal-proxy] final literal-log upload failed: {exc}", flush=True)


@contextmanager
def maybe_serve_literal_proxy(
    enabled: bool,
    upstream_endpoint: str,
    *,
    experiments_dir: str | Path,
    job_name: str,
    host: str = DEFAULT_LITERAL_PROXY_HOST,
    port: int = DEFAULT_LITERAL_PROXY_PORT,
) -> Iterator[str]:
    """Flag-gated wrapper around :func:`serve_record_proxy`.

    When ``enabled`` is False (the default), this is a NULL context manager: it
    yields ``upstream_endpoint`` UNCHANGED and starts no server, so the caller's
    endpoint handoff is byte-identical to today. When True, it serves a co-located
    RecordProxy in front of ``upstream_endpoint`` and yields the proxy base URL.
    """
    if not enabled:
        yield upstream_endpoint
        return
    # Per-serve token minted ONCE so the staging path and the remote URI share it: a
    # preempt-resume serve then writes a DISTINCT ``logs/<slug>__<token>_literal.jsonl``
    # instead of clobbering attempt-0's populated log (the correlator unions all).
    token = serve_token()
    # Remote experiments_dir (iris ``gs://…``): the proxy cannot append to object
    # storage, so stage the log on the LOCAL worker FS and upload it to the durable
    # remote URI (periodic + on-exit). Local experiments_dir (local runs / tests):
    # write directly under ``<exp>/logs/`` with no upload — byte-identical to before.
    remote_uri = literal_log_remote_uri(experiments_dir, job_name, token)
    if remote_uri is not None:
        log_path: str | Path = _local_staging_path(job_name, token)
    else:
        log_path = literal_log_path(experiments_dir, job_name, token)
    # Publish the LOCAL literal-log path so the in-loop RL glue (the opencode literal
    # bridge, terminal_bench_generator._maybe_correlate_opencode_rollout_details) can
    # find the shared RecordProxy log and correlate each trial's rollout_details by its
    # per-trial x-ot-trial-id. This is the worker-side file the proxy APPENDS to (the
    # staging path when experiments_dir is remote, else the direct log); the correlator
    # reads it locally, never the gs:// upload. Single source of truth for the path.
    os.environ["OTAGENT_LITERAL_LOG_PATH"] = str(log_path)
    with serve_record_proxy(upstream_endpoint, log_path, host=host, port=port, remote_uri=remote_uri) as proxy_endpoint:
        yield proxy_endpoint
