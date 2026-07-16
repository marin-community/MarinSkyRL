"""ingress_utils.py — native controller-ingress wiring helpers (capability-URL scheme).

Shared by the RL / datagen launchers and the eval listener when
``--ingress-mode controller`` is selected. Under the default ``pinggy`` mode
NONE of this is reached, so the legacy path stays byte-identical.

The controller path replaces the pinggy tunnel with the native iris
capability-URL scheme that shipped in marin #6857 (``/proxy/t/*``): the co-located
vLLM (or the RecordProxy in the literal combo) is registered with the iris
controller under ``ENDPOINT_ACCESS_LINK``, then a scoped capability token is
minted for it and carried IN THE URL PATH:

    api_base = https://<ingress_host>/proxy/t/<token>/<encoded_endpoint>/v1

Possession of the URL is the credential — there is NO auth header, and the
sandbox-facing ``api_key`` is an unused dummy (installed OpenAI-compatible agents
still require a non-empty key string, so we inject one, but it never rides the
wire). ``<encoded_endpoint>`` is the registered wire name with a leading ``/``
dropped and ``/`` -> ``.`` (the exact encoding of ``rigging.connect.capability_path``
/ ``proxy_path``); our single-segment ``otagent-<slug>`` name encodes to itself.

TOKEN LIFETIME. The controller clamps a minted token to
``MAX_ENDPOINT_TOKEN_TTL_SECONDS`` = 24h (``DEFAULT`` = 1h). The endpoint
REGISTRATION is separately lease-renewed for the whole run (see
:class:`ControllerEndpointRegistration`); only the token expires. So the api_base
is resolved through :func:`capability_api_base`, which mints a 24h token, caches
it worker-side keyed by endpoint name, and re-mints when within
``TOKEN_REFRESH_MARGIN_SECONDS`` of expiry.

INJECTION CADENCE (important). The launchers bake the resolved api_base into the
harbor command once per harbor spawn (``_run_harbor`` -> ``build_harbor_command``),
so it is a PER-HARBOR-SPAWN value, not a per-trial one: a single harbor process
uses one api_base string for its whole lifetime. The worker-side cache therefore
refreshes the token across harbor RE-SPAWNS (resume / campaign refills), not
across trials within one running harbor process. A harbor run that stays up
longer than the token TTL will outlive its token — keep individual harbor runs
under 24h, or re-spawn to re-mint. There is no per-trial base_url resolution hook
in the current OT-Agent->harbor plumbing.
"""

from __future__ import annotations

import os
import re
import threading
import time
from dataclasses import dataclass
from typing import Callable, Dict, Optional, Protocol, Tuple

# The sandbox-facing api_key. The capability token rides in the URL path, so no
# bearer is needed; but installed OpenAI-compatible agents refuse to start
# without SOME non-empty key, so we hand them this inert placeholder. It is
# NOT a secret and never authenticates anything.
DUMMY_API_KEY = "capability-url-no-auth-header"

# Bespoke placeholder var the agent's SANDBOX key is sourced from. Kept SEPARATE
# from OPENAI_API_KEY so the launcher never clobbers the real host OPENAI_API_KEY:
# the nemotron-gym LLM-judge verifiers render `[verifier.env] OPENAI_API_KEY` from
# the WORKER env, so a placeholder there → every judge 401s → uniform reward 0.0
# (2026-07-13, identity-following #29). harbor's opencode agent sources its sandbox
# key from OPENCODE_DUMMY_KEY (see harbor agents/installed/opencode.py).
AGENT_DUMMY_KEY_VAR = "OPENCODE_DUMMY_KEY"
# Legacy vars other OpenAI-compatible agents read directly (qwen/codex/hermes/trae →
# OPENAI_API_KEY, openhands → LLM_API_KEY). Filled ONLY when absent (setdefault) so a
# real host OPENAI_API_KEY survives for the judge while keyless workers still start.
_AGENT_KEY_ENV_VARS = ("OPENAI_API_KEY", "LLM_API_KEY")

# Env var iris sets in-cluster to the externally-visible host a task should
# advertise its services under (``JobInfo.advertise_host``). The controller
# resolves a registered endpoint by connecting to its ``address``, so the
# upstream (raw vLLM, or the co-located RecordProxy) must be reachable at THIS
# host — a loopback ``127.0.0.1`` would only be reachable from inside the task.
# Off-cluster / in tests the var is unset and we fall back to loopback.
ADVERTISE_HOST_ENV = "IRIS_ADVERTISE_HOST"
DEFAULT_ADVERTISE_HOST = "127.0.0.1"

# The raw vLLM HTTP port the RL/datagen servers bind on the task node.
DEFAULT_VLLM_PORT = 8000

# Token TTL we request when minting (clamped server-side to the controller's
# MAX_ENDPOINT_TOKEN_TTL_SECONDS, currently 24h) and the safety margin at which a
# cached token is re-minted rather than reused.
DEFAULT_TOKEN_TTL_HOURS = 24.0
TOKEN_REFRESH_MARGIN_SECONDS = 2 * 3600  # re-mint when <2h remains


def encode_endpoint_name(name: str) -> str:
    """Encode a wire endpoint name for the ``/proxy`` path (``/`` -> ``.``, leading ``/`` dropped).

    Byte-identical to ``rigging.connect.proxy_path`` / ``capability_path``'s
    ``name.strip('/').replace('/', '.')``. Replicated here (rather than imported)
    so the pure-string helpers stay usable on a launch host whose pinned iris
    predates the capability APIs; the worker runtime (post pin-bump) agrees.
    """
    return name.strip("/").replace("/", ".")


def controller_endpoint_name(job_name: Optional[str]) -> str:
    """Deterministic iris endpoint wire name for a job's vLLM.

    ``otagent-<sanitized-job-name>`` — unique per job, a single DOT-FREE path
    segment, so ``encode_endpoint_name`` is the identity and the registered name,
    the minted token's audience, and the capability path all agree. Anything
    outside ``[A-Za-z0-9_-]`` (including dots and slashes) maps to ``-``.
    """
    slug = re.sub(r"[^A-Za-z0-9_-]+", "-", (job_name or "job")).strip("-_").lower()
    return f"otagent-{slug or 'job'}"


def build_capability_api_base(ingress_host: str, endpoint_name: str, token: str) -> str:
    """``https://<ingress_host>/proxy/t/<token>/<encoded_endpoint>/v1``.

    The capability URL for an OpenAI server: the scoped token rides in the path
    and no auth header is needed. ``ingress_host`` may omit the scheme.
    """
    host = (ingress_host or "").rstrip("/")
    if not (host.startswith("http://") or host.startswith("https://")):
        host = f"https://{host}"
    return f"{host}/proxy/t/{token}/{encode_endpoint_name(endpoint_name)}/v1"


def inject_ingress_agent_key(env: Optional[dict] = None) -> bool:
    """Publish the inert :data:`DUMMY_API_KEY` for the agents WITHOUT clobbering a
    real ``OPENAI_API_KEY``.

    The capability token is in the URL path, so no real bearer is needed; agents
    just refuse to start without a non-empty key. Sets the bespoke
    :data:`AGENT_DUMMY_KEY_VAR` (``OPENCODE_DUMMY_KEY``) unconditionally, and only
    *fills in* an absent ``OPENAI_API_KEY``/``LLM_API_KEY`` (``setdefault``) — so a
    real host ``OPENAI_API_KEY`` needed by the LLM-judge verifiers is preserved.
    Always returns True (there is no secret to be missing).
    """
    env = os.environ if env is None else env
    # Always publish the bespoke placeholder (harbor's opencode sources it).
    env[AGENT_DUMMY_KEY_VAR] = DUMMY_API_KEY
    # For the legacy agent-facing key vars, only fill an ABSENT value — never
    # overwrite a real host OPENAI_API_KEY (the LLM-judge verifiers need it).
    for var in _AGENT_KEY_ENV_VARS:
        env.setdefault(var, DUMMY_API_KEY)
    return True


# --------------------------------------------------------------------------- #
# Capability-token minting + worker-side cache
# --------------------------------------------------------------------------- #


class CapabilityMinter(Protocol):
    """Mints a scoped capability token for a registered endpoint.

    Returns ``(token, expires_at_epoch_seconds)``. The in-cluster controller-RPC
    adapter and unit-test fakes both satisfy it, so the cache is testable
    in-process without a live controller.
    """

    def mint(self, endpoint_name: str, ttl_hours: float) -> Tuple[str, float]: ...


@dataclass
class _CachedToken:
    token: str
    expires_at: float  # epoch seconds


class CapabilityTokenCache:
    """Worker-side cache of scoped capability tokens, keyed by endpoint name.

    Mints on first use and re-mints when a cached token is within
    :data:`TOKEN_REFRESH_MARGIN_SECONDS` of expiry, so every resolve of the
    api_base hands out a token with ample life left. Thread-safe: harbor spawns
    and lease renewers touch it from different threads.
    """

    def __init__(self, minter: CapabilityMinter, *, ttl_hours: float = DEFAULT_TOKEN_TTL_HOURS) -> None:
        self._minter = minter
        self._ttl_hours = ttl_hours
        self._lock = threading.Lock()
        self._cache: Dict[str, _CachedToken] = {}

    def token_for(self, endpoint_name: str, *, now: Optional[float] = None) -> str:
        now = time.time() if now is None else now
        with self._lock:
            cached = self._cache.get(endpoint_name)
            if cached is not None and cached.expires_at - now > TOKEN_REFRESH_MARGIN_SECONDS:
                return cached.token
            token, expires_at = self._minter.mint(endpoint_name, self._ttl_hours)
            if not token:
                raise RuntimeError(
                    f"minting a capability token for {endpoint_name} returned an empty "
                    "token; refusing to build an unreachable api_base."
                )
            self._cache[endpoint_name] = _CachedToken(token=token, expires_at=expires_at)
            return token


class _ControllerCapabilityMinter:
    """Mints via the in-cluster controller ``MintEndpointToken`` RPC.

    Builds a ``ControllerServiceClientSync`` from the task's
    ``IRIS_CONTROLLER_ADDRESS`` (network-level trust in-cluster, no explicit
    credentials — mirroring the leased ``EndpointClient``). The mint is
    authorized to the endpoint's owning user or an admin; an in-cluster task
    registering its own endpoint is that owner.
    """

    def __init__(self) -> None:
        from iris.cluster.client.job_info import get_job_info
        from iris.rpc.compression import IRIS_RPC_COMPRESSIONS
        from iris.rpc.controller_connect import ControllerServiceClientSync

        info = get_job_info()
        if info is None or not info.controller_address:
            raise RuntimeError(
                "capability-token minting requires an in-cluster iris task "
                "(IRIS_TASK_ID + IRIS_CONTROLLER_ADDRESS); none found."
            )
        self._stub = ControllerServiceClientSync(
            info.controller_address,
            accept_compression=IRIS_RPC_COMPRESSIONS,
            send_compression=None,
        )

    def mint(self, endpoint_name: str, ttl_hours: float) -> Tuple[str, float]:
        from iris.rpc import controller_pb2
        from iris.time_proto import duration_to_proto
        from rigging.timing import Duration

        request = controller_pb2.Controller.MintEndpointTokenRequest(
            endpoint_name=endpoint_name,
            ttl=duration_to_proto(Duration.from_hours(ttl_hours)),
        )
        resp = self._stub.mint_endpoint_token(request)
        return resp.token, resp.expires_at.epoch_ms / 1000.0


# Process-wide cache; the default minter is constructed lazily on first resolve
# (it needs the in-cluster controller address, unavailable at import time).
_TOKEN_CACHE: Optional[CapabilityTokenCache] = None
_TOKEN_CACHE_LOCK = threading.Lock()


def _default_token_cache() -> CapabilityTokenCache:
    global _TOKEN_CACHE
    with _TOKEN_CACHE_LOCK:
        if _TOKEN_CACHE is None:
            _TOKEN_CACHE = CapabilityTokenCache(_ControllerCapabilityMinter())
        return _TOKEN_CACHE


def capability_api_base(
    ingress_host: str,
    endpoint_name: str,
    *,
    cache: Optional[CapabilityTokenCache] = None,
    now: Optional[float] = None,
) -> str:
    """Resolve the current capability api_base for a REGISTERED endpoint.

    Mints (or reuses a cached, still-fresh) scoped token and returns
    ``https://<ingress_host>/proxy/t/<token>/<encoded_endpoint>/v1``. The endpoint
    MUST already be registered with ``ENDPOINT_ACCESS_LINK`` (see
    :func:`register_controller_endpoint`), else the mint has nothing to resolve.
    ``cache`` is injectable for tests; production uses the process-wide worker
    cache backed by the controller RPC.
    """
    token_cache = cache if cache is not None else _default_token_cache()
    token = token_cache.token_for(endpoint_name, now=now)
    return build_capability_api_base(ingress_host, endpoint_name, token)


def build_controller_endpoint_meta(
    ingress_host: str,
    endpoint_name: str,
    *,
    cache: Optional[CapabilityTokenCache] = None,
) -> Dict[str, str]:
    """Endpoint metadata dict for controller mode (capability api_base + dummy key).

    metrics_endpoint is intentionally omitted: the capability route fronts only
    the ``/v1`` inference surface, so ``/metrics`` is not reachable through it.
    """
    return {
        "api_base": capability_api_base(ingress_host, endpoint_name, cache=cache),
        "api_key": DUMMY_API_KEY,
    }


# --------------------------------------------------------------------------- #
# Endpoint registration (shared by the controller path and the +literal combo)
# --------------------------------------------------------------------------- #
#
# The capability api_base only resolves once the vLLM (or, in the literal combo,
# the co-located RecordProxy) is REGISTERED with the iris controller under
# ``<endpoint_name>`` AND with ``ENDPOINT_ACCESS_LINK`` — a PRIVATE (default)
# endpoint rejects a scoped capability token (the controller's proxy authorizer
# returns 403 "endpoint-scoped token cannot access this endpoint"). The helpers
# below register with LINK access and are shared so the plain and the
# record_literal combo register the SAME name (only the address differs).
#
# NAMESPACING: we register through the leased :class:`EndpointClient`
# (``iris.cluster.client.endpoint_client``) directly rather than through
# ``ctx.registry`` — the latter auto-prefixes the name with the job namespace,
# which would break the fixed single-segment ``otagent-<slug>`` name the mint's
# token audience and the capability path both use. ``EndpointClient.register``
# does NOT prefix, so the wire name stays that single segment.
#
# LEASING: iris endpoints are LEASED. ``EndpointClient`` owns the RPC stub AND a
# background ``EndpointLeaseRenewer`` daemon — the lease keeps the controller
# serving the endpoint, and a one-shot register with no renewal expires within
# minutes. So we build a dedicated ``EndpointClient`` (its renewer running),
# register through it with LINK access, and KEEP IT ALIVE for the whole harbor
# run via the returned :class:`ControllerEndpointRegistration`, then ``close()``
# it (stops renewing + unregisters) on run exit. The token expiring is orthogonal
# — the lease renews the registration; the token cache re-mints the token.


class EndpointRegistrar(Protocol):
    """The ``register(name, address, metadata, access) -> endpoint_id`` shape we drive.

    The in-cluster leased-``EndpointClient`` adapter and unit-test fakes both
    satisfy it. An implementation MAY also expose ``close()`` to stop lease
    renewal and unregister; the registration handle calls it on teardown.
    """

    def register(
        self,
        name: str,
        address: str,
        metadata: Optional[Dict[str, str]] = None,
        access: Optional[int] = None,
    ) -> str: ...


@dataclass
class ControllerEndpointRegistration:
    """A live controller endpoint registration whose lease is being renewed.

    Holds the real ``endpoint_id`` and a ``close`` callable that stops the
    background lease renewer and unregisters the endpoint. The caller MUST keep
    this handle alive for the whole harbor run and call :meth:`close` on exit; a
    dropped handle lets the lease lapse and the ``/proxy`` route starts 404-ing.
    """

    endpoint_id: str
    _close: Callable[[], None]

    def close(self) -> None:
        """Stop lease renewal and unregister the endpoint (best-effort, idempotent)."""
        self._close()


class _LeasedEndpointRegistrar:
    """Registers through a dedicated leased iris :class:`EndpointClient`.

    Owns the ``EndpointClient`` (and thus its background ``EndpointLeaseRenewer``
    daemon), so ``register`` returns the real endpoint_id and keeps the lease
    renewed until ``close``. Registers WITHOUT namespace-prefixing and with
    ``ENDPOINT_ACCESS_LINK`` so the capability token minted for the single
    ``otagent-<slug>`` wire name resolves and is accepted by the proxy.
    """

    def __init__(self, client: "EndpointClient", task_attempt: "TaskAttempt") -> None:  # noqa: F821
        self._client = client
        self._task_attempt = task_attempt

    def register(
        self,
        name: str,
        address: str,
        metadata: Optional[Dict[str, str]] = None,
        access: Optional[int] = None,
    ) -> str:
        from iris.cluster.types import EndpointAccess

        access_mode = access if access is not None else EndpointAccess.ENDPOINT_ACCESS_LINK
        return self._client.register(
            name, address, self._task_attempt, metadata or {}, access=access_mode
        )

    def close(self) -> None:
        # Stops the renewer daemon and best-effort unregisters everything still
        # registered, then disconnects the stub.
        self._client.close()


def controller_upstream_address(port: int, *, env: Optional[dict] = None) -> str:
    """``http://<advertise-host>:<port>`` — the address the controller resolves to.

    ``<advertise-host>`` is ``IRIS_ADVERTISE_HOST`` (the task's externally-visible
    host, set by iris in-cluster) falling back to ``127.0.0.1`` off-cluster. The
    controller connects here to reach the registered upstream, so it must NOT be a
    loopback address on a live cluster.
    """
    env = os.environ if env is None else env
    host = env.get(ADVERTISE_HOST_ENV) or DEFAULT_ADVERTISE_HOST
    return f"http://{host}:{port}"


def _default_endpoint_registrar() -> _LeasedEndpointRegistrar:
    """Build the in-cluster leased-``EndpointClient`` registrar.

    Constructs a dedicated ``EndpointClient`` from the task's
    ``IRIS_CONTROLLER_ADDRESS`` (network-level trust in-cluster, no credentials)
    and the task identity from :func:`iris.cluster.client.job_info.get_job_info`.
    Raises loudly (never returns ``None``) when iris is unavailable or we are not
    inside a task — a silent no-op here is exactly what produces a 404-ing run.
    """
    from iris.cluster.client.endpoint_client import EndpointClient
    from iris.cluster.client.job_info import get_job_info
    from iris.rpc.compression import IRIS_RPC_COMPRESSIONS
    from iris.rpc.controller_connect import EndpointServiceClientSync

    info = get_job_info()
    if info is None or not info.controller_address:
        raise RuntimeError(
            "controller endpoint registration requires an in-cluster iris task "
            "(IRIS_TASK_ID + IRIS_CONTROLLER_ADDRESS); none found. Registration "
            "cannot proceed — the /proxy route would 404."
        )
    stub = EndpointServiceClientSync(
        info.controller_address,
        accept_compression=IRIS_RPC_COMPRESSIONS,
        send_compression=None,
    )
    return _LeasedEndpointRegistrar(EndpointClient(stub), info.task_attempt)


def register_controller_endpoint(
    endpoint_name: str,
    address: str,
    *,
    registrar: Optional[EndpointRegistrar] = None,
    metadata: Optional[Dict[str, str]] = None,
) -> ControllerEndpointRegistration:
    """Register ``address`` under ``endpoint_name`` (``ENDPOINT_ACCESS_LINK``).

    Registers through a leased ``EndpointClient`` (its background lease renewer
    keeps the controller serving the endpoint) so the capability route resolves
    to ``address`` for the whole run. LINK access is what lets the minted scoped
    token reach it. ``registrar`` is injectable for unit tests; in production it
    is the in-cluster :func:`_default_endpoint_registrar`.

    Returns a :class:`ControllerEndpointRegistration` handle with the REAL
    ``endpoint_id`` and a ``close()`` that stops renewal + unregisters. The caller
    MUST keep the handle alive for the whole run and ``close()`` it on exit.

    Raises if no registrar is available or the register call yields no id — a
    broken registration must fail loud, not silently 404 the run.
    """
    reg = registrar if registrar is not None else _default_endpoint_registrar()
    endpoint_id = reg.register(endpoint_name, address, metadata or {})
    if not endpoint_id:
        raise RuntimeError(
            f"controller endpoint registration for {endpoint_name} -> {address} "
            "returned no endpoint_id; refusing to proceed (would 404 the run)."
        )
    close = getattr(reg, "close", None)
    return ControllerEndpointRegistration(
        endpoint_id=endpoint_id,
        _close=close if callable(close) else (lambda: None),
    )


def controller_registration_plan(
    job_name: Optional[str],
    *,
    record_literal: bool,
    proxy_port: int,
    vllm_port: int = DEFAULT_VLLM_PORT,
    env: Optional[dict] = None,
) -> Tuple[str, str]:
    """Compute ``(endpoint_name, register_address)`` for controller mode.

    The single decision point the launchers share so the plain and the
    ``record_literal`` combo stay consistent:

      * ``endpoint_name`` — the same ``otagent-<slug>`` either way, so the
        capability api_base (built after register+mint) is stable whether or not
        literal capture is on.
      * ``register_address`` — the co-located RecordProxy's ``proxy_port`` when
        ``record_literal`` is set (controller -> RecordProxy -> vLLM, so literal
        tokens are captured on the served path), otherwise raw vLLM's ``vllm_port``.

    The api_base is NOT returned here: it can only be built AFTER the endpoint is
    registered and a token minted (:func:`capability_api_base`).
    """
    endpoint_name = controller_endpoint_name(job_name)
    port = proxy_port if record_literal else vllm_port
    register_address = controller_upstream_address(port, env=env)
    return endpoint_name, register_address
