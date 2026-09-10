"""Version-bound worker sessions around prepared K10 stream runners.

Native preparation must bind actual rank-local tensors and groups before these
methods are exposed. Replica verification is an explicit full-byte callback,
run while the learner lease is held; no digest/count-only default is provided.
"""

from dataclasses import asdict, dataclass
from enum import StrEnum
import hashlib
import json
from threading import Lock

import torch
import torch.distributed as dist

from skyrl_train.weight_sync.shard_memory import device_memory
from skyrl_train.weight_sync.shard_observations import port_counters

from skyrl_train.weight_sync.shard_replay import ShardReplay


class ShardPhase(StrEnum):
    PREPARED = "prepared"
    FROZEN = "frozen"
    VERIFIED = "verified"
    RUNNING = "running"
    INSTALLED = "installed"
    REPLAY_READY = "replay-ready"
    REPLAYING = "replaying"
    FAILED = "failed"
    CLOSED = "closed"


def shard_manifest_id(runner):
    payload = {
        "schema": 1,
        "dense_chunk_bytes": runner.dense_chunk_bytes,
        "replica_plan_id": getattr(runner, "replica_plan_id", None),
        "schedule": asdict(runner.schedule),
        "experts": [asdict(value) for _, value in sorted(runner.expert_views.items())],
        "dense": [asdict(value) for value in runner.dense_plan],
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def storage_versions(sources):
    return {
        name: (
            value.data_ptr(),
            tuple(value.shape),
            tuple(value.stride()),
            str(value.dtype),
            str(value.device),
            value._version,
        )
        for name, value in sources.items()
    }


@dataclass(frozen=True)
class SourceReplicaProof:
    manifest_id: str
    publication_id: int
    rank: int
    compared_bytes: int
    expected_bytes: int
    mismatches: int
    source_versions: dict
    method: str


class ShardSession:
    def __init__(
        self,
        runner,
        *,
        policy_access,
        replica_verifier,
        owned_groups,
        retained_proof_workspace_bytes=0,
        inventory_validator=None,
    ):
        if runner.completed or runner.manifest_id is not None:
            raise ValueError("Only an unused stream runner may be bound")
        self.runner = runner
        self.manifest_id = shard_manifest_id(runner)
        self.policy_access = policy_access
        self.replica_verifier = replica_verifier
        self.owned_groups = tuple(owned_groups)
        self.phase = ShardPhase.PREPARED
        self.publication_id = None
        self.token = None
        self.versions = None
        self.lock = Lock()
        self.proof = None
        self.installed_versions = None
        self.replay_state = None
        self.retained_proof_workspace_bytes = retained_proof_workspace_bytes
        self.inventory_validator = inventory_validator
        self.last_completed_publication = None
        self.source_layout = {name: value[:-1] for name, value in storage_versions(runner.sources).items()}
        if runner.rank in runner.trainers and (policy_access is None or replica_verifier is None):
            raise ValueError("Trainer stream requires an enforceable lease and exact replica verifier")
        if runner.rank in runner.receivers and (policy_access is not None or replica_verifier is not None):
            raise ValueError("Receiver session cannot claim learner ownership")

    def adopt_preparation_lease(self, token, versions):
        """Retain preparation ownership until publication finish, close or failed begin."""
        with self.lock:
            if self.phase is not ShardPhase.PREPARED or self.token is not None or self.policy_access is None:
                raise ValueError("Only a fresh policy session can adopt preparation ownership")
            if storage_versions(self.runner.sources) != versions:
                raise ValueError("Policy sources changed before preparation ownership transfer")
            self.policy_access.transfer(token, "shard-publication")
            self.token = token
            self.versions = versions

    def identity(self, manifest_id, publication_id):
        if manifest_id != self.manifest_id or type(publication_id) is not int or publication_id < 0:
            raise ValueError("Shard manifest/publication identity is invalid")
        if self.publication_id is not None and publication_id != self.publication_id:
            raise ValueError("Shard call belongs to a different publication")

    def receipt(self):
        return {
            "rank": self.runner.rank,
            "manifest_id": self.manifest_id,
            "publication_id": self.publication_id,
            "phase": self.phase.value,
        }

    def begin(self, manifest_id, publication_id):
        with self.lock:
            self.identity(manifest_id, publication_id)
            if self.phase is not ShardPhase.PREPARED:
                raise ValueError("Shard session is already active")
            if self.last_completed_publication is not None and publication_id <= self.last_completed_publication:
                raise ValueError("Shard publication must advance beyond the completed version")
            if self.policy_access is not None and self.token is None:
                self.token = self.policy_access.acquire("shard-publication")
            try:
                inventory = self.inventory_validator(publication_id) if self.inventory_validator is not None else None
                current = storage_versions(self.runner.sources)
                if {name: value[:-1] for name, value in current.items()} != self.source_layout:
                    raise ValueError("Prepared learner source storage changed between publications")
                if self.versions is not None and current != self.versions:
                    raise ValueError("Policy sources changed after preparation ownership transfer")
                self.versions = current
                self.runner.begin(manifest_id=manifest_id, publication_id=publication_id)
            except BaseException:
                if self.token is not None:
                    self.policy_access.release(self.token)
                    self.token = None
                raise
            self.publication_id = publication_id
            self.phase = ShardPhase.FROZEN
            return {**self.receipt(), "live_inventory": inventory}

    def verify_replicas(self, manifest_id, publication_id):
        with self.lock:
            self.identity(manifest_id, publication_id)
            if self.phase is not ShardPhase.FROZEN or self.replica_verifier is None:
                raise ValueError("Replica proof requires a frozen learner session")
            proof = self.replica_verifier(self.runner.sources, manifest_id, publication_id, self.runner.rank)
            expected_bytes = sum(value.numel() * value.element_size() for value in self.runner.sources.values())
            if (
                not isinstance(proof, SourceReplicaProof)
                or proof.manifest_id != manifest_id
                or type(proof.publication_id) is not int
                or proof.publication_id != publication_id
                or type(proof.rank) is not int
                or proof.rank != self.runner.rank
                or type(proof.compared_bytes) is not int
                or proof.compared_bytes != expected_bytes
                or type(proof.expected_bytes) is not int
                or proof.expected_bytes != expected_bytes
                or type(proof.mismatches) is not int
                or proof.mismatches != 0
                or proof.method != "full-byte-comparison"
                or proof.source_versions != self.versions
                or storage_versions(self.runner.sources) != self.versions
            ):
                self.phase = ShardPhase.FAILED
                raise ValueError("Exact source-replica proof is incomplete, stale or mismatched")
            memory = getattr(self.replica_verifier, "last_receipt", None)
            if memory is not None and memory.get("proof_memory_within_limit") is False:
                self.phase = ShardPhase.FAILED
                raise ValueError("Source-copy proof exceeds the complete additional scratch limit")
            self.proof = proof
            self.phase = ShardPhase.VERIFIED
            return {
                **self.receipt(),
                "proof": asdict(proof),
                "replica_groups": getattr(self.replica_verifier, "last_receipt", None),
            }

    def run(self, manifest_id, publication_id):
        with self.lock:
            self.identity(manifest_id, publication_id)
            required = ShardPhase.VERIFIED if self.policy_access is not None else ShardPhase.FROZEN
            if self.phase is not required or storage_versions(self.runner.sources) != self.versions:
                raise ValueError("Shard install requires unchanged frozen weights and completed replica proof")
            self.phase = ShardPhase.RUNNING
        try:
            device = self.runner.scratch.device
            memory_before = device_memory(device)
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)
            ports_before = port_counters()
            result = self.runner.run(manifest_id=manifest_id, publication_id=publication_id)
            ports_after = port_counters()
            memory_after = device_memory(device)
            if storage_versions(self.runner.sources) != self.versions:
                raise ValueError("Learner source changed during shard installation")
        except BaseException:
            with self.lock:
                self.phase = ShardPhase.FAILED
            raise
        with self.lock:
            self.phase = ShardPhase.INSTALLED
            self.installed_versions = storage_versions(self.runner.parameters)
        return {
            **self.receipt(),
            "stream": result,
            "install_observations": {
                "ports_before": ports_before,
                "ports_after": ports_after,
                "memory_before": memory_before,
                "memory_after": memory_after,
                "memory_scope": "Torch allocator peak plus device-free endpoints; external allocator peak unmeasured",
            },
        }

    def prepare_replay(self, manifest_id, publication_id):
        with self.lock:
            self.identity(manifest_id, publication_id)
            if self.phase is not ShardPhase.INSTALLED or storage_versions(self.runner.sources) != self.versions:
                raise ValueError("Replay requires the unchanged installed publication")
            if storage_versions(self.runner.parameters) != self.installed_versions:
                raise ValueError("Installed storage changed before replay preparation")
            try:
                self.replay_state = ShardReplay(
                    self.runner, retained_proof_workspace_bytes=self.retained_proof_workspace_bytes
                )
            except BaseException:
                self.phase = ShardPhase.FAILED
                raise
            self.phase = ShardPhase.REPLAY_READY
            return {
                **self.receipt(),
                "expected_bytes": self.replay_state.expected_bytes,
                "memory_before": self.replay_state.memory_before,
            }

    def replay(self, manifest_id, publication_id):
        with self.lock:
            self.identity(manifest_id, publication_id)
            if self.phase is not ShardPhase.REPLAY_READY:
                raise ValueError("Replay requires complete prechecked receiver inventory")
            if (
                storage_versions(self.runner.sources) != self.versions
                or storage_versions(self.runner.parameters) != self.installed_versions
            ):
                raise ValueError("Frozen source or installed storage changed before replay")
            self.phase = ShardPhase.REPLAYING
        try:
            result = self.replay_state.run()
            if (
                storage_versions(self.runner.sources) != self.versions
                or storage_versions(self.runner.parameters) != self.installed_versions
            ):
                raise ValueError("Replay modified source or installed storage")
        except BaseException:
            with self.lock:
                self.phase = ShardPhase.FAILED
            raise
        with self.lock:
            self.phase = (
                ShardPhase.VERIFIED
                if result["mismatches"] == 0 and result["replay_memory_within_limit"] is not False
                else ShardPhase.FAILED
            )
        return {**self.receipt(), **result}

    def finish(self, manifest_id, publication_id):
        """Release frozen ownership after replay while retaining warmed groups and buffers."""
        with self.lock:
            self.identity(manifest_id, publication_id)
            if (
                self.phase is not ShardPhase.VERIFIED
                or self.replay_state is None
                or not self.replay_state.executed
                or storage_versions(self.runner.sources) != self.versions
                or storage_versions(self.runner.parameters) != self.installed_versions
            ):
                raise ValueError("Persistent publication finish requires unchanged, fully replayed weights")
            self.runner.reset(manifest_id=manifest_id, publication_id=publication_id)
            if self.token is not None:
                self.policy_access.release(self.token)
                self.token = None
            self.last_completed_publication = publication_id
            self.publication_id = None
            self.versions = None
            self.installed_versions = None
            self.proof = None
            self.replay_state = None
            self.phase = ShardPhase.PREPARED
            return {**self.receipt(), "publication_id": publication_id, "groups_retained": True}

    def close(self, manifest_id, publication_id):
        with self.lock:
            self.identity(manifest_id, publication_id)
            if self.phase in (ShardPhase.RUNNING, ShardPhase.REPLAYING, ShardPhase.CLOSED):
                raise ValueError("Cannot close a running or already closed shard session")
            closed_from = self.phase.value
            if self.publication_id is None:
                self.publication_id = publication_id
            errors = []
            if self.versions is not None and storage_versions(self.runner.sources) != self.versions:
                errors.append("Frozen learner source changed before session close")
            for group in self.owned_groups:
                try:
                    dist.destroy_process_group(group)
                except Exception as error:
                    errors.append(f"{type(error).__name__}: {error}")
            if self.token is not None:
                self.policy_access.release(self.token)
                self.token = None
            self.phase = ShardPhase.CLOSED
            result = {**self.receipt(), "closed_from": closed_from, "cleanup_errors": errors}
            if errors:
                raise RuntimeError(f"Shard cleanup failed: {errors}")
            return result


def bind_worker_shard_stream(
    worker,
    runner,
    *,
    policy_access,
    replica_verifier,
    owned_groups,
    retained_proof_workspace_bytes=0,
    inventory_validator=None,
):
    if getattr(worker, "_shard_stream_session", None) is not None:
        raise ValueError("Close the previous shard worker session before binding another")
    worker._shard_stream_session = ShardSession(
        runner,
        policy_access=policy_access,
        replica_verifier=replica_verifier,
        owned_groups=owned_groups,
        retained_proof_workspace_bytes=retained_proof_workspace_bytes,
        inventory_validator=inventory_validator,
    )
    return worker._shard_stream_session.manifest_id


def worker_shard_call(worker, method, manifest_id, publication_id):
    state = getattr(worker, "_shard_stream_session", None)
    if state is None:
        raise ValueError("Native shard preparation has not bound a runner")
    result = getattr(state, method)(manifest_id, publication_id)
    if method == "close":
        del worker._shard_stream_session
    return result
