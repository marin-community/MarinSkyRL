import copy
import hashlib
import logging
import itertools
import json
import io
from datetime import timedelta

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as multiprocessing
import requests
import sys
from rigging.telemetry.metrics import MetricSnapshotPublisher
from rigging.telemetry.prometheus import PrometheusCollector, PrometheusScraper
from hydra import compose, initialize_config_dir
from pathlib import Path
from loguru import logger
from types import SimpleNamespace
from cloud.iris.env_vars import ALL_RUNTIME_SCOPES, EnvVarManager

from skyrl_train.weight_sync.initial_readback import run_initial_readback, validate_initial_readback
from skyrl_train.weight_sync.readback_diagnostics import (
    ENVIRONMENT_KEYS,
    HASH_CHUNK_BYTES,
    parameter_digests,
    receipt_chunks,
    reassemble_receipt,
    tensor_sha256,
    validate_replica_digests,
    network_log_readback,
    persist_readback,
    persist_payload,
    serialize_receipt,
    FULL_READBACK_EVERY,
)
from skyrl_train.weight_sync.receiver_readback_rpc import call_all_receiver_workers
from skyrl_train.utils.tracking import Tracking


@pytest.mark.parametrize("length", [1, 7, HASH_CHUNK_BYTES + 17])
def test_tensor_digest_preserves_exact_storage_and_tail(length):
    tensor = torch.arange(length, dtype=torch.int64).to(torch.uint8)
    expected = hashlib.sha256(memoryview(tensor.numpy())).hexdigest()
    assert tensor_sha256(tensor) == expected
    tensor[-1] ^= 1
    assert tensor_sha256(tensor) != expected


def test_digest_distinguishes_bfloat16_bit_patterns_without_numeric_conversion():
    first = torch.tensor([0x7FC0, 0, 0x8000], dtype=torch.uint16).view(torch.bfloat16)
    second = torch.tensor([0x7FC1, 0, 0x8000], dtype=torch.uint16).view(torch.bfloat16)
    assert tensor_sha256(first) != tensor_sha256(second)
    assert tensor_sha256(first) == hashlib.sha256(bytes.fromhex("c07f00000080")).hexdigest()


def test_tensor_digest_rejects_hidden_full_storage_copy():
    with pytest.raises(ValueError, match="contiguous"):
        tensor_sha256(torch.arange(12).reshape(3, 4).t())


def test_receipt_unicode_and_missing_mixed_or_corrupt_parts():
    receipt = {"name": "λ🧊" * 2000, "nested": {"value": [1, 2]}}
    chunks = receipt_chunks(receipt)
    assert len(chunks) > 1 and max(len(part["payload"].encode()) for part in chunks) <= 3072
    assert reassemble_receipt(list(reversed(chunks))) == receipt
    for damaged in (chunks[:-1], [chunks[0], *chunks[:-1]]):
        with pytest.raises(ValueError, match="Incomplete"):
            reassemble_receipt(damaged)
    corrupt = copy.deepcopy(chunks)
    corrupt[-1]["payload"] = corrupt[-1]["payload"].replace("}", "]", 1)
    with pytest.raises(ValueError, match="hash"):
        reassemble_receipt(corrupt)


def test_actual_metrics_poller_recovers_after_endpoint_connection_failure(monkeypatch):
    observed = []

    class Response:
        status_code = 200
        headers = {}
        encoding = "utf-8"

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def iter_content(self, chunk_size):
            yield b"# TYPE ray_tasks gauge\nray_tasks 3\n"

    def refused(*args, **kwargs):
        raise requests.ConnectionError("metrics endpoint not listening yet")

    def process(families):
        observed.extend(families)
        return ()

    collector = PrometheusCollector(
        metric_source="ray",
        scraper=PrometheusScraper("http://127.0.0.1:8090/metrics"),
        processor=process,
        publisher=MetricSnapshotPublisher(max_records=8),
    )
    monkeypatch.setattr(requests, "get", refused)
    collector.poll_once()
    assert observed == []
    monkeypatch.setattr(requests, "get", lambda *args, **kwargs: Response())
    collector.poll_once()
    assert [(family.name, family.samples[0].value) for family in observed] == [("ray_tasks", 3.0)]


def test_diagnostic_console_tracker_never_initializes_wandb(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("Diagnostic initialized W&B")

    monkeypatch.setitem(sys.modules, "wandb", SimpleNamespace(init=forbidden))
    stream = io.StringIO()
    sink = logger.add(stream)
    try:
        tracker = Tracking(project_name="readback", experiment_name="zero-update", backends="console")
        tracker.log({"native_readback_updates": 0}, step=0)
        tracker.finish(exit_code=0)
    finally:
        logger.remove(sink)
    assert "native_readback_updates" in stream.getvalue()


def test_typed_nccl_readback_diagnostics_reach_every_runtime_scope():
    config_dir = str(Path(__file__).parents[3] / "skyrl_train/config")
    with initialize_config_dir(config_dir=config_dir, version_base=None):
        cfg = compose(config_name="ppo_base_config", overrides=["trainer.weight_sync_nccl_diagnostics=true"])
    manager = EnvVarManager.from_config(cfg, environ={})
    for scope in ALL_RUNTIME_SCOPES:
        values = manager.environment_for(scope)
        assert values["NCCL_DEBUG"] == "INFO" and values["NCCL_DEBUG_SUBSYS"] == "INIT,NET"
        assert values["NCCL_DEBUG_FILE"] == "/tmp/skyrl-weight-sync-nccl.%h.%p.log"
        assert "VLLM_BATCH_INVARIANT" not in values


def test_native_network_capture_is_bounded_and_reports_truncation(tmp_path, monkeypatch):
    path = tmp_path / "nccl.log"
    path.write_text("unrelated\n" * 40000 + "host NCCL INFO NET/IB : Using mlx5_0\n")
    monkeypatch.setenv("NCCL_DEBUG_FILE", str(path))
    receipt = network_log_readback()
    assert receipt["network_lines"] == ["host NCCL INFO NET/IB : Using mlx5_0"]
    assert receipt["truncated"] and receipt["tail_bytes"] == 262144


@pytest.mark.asyncio
async def test_receiver_readback_retains_each_dp_core_and_rejects_missing_core():
    class NativeCoreBoundary:
        core_engines = [bytes([0, 0]), bytes([1, 0])]
        engine_ranks_managed = [0, 1]

        async def _call_utility_async(self, utility, method, timeout, args, kwargs, *, engine):
            assert utility == "collective_rpc" and method == "read_weight_sync_environment"
            return [{"origin": int.from_bytes(engine, "little")}]

    core = NativeCoreBoundary()
    engine = SimpleNamespace(
        engine_core=core,
        vllm_config=SimpleNamespace(
            parallel_config=SimpleNamespace(
                data_parallel_size=2,
                data_parallel_index=0,
                data_parallel_size_local=2,
                data_parallel_rank_local=None,
                local_engines_only=False,
            )
        ),
    )
    rows = await call_all_receiver_workers(engine, "read_weight_sync_environment")
    assert [row["origin"] for row in rows] == [0, 1]
    assert all(row["receiver_transport"]["managed_dp_ranks"] == [0, 1] for row in rows)
    core.core_engines = [bytes([0, 0])]
    with pytest.raises(ValueError, match="every configured managed DP core"):
        await call_all_receiver_workers(engine, "read_weight_sync_environment")


def _distributed_digests(rank, rendezvous, output, mutate):
    dist.init_process_group("gloo", init_method=rendezvous, rank=rank, world_size=2, timeout=timedelta(seconds=20))
    try:
        groups = [dist.new_group([member]) for member in range(2)]
        model = torch.nn.Linear(3, 2, bias=False)
        model.weight.data.fill_(1 + int(mutate and rank == 1))
        model.register_parameter("expert", torch.nn.Parameter(torch.full((3,), float(rank))))
        model.expert.allreduce = False
        row = {
            "rank": rank,
            "digests": parameter_digests([model])["digests"],
            "replica_ranks": {
                "dense": dist.get_process_group_ranks(dist.group.WORLD),
                "expert": dist.get_process_group_ranks(groups[rank]),
            },
        }
        rows = [None, None]
        dist.all_gather_object(rows, row)
        if rank == 0:
            try:
                result = {"comparisons": validate_replica_digests(rows)}
            except ValueError as error:
                result = {"error": str(error)}
            with open(output, "w") as stream:
                json.dump(result, stream)
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("mutate", [False, True])
def test_actual_gloo_allgather_compares_replicas_without_comparing_expert_shards(tmp_path, mutate):
    output = tmp_path / "result.json"
    multiprocessing.start_processes(
        _distributed_digests,
        args=(f"file://{tmp_path / 'rendezvous'}", str(output), mutate),
        nprocs=2,
        start_method="fork",
    )
    receipt = json.loads(output.read_text())
    if mutate:
        assert "dense parameter digest mismatch" in receipt["error"]
    else:
        assert [row["ranks"] for row in receipt["comparisons"]] == [[0, 1], [0], [1]]


def _readbacks():
    digest = {"parameters": 1, "bytes": 2, "sha256": "x" * 64}
    identity = {
        "rank": 0,
        "tp_rank": 0,
        "pp_rank": 0,
        "ep_rank": 0,
        "digests": {"dense": digest, "expert": digest},
        "replica_ranks": {"dense": [0], "expert": [0]},
    }
    environment = {"values": dict.fromkeys(ENVIRONMENT_KEYS)}
    policy = [
        {
            **identity,
            "environment": environment,
            "all_rank_digests": [identity],
            "expert_samples": [],
            "expert_layouts": [],
        }
    ]
    receivers = [
        [
            {
                "rank": 0,
                "world_size": 1,
                "environment": copy.deepcopy(environment),
                "parallel_config": {"tensor_parallel_size": 1},
                "layers": [],
            }
        ]
    ]
    return policy, receivers


def _geometry():
    return {
        "policy_ranks": 1,
        "receiver_engines": 1,
        "tp_rank": 1,
        "pp_rank": 1,
        "ep_rank": 1,
        "receiver_ranks_per_engine": 1,
        "receiver_parallel": {"tensor_parallel_size": 1},
    }


@pytest.mark.parametrize("damage", ["missing_receiver", "missing_allgather", "invariance", "nccl"])
def test_initial_readback_rejects_incomplete_or_asymmetric_native_state(damage):
    policy, receivers = _readbacks()
    if damage == "missing_receiver":
        receivers = [[]]
    elif damage == "missing_allgather":
        policy[0]["all_rank_digests"] = None
    elif damage == "invariance":
        receivers[0][0]["environment"]["values"]["VLLM_BATCH_INVARIANT"] = "1"
    else:
        receivers[0][0]["environment"]["values"]["NCCL_PROTO"] = "Simple"
    with pytest.raises(ValueError):
        validate_initial_readback(policy, receivers, _geometry())


@pytest.mark.asyncio
async def test_zero_update_lifecycle_reads_installed_weights_without_training(capsys, tmp_path):
    policy, receivers = _readbacks()

    class NativeBoundary:
        initialized = False
        synced = False
        drained = False

        def init_weight_sync_state(self):
            self.initialized = True

        async def async_sync_policy_weights_to_inference_engines(self):
            assert self.initialized
            self.synced = True

        async def _drain_policy_event_loops(self):
            assert self.synced
            self.drained = True

        async def read_policy(self):
            assert self.drained
            return policy[0]

        def async_run_ray_method(self, dispatch, method):
            if method == "read_weight_sync_environment":
                return [self.read_policy_environment()]
            return [self.read_policy()]

        async def read_policy_environment(self):
            return {"rank": 0, "environment": policy[0]["environment"]}

        async def read_weight_sync_environment(self):
            return receivers

        async def read_publication_receiver_state(self):
            assert self.drained
            return receivers

        async def train(self):
            pytest.fail("Readback entered training")

    trainer = NativeBoundary()
    trainer.policy_model = trainer
    trainer.inference_engine_client = trainer
    receipt = await run_initial_readback(trainer, str(tmp_path), _geometry())
    assert receipt["updates"] == 0 and receipt["initial_syncs"] == 1
    assert receipt["policy"] == policy and receipt["receivers"] == receivers
    assert receipt["replica_digest_comparisons"][0]["bytes"] == 2
    assert json.loads(Path(receipt["durable_pre_sync"]["uri"]).read_text()) == receipt["pre_sync_environment"]
    assert receipt["precursor_coverage"]["policy_expert_samples"] is False
    chunks = [
        json.loads(line.removeprefix("WEIGHT_SYNC_PRE_GROUP_CHUNK ")) for line in capsys.readouterr().out.splitlines()
    ]
    assert reassemble_receipt(chunks) == receipt["pre_sync_environment"]

    def failed_group_init():
        raise RuntimeError("native group rendezvous failed")

    trainer.init_weight_sync_state = failed_group_init
    with pytest.raises(RuntimeError, match="rendezvous"):
        await run_initial_readback(trainer, str(tmp_path), _geometry())
    failed_chunks = [
        json.loads(line.removeprefix("WEIGHT_SYNC_PRE_GROUP_CHUNK ")) for line in capsys.readouterr().out.splitlines()
    ]
    assert reassemble_receipt(failed_chunks) == receipt["pre_sync_environment"]


def _object_store(monkeypatch, *, etag_of=None, on_stat=None):
    """An in-memory stand-in for S3 that records how each object was verified."""
    from skyrl_train.weight_sync import readback_diagnostics as module

    stored: dict[str, bytes] = {}
    reads: list[str] = []

    def write(path, payload):
        stored[path] = payload

    def read(path):
        reads.append(path)
        return stored[path]

    def stat(path):
        if on_stat is not None:
            on_stat(path)
        payload = stored[path]
        etag = etag_of(payload) if etag_of else hashlib.md5(payload).hexdigest()
        return {"ETag": f'"{etag}"', "size": len(payload)}

    monkeypatch.setattr(module, "write_bytes_atomic", write)
    monkeypatch.setattr(module, "read_bytes", read)
    monkeypatch.setattr(module, "stat_object", stat)
    monkeypatch.setattr(module, "_RECEIPTS_WRITTEN", itertools.count(1))
    return stored, reads


def test_matching_etag_verifies_the_stored_object_without_transferring_it(monkeypatch):
    stored, reads = _object_store(monkeypatch)
    receipt = persist_readback("s3://bucket/prefix", "stage", {"rows": list(range(64))})
    assert receipt["verify"] == "etag-md5"
    assert receipt["etag"] == receipt["md5"] == hashlib.md5(stored[receipt["uri"]]).hexdigest()
    assert receipt["sha256"] == hashlib.sha256(stored[receipt["uri"]]).hexdigest()
    assert reads == [], "the cheap verify must not read the object back"


def test_one_receipt_in_every_full_readback_every_is_still_read_in_full(monkeypatch):
    _, reads = _object_store(monkeypatch)
    verifies = [persist_readback("s3://bucket/p", f"s{i}", {"i": i})["verify"] for i in range(2 * FULL_READBACK_EVERY)]
    assert verifies.count("full-readback-sampled") == 2
    assert len(reads) == 2 and verifies[FULL_READBACK_EVERY - 1] == "full-readback-sampled"


@pytest.mark.parametrize(
    "etag_of",
    [
        pytest.param(lambda payload: hashlib.md5(payload).hexdigest() + "-3", id="multipart"),
        pytest.param(lambda payload: hashlib.sha256(payload).hexdigest(), id="not-a-payload-md5"),
        pytest.param(lambda payload: "", id="absent"),
    ],
)
def test_an_unusable_etag_falls_back_to_the_full_read_rather_than_failing(monkeypatch, etag_of):
    _, reads = _object_store(monkeypatch, etag_of=etag_of)
    receipt = persist_readback("s3://bucket/prefix", "stage", {"rows": [1, 2, 3]})
    assert receipt["verify"] == "full-readback-after-etag-unusable"
    assert reads == [receipt["uri"]]


def test_metadata_failure_does_not_fail_the_write(monkeypatch):
    def boom(path):
        raise TimeoutError("head timed out")

    _, reads = _object_store(monkeypatch, on_stat=boom)
    receipt = persist_readback("s3://bucket/prefix", "stage", {"rows": [1]})
    assert receipt["verify"] == "full-readback-after-etag-unusable"
    assert reads == [receipt["uri"]]


def test_corruption_is_still_rejected_on_both_paths(monkeypatch):
    from skyrl_train.weight_sync import readback_diagnostics as module

    stored, _ = _object_store(monkeypatch)
    original_write = module.write_bytes_atomic

    def corrupting_write(path, payload):
        original_write(path, payload[:-1] + b" ")

    monkeypatch.setattr(module, "write_bytes_atomic", corrupting_write)
    # The cheap path sees a digest that does not match, falls through, and the full
    # read-back -- still the authority -- rejects it.
    with pytest.raises(ValueError, match="differs from the original receipt"):
        persist_readback("s3://bucket/prefix", "stage", {"rows": [1, 2, 3]})


def test_a_local_path_keeps_the_full_read_back(monkeypatch, tmp_path):
    receipt = persist_readback(str(tmp_path), "stage", {"rows": [7]})
    assert receipt["verify"] in ("full-readback", "full-readback-sampled")
    assert "etag" not in receipt


def test_serialized_receipt_is_the_bytes_persist_readback_would_write(monkeypatch):
    stored, _ = _object_store(monkeypatch)
    receipt = {"phase": "installed-before-replay", "result": {"rows": list(range(32))}}
    written = persist_readback("s3://bucket/prefix", "stage", receipt)
    assert serialize_receipt(receipt) == stored[written["uri"]]


def test_a_frozen_receipt_ignores_later_mutation_of_the_live_result(monkeypatch):
    stored, _ = _object_store(monkeypatch)
    result = {"phase_seconds": {"install": 0.5}}
    frozen = serialize_receipt({"phase": "installed-before-replay", "result": result})
    # The interval result gains finish and timing fields after the row is captured.
    result["phase_seconds"]["finish"] = 0.013
    result["durable_receipt"] = {"uri": "s3://bucket/prefix/later.json"}
    written = persist_payload("s3://bucket/prefix", "stage", frozen)
    replayed = json.loads(stored[written["uri"]])
    assert replayed["result"] == {"phase_seconds": {"install": 0.5}}
    assert written["bytes"] == len(frozen)


def _publication_stub(deferred):
    from types import SimpleNamespace

    return SimpleNamespace(
        deferred_payloads=deferred, output_uri="s3://bucket/prefix", preparation_id="prep", receipt_index=0
    )


def test_a_deferred_payload_round_trips_including_on_the_failure_path(monkeypatch):
    from skyrl_train.weight_sync.shard_training import ShardTrainingPublication

    stored, _ = _object_store(monkeypatch)
    stub = _publication_stub([])
    row = {"phase": "installed-before-replay", "result": {"rows": [{"rank": i} for i in range(8)]}}
    assert ShardTrainingPublication.capture(stub, row) is None
    assert stub.deferred_payloads == [serialize_receipt(row)]

    # The failure path writes the same payloads after deferred_payloads is cleared, and must
    # write them as payloads: capture() would serialize an already-serialized payload.
    payloads, stub.deferred_payloads = stub.deferred_payloads, None
    written = ShardTrainingPublication.capture_payload(stub, payloads[0])
    assert json.loads(stored[written["uri"]]) == row
    with pytest.raises(TypeError):
        ShardTrainingPublication.capture(stub, payloads[0])


@pytest.mark.parametrize(
    "etag_of,reason,level",
    [
        pytest.param(lambda p: hashlib.md5(p).hexdigest() + "-3", "etag-multipart", "WARNING", id="multipart"),
        pytest.param(lambda p: "", "etag-absent", "WARNING", id="absent"),
        pytest.param(lambda p: hashlib.sha256(p).hexdigest(), "etag-digest-differs", "ERROR", id="digest-differs"),
    ],
)
def test_each_fallback_is_classified_and_reported_at_its_own_level(monkeypatch, caplog, etag_of, reason, level):
    _object_store(monkeypatch, etag_of=etag_of)
    with caplog.at_level(logging.WARNING, logger="skyrl_train.weight_sync.readback_diagnostics"):
        receipt = persist_readback("s3://bucket/prefix", "stage", {"rows": [1, 2, 3]})
    assert receipt["verify"] == "full-readback-after-etag-unusable"
    assert receipt["fallback_reason"] == reason
    assert [r.levelname for r in caplog.records] == [level]
    assert reason in caplog.records[0].getMessage()


def test_a_size_disagreement_is_reported_as_a_content_problem(monkeypatch, caplog):
    from skyrl_train.weight_sync import readback_diagnostics as module

    _object_store(monkeypatch)
    monkeypatch.setattr(module, "stat_object", lambda p: {"ETag": '"' + "0" * 32 + '"', "size": 1})
    with caplog.at_level(logging.WARNING, logger="skyrl_train.weight_sync.readback_diagnostics"):
        receipt = persist_readback("s3://bucket/prefix", "stage", {"rows": [1]})
    assert receipt["fallback_reason"] == "stored-size-differs"
    assert caplog.records[0].levelname == "ERROR"


def test_metadata_failure_is_a_warning_not_an_error(monkeypatch, caplog):
    def boom(path):
        raise TimeoutError("head timed out")

    _object_store(monkeypatch, on_stat=boom)
    with caplog.at_level(logging.WARNING, logger="skyrl_train.weight_sync.readback_diagnostics"):
        receipt = persist_readback("s3://bucket/prefix", "stage", {"rows": [1]})
    assert receipt["fallback_reason"] == "metadata-unavailable:TimeoutError"
    assert caplog.records[0].levelname == "WARNING"


def test_by_design_fallbacks_are_silent(monkeypatch, caplog, tmp_path):
    # A non-S3 store and the sampling schedule are expected, not degradations.
    _object_store(monkeypatch)
    with caplog.at_level(logging.WARNING, logger="skyrl_train.weight_sync.readback_diagnostics"):
        local = persist_readback(str(tmp_path), "stage", {"rows": [1]})
        first = persist_readback("s3://bucket/prefix", "sampled", {"rows": [2]})
    assert local["fallback_reason"] == "non-s3-store"
    assert first["verify"] in ("etag-md5", "full-readback-sampled")
    assert caplog.records == []


def test_the_cheap_path_logs_nothing(monkeypatch, caplog):
    _object_store(monkeypatch)
    with caplog.at_level(logging.WARNING, logger="skyrl_train.weight_sync.readback_diagnostics"):
        receipt = persist_readback("s3://bucket/prefix", "stage", {"rows": list(range(8))})
    assert receipt["verify"] == "etag-md5" and caplog.records == []
