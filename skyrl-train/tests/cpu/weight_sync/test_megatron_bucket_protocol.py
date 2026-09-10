"""Actual Megatron interval and receiver operations across simulated CUDA/NCCL I/O."""

import ast
import asyncio
import json
from contextlib import nullcontext
from pathlib import Path
from threading import Condition, main_thread, current_thread
from types import SimpleNamespace
import weakref

import pytest
import torch

from skyrl_train.weight_sync import WeightChunk
from skyrl_train.weight_sync import megatron_bucket_protocol as protocol
from skyrl_train.weight_sync.policy_weight_access import PolicyWeightAccess
from tests.cpu.weight_sync.test_worker_bucket_protocol import native_protocol as native_protocol


def policy_methods():
    path = Path(__file__).parents[3] / "skyrl_train/workers/megatron/megatron_worker.py"
    module = ast.parse(path.read_text())
    cls = next(
        node for node in module.body if isinstance(node, ast.ClassDef) and node.name == "MegatronPolicyWorkerBase"
    )
    methods = [
        node
        for node in cls.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and (
            node.name in {"ppo_train", "diagnostic_bucket_install_and_replay", "prepare_reference_timing"}
            or node.name.endswith("_bucket_timing")
        )
    ]
    assert len(methods) == 8
    selected = ast.ClassDef(name="ActualPolicyMethods", bases=[], keywords=[], body=methods, decorator_list=[])
    tree = ast.fix_missing_locations(ast.Module(body=[selected], type_ignores=[]))
    namespace = {"mpu": SimpleNamespace(get_data_parallel_rank=lambda: 0, get_expert_data_parallel_rank=lambda: 0)}
    exec(compile(tree, str(path), "exec"), namespace)
    return namespace["ActualPolicyMethods"]


@pytest.fixture
def gate(request, monkeypatch, tmp_path):
    case = request.getfixturevalue("native_protocol")
    monkeypatch.setenv("IRIS_ATTEMPT_UID", "cpu-timing-original-attempt")
    monkeypatch.setattr(protocol, "BUCKET_BYTES", 32)
    monkeypatch.setattr(protocol, "MAX_REPLAY_EXTRA_BYTES", 16)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: torch.device("cpu"))
    monkeypatch.setattr(torch.cuda, "device", lambda device: nullcontext())
    monkeypatch.setattr(torch.cuda, "stream", lambda stream: nullcontext())
    monkeypatch.setattr(torch.distributed, "get_rank", lambda: 0)
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda: 1)
    monkeypatch.setattr(torch.distributed, "barrier", lambda: None)
    monkeypatch.setattr(
        torch.distributed, "all_gather_object", lambda output, value, group=None: output.__setitem__(0, value)
    )
    monkeypatch.setattr(torch.distributed, "new_group", lambda **kwargs: "cpu-gloo-group")
    monkeypatch.setattr(torch.distributed, "destroy_process_group", lambda group: None)
    condition = Condition()
    wire = []

    def broadcast(tensor, src, group=None):
        assert src == 0
        if group is None:
            return  # Actual one-rank policy WORLD has no peer to copy from.
        assert group == "native-custom-group"
        with condition:
            if current_thread() is not main_thread():
                wire.append(tensor.clone())
                condition.notify_all()
            else:
                assert condition.wait_for(lambda: bool(wire), timeout=5), "sender failed to reach native broadcast"
                received = wire.pop(0)
                assert tensor.shape == received.shape
                tensor.copy_(received)

    monkeypatch.setattr(torch.distributed, "broadcast", broadcast)
    policy = policy_methods()()
    policy._policy_weight_access = PolicyWeightAccess()
    policy._completed_update = None
    policy._model_update_group = "native-custom-group"
    policy.use_cuda_ipc = False
    source_parameter = torch.nn.Parameter(torch.ones(2))
    model = torch.nn.Module()
    model.register_parameter("weight", source_parameter)
    policy.actor_module = [model]
    sources = case.parts[1]
    policy.weight_extractor = SimpleNamespace(
        enable_bucketing=False,
        model_type="grug_moe",
        extract_weights=lambda dtype: iter(
            [
                WeightChunk(names=[name], tensors=[tensor], dtypes=[str(tensor.dtype)], shapes=[list(tensor.shape)])
                for name, tensor in sources.items()
            ]
        ),
    )
    tasks = []
    for name, tensor in sources.items():
        expert = tensor.ndim == 3
        kind = "GrugStackedExpertMapping" if expert else "AutoMapping"
        for index, value in enumerate(tensor.unbind(0) if expert else (tensor,)):
            mapping = type(kind, (), {})()
            mapping.hf_param = name
            tasks.append(
                SimpleNamespace(
                    global_param_name=f"{name}{index}" if expert else name,
                    mapping=mapping,
                    param_weight=value,
                )
            )
    policy.bridge = SimpleNamespace(get_conversion_tasks=lambda model: tasks)
    policy.provider = SimpleNamespace(tensor_model_parallel_size=1, num_moe_experts=4)
    policy.cfg = SimpleNamespace(
        trainer=SimpleNamespace(weight_sync_readback_output=str(tmp_path / "receiver-finishes")),
        generator=SimpleNamespace(
            model_dtype="bfloat16",
            num_inference_engines=1,
            inference_engine_tensor_parallel_size=1,
            inference_engine_data_parallel_size=1,
            inference_engine_pipeline_parallel_size=1,
        ),
    )

    class Client:
        def __init__(self):
            self.entered = asyncio.Event()
            self.allow_prepare = asyncio.Event()
            self.allow_prepare.set()
            self.receives = 0
            self.corrupt_after_install = False
            self.restart_after_install = False

        async def prepare_diagnostic_weight_sync_buckets(self, payload, manifest_id):
            self.entered.set()
            await self.allow_prepare.wait()
            return [[case.worker.prepare_diagnostic_weight_sync_buckets(payload, manifest_id)]]

        async def begin_diagnostic_weight_sync(self, manifest_id, publication_id):
            return [[case.worker.begin_diagnostic_weight_sync(manifest_id, publication_id)]]

        async def receive_diagnostic_weight_sync_bucket(self, bucket_id, replay=False, **identity):
            self.receives += 1

            # Real receiver actors block in another process. Let this single-process
            # fixture wait for posted transport without blocking the driver's loop.
            def transport_ready():
                with condition:
                    assert condition.wait_for(lambda: bool(wire), timeout=5)

            await asyncio.to_thread(transport_ready)
            result = case.worker.receive_diagnostic_weight_sync_bucket(bucket_id, replay=replay, **identity)
            if replay and self.restart_after_install:
                result["identity"]["pid"] += 1
            return [[result]]

        async def finish_diagnostic_weight_sync_install(self, **identity):
            result = case.worker.finish_diagnostic_weight_sync_install(**identity)
            if self.corrupt_after_install:
                case.parts[2]["model.embed_tokens.weight"].view(torch.uint8).view(-1)[0] ^= 1
            return [[result]]

        async def finish_diagnostic_weight_sync_replay(self, **identity):
            return [[case.worker.finish_diagnostic_weight_sync_replay(**identity)]]

        async def close_diagnostic_weight_sync_buckets(self, **identity):
            case.worker.close_diagnostic_weight_sync_buckets(**identity)

    return SimpleNamespace(policy=policy, client=Client(), receiver=case, parameter=source_parameter)


@pytest.mark.asyncio
async def test_actual_awaited_interval_installs_and_replays_every_receiver_byte(gate):
    receipt = await gate.policy.diagnostic_bucket_install_and_replay(gate.client)
    assert receipt["exclusive_weight_owner"] == "bucket-install-and-replay"
    assert receipt["completed_update_before"] is None and receipt["completed_update_after"] is None
    assert gate.policy._policy_weight_access.owner is None
    install, replay = receipt["phases"]["install"], receipt["phases"]["replay"]
    assert install["receivers"][0]["completed_slots"] == [0, 1]
    assert replay["receivers"][0]["coverage"] == 1.0 and replay["receivers"][0]["mismatches"] == 0
    assert replay["receivers"][0]["compared_bytes"] == receipt["prepared_receivers"][0]["installed_parameter_bytes"]
    assert gate.client.receives == 2 * install["sender"]["bucket_count"]
    assert install["sender"]["wire_bytes"] == replay["sender"]["wire_bytes"]
    assert install["seconds"] > 0 and replay["seconds"] > 0


@pytest.mark.asyncio
async def test_actual_ppo_entry_cannot_enter_while_diagnostic_is_awaiting_io(gate):
    gate.client.allow_prepare.clear()
    task = asyncio.create_task(gate.policy.diagnostic_bucket_install_and_replay(gate.client))
    await asyncio.wait_for(gate.client.entered.wait(), timeout=5)
    with pytest.raises(RuntimeError, match="already owned by bucket-install-and-replay"):
        gate.policy.ppo_train(None)
    with pytest.raises(RuntimeError, match="already owned"):
        await gate.policy.diagnostic_bucket_install_and_replay(gate.client)
    gate.client.allow_prepare.set()
    receipt = await asyncio.wait_for(task, timeout=10)
    assert receipt["completed_update_after"] is None
    assert gate.policy._policy_weight_access.owner is None


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["corruption", "restart"])
async def test_native_byte_error_or_receiver_restart_rejects_full_proof(gate, failure):
    gate.client.corrupt_after_install = failure == "corruption"
    gate.client.restart_after_install = failure == "restart"
    with pytest.raises(ValueError, match="Full-byte replay|identity changed"):
        await gate.policy.diagnostic_bucket_install_and_replay(gate.client)
    assert gate.policy._policy_weight_access.owner is None


@pytest.mark.asyncio
async def test_frozen_parameter_mutation_before_install_rejects(gate):
    gate.client.allow_prepare.clear()
    task = asyncio.create_task(gate.policy.diagnostic_bucket_install_and_replay(gate.client))
    await asyncio.wait_for(gate.client.entered.wait(), timeout=5)
    with torch.no_grad():
        gate.parameter.add_(1)
    gate.client.allow_prepare.set()
    with pytest.raises(ValueError, match="changed inside"):
        await asyncio.wait_for(task, timeout=10)
    assert gate.client.receives == 0
    assert gate.policy._policy_weight_access.owner is None


@pytest.mark.asyncio
async def test_new_sender_replay_allocation_is_not_excluded_from_scratch_gate(gate, monkeypatch):
    finish_receiver = gate.client.finish_diagnostic_weight_sync_replay

    async def receiver_then_sender_allocator():
        receipt = await finish_receiver()
        # Separate native processes have separate allocators. Switch the CPU
        # boundary back to the sender's larger peak after receiver completion.
        monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda device: 4017)
        return receipt

    gate.client.finish_diagnostic_weight_sync_replay = receiver_then_sender_allocator
    with pytest.raises(ValueError, match="Sender replay introduced"):
        await gate.policy.diagnostic_bucket_install_and_replay(gate.client)
    assert gate.policy._policy_weight_access.owner is None


@pytest.mark.asyncio
async def test_receiver_prepare_rejection_closes_allocated_state(gate):
    prepare = gate.client.prepare_diagnostic_weight_sync_buckets

    async def wrong_manifest(payload, manifest_id):
        result = await prepare(payload, manifest_id)
        result[0][0]["manifest_id"] = "wrong-manifest"
        return result

    gate.client.prepare_diagnostic_weight_sync_buckets = wrong_manifest
    with pytest.raises(ValueError, match="prepared a different manifest"):
        await gate.policy.diagnostic_bucket_install_and_replay(gate.client)
    assert not hasattr(gate.receiver.worker, "_diagnostic_bucket_state")
    assert gate.policy._policy_weight_access.owner is None


@pytest.mark.asyncio
async def test_completed_exporter_is_released_before_replay_can_allocate(gate, monkeypatch):
    timed_sender = protocol.StreamingBucketSender
    replay_sender = protocol.FrozenViewBucketSender
    references = []

    class ObservedTimedSender(timed_sender):
        def __init__(self, *args):
            super().__init__(*args)
            references.append(weakref.ref(self))

    class ObservedReplaySender(replay_sender):
        def __init__(self, *args):
            assert references and references[0]() is None, "timed conversion storage remained live at replay entry"
            super().__init__(*args)

    monkeypatch.setattr(protocol, "StreamingBucketSender", ObservedTimedSender)
    monkeypatch.setattr(protocol, "FrozenViewBucketSender", ObservedReplaySender)
    await gate.policy.diagnostic_bucket_install_and_replay(gate.client)


@pytest.mark.asyncio
async def test_replay_only_catalogue_gpu_allocation_is_included_in_proof_gate(gate, monkeypatch):
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda device: 4017)
    with pytest.raises(ValueError, match="source catalogue introduced"):
        await gate.policy.diagnostic_bucket_install_and_replay(gate.client)
    assert gate.client.receives == 0
    assert not hasattr(gate.receiver.worker, "_diagnostic_bucket_state")
    assert gate.policy._policy_weight_access.owner is None


def test_external_dp_bucket_receipts_keep_every_actor_identity():
    actors = [
        [{"identity": {"rank": rank, "world_size": 8, "host": "receiver", "gpu_uuid": f"gpu-{rank}"}}]
        for rank in range(8)
    ]
    rows, identities = protocol.receiver_rows(actors, engine_count=1, ranks_per_engine=8, data_parallel_size=8)
    assert [row["identity"]["rank"] for row in rows] == list(range(8))
    assert len(identities) == 8
    with pytest.raises(ValueError, match="external DP actors"):
        protocol.receiver_rows(actors[:-1], engine_count=1, ranks_per_engine=8, data_parallel_size=8)
    actors[7] = actors[0]
    with pytest.raises(ValueError, match="factory slice"):
        protocol.receiver_rows(actors, engine_count=1, ranks_per_engine=8, data_parallel_size=8)


@pytest.mark.asyncio
async def test_failed_memory_gate_retains_raw_receiver_receipt(gate, monkeypatch):
    original = gate.client.finish_diagnostic_weight_sync_replay

    async def finish_with_excess_allocation():
        monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda device: 4017)
        return await original()

    monkeypatch.setattr(gate.client, "finish_diagnostic_weight_sync_replay", finish_with_excess_allocation)
    with pytest.raises(ValueError, match="allocation gate failed"):
        await gate.policy.diagnostic_bucket_install_and_replay(gate.client)
    paths = list(Path(gate.policy.cfg.trainer.weight_sync_readback_output).glob("bucket-replay-receivers-*.json"))
    assert len(paths) == 1
    raw = json.loads(paths[0].read_text())
    row = raw["receivers"][0]
    assert raw["validation_status"] == "not_yet_validated"
    assert row["replay_peak_extra_bytes"] == 17 and not row["replay_memory_within_limit"]
    assert row["compared_bytes"] == row["expected_bytes"] and row["mismatches"] == 0
    assert gate.policy._policy_weight_access.owner is None


@pytest.mark.asyncio
async def test_cached_timing_refreshes_materialized_exports_after_optimizer_updates(gate):
    sources = gate.receiver.parts[1]
    tensors = [*sources.values(), gate.parameter]
    for tensor in tensors:
        tensor.requires_grad_(True)
    optimizer = torch.optim.SGD(tensors, lr=2.0)

    def materialize(dtype):
        for name, tensor in sources.items():
            value = (
                torch.stack(tuple(tensor.unbind(0))) if tensor.ndim == 3 else torch.cat(tensor.chunk(2, dim=0), dim=0)
            )
            yield WeightChunk(names=[name], tensors=[value], dtypes=[str(value.dtype)], shapes=[list(value.shape)])

    gate.policy.weight_extractor.extract_weights = materialize
    await gate.policy.prepare_bucket_timing(gate.client)
    pointers = [buffer.data_ptr() for buffer in gate.policy._bucket_timing_session.prepared.buffers]
    previous = None
    for version in (1, 2):
        for tensor in tensors:
            tensor.grad = torch.ones_like(tensor)
        optimizer.step()
        optimizer.zero_grad()
        gate.policy._completed_update = version
        await gate.policy.begin_bucket_timing(gate.client, version)
        with pytest.raises(RuntimeError, match="already owned by bucket-timing"):
            gate.policy.ppo_train(None)
        install = await gate.policy.install_bucket_timing(gate.client, version)
        assert gate.policy._policy_weight_access.owner == "bucket-timing"
        paths = Path(gate.policy.cfg.trainer.weight_sync_readback_output)
        assert not list(paths.glob(f"bucket-*-receivers-sync-{version}-*.json"))
        proof = await gate.policy.replay_bucket_timing(gate.client, version)
        assert proof["replay"]["receivers"][0]["mismatches"] == 0
        assert install["install"]["sender"]["wire_bytes"] == proof["replay"]["sender"]["wire_bytes"]
        assert len(list(paths.glob(f"bucket-*-receivers-sync-{version}-*.json"))) == 2
        assert gate.policy._policy_weight_access.owner is None
        assert [buffer.data_ptr() for buffer in gate.policy._bucket_timing_session.prepared.buffers] == pointers
        installed = {name: tensor.detach().clone() for name, tensor in gate.receiver.parts[2].items()}
        if previous is not None:
            assert all(not torch.equal(installed[name], old) for name, old in previous.items())
        previous = installed
    with pytest.raises(ValueError, match="active weight-sync version"):
        await gate.policy.close_bucket_timing(gate.client, 1)
    assert hasattr(gate.policy, "_bucket_timing_session")
    await gate.policy.close_bucket_timing(gate.client, 2)
    assert not hasattr(gate.policy, "_bucket_timing_session")


@pytest.mark.asyncio
async def test_cancelled_timing_install_releases_policy_and_receiver_storage(gate):
    await gate.policy.prepare_bucket_timing(gate.client)
    await gate.policy.begin_bucket_timing(gate.client, 1)
    entered = asyncio.Event()

    async def pending_finish(**identity):
        entered.set()
        await asyncio.Event().wait()

    gate.client.finish_diagnostic_weight_sync_install = pending_finish
    task = asyncio.create_task(gate.policy.install_bucket_timing(gate.client, 1))
    await asyncio.wait_for(entered.wait(), timeout=5)
    with pytest.raises(RuntimeError, match="already owned"):
        gate.policy.ppo_train(None)
    with pytest.raises(RuntimeError, match="invalid during phase"):
        await gate.policy.close_bucket_timing(gate.client, 1)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert gate.policy._policy_weight_access.owner is None
    assert not hasattr(gate.policy, "_bucket_timing_session")
    assert not hasattr(gate.receiver.worker, "_diagnostic_bucket_state")


@pytest.mark.asyncio
async def test_cached_timing_source_storage_change_closes_prepared_receiver(gate):
    await gate.policy.prepare_bucket_timing(gate.client)
    source = next(iter(gate.receiver.parts[1].values()))
    source.data = source.clone()
    with pytest.raises(ValueError, match="mapping/storage changed"):
        await gate.policy.begin_bucket_timing(gate.client, 1)
    assert gate.policy._policy_weight_access.owner is None
    assert not hasattr(gate.policy, "_bucket_timing_session")
    assert not hasattr(gate.receiver.worker, "_diagnostic_bucket_state")


@pytest.mark.asyncio
async def test_cached_timing_replay_failure_retains_raw_values_then_releases(gate, monkeypatch):
    await gate.policy.prepare_bucket_timing(gate.client)
    await gate.policy.begin_bucket_timing(gate.client, 1)
    await gate.policy.install_bucket_timing(gate.client, 1)
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda device: 4017)
    with pytest.raises(ValueError, match="allocation gate failed"):
        await gate.policy.replay_bucket_timing(gate.client, 1)
    path = next(Path(gate.policy.cfg.trainer.weight_sync_readback_output).glob("bucket-replay-receivers-sync-1-*.json"))
    raw = json.loads(path.read_text())
    assert raw["receivers"][0]["replay_peak_extra_bytes"] == 17
    assert gate.policy._policy_weight_access.owner is None
    assert not hasattr(gate.receiver.worker, "_diagnostic_bucket_state")


@pytest.mark.asyncio
async def test_reference_timing_defers_storage_binding_until_after_native_load(gate):
    from skyrl_train.weight_sync.reference_bucket_protocol import begin_reference_sync, finish_reference_sync
    from skyrl_train.weight_sync.wire_inventory import WireInventory
    from tests.cpu.weight_sync.test_reference_bucket_protocol import replace_and_load

    gate.policy.cfg.generator.weight_sync_wire_inventory = True

    async def begin_reference(manifest_id, publication_id):
        return [[begin_reference_sync(gate.receiver.worker, manifest_id, publication_id)]]

    async def finish_reference(manifest_id, publication_id):
        return [[finish_reference_sync(gate.receiver.worker, manifest_id, publication_id)]]

    async def original_broadcast(client):
        assert gate.policy._policy_weight_access.owner == "bucket-timing"
        replace_and_load(gate.receiver)
        inventory = WireInventory()
        for name, tensor in gate.receiver.parts[1].items():
            inventory.observe(name, tensor)
        gate.policy._weight_sync_wire_receipt = inventory.finish(completed_update=gate.policy._completed_update)

    gate.client.begin_reference_bucket_sync = begin_reference
    gate.client.finish_reference_bucket_sync = finish_reference
    gate.policy.broadcast_to_inference_engines = original_broadcast
    gate.policy.read_weight_sync_wire_inventory = lambda: gate.policy._weight_sync_wire_receipt
    await gate.policy.prepare_reference_timing(gate.client)
    for version in (1, 2):
        for tensor in gate.receiver.parts[1].values():
            tensor.add_(2)
        gate.policy._completed_update = version
        await gate.policy.begin_bucket_timing(gate.client, version)
        install = await gate.policy.install_bucket_timing(gate.client, version)
        assert install["install"]["installation_mode"] == "original_loader"
        assert install["install"]["receivers"] is None
        assert gate.receiver.worker._diagnostic_bucket_state["receiver"].parameters == {}
        proof = await gate.policy.replay_bucket_timing(gate.client, version)
        assert proof["replay"]["receivers"][0]["mismatches"] == 0
        assert install["install"]["sender"]["wire_bytes"] == proof["replay"]["sender"]["wire_bytes"]
        assert gate.policy._policy_weight_access.owner is None
    await gate.policy.close_bucket_timing(gate.client, 2)


@pytest.mark.asyncio
async def test_timing_restart_cannot_repeat_a_recorded_measurement(gate):
    from skyrl_train.weight_sync.bucket_qualification import mark_measurement_once

    marker = mark_measurement_once(gate.policy.cfg.trainer.weight_sync_readback_output)
    original = Path(marker["uri"]).read_bytes()
    await gate.policy.prepare_bucket_timing(gate.client)
    with pytest.raises(ValueError, match="previous attempt"):
        await gate.policy.begin_bucket_timing(gate.client, 1)
    assert Path(marker["uri"]).read_bytes() == original
    assert not hasattr(gate.policy, "_bucket_timing_session")
    assert not hasattr(gate.receiver.worker, "_diagnostic_bucket_state")
    with gate.policy._policy_weight_access.hold("ppo"):
        pass


@pytest.mark.asyncio
async def test_preparation_observes_actual_sender_and_receiver_environment(gate, monkeypatch):
    monkeypatch.setenv("VLLM_BATCH_INVARIANT", "0")
    monkeypatch.setenv("NCCL_NTHREADS", "256")
    monkeypatch.setenv("NCCL_P2P_NET_DISABLE", "0")
    monkeypatch.setenv("UNRELATED_SECRET", "must-never-enter-receipt")
    receipt = await gate.policy.prepare_bucket_timing(gate.client)
    for row in [receipt, *receipt["prepared_receivers"]]:
        values = row["environment"]["values"]
        assert values["VLLM_BATCH_INVARIANT"] == "0"
        assert values["NCCL_NTHREADS"] == "256"
        assert values["NCCL_P2P_NET_DISABLE"] == "0"
        assert "UNRELATED_SECRET" not in values
    await gate.policy.close_bucket_timing(gate.client, None)


@pytest.mark.asyncio
async def test_pipeline_packs_next_bucket_while_receiver_receipt_is_held(gate, monkeypatch):
    gate.policy.cfg.generator.weight_sync_bucket_pipeline = True
    held = asyncio.Event()
    release = asyncio.Event()
    prefetched = asyncio.Event()
    receive = gate.client.receive_diagnostic_weight_sync_bucket
    pack = protocol.StreamingBucketSender.pack_next_bucket

    async def held_receive(bucket_id, replay=False, **identity):
        if bucket_id == 0 and not replay:
            held.set()
        result = await receive(bucket_id, replay=replay, **identity)
        if bucket_id == 0 and not replay:
            await release.wait()
        return result

    def observed_pack(sender):
        result = pack(sender)
        if sender.next_bucket == 2:
            assert held.is_set() and not release.is_set()
            prefetched.set()
        return result

    monkeypatch.setattr(gate.client, "receive_diagnostic_weight_sync_bucket", held_receive)
    monkeypatch.setattr(protocol.StreamingBucketSender, "pack_next_bucket", observed_pack)
    task = asyncio.create_task(gate.policy.diagnostic_bucket_install_and_replay(gate.client))
    try:
        await asyncio.wait_for(prefetched.wait(), timeout=5)
    finally:
        release.set()
    receipt = await asyncio.wait_for(task, timeout=10)
    install, replay = receipt["phases"]["install"], receipt["phases"]["replay"]
    assert install["pipeline_enabled"] and not replay["pipeline_enabled"]
    assert install["sender"]["send_completion_joined"]
    assert replay["receivers"][0]["coverage"] == 1.0
    assert replay["receivers"][0]["mismatches"] == 0
    assert replay["receivers"][0]["compared_bytes"] == receipt["prepared_receivers"][0]["installed_parameter_bytes"]


@pytest.mark.asyncio
async def test_pipeline_prefetch_failure_joins_held_receiver_before_cleanup(gate, monkeypatch):
    gate.policy.cfg.generator.weight_sync_bucket_pipeline = True
    held = asyncio.Event()
    joined = asyncio.Event()
    receive = gate.client.receive_diagnostic_weight_sync_bucket
    pack = protocol.StreamingBucketSender.pack_next_bucket

    async def held_receive(bucket_id, replay=False, **identity):
        if bucket_id == 0 and not replay:
            held.set()
            try:
                await receive(bucket_id, replay=replay, **identity)
                await asyncio.Event().wait()
            finally:
                joined.set()
        return await receive(bucket_id, replay=replay, **identity)

    def failed_prefetch(sender):
        if sender.next_bucket == 1:
            assert held.is_set()
            raise RuntimeError("prefetch export failed")
        return pack(sender)

    monkeypatch.setattr(gate.client, "receive_diagnostic_weight_sync_bucket", held_receive)
    monkeypatch.setattr(protocol.StreamingBucketSender, "pack_next_bucket", failed_prefetch)
    with pytest.raises(RuntimeError, match="prefetch export failed"):
        await asyncio.wait_for(gate.policy.diagnostic_bucket_install_and_replay(gate.client), timeout=10)
    assert joined.is_set()
    assert gate.policy._policy_weight_access.owner is None


@pytest.mark.asyncio
async def test_reference_timing_rejects_bucket_pipeline_before_preparation(gate):
    gate.policy.cfg.generator.weight_sync_bucket_pipeline = True
    gate.policy.cfg.generator.weight_sync_wire_inventory = True
    with pytest.raises(ValueError, match="Bucket pipeline requires bucket timing mode"):
        await gate.policy.prepare_reference_timing(gate.client)
    assert not gate.client.entered.is_set()
    assert gate.policy._policy_weight_access.owner is None


@pytest.mark.asyncio
async def test_pipeline_posts_each_receiver_before_advancing_its_export(gate, monkeypatch):
    gate.policy.cfg.generator.weight_sync_bucket_pipeline = True
    posted = []
    create = asyncio.create_task
    pack = protocol.StreamingBucketSender.pack_next_bucket

    def observed_create(coro, **kwargs):
        if coro.cr_code.co_name == "receive_diagnostic_weight_sync_bucket":
            posted.append(coro.cr_frame.f_locals["bucket_id"])
        return create(coro, **kwargs)

    def observed_pack(sender):
        assert posted and posted[-1] == sender.next_bucket
        return pack(sender)

    monkeypatch.setattr(asyncio, "create_task", observed_create)
    monkeypatch.setattr(protocol.StreamingBucketSender, "pack_next_bucket", observed_pack)
    result = await gate.policy.diagnostic_bucket_install_and_replay(gate.client)
    assert result["phases"]["replay"]["receivers"][0]["mismatches"] == 0


@pytest.mark.asyncio
async def test_repeated_cancellation_joins_native_send_before_releasing_policy(gate, monkeypatch):
    gate.policy.cfg.generator.weight_sync_bucket_pipeline = True
    from threading import Event

    entered, release = Event(), Event()
    original = torch.distributed.broadcast

    def held_broadcast(tensor, src, group=None):
        if group == "native-custom-group" and current_thread() is not main_thread():
            entered.set()
            assert release.wait(timeout=5)
        return original(tensor, src, group=group)

    monkeypatch.setattr(torch.distributed, "broadcast", held_broadcast)
    task = asyncio.create_task(gate.policy.diagnostic_bucket_install_and_replay(gate.client))
    try:
        assert await asyncio.to_thread(entered.wait, 5)
        task.cancel()
        await asyncio.sleep(0)
        task.cancel()
        await asyncio.sleep(0)
        with pytest.raises(asyncio.TimeoutError):
            await asyncio.wait_for(asyncio.shield(task), timeout=0.05)
        assert not task.done()
        assert gate.policy._policy_weight_access.owner == "bucket-install-and-replay"
    finally:
        release.set()
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(task, timeout=10)
    assert gate.policy._policy_weight_access.owner is None
