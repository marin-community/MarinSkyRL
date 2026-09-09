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
        and node.name in {"ppo_train", "diagnostic_bucket_install_and_replay"}
    ]
    assert len(methods) == 2
    selected = ast.ClassDef(name="ActualPolicyMethods", bases=[], keywords=[], body=methods, decorator_list=[])
    tree = ast.fix_missing_locations(ast.Module(body=[selected], type_ignores=[]))
    namespace = {"mpu": SimpleNamespace(get_data_parallel_rank=lambda: 0, get_expert_data_parallel_rank=lambda: 0)}
    exec(compile(tree, str(path), "exec"), namespace)
    return namespace["ActualPolicyMethods"]


@pytest.fixture
def gate(request, monkeypatch, tmp_path):
    case = request.getfixturevalue("native_protocol")
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

        async def receive_diagnostic_weight_sync_bucket(self, bucket_id, replay=False):
            self.receives += 1
            result = case.worker.receive_diagnostic_weight_sync_bucket(bucket_id, replay=replay)
            if replay and self.restart_after_install:
                result["identity"]["pid"] += 1
            return [[result]]

        async def finish_diagnostic_weight_sync_install(self):
            result = case.worker.finish_diagnostic_weight_sync_install()
            if self.corrupt_after_install:
                case.parts[2]["model.embed_tokens.weight"].view(torch.uint8).view(-1)[0] ^= 1
            return [[result]]

        async def finish_diagnostic_weight_sync_replay(self):
            return [[case.worker.finish_diagnostic_weight_sync_replay()]]

        async def close_diagnostic_weight_sync_buckets(self):
            case.worker.close_diagnostic_weight_sync_buckets()

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
