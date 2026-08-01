import asyncio
import json

from skyrl_train.distributed import collective_phase_diagnostics


class FakeProcessGroup:
    def __init__(self, sequence_number: int):
        self.sequence_number = sequence_number
        self.reads = 0

    def _get_sequence_number_for_group(self) -> int:
        self.reads += 1
        return self.sequence_number


class FakeDeviceMesh:
    mesh_dim_names = ("fsdp", "ep")
    shape = (2, 2)

    def __init__(self, fsdp_group: FakeProcessGroup, ep_group: FakeProcessGroup):
        self.groups = {"fsdp": fsdp_group, "ep": ep_group}

    def get_coordinate(self) -> list[int]:
        return [1, 0]

    def get_group(self, name: str) -> FakeProcessGroup:
        return self.groups[name]


def test_phase_record_includes_world_and_mesh_group_sequences(monkeypatch):
    monkeypatch.setenv("SKYRL_COLLECTIVE_PHASE_DIAGNOSTICS", "1")
    world = FakeProcessGroup(31)
    fsdp = FakeProcessGroup(17)
    ep = FakeProcessGroup(23)
    mesh = FakeDeviceMesh(fsdp, ep)
    monkeypatch.setattr(collective_phase_diagnostics, "_default_process_group", lambda: world)

    with collective_phase_diagnostics.region(
        mesh,
        kind="policy_training_step",
        rank=2,
        metadata={"global_step": 3, "local_step": 7},
    ) as region_id:
        messages: list[str] = []
        monkeypatch.setattr(collective_phase_diagnostics.logger, "info", messages.append)
        record = collective_phase_diagnostics.log_phase("model_forward_enter")

        assert record is not None
        assert record.region_id == region_id
        assert record.kind == "policy_training_step"
        assert record.rank == 2
        assert record.phase == "model_forward_enter"
        assert record.metadata == {"global_step": 3, "local_step": 7}
        assert record.mesh_dim_names == ("fsdp", "ep")
        assert record.mesh_shape == (2, 2)
        assert record.mesh_coordinate == (1, 0)
        assert record.sequence_numbers == {"world": 31, "fsdp": 17, "ep": 23}

        payload = json.loads(messages[0].removeprefix(collective_phase_diagnostics.LOG_PREFIX))
        assert payload["region_id"] == region_id
        assert payload["sequence_numbers"] == {"world": 31, "fsdp": 17, "ep": 23}

    assert collective_phase_diagnostics.log_phase("stale_phase") is None


def test_moe_boundary_guard_crosses_asyncio_to_thread_and_resets_by_phase(monkeypatch):
    monkeypatch.setenv("SKYRL_COLLECTIVE_PHASE_DIAGNOSTICS", "1")
    world = FakeProcessGroup(1)
    mesh = FakeDeviceMesh(FakeProcessGroup(2), FakeProcessGroup(3))
    monkeypatch.setattr(collective_phase_diagnostics, "_default_process_group", lambda: world)

    messages: list[str] = []
    monkeypatch.setattr(collective_phase_diagnostics.logger, "info", messages.append)
    with collective_phase_diagnostics.region(mesh, kind="policy_training_step", rank=0) as region_id:
        collective_phase_diagnostics.log_phase("model_forward_enter", reset_moe_boundary=True)

        async def record_forward_boundary() -> None:
            await asyncio.to_thread(collective_phase_diagnostics.log_moe_ep_boundary_once)
            await asyncio.to_thread(collective_phase_diagnostics.log_moe_ep_boundary_once)

        asyncio.run(record_forward_boundary())
        collective_phase_diagnostics.log_phase("backward_enter", reset_moe_boundary=True)
        collective_phase_diagnostics.log_moe_ep_boundary_once()
        collective_phase_diagnostics.log_moe_ep_boundary_once()

    payloads = [json.loads(message.removeprefix(collective_phase_diagnostics.LOG_PREFIX)) for message in messages]
    assert [payload["phase"] for payload in payloads] == [
        "model_forward_enter",
        "moe_ep_a2a_first",
        "backward_enter",
        "moe_ep_a2a_first",
    ]
    assert [payload["event_index"] for payload in payloads] == [0, 1, 2, 3]
    assert {payload["region_id"] for payload in payloads} == {region_id}


def test_disabled_diagnostics_do_not_read_process_groups(monkeypatch):
    monkeypatch.delenv("SKYRL_COLLECTIVE_PHASE_DIAGNOSTICS", raising=False)
    world = FakeProcessGroup(1)
    fsdp = FakeProcessGroup(2)
    ep = FakeProcessGroup(3)
    mesh = FakeDeviceMesh(fsdp, ep)
    monkeypatch.setattr(collective_phase_diagnostics, "_default_process_group", lambda: world)

    messages: list[str] = []
    monkeypatch.setattr(collective_phase_diagnostics.logger, "info", messages.append)
    with collective_phase_diagnostics.region(mesh, kind="policy_training_step", rank=0) as region_id:
        assert region_id is None
        assert collective_phase_diagnostics.log_phase("model_forward_enter") is None
        assert collective_phase_diagnostics.log_moe_ep_boundary_once() is None

    assert [world.reads, fsdp.reads, ep.reads] == [0, 0, 0]
    assert messages == []


def test_capture_failures_warn_without_interrupting_training(monkeypatch):
    monkeypatch.setenv("SKYRL_COLLECTIVE_PHASE_DIAGNOSTICS", "1")
    mesh = FakeDeviceMesh(FakeProcessGroup(2), FakeProcessGroup(3))
    monkeypatch.setattr(mesh, "get_coordinate", lambda: (_ for _ in ()).throw(RuntimeError("coordinate lost")))
    monkeypatch.setattr(collective_phase_diagnostics, "_default_process_group", lambda: FakeProcessGroup(1))

    warnings: list[str] = []
    monkeypatch.setattr(collective_phase_diagnostics.logger, "warning", warnings.append)
    with collective_phase_diagnostics.region(mesh, kind="policy_training_step", rank=0):
        assert collective_phase_diagnostics.log_phase("backward_enter") is None

    assert len(warnings) == 1
    assert "coordinate lost" in warnings[0]
