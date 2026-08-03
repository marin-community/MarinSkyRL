import asyncio
import json

import pytest

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
        kind=collective_phase_diagnostics.CollectiveRegionKind.POLICY_TRAINING_STEP,
        rank=2,
        metadata=collective_phase_diagnostics.CollectiveRegionMetadata(global_step=3, local_step=7),
    ):
        messages: list[str] = []
        monkeypatch.setattr(collective_phase_diagnostics.logger, "info", messages.append)
        collective_phase_diagnostics.log_phase(collective_phase_diagnostics.CollectivePhase.MODEL_FORWARD_ENTER)

        payload = json.loads(messages[0].removeprefix(collective_phase_diagnostics.LOG_PREFIX))
        assert payload["region_id"] > 0
        assert payload["kind"] == "policy_training_step"
        assert payload["rank"] == 2
        assert payload["phase"] == "model_forward_enter"
        assert payload["metadata"] == {"global_step": 3, "local_step": 7}
        assert payload["snapshot"]["mesh_dim_names"] == ["fsdp", "ep"]
        assert payload["snapshot"]["mesh_shape"] == [2, 2]
        assert payload["snapshot"]["mesh_coordinate"] == [1, 0]
        assert payload["snapshot"]["sequence_numbers"] == {"world": 31, "fsdp": 17, "ep": 23}

    collective_phase_diagnostics.log_phase(collective_phase_diagnostics.CollectivePhase.FORWARD_EXIT)
    assert len(messages) == 1


def test_moe_boundary_guard_crosses_asyncio_to_thread_and_resets_by_phase(monkeypatch):
    monkeypatch.setenv("SKYRL_COLLECTIVE_PHASE_DIAGNOSTICS", "1")
    world = FakeProcessGroup(1)
    mesh = FakeDeviceMesh(FakeProcessGroup(2), FakeProcessGroup(3))
    monkeypatch.setattr(collective_phase_diagnostics, "_default_process_group", lambda: world)

    messages: list[str] = []
    monkeypatch.setattr(collective_phase_diagnostics.logger, "info", messages.append)
    with collective_phase_diagnostics.region(
        mesh,
        kind=collective_phase_diagnostics.CollectiveRegionKind.POLICY_TRAINING_STEP,
        rank=0,
    ):
        collective_phase_diagnostics.start_phase(collective_phase_diagnostics.CollectivePhase.MODEL_FORWARD_ENTER)

        async def record_forward_boundary() -> None:
            await asyncio.to_thread(collective_phase_diagnostics.log_moe_ep_boundary_once)
            await asyncio.to_thread(collective_phase_diagnostics.log_moe_ep_boundary_once)

        asyncio.run(record_forward_boundary())
        collective_phase_diagnostics.start_phase(collective_phase_diagnostics.CollectivePhase.BACKWARD_ENTER)
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
    assert len({payload["region_id"] for payload in payloads}) == 1


def test_disabled_diagnostics_do_not_read_process_groups(monkeypatch):
    monkeypatch.delenv("SKYRL_COLLECTIVE_PHASE_DIAGNOSTICS", raising=False)
    world = FakeProcessGroup(1)
    fsdp = FakeProcessGroup(2)
    ep = FakeProcessGroup(3)
    mesh = FakeDeviceMesh(fsdp, ep)
    monkeypatch.setattr(collective_phase_diagnostics, "_default_process_group", lambda: world)

    messages: list[str] = []
    monkeypatch.setattr(collective_phase_diagnostics.logger, "info", messages.append)
    with collective_phase_diagnostics.region(
        mesh,
        kind=collective_phase_diagnostics.CollectiveRegionKind.POLICY_TRAINING_STEP,
        rank=0,
    ):
        collective_phase_diagnostics.log_phase(collective_phase_diagnostics.CollectivePhase.MODEL_FORWARD_ENTER)
        collective_phase_diagnostics.log_moe_ep_boundary_once()

    assert [world.reads, fsdp.reads, ep.reads] == [0, 0, 0]
    assert messages == []


def test_enabled_capture_failure_propagates(monkeypatch):
    monkeypatch.setenv("SKYRL_COLLECTIVE_PHASE_DIAGNOSTICS", "1")
    mesh = FakeDeviceMesh(FakeProcessGroup(2), FakeProcessGroup(3))
    monkeypatch.setattr(mesh, "get_coordinate", lambda: (_ for _ in ()).throw(RuntimeError("coordinate lost")))
    monkeypatch.setattr(collective_phase_diagnostics, "_default_process_group", lambda: FakeProcessGroup(1))

    with collective_phase_diagnostics.region(
        mesh,
        kind=collective_phase_diagnostics.CollectiveRegionKind.POLICY_TRAINING_STEP,
        rank=0,
    ):
        with pytest.raises(RuntimeError, match="coordinate lost"):
            collective_phase_diagnostics.log_phase(collective_phase_diagnostics.CollectivePhase.BACKWARD_ENTER)
