import asyncio
import json
from unittest.mock import patch

from skyrl_train.distributed import collective_phase_diag


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
    monkeypatch.setenv("SKYRL_COLLECTIVE_COUNT_DIAG", "1")
    world = FakeProcessGroup(31)
    fsdp = FakeProcessGroup(17)
    ep = FakeProcessGroup(23)
    mesh = FakeDeviceMesh(fsdp, ep)
    monkeypatch.setattr(collective_phase_diag, "_default_process_group", lambda: world)

    region_id = collective_phase_diag.begin_region(
        mesh,
        kind="policy_training_step",
        rank=2,
        metadata={"global_step": 3, "local_step": 7},
    )
    with patch.object(collective_phase_diag.logger, "info") as info:
        record = collective_phase_diag.log_phase("model_forward_enter")

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

    parsed = collective_phase_diag.parse_log_line(info.call_args.args[0])
    assert parsed == record
    collective_phase_diag.end_region()
    assert collective_phase_diag.log_phase("stale_phase") is None


def test_moe_boundary_guard_crosses_asyncio_to_thread_and_resets_by_phase(monkeypatch):
    monkeypatch.setenv("SKYRL_COLLECTIVE_COUNT_DIAG", "1")
    world = FakeProcessGroup(1)
    mesh = FakeDeviceMesh(FakeProcessGroup(2), FakeProcessGroup(3))
    monkeypatch.setattr(collective_phase_diag, "_default_process_group", lambda: world)

    region_id = collective_phase_diag.begin_region(mesh, kind="policy_training_step", rank=0)
    messages: list[str] = []
    with patch.object(collective_phase_diag.logger, "info", messages.append):
        collective_phase_diag.log_phase("model_forward_enter", reset_moe_boundary=True)

        async def record_forward_boundary() -> None:
            await asyncio.to_thread(collective_phase_diag.log_moe_ep_boundary_once)
            await asyncio.to_thread(collective_phase_diag.log_moe_ep_boundary_once)

        asyncio.run(record_forward_boundary())
        collective_phase_diag.log_phase("backward_enter", reset_moe_boundary=True)
        collective_phase_diag.log_moe_ep_boundary_once()
        collective_phase_diag.log_moe_ep_boundary_once()

    records = [collective_phase_diag.parse_log_line(message) for message in messages]
    assert [record.phase for record in records] == [
        "model_forward_enter",
        "moe_ep_a2a_first",
        "backward_enter",
        "moe_ep_a2a_first",
    ]
    assert [record.event_index for record in records] == [0, 1, 2, 3]
    assert {record.region_id for record in records} == {region_id}


def test_disabled_diagnostics_do_not_read_process_groups(monkeypatch):
    monkeypatch.delenv("SKYRL_COLLECTIVE_COUNT_DIAG", raising=False)
    world = FakeProcessGroup(1)
    fsdp = FakeProcessGroup(2)
    ep = FakeProcessGroup(3)
    mesh = FakeDeviceMesh(fsdp, ep)
    monkeypatch.setattr(collective_phase_diag, "_default_process_group", lambda: world)

    with patch.object(collective_phase_diag.logger, "info") as info:
        assert collective_phase_diag.begin_region(mesh, kind="policy_training_step", rank=0) is None
        assert collective_phase_diag.log_phase("model_forward_enter") is None
        assert collective_phase_diag.log_moe_ep_boundary_once() is None

    assert [world.reads, fsdp.reads, ep.reads] == [0, 0, 0]
    info.assert_not_called()


def test_capture_failures_warn_without_interrupting_training(monkeypatch):
    monkeypatch.setenv("SKYRL_COLLECTIVE_COUNT_DIAG", "1")
    mesh = FakeDeviceMesh(FakeProcessGroup(2), FakeProcessGroup(3))
    monkeypatch.setattr(mesh, "get_coordinate", lambda: (_ for _ in ()).throw(RuntimeError("coordinate lost")))
    monkeypatch.setattr(collective_phase_diag, "_default_process_group", lambda: FakeProcessGroup(1))
    collective_phase_diag.begin_region(mesh, kind="policy_training_step", rank=0)

    with patch.object(collective_phase_diag.logger, "warning") as warning:
        assert collective_phase_diag.log_phase("backward_enter") is None

    warning.assert_called_once()


def _record(
    rank: int,
    coordinate: tuple[int, int],
    event_index: int,
    phase: str,
    *,
    world: int,
    fsdp: int,
    ep: int,
) -> collective_phase_diag.CollectivePhaseRecord:
    return collective_phase_diag.CollectivePhaseRecord(
        region_id=4,
        event_index=event_index,
        kind="policy_training_step",
        rank=rank,
        phase=phase,
        metadata={"global_step": 3, "local_step": 7},
        mesh_dim_names=("fsdp", "ep"),
        mesh_shape=(2, 2),
        mesh_coordinate=coordinate,
        sequence_numbers={"world": world, "fsdp": fsdp, "ep": ep},
    )


def test_first_divergence_reports_the_process_group_and_rank_pair():
    records = []
    coordinates = ((0, 0), (0, 1), (1, 0), (1, 1))
    for rank, coordinate in enumerate(coordinates):
        records.append(_record(rank, coordinate, 0, "backward_enter", world=40, fsdp=12, ep=20))
        records.append(
            _record(
                rank,
                coordinate,
                1,
                "moe_ep_a2a_first",
                world=40,
                fsdp=12,
                ep=22 if rank == 1 else 21,
            )
        )

    divergence = collective_phase_diag.find_first_divergence(records)

    assert divergence is not None
    assert divergence.region_id == 4
    assert divergence.event_index == 1
    assert divergence.phase == "moe_ep_a2a_first"
    assert divergence.group_name == "ep"
    assert divergence.reference_rank == 0
    assert divergence.divergent_rank == 1
    assert divergence.kind == collective_phase_diag.DivergenceKind.SEQUENCE
    assert divergence.expected_sequence == 21
    assert divergence.actual_sequence == 22


def test_first_divergence_reports_a_rank_that_never_reaches_the_boundary():
    coordinates = ((0, 0), (0, 1), (1, 0), (1, 1))
    records = [
        _record(rank, coordinate, 0, "backward_enter", world=40, fsdp=12, ep=20)
        for rank, coordinate in enumerate(coordinates)
    ]
    records.extend(
        _record(rank, coordinate, 1, "moe_ep_a2a_first", world=40, fsdp=13, ep=21)
        for rank, coordinate in enumerate(coordinates)
        if rank != 2
    )

    divergence = collective_phase_diag.find_first_divergence(records)

    assert divergence is not None
    assert divergence.region_id == 4
    assert divergence.event_index == 1
    assert divergence.group_name == "world"
    assert divergence.reference_rank == 0
    assert divergence.divergent_rank == 2
    assert divergence.kind == collective_phase_diag.DivergenceKind.MISSING_RANK
    assert divergence.actual_sequence is None


def test_matching_phase_records_have_no_divergence():
    records = []
    for rank, coordinate in enumerate(((0, 0), (0, 1), (1, 0), (1, 1))):
        records.append(_record(rank, coordinate, 0, "backward_enter", world=40, fsdp=12, ep=20))
        records.append(_record(rank, coordinate, 1, "moe_ep_a2a_first", world=40, fsdp=13, ep=21))

    assert collective_phase_diag.find_first_divergence(records) is None


def test_cli_reports_the_first_divergence(tmp_path, capsys):
    records = []
    for rank, coordinate in enumerate(((0, 0), (0, 1), (1, 0), (1, 1))):
        records.append(_record(rank, coordinate, 0, "backward_enter", world=40, fsdp=12, ep=20))
        records.append(
            _record(
                rank,
                coordinate,
                1,
                "moe_ep_a2a_first",
                world=40,
                fsdp=12,
                ep=22 if rank == 1 else 21,
            )
        )
    log_path = tmp_path / "workers.log"
    log_path.write_text(
        "ordinary worker line\n" + "\n".join(collective_phase_diag.format_log_record(record) for record in records),
        encoding="utf-8",
    )

    collective_phase_diag.main([str(log_path)])

    report = json.loads(capsys.readouterr().out)
    assert report["record_count"] == 8
    assert report["divergence"]["kind"] == "sequence"
    assert report["divergence"]["group_name"] == "ep"
