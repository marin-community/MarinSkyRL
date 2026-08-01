from tests.collective_schedule import (
    CollectiveBoundary,
    CollectiveEvent,
    RankCollectiveSchedule,
    assert_collective_schedules_match,
    find_first_collective_divergence,
)


def _schedule(rank, coordinate, *, ep_events, fsdp_events, ep_boundaries=(0, 3), fsdp_boundaries=(0, 1)):
    labels = ("layer0:original:enter", "layer0:original:exit")
    return RankCollectiveSchedule(
        rank=rank,
        mesh_dim_names=("fsdp", "ep"),
        mesh_shape=(2, 2),
        mesh_coordinate=coordinate,
        events={
            "ep": tuple(CollectiveEvent(operation, index + 10) for index, operation in enumerate(ep_events)),
            "fsdp": tuple(CollectiveEvent(operation, index + 20) for index, operation in enumerate(fsdp_events)),
        },
        boundaries=tuple(
            CollectiveBoundary(label, {"ep": ep_sequence, "fsdp": fsdp_sequence})
            for label, ep_sequence, fsdp_sequence in zip(labels, ep_boundaries, fsdp_boundaries)
        ),
    )


def test_collective_schedules_match_with_independent_ep_and_fsdp_groups():
    schedules = [
        _schedule(0, (0, 0), ep_events=("EP0_DISPATCH", "EP0_COMBINE"), fsdp_events=("FSDP0_AG", "FSDP0_RS")),
        _schedule(
            1,
            (0, 1),
            ep_events=("EP0_DISPATCH", "EP0_COMBINE"),
            fsdp_events=("FSDP1_AG",),
            fsdp_boundaries=(0, 2),
        ),
        _schedule(
            2,
            (1, 0),
            ep_events=("EP1_COUNTS", "EP1_DISPATCH", "EP1_COMBINE"),
            fsdp_events=("FSDP0_AG", "FSDP0_RS"),
            ep_boundaries=(0, 4),
        ),
        _schedule(
            3,
            (1, 1),
            ep_events=("EP1_COUNTS", "EP1_DISPATCH", "EP1_COMBINE"),
            fsdp_events=("FSDP1_AG",),
            ep_boundaries=(0, 4),
            fsdp_boundaries=(0, 2),
        ),
    ]

    assert_collective_schedules_match(schedules, "ep")
    assert_collective_schedules_match(schedules, "fsdp")


def test_collective_schedule_reports_first_missing_operation():
    schedules = [
        _schedule(0, (0, 0), ep_events=("COUNTS", "DISPATCH", "COMBINE"), fsdp_events=("ALLGATHER",)),
        _schedule(1, (0, 1), ep_events=("COUNTS", "DISPATCH"), fsdp_events=("ALLGATHER",)),
        _schedule(2, (1, 0), ep_events=("COUNTS", "DISPATCH", "COMBINE"), fsdp_events=("ALLGATHER",)),
        _schedule(3, (1, 1), ep_events=("COUNTS", "DISPATCH", "COMBINE"), fsdp_events=("ALLGATHER",)),
    ]

    divergence = find_first_collective_divergence(schedules, "ep")

    assert divergence is not None
    assert divergence.fixed_coordinate == (("fsdp", 0),)
    assert divergence.reference_rank == 0
    assert divergence.divergent_rank == 1
    assert divergence.sequence_kind == "operation"
    assert divergence.sequence_index == 2
    assert divergence.expected == "COMBINE"
    assert divergence.actual == "<end>"


def test_collective_schedule_reports_layer_boundary_sequence_drift():
    schedules = [
        _schedule(0, (0, 0), ep_events=("A2A",), fsdp_events=("AG",)),
        _schedule(1, (0, 1), ep_events=("A2A",), fsdp_events=("AG",), ep_boundaries=(0, 4)),
        _schedule(2, (1, 0), ep_events=("A2A",), fsdp_events=("AG",)),
        _schedule(3, (1, 1), ep_events=("A2A",), fsdp_events=("AG",)),
    ]

    divergence = find_first_collective_divergence(schedules, "ep")

    assert divergence is not None
    assert divergence.sequence_kind == "boundary"
    assert divergence.sequence_index == 1
    assert divergence.expected == "layer0:original:exit at sequence +3"
    assert divergence.actual == "layer0:original:exit at sequence +4"
