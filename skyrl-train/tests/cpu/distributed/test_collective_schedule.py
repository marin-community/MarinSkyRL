from tests.collective_schedule import (
    CollectiveBoundary,
    CollectiveEvent,
    RankCollectiveSchedule,
    assert_collective_schedules_match,
    find_first_collective_divergence,
)


def _schedule(rank, coordinate, *, ep_events, dp_events, ep_boundaries=(0, 3), dp_boundaries=(0, 1)):
    labels = ("layer0:original:enter", "layer0:original:exit")
    return RankCollectiveSchedule(
        rank=rank,
        mesh_dim_names=("dp", "ep"),
        mesh_shape=(2, 2),
        mesh_coordinate=coordinate,
        events={
            "ep": tuple(CollectiveEvent(operation, index + 10) for index, operation in enumerate(ep_events)),
            "dp": tuple(CollectiveEvent(operation, index + 20) for index, operation in enumerate(dp_events)),
        },
        boundaries=tuple(
            CollectiveBoundary(label, {"ep": ep_sequence, "dp": dp_sequence})
            for label, ep_sequence, dp_sequence in zip(labels, ep_boundaries, dp_boundaries)
        ),
    )


def test_collective_schedules_match_with_independent_ep_and_dp_groups():
    schedules = [
        _schedule(0, (0, 0), ep_events=("EP0_DISPATCH", "EP0_COMBINE"), dp_events=("DP0_AG", "DP0_RS")),
        _schedule(
            1,
            (0, 1),
            ep_events=("EP0_DISPATCH", "EP0_COMBINE"),
            dp_events=("DP1_AG",),
            dp_boundaries=(0, 2),
        ),
        _schedule(
            2,
            (1, 0),
            ep_events=("EP1_COUNTS", "EP1_DISPATCH", "EP1_COMBINE"),
            dp_events=("DP0_AG", "DP0_RS"),
            ep_boundaries=(0, 4),
        ),
        _schedule(
            3,
            (1, 1),
            ep_events=("EP1_COUNTS", "EP1_DISPATCH", "EP1_COMBINE"),
            dp_events=("DP1_AG",),
            ep_boundaries=(0, 4),
            dp_boundaries=(0, 2),
        ),
    ]

    assert_collective_schedules_match(schedules, "ep")
    assert_collective_schedules_match(schedules, "dp")


def test_collective_schedule_reports_first_missing_operation():
    schedules = [
        _schedule(0, (0, 0), ep_events=("COUNTS", "DISPATCH", "COMBINE"), dp_events=("ALLGATHER",)),
        _schedule(1, (0, 1), ep_events=("COUNTS", "DISPATCH"), dp_events=("ALLGATHER",)),
        _schedule(2, (1, 0), ep_events=("COUNTS", "DISPATCH", "COMBINE"), dp_events=("ALLGATHER",)),
        _schedule(3, (1, 1), ep_events=("COUNTS", "DISPATCH", "COMBINE"), dp_events=("ALLGATHER",)),
    ]

    divergence = find_first_collective_divergence(schedules, "ep")

    assert divergence is not None
    assert divergence.fixed_coordinate == (("dp", 0),)
    assert divergence.reference_rank == 0
    assert divergence.divergent_rank == 1
    assert divergence.sequence_kind == "operation"
    assert divergence.sequence_index == 2
    assert divergence.expected == "COMBINE"
    assert divergence.actual == "<end>"


def test_collective_schedule_reports_layer_boundary_sequence_drift():
    schedules = [
        _schedule(0, (0, 0), ep_events=("A2A",), dp_events=("AG",)),
        _schedule(1, (0, 1), ep_events=("A2A",), dp_events=("AG",), ep_boundaries=(0, 4)),
        _schedule(2, (1, 0), ep_events=("A2A",), dp_events=("AG",)),
        _schedule(3, (1, 1), ep_events=("A2A",), dp_events=("AG",)),
    ]

    divergence = find_first_collective_divergence(schedules, "ep")

    assert divergence is not None
    assert divergence.sequence_kind == "boundary"
    assert divergence.sequence_index == 1
    assert divergence.expected == "layer0:original:exit at sequence +3"
    assert divergence.actual == "layer0:original:exit at sequence +4"
