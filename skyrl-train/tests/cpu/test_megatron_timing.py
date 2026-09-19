from skyrl_train.megatron_timing import (
    FINAL_BARRIER,
    FORWARD_BACKWARD_SCHEDULER,
    OPTIMIZER_STEP,
    PIPELINE_METRIC_BROADCAST,
    RESIDUAL,
    TOTAL,
    WORLD_METRIC_REDUCTION,
    MegatronTrainTimings,
    publish_megatron_train_timings,
)


class FakeClock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


class RecordingHistogram:
    def __init__(self) -> None:
        self.records: list[tuple[float, dict[str, str]]] = []

    def record(self, value: float, *, attributes: dict[str, str]) -> None:
        self.records.append((value, attributes))


def test_disabled_megatron_timings_do_not_read_the_clock_or_publish() -> None:
    def unexpected_clock_read() -> float:
        raise AssertionError("disabled timing read the clock")

    timing = MegatronTrainTimings(enabled=False, clock=unexpected_clock_read)
    with timing.span(OPTIMIZER_STEP):
        pass

    assert timing.finish() == ()


def test_megatron_timings_close_total_with_signed_residual() -> None:
    clock = FakeClock()
    timing = MegatronTrainTimings(enabled=True, clock=clock)

    clock.advance(1.0)
    with timing.span(FORWARD_BACKWARD_SCHEDULER):
        clock.advance(2.0)
    with timing.span(PIPELINE_METRIC_BROADCAST):
        clock.advance(0.5)
    with timing.span(OPTIMIZER_STEP):
        clock.advance(1.25)
    with timing.span(WORLD_METRIC_REDUCTION):
        clock.advance(0.25)
    with timing.span(WORLD_METRIC_REDUCTION):
        clock.advance(0.5)
    with timing.span(FINAL_BARRIER):
        clock.advance(0.75)
    clock.advance(0.75)

    observations = {observation.phase: observation for observation in timing.finish()}

    assert observations[TOTAL].seconds == 7.0
    assert observations[WORLD_METRIC_REDUCTION].seconds == 0.75
    assert observations[RESIDUAL].seconds == 1.75
    assert sum(observations[phase].seconds for phase in observations if phase != TOTAL) == 7.0
    assert observations[FORWARD_BACKWARD_SCHEDULER].parent_phase == TOTAL
    assert observations[TOTAL].parent_phase is None


def test_publish_megatron_timings_preserves_worker_identity_and_clock_domain() -> None:
    clock = FakeClock()
    timing = MegatronTrainTimings(enabled=True, clock=clock)
    clock.advance(3.0)
    recorder = RecordingHistogram()

    publish_megatron_train_timings(
        timing.finish(),
        step=7,
        rank=3,
        outcome="success",
        recorder=recorder,
    )

    records_by_phase = {attributes["phase"]: (value, attributes) for value, attributes in recorder.records}
    assert records_by_phase[TOTAL][0] == 3.0
    assert records_by_phase[RESIDUAL][0] == 3.0
    assert records_by_phase[TOTAL][1] == {
        "backend": "megatron",
        "clock_domain": "cpu_dispatch_wall",
        "outcome": "success",
        "phase": TOTAL,
        "rank": "3",
        "role": "worker",
        "step": "7",
        "root": TOTAL,
    }
