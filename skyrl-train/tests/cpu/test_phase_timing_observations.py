from skyrl_train.timing_observability import phase_timing_observations, publish_step_timings


def test_overlapping_async_spans_remain_inclusive_instead_of_claiming_exclusive_time():
    observations = phase_timing_observations({"step": 4.0, "generate": 3.0, "run_training": 3.0})

    assert {item.name: item.duration_seconds for item in observations} == {
        "step": 4.0,
        "generate": 3.0,
        "run_training": 3.0,
    }
    assert {item.name: item.parent for item in observations} == {
        "step": None,
        "generate": "step",
        "run_training": "step",
    }


def test_unknown_spans_are_not_published():
    calls = []

    class _Sink:
        def publish(self, observations, step):
            calls.append((observations, step))

    publish_step_timings({"step": 2.0, "something_new": 1.0}, step=7, sinks=(_Sink(),))

    observations, step = calls[0]
    assert [item.name for item in observations] == ["step"]
    assert step == 7
