import pytest

from skyrl_train.learner_memory import LearnerCudaMetrics


def test_spans_reset_peaks_and_an_overlapping_span_suppresses_them(fake_cuda, delivered_telemetry):
    memory = LearnerCudaMetrics(enabled=True, rank=11)
    with memory.span("forward", step=7):
        fake_cuda.use_memory(500, 600)
        fake_cuda.use_memory(120, 200)
    with memory.span("broadcast_to_inference_engines", step=7):
        fake_cuda.use_memory(300, 400)
        with LearnerCudaMetrics(enabled=True, rank=11).span("forward", step=8):
            fake_cuda.use_memory(100, 160)

    forward_enter, forward_exit, publish_enter, publish_exit = delivered_telemetry.select("cuda_memory_observation")
    assert forward_enter["attributes"]["outcome"] == "started"
    assert forward_exit["body"] == {
        "allocated_bytes": 120,
        "reserved_bytes": 200,
        "device_free_bytes": 2000,
        "device_total_bytes": 4096,
        "peak_allocated_bytes": 500,
        "peak_reserved_bytes": 600,
    }
    assert publish_enter["body"]["allocated_bytes"] == 120
    assert "peak_allocated_bytes" not in publish_exit["body"]
    assert publish_exit["attributes"] == {
        "backend": "megatron",
        "role": "worker",
        "worker_role": "policy",
        "rank": "11",
        "cuda_device": "2",
        "gpu_uuid": "GPU-physical-two",
        "allocator_backend": "native",
        "phase": "broadcast_to_inference_engines",
        "boundary": "exit",
        "outcome": "success",
        "step": "7",
        "scope_overlap": "true",
    }


@pytest.mark.parametrize(
    ("failure", "at_exit"),
    [
        ("identity", False),
        ("allocator", False),
        ("reset", False),
        ("sample", False),
        ("export", False),
        ("sample", True),
        ("export", True),
        (None, False),
    ],
)
def test_an_observation_failure_never_replaces_the_training_outcome(fake_cuda, delivered_telemetry, failure, at_exit):
    memory = LearnerCudaMetrics(enabled=True, rank=3)
    training_error = RuntimeError("optimizer failed")
    with pytest.MonkeyPatch.context() as patch:

        def fail_observation():
            if failure == "export":
                patch.setattr("skyrl_train.learner_memory.record_event", lambda *args, **kwargs: 1 / 0)
            elif failure == "allocator":
                fake_cuda.backend = "cudaMallocAsync"
            else:
                fake_cuda.failure = failure

        if not at_exit:
            fail_observation()
        with pytest.raises(RuntimeError) as caught:
            with memory.span("ppo_train", step=4):
                if at_exit:
                    fail_observation()
                raise training_error
    assert caught.value is training_error
    assert memory.enabled is (failure is None)
    fake_cuda.failure, fake_cuda.backend = None, "native"
    # The device is free again for a new recorder.
    with LearnerCudaMetrics(enabled=True, rank=3).span("broadcast_to_inference_engines", step=4):
        fake_cuda.use_memory(1400, 1500)
    assert delivered_telemetry.select("cuda_memory_observation")[-1]["body"]["peak_allocated_bytes"] == 1400
