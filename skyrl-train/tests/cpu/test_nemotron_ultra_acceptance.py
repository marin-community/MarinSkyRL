import json

from ci.nemotron_ultra.gate import check_log, expected_coverage


def test_acceptance_gate_requires_every_phase_generator_pair():
    coverage = {key: 2 for key in expected_coverage()}
    metrics = {
        **coverage,
        "policy/policy_loss": 0.1,
        "policy/final_loss": 0.2,
        "generate/num_failed_trajectories": 0,
    }
    log = "\n".join(
        (
            f'NEMOTRON_ULTRA_SAMPLE {{"rows": {len(coverage)}}}',
            f"WANDB_MIRROR kind=train step=1 metrics={json.dumps(metrics)}",
        )
    )

    assert check_log(log) == []

    del metrics[next(iter(coverage))]
    failures = check_log(
        "\n".join(
            (
                f'NEMOTRON_ULTRA_SAMPLE {{"rows": {len(coverage)}}}',
                f"WANDB_MIRROR kind=train step=1 metrics={json.dumps(metrics)}",
            )
        )
    )
    assert any("expected 2 completed rollouts" in failure for failure in failures)
