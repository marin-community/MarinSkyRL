from infra.rl_metrics import (
    ENTROPY_KEYS,
    GRAD_NORM_KEYS,
    POLICY_LOSS_KEYS,
    REWARD_KEYS,
    metric_value,
    parse_training_metrics_result,
)


def test_parse_training_metrics_extracts_standard_rl_status_fields() -> None:
    log = "\n".join(
        [
            'WANDB_MIRROR kind=train step=6 metrics={"broken": ',
            (
                "\x1b[32m(skyrl_entrypoint pid=8444) WANDB_MIRROR kind=train step=7 metrics="
                '{"reward/avg_raw_reward": -0.9194, "policy/policy_loss": 2.77e-09, '
                '"policy/raw_grad_norm": 0.0371, "policy/policy_entropy": 1.178}\x1b[0m'
            ),
        ]
    )

    result = parse_training_metrics_result(log)
    records = result.records

    assert len(records) == 1
    assert result.malformed_lines == 1
    assert records[0].step == 7
    assert records[0].metrics["trainer/global_step"] == 7
    assert metric_value(records[0].metrics, *REWARD_KEYS) == -0.9194
    assert metric_value(records[0].metrics, *POLICY_LOSS_KEYS) == 2.77e-09
    assert metric_value(records[0].metrics, *GRAD_NORM_KEYS) == 0.0371
    assert metric_value(records[0].metrics, *ENTROPY_KEYS) == 1.178


def test_parse_training_metrics_preserves_latest_valid_record_and_reports_malformed_lines() -> None:
    log = "\n".join(
        [
            'WANDB_MIRROR kind=train step=1 metrics={"reward/avg_raw_reward": 0.25}',
            (
                'WANDB_MIRROR kind=train step=2 metrics={"reward/avg_raw_reward": 0.5, '
                '"policy/policy_loss": 0.01, "policy/raw_grad_norm": 0.2}'
            ),
            'WANDB_MIRROR kind=train step=3 metrics={"reward/avg_raw_reward": ',
        ]
    )

    result = parse_training_metrics_result(log)

    assert result.malformed_lines == 1
    assert len(result.records) == 2
    assert result.records[-1].step == 2
    assert result.records[-1].metrics == {
        "reward/avg_raw_reward": 0.5,
        "policy/policy_loss": 0.01,
        "policy/raw_grad_norm": 0.2,
        "trainer/global_step": 2,
    }
