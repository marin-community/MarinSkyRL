from infra.rl_metrics import (
    ENTROPY_KEYS,
    GRAD_NORM_KEYS,
    POLICY_LOSS_KEYS,
    REWARD_KEYS,
    metric_value,
    parse_training_metrics_result,
)
from scripts.iris.watch_coreweave_rl import parse_metrics


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

    records = parse_training_metrics_result(log).records

    assert len(records) == 1
    assert records[0].step == 7
    assert records[0].metrics["trainer/global_step"] == 7
    assert metric_value(records[0].metrics, *REWARD_KEYS) == -0.9194
    assert metric_value(records[0].metrics, *POLICY_LOSS_KEYS) == 2.77e-09
    assert metric_value(records[0].metrics, *GRAD_NORM_KEYS) == 0.0371
    assert metric_value(records[0].metrics, *ENTROPY_KEYS) == 1.178


def test_watcher_uses_the_shared_latest_training_record(tmp_path) -> None:
    finelog = tmp_path / "finelog.log"
    finelog.write_text(
        "\n".join(
            [
                'WANDB_MIRROR kind=train step=1 metrics={"reward/avg_raw_reward": 0.25}',
                (
                    'WANDB_MIRROR kind=train step=2 metrics={"reward/avg_raw_reward": 0.5, '
                    '"policy/policy_loss": 0.01, "policy/raw_grad_norm": 0.2}'
                ),
            ]
        )
    )

    step, total, metrics, error = parse_metrics(finelog)

    assert (step, total, error) == (2, None, None)
    assert metrics == {
        "reward/avg_raw_reward": 0.5,
        "policy/policy_loss": 0.01,
        "policy/raw_grad_norm": 0.2,
        "trainer/global_step": 2,
    }


def test_watcher_reports_a_malformed_record_after_valid_metrics(tmp_path) -> None:
    finelog = tmp_path / "finelog.log"
    finelog.write_text(
        "\n".join(
            [
                'WANDB_MIRROR kind=train step=2 metrics={"reward/avg_raw_reward": 0.5}',
                'WANDB_MIRROR kind=train step=3 metrics={"reward/avg_raw_reward": ',
            ]
        )
    )

    step, total, metrics, error = parse_metrics(finelog)

    assert (step, total) == (2, None)
    assert metrics["reward/avg_raw_reward"] == 0.5
    assert error == "1 WANDB_MIRROR train lines failed JSON parse"
