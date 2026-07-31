from infra.rl_analysis.training_metrics import parse_training_metrics
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

    records = parse_training_metrics(log)

    assert len(records) == 1
    assert records[0].step == 7
    assert records[0].metrics["trainer/global_step"] == 7
    assert records[0].reward == -0.9194
    assert records[0].policy_loss == 2.77e-09
    assert records[0].grad_norm == 0.0371
    assert records[0].entropy == 1.178


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
