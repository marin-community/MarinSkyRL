import sys

import pytest

from rl_cleanup import parse_skyrl_metrics


def test_agentic_format_writes_reward_ema_table(tmp_path, monkeypatch):
    log_path = tmp_path / "trainer.out"
    log_path.write_text(
        "\n".join(
            [
                "{'async/staleness_max': 0, 'reward/avg_raw_reward': 0.2, 'trainer/global_step': 20}",
                "{'async/staleness_max': 0, 'reward/avg_raw_reward': 0.8, 'trainer/global_step': 40}",
                "{'async/staleness_max': 0, 'reward/avg_raw_reward': 0.5, 'trainer/global_step': 60}",
            ]
        )
    )
    output_path = tmp_path / "results"
    monkeypatch.setattr(parse_skyrl_metrics, "generate_reward_plot", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "parse_skyrl_metrics.py",
            str(log_path),
            str(output_path),
            "--format",
            "agentic",
            "--save_every",
            "20",
        ],
    )

    parse_skyrl_metrics.main()

    report = next(output_path.glob("*_metrics_report.md")).read_text()
    assert "## Best Checkpoint (trailing-5 EMA of reward/avg_raw_reward)" in report
    assert "| 40 | 0.8000 | 0.4000 | yes |" in report


def test_checkpoint_selection_calculates_trailing_five_ema():
    selection = parse_skyrl_metrics.select_best_checkpoint(
        [
            {"trainer/global_step": 20, "reward/avg_raw_reward": 0.2},
            {"trainer/global_step": 40, "reward/avg_raw_reward": 0.8},
        ],
        save_every=20,
    )
    assert selection.ema[40] == pytest.approx(0.4)


def test_checkpoint_selection_rejects_an_invalid_cap(tmp_path):
    (tmp_path / "latest_ckpt_global_step.txt").write_text("not-a-step")

    with pytest.raises(ValueError, match="latest_ckpt_global_step.txt"):
        parse_skyrl_metrics.select_best_checkpoint(
            [{"trainer/global_step": 40, "reward/avg_raw_reward": 0.8}],
            run_dir=tmp_path,
            save_every=20,
        )
