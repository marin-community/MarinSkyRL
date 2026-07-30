import sys

from rl_cleanup import parse_skyrl_metrics


def test_agentic_format_prints_reward_ema_table(tmp_path, monkeypatch, capsys):
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

    output = capsys.readouterr().out
    assert "BEST-CHECKPOINT SELECTOR (agentic GRPO, trailing-5 EMA)" in output
    assert "    40 |     0.8000 |     0.4000" in output
    report = next(output_path.glob("*_metrics_report.md")).read_text()
    assert "## Best Checkpoint (trailing-5 EMA of reward/avg_raw_reward)" in report
    assert "| 40 | 0.8000 | 0.4000 | yes |" in report
