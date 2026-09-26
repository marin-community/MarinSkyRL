import json
import subprocess
import sys
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[2] / "examples/cat_count/cpu_canary.py"


def _run(*arguments: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(SCRIPT), *arguments],
        text=True,
        capture_output=True,
        timeout=1800,
        check=False,
    )


def _records(output: str) -> list[dict]:
    return [json.loads(line) for line in output.splitlines() if line.startswith("{")]


def test_cat_count_cpu_learns_and_flipped_advantage_fails(tmp_path):
    checkpoint = str(tmp_path / "base")
    pretrained = _run("pretrain", "--out", checkpoint)
    assert pretrained.returncode == 0, pretrained.stderr

    positive = _run("rl", "--ckpt", checkpoint, "--seed", "0")
    assert positive.returncode == 0, positive.stdout + positive.stderr
    positive_records = _records(positive.stdout)
    assert positive_records[-1]["verdict"] == "PASS"
    assert positive_records[-1]["step"] <= 100
    assert positive_records[-1]["train_reward"][0] < positive_records[-1]["train_reward"][1]
    assert positive_records[-1]["heldout_reward"][0] < positive_records[-1]["heldout_reward"][1]
    assert positive_records[-1]["train_reward"][1] >= 0.9
    assert positive_records[-1]["heldout_reward"][1] >= 0.9

    negative = _run("rl", "--ckpt", checkpoint, "--seed", "0", "--flip-advantage", "--max_steps", "20")
    assert negative.returncode == 1, negative.stdout + negative.stderr
    negative_records = _records(negative.stdout)
    assert negative_records[-1]["verdict"] == "FAIL"
    assert negative_records[-2]["eval"]["step"] == 20
    assert negative_records[-2]["eval"]["train_reward"] < 0.9
    assert negative_records[-2]["eval"]["heldout_reward"] < 0.9
