"""Import-boundary tests for the CPU-only launcher installation."""

import json
from pathlib import Path
import subprocess
import sys


REPOSITORY_ROOT = Path(__file__).parents[3]


def test_importing_job_does_not_import_training_stacks() -> None:
    program = """
import json
import sys

import cloud.iris.job

blocked = ("flash_attn", "ray", "skyrl_train", "torch", "vllm")
print(json.dumps(sorted(name for name in blocked if name in sys.modules)))
"""
    result = subprocess.run(
        [sys.executable, "-c", program],
        cwd=REPOSITORY_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    assert json.loads(result.stdout) == []


def test_importing_hf_export_does_not_import_training_stacks() -> None:
    """The launcher's export step imports skyrl_train.hf_export in a torch-free environment."""
    program = """
import json
import sys

import skyrl_train.hf_export

blocked = ("flash_attn", "ray", "torch", "vllm")
print(json.dumps(sorted(name for name in blocked if name in sys.modules)))
"""
    result = subprocess.run(
        [sys.executable, "-c", program],
        cwd=REPOSITORY_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    assert json.loads(result.stdout) == []
