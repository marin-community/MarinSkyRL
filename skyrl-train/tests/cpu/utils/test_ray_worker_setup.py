import os
import subprocess
import sys
from pathlib import Path

from skyrl_train.env_vars import (
    DEBUG_ARTIFACT_DIR_ENV,
    DEBUG_MODE_ENV,
)
from skyrl_train.numa_policy import NUMA_AFFINITY_ENV
from skyrl_train.worker_setup import INCOMPATIBLE_NCCL_ENVIRONMENT, configure_worker_process


def _run_worker_setup_probe() -> None:
    configure_worker_process()
    configure_worker_process()

    assert not set(INCOMPATIBLE_NCCL_ENVIRONMENT).intersection(os.environ)
    assert not {"ray", "torch", "transformers"}.intersection(sys.modules)
    print("ok")


def test_ray_worker_setup_prepares_process_before_torch_import(tmp_path: Path) -> None:
    package_root = Path(__file__).parents[3]
    repository_root = Path(__file__).parents[4]
    python_path = os.pathsep.join(filter(None, (str(repository_root), str(package_root), os.environ.get("PYTHONPATH"))))
    result = subprocess.run(
        [sys.executable, __file__],
        check=True,
        capture_output=True,
        text=True,
        env={
            **os.environ,
            "PYTHONPATH": python_path,
            DEBUG_ARTIFACT_DIR_ENV: str(tmp_path),
            DEBUG_MODE_ENV: "light",
            NUMA_AFFINITY_ENV: "0",
            **dict.fromkeys(INCOMPATIBLE_NCCL_ENVIRONMENT, "1"),
        },
    )

    assert result.stdout.strip() == "ok"
    assert len(list((tmp_path / "processes").glob("ray-worker-bootstrap.*.json"))) == 1
    assert not list((tmp_path / "stacks").glob("*.log"))


if __name__ == "__main__":
    _run_worker_setup_probe()
