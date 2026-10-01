import sys
from pathlib import Path

import pytest
import ray


@ray.remote
def worker_executable() -> str:
    return sys.executable


@pytest.mark.usefixtures("ray_init")
def test_ray_worker_reuses_test_environment():
    assert Path(ray.get(worker_executable.remote())).resolve() == Path(sys.executable).resolve()
