import asyncio
import os
import subprocess
import sys
from pathlib import Path

import pytest

from skyrl_train.numa_policy import NUMA_AFFINITY_ENV
from skyrl_train.worker_setup import INCOMPATIBLE_NCCL_ENVIRONMENT, configure_worker_process


def _run_worker_setup_probe() -> None:
    configure_worker_process()
    configure_worker_process()

    assert os.environ["UV_USE_IO_URING"] == "0"
    assert not set(INCOMPATIBLE_NCCL_ENVIRONMENT).intersection(os.environ)
    assert isinstance(asyncio.get_event_loop_policy(), asyncio.DefaultEventLoopPolicy)

    loop = asyncio.new_event_loop()
    assert isinstance(loop, asyncio.SelectorEventLoop)
    loop.close()

    try:
        import uvloop
    except ImportError:
        pass
    else:
        loop = uvloop.new_event_loop()
        assert isinstance(loop, asyncio.SelectorEventLoop)
        loop.close()
        uvloop.install()
        assert isinstance(asyncio.get_event_loop_policy(), asyncio.DefaultEventLoopPolicy)

    assert not {"ray", "torch", "transformers"}.intersection(sys.modules)
    print("ok")


@pytest.mark.parametrize(
    "blocking_wait_environment",
    (
        {},
        dict.fromkeys(INCOMPATIBLE_NCCL_ENVIRONMENT, "1"),
    ),
    ids=("default", "inherited-blocking-wait"),
)
def test_ray_worker_setup_prepares_process_before_torch_import(blocking_wait_environment: dict[str, str]) -> None:
    package_root = Path(__file__).parents[3]
    python_path = os.pathsep.join(filter(None, (str(package_root), os.environ.get("PYTHONPATH"))))
    result = subprocess.run(
        [sys.executable, __file__],
        check=True,
        capture_output=True,
        text=True,
        env={
            **os.environ,
            "PYTHONPATH": python_path,
            NUMA_AFFINITY_ENV: "0",
            **blocking_wait_environment,
        },
    )

    assert result.stdout.strip() == "ok"
    for variable in blocking_wait_environment:
        assert variable in result.stderr


if __name__ == "__main__":
    _run_worker_setup_probe()
