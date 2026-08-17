import asyncio
import os
import subprocess
import sys
from pathlib import Path

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


def test_ray_worker_setup_prepares_process_before_torch_import() -> None:
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
            NUMA_AFFINITY_ENV: "0",
            **dict.fromkeys(INCOMPATIBLE_NCCL_ENVIRONMENT, "1"),
        },
    )

    assert result.stdout.strip() == "ok"


if __name__ == "__main__":
    _run_worker_setup_probe()
