import asyncio
import os
import subprocess
import sys
from pathlib import Path

from skyrl_train import worker_setup
from skyrl_train.worker_setup import force_stock_asyncio_in_worker


def _run_worker_setup_probe() -> None:
    force_stock_asyncio_in_worker()
    force_stock_asyncio_in_worker()

    assert os.environ["UV_USE_IO_URING"] == "0"
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


def test_ray_worker_setup_installs_stock_asyncio_without_loading_torch():
    package_root = Path(__file__).parents[3]
    python_path = os.pathsep.join(filter(None, (str(package_root), os.environ.get("PYTHONPATH"))))
    result = subprocess.run(
        [sys.executable, __file__],
        check=True,
        capture_output=True,
        text=True,
        env={**os.environ, "PYTHONPATH": python_path},
    )

    assert result.stdout.strip() == "ok"


def test_ray_worker_setup_installs_host_memory_policy_before_actor_threads(monkeypatch):
    calls = []
    monkeypatch.setattr(worker_setup, "set_host_memory_policy", lambda: calls.append("memory-policy"))

    force_stock_asyncio_in_worker()

    assert calls == ["memory-policy"]


if __name__ == "__main__":
    _run_worker_setup_probe()
