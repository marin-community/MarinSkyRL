import asyncio
import os
import subprocess
import sys

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
    result = subprocess.run(
        [sys.executable, __file__],
        check=True,
        capture_output=True,
        text=True,
    )

    assert result.stdout.strip() == "ok"


if __name__ == "__main__":
    _run_worker_setup_probe()
