"""Dependency-light Ray worker bootstrap.

Ray imports this module before assigning an actor-specific CUDA visibility mask.
Keep both :mod:`skyrl_train` and this module free of torch/transformers/ray imports
so CUDA cannot be initialized against the driver's device view first.
"""

import asyncio
import os


def force_stock_asyncio_in_worker() -> None:
    """Use stock asyncio and disable uvloop's libuv paths in Ray workers."""

    os.environ["UV_USE_IO_URING"] = "0"
    asyncio.set_event_loop_policy(asyncio.DefaultEventLoopPolicy())

    try:
        import uvloop  # noqa: PLC0415 - optional dependency in some worker environments
    except ImportError:
        return

    def _stock_new_event_loop():
        return asyncio.SelectorEventLoop()

    def _stock_install():
        asyncio.set_event_loop_policy(asyncio.DefaultEventLoopPolicy())

    uvloop.new_event_loop = _stock_new_event_loop
    uvloop.install = _stock_install
    uvloop.EventLoopPolicy = asyncio.DefaultEventLoopPolicy
    if hasattr(uvloop, "Loop"):
        uvloop.Loop = asyncio.SelectorEventLoop
