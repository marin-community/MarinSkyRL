import os
import sys
from collections.abc import Iterator
from contextlib import contextmanager

import pytest

# CPU tests already run in the locked uv environment. Ray's uv hook would package
# this checkout and create another environment for every local Ray session.
os.environ["RAY_ENABLE_UV_RUN_RUNTIME_ENV"] = "0"

import ray  # noqa: E402
import torch.distributed as dist  # noqa: E402


def _kill_registry_actors() -> None:
    registry_module = sys.modules.get("skyrl_train.utils.function_registry")
    if registry_module is None:
        return
    for registry in registry_module.BaseFunctionRegistry.__subclasses__():
        registry.shutdown_actor()


@contextmanager
def _local_ray_session() -> Iterator[None]:
    if not ray.is_initialized():
        ray.init()
    try:
        yield
    finally:
        if ray.is_initialized():
            _kill_registry_actors()
            ray.shutdown()


@pytest.fixture
def ray_init() -> Iterator[None]:
    """Run one Ray-dependent CPU test in a local session."""
    with _local_ray_session():
        yield


@pytest.fixture(scope="module")
def ray_module() -> Iterator[None]:
    """Share a local Ray session across an actor-heavy test module."""
    with _local_ray_session():
        yield


@pytest.fixture(scope="module")
def single_rank_group():
    """A world-size-1 gloo process group so distributed collectives (TP
    all-reduces, broadcast_object_list) run as no-ops on one CPU process."""
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29591")
    created = False
    if not dist.is_initialized():
        dist.init_process_group("gloo", rank=0, world_size=1)
        created = True
    try:
        yield dist.group.WORLD
    finally:
        if created:
            dist.destroy_process_group()
