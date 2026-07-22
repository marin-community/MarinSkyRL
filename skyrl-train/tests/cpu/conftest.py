import os
import sys
from collections.abc import Iterator
from contextlib import contextmanager

import pytest

# CPU tests already run in the locked uv environment. Ray's uv hook would package
# this checkout and create another environment for every local Ray session.
os.environ["RAY_ENABLE_UV_RUN_RUNTIME_ENV"] = "0"

import ray  # noqa: E402


def _kill_registry_actors() -> None:
    registry_module = sys.modules.get("skyrl_train.utils.ppo_utils")
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
