import os
import sys
from collections.abc import Iterator
from contextlib import contextmanager

import pytest

# CPU tests already run in the locked uv environment. Ray's uv hook would package
# this checkout and create another environment for every local Ray session.
os.environ["RAY_ENABLE_UV_RUN_RUNTIME_ENV"] = "0"

import ray  # noqa: E402


REGISTRY_ACTOR_NAMES = ("advantage_estimator_registry", "policy_loss_registry")
REGISTRY_CLASS_NAMES = ("AdvantageEstimatorRegistry", "PolicyLossRegistry")


def _kill_registry_actors() -> None:
    for actor_name in REGISTRY_ACTOR_NAMES:
        try:
            actor = ray.get_actor(actor_name)
        except ValueError:
            continue
        ray.kill(actor)

    registry_module = sys.modules.get("skyrl_train.utils.ppo_utils")
    if registry_module is None:
        return
    for registry_name in REGISTRY_CLASS_NAMES:
        registry = getattr(registry_module, registry_name)
        registry._ray_actor = None
        registry._synced_to_actor = False


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
