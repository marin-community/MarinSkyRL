import os
from uuid import uuid4

import pytest
import ray
from tests.gpu.utils import ray_init_for_tests


@pytest.fixture
def megatron_checkpoint_path():
    prefix = os.environ.get("MARIN_TEMP_PREFIX", os.environ.get("MARIN_PREFIX", ""))
    if not prefix.startswith("s3://"):
        raise RuntimeError("Megatron checkpoint tests require a CoreWeave S3 MARIN_TEMP_PREFIX")
    return f"{prefix.rstrip('/')}/megatron-checkpoint-tests/{uuid4().hex}"


@pytest.fixture
def ray_init_fixture():
    if ray.is_initialized():
        ray.shutdown()
    ray_init_for_tests()
    yield
    # call ray shutdown after a test regardless
    ray.shutdown()
