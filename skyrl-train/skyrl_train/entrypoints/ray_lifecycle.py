import atexit
import os

import ray
from loguru import logger


def shutdown_ray() -> None:
    owner = os.environ.get("SKYRL_RAY_CLUSTER_OWNER")
    if not owner:
        ray.shutdown()
        return

    atexit.unregister(ray.shutdown)
    logger.info(f"Leaving Ray cluster teardown to {owner}")
