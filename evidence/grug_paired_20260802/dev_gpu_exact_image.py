#!/usr/bin/env python3
"""Run Marin's dev_gpu CLI with one immutable task-image override."""

import os
import runpy

from iris.client import IrisClient


task_image = os.environ["DEV_GPU_TASK_IMAGE"]
original_submit = IrisClient.submit


def submit_with_exact_image(self, *args, **kwargs):
    requested = kwargs.get("task_image")
    if requested not in (None, task_image):
        raise RuntimeError(f"conflicting dev-GPU task image: {requested}")
    kwargs["task_image"] = task_image
    return original_submit(self, *args, **kwargs)


IrisClient.submit = submit_with_exact_image
runpy.run_path(
    "/home/romain/dev/marin-wt/grug-training-perf-gap-20260731/scripts/iris/dev_gpu.py",
    run_name="__main__",
)
