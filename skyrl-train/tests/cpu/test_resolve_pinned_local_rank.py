"""Device pinning for Ray training actors: LOCAL_RANK selection and the per-actor CUDA environment.

Regression context: on GH200 nodes, actors sharing one whole-node {GPU:4} bundle all saw
ray.get_gpu_ids() == [0] and stacked on GPU 0. Per-GPU bundles give each actor its own physical id.
"""

import pytest

from skyrl_train.utils.utils import resolve_actor_cuda_env, resolve_pinned_local_rank


@pytest.mark.parametrize(
    ("overrides", "expected"),
    [
        pytest.param({"noset_visible_devices": True, "ray_gpu_ids": [3]}, "3", id="noset-pins-ray-id"),
        pytest.param({"cuda_visible_devices": "2"}, "0", id="ray-masked-single-device"),
        pytest.param({"launcher_local_rank": 2}, "2", id="whole-node-uses-launcher-rank"),
        pytest.param({"launcher_local_rank": 9, "ray_gpu_ids": [1]}, "1", id="whole-node-rank-out-of-range"),
        pytest.param({"ray_gpu_ids": [3], "pin_to_ray_gpu_id": True}, "3", id="per-gpu-bundle-pins-ray-id"),
        pytest.param(
            {"ray_gpu_ids": [99], "launcher_local_rank": 1, "pin_to_ray_gpu_id": True},
            "1",
            id="per-gpu-bundle-ray-id-out-of-range",
        ),
        # A shared whole-node bundle reports [0] to every actor, so pinning to the Ray id collapses
        # all ranks onto GPU 0; this is why pinning requires per-GPU bundles.
        pytest.param(
            {"ray_gpu_ids": [0], "launcher_local_rank": 3, "pin_to_ray_gpu_id": True},
            "0",
            id="shared-bundle-pin-collapses",
        ),
    ],
)
def test_resolve_pinned_local_rank(overrides, expected):
    arguments = {
        "noset_visible_devices": False,
        "cuda_visible_devices": None,
        "ray_gpu_ids": [0],
        "launcher_local_rank": 0,
        "device_count": 4,
        "pin_to_ray_gpu_id": False,
    }
    assert resolve_pinned_local_rank(**(arguments | overrides)) == expected


@pytest.mark.parametrize(
    ("noset_visible_devices", "cuda_visible_devices", "ray_gpu_ids", "expected"),
    [
        pytest.param(False, None, [3], {"CUDA_VISIBLE_DEVICES": "3"}, id="unmasked-masks-to-physical-id"),
        # An id outside an already-masked single-device view is not addressable; keep Ray's mask.
        pytest.param(False, "2", [2], {}, id="ray-masked-single-device-untouched"),
        pytest.param(True, "0,1,2,3", [2], {"CUDA_VISIBLE_DEVICES": "2"}, id="noset-masks-to-physical-id"),
        pytest.param(False, "0,1", [1], {"CUDA_VISIBLE_DEVICES": "1"}, id="multi-device-view"),
    ],
)
def test_resolve_actor_cuda_env_masks_each_actor_to_one_physical_gpu(
    noset_visible_devices, cuda_visible_devices, ray_gpu_ids, expected
):
    env = resolve_actor_cuda_env(
        noset_visible_devices=noset_visible_devices,
        cuda_visible_devices=cuda_visible_devices,
        ray_gpu_ids=ray_gpu_ids,
    )
    assert env == {"CUDA_DEVICE_ORDER": "PCI_BUS_ID", "LOCAL_RANK": "0"} | expected
