"""
Unit tests for resolve_pinned_local_rank — the pure LOCAL_RANK / set_device
decision extracted from DistributedTorchRayActor.__init__.

This is the load-bearing GH200 device-pinning logic. It must:
  - keep the a3 8B venv-Ray path byte-identical ("0" via single-device CVD mask),
  - keep the NOSET path pinning to the Ray physical id,
  - in per-GPU-bundle mode (the fix), pin to the actor's OWN bundle's physical
    GPU id (distinct per actor) even when the SIF Ray leaves CVD unmasked,
  - reproduce that the LEGACY whole-node modulo path is what it was (regression
    guard), and
  - reproduce the exact 80B SIF failure the fix targets: 4 actors, CVD unset,
    whole-node bundle -> get_gpu_ids()==[0] for all -> modulo gives 0,1,2,3
    (so modulo "spreads" but is positional/logical, not physical) while
    per-GPU bundles give the true distinct physical ids.

uv run --isolated --extra dev pytest tests/cpu/test_resolve_pinned_local_rank.py
"""

from skyrl_train.utils.utils import resolve_pinned_local_rank


def _resolve(**kw):
    base = dict(
        noset_visible_devices=False,
        cuda_visible_devices=None,
        ray_gpu_ids=[0],
        launcher_local_rank=0,
        device_count=4,
        pin_to_ray_gpu_id=False,
    )
    base.update(kw)
    return resolve_pinned_local_rank(**base)


# --- Case 1: NOSET set -> Ray doesn't mask CVD, pin to physical id ----------


def test_noset_pins_to_ray_gpu_id():
    assert _resolve(noset_visible_devices=True, ray_gpu_ids=[3]) == "3"


# --- Case 2: CVD masked to a single device (a3 8B venv-Ray) -> "0" ----------


def test_cvd_masked_single_device_is_zero():
    # a3 path: older venv Ray masks CVD to exactly one device.
    assert _resolve(cuda_visible_devices="2") == "0"
    assert _resolve(cuda_visible_devices="0") == "0"


# --- Case 4: legacy whole-node bundle (pin off), CVD unset ------------------


def test_legacy_modulo_when_cvd_unset_and_pin_off():
    # The current production behavior: launcher local_rank (rank % gpus_per_node).
    assert _resolve(cuda_visible_devices=None, launcher_local_rank=2, pin_to_ray_gpu_id=False) == "2"


def test_legacy_modulo_out_of_range_falls_back_to_ray_id():
    assert (
        _resolve(cuda_visible_devices=None, launcher_local_rank=9, device_count=4, ray_gpu_ids=[1])
        == "1"
    )


# --- Case 3: per-GPU {GPU:1} bundle (THE FIX) -------------------------------


def test_per_gpu_bundle_pins_to_distinct_ray_gpu_id():
    # Each actor owns its own 1-GPU bundle, so get_gpu_ids() is its distinct
    # physical id. With CVD unset, that physical id IS the set_device index.
    assert _resolve(cuda_visible_devices=None, ray_gpu_ids=[3], pin_to_ray_gpu_id=True) == "3"
    assert _resolve(cuda_visible_devices=None, ray_gpu_ids=[1], pin_to_ray_gpu_id=True) == "1"


def test_per_gpu_bundle_four_actors_get_distinct_devices():
    # Simulate one node's 4 policy actors in per-GPU-bundle mode. Each actor's
    # bundle is a distinct physical GPU -> get_gpu_ids() returns [0],[1],[2],[3].
    results = [
        _resolve(cuda_visible_devices=None, ray_gpu_ids=[g], pin_to_ray_gpu_id=True)
        for g in (0, 1, 2, 3)
    ]
    assert results == ["0", "1", "2", "3"]
    assert len(set(results)) == 4  # distinct -> no GPU-0 collision


def test_per_gpu_bundle_falls_back_when_ray_id_out_of_range():
    # Defensive: if get_gpu_ids() returned something nonsensical, fall back to
    # the launcher local_rank rather than crashing.
    assert (
        _resolve(cuda_visible_devices=None, ray_gpu_ids=[99], device_count=4,
                 launcher_local_rank=1, pin_to_ray_gpu_id=True)
        == "1"
    )


# --- The exact 80B-SIF pathology this fix targets ---------------------------


def test_whole_node_bundle_sif_collision_reproduction():
    """In the SIF, 4 actors SHARE one {GPU:4} bundle -> Ray's get_gpu_ids()
    returned [0] for ALL of them (notes: job 626872). The OLD get_gpu_ids
    approach (attempt 7dd2b9ce) therefore collapsed every rank onto GPU 0.

    With pin_to_ray_gpu_id=False (whole-node mode), the current code instead
    uses the launcher modulo (0,1,2,3) — distinct logical indices, but
    positional, not physical, and racing on CVD ordering. This test pins that
    documented behavior so a regression is visible.
    """
    # OLD get_gpu_ids approach would have produced all "0":
    all_zero = [_resolve(cuda_visible_devices=None, ray_gpu_ids=[0], pin_to_ray_gpu_id=True,
                         launcher_local_rank=r, device_count=4) for r in range(4)]
    # Because each shared-bundle actor reports [0], pin-to-ray-id gives "0" for
    # all -> THIS is why per-GPU bundles (not shared {GPU:4}) are required.
    assert all_zero == ["0", "0", "0", "0"]

    # Current whole-node modulo path (pin off) spreads logically:
    modulo = [_resolve(cuda_visible_devices=None, ray_gpu_ids=[0], pin_to_ray_gpu_id=False,
                       launcher_local_rank=r, device_count=4) for r in range(4)]
    assert modulo == ["0", "1", "2", "3"]
