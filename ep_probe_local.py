"""Ray per-GPU-bundle device-assignment PROBE (root-cause confirmation).

Reproduces the exact failing condition of job 646698's policy PG:
  - a {GPU:1,CPU:1} PACK placement group spanning N GPUs per node,
  - actors requested with num_gpus=1, scheduled one-per-bundle (the
    get_reordered_bundle_indices ordering),
  - under SIF Ray that leaves CUDA_VISIBLE_DEVICES UNMASKED (case 3).

For each actor it reports the four quantities that decide whether the
per-GPU-bundle pin is sound:
  ray.get_gpu_ids()            <- what resolve_pinned_local_rank() trusts
  CUDA_VISIBLE_DEVICES         <- whether Ray masked it (expect <unset> on SIF)
  resolved LOCAL_RANK          <- the value we'd torch.cuda.set_device()
  physical GPU UUID @ that idx <- the ACTUAL silicon the pin lands on

PASS  := every actor's (node, physical-UUID) is DISTINCT
         (i.e. no two ranks share a physical GPU).
"""
import os
import socket
import ray


@ray.remote(num_gpus=1, num_cpus=1)
class Probe:
    def report(self, launcher_local_rank, rank):
        import torch
        from skyrl_train.utils import ray_noset_visible_devices
        from skyrl_train.utils.utils import resolve_pinned_local_rank

        gpu_ids = ray.get_gpu_ids()
        cvd = os.environ.get("CUDA_VISIBLE_DEVICES")
        noset = ray_noset_visible_devices()
        dc = torch.cuda.device_count()
        # exactly what worker.py __init__ computes when pin_to_ray_gpu_id=True
        lr = resolve_pinned_local_rank(
            noset_visible_devices=noset,
            cuda_visible_devices=cvd,
            ray_gpu_ids=gpu_ids,
            launcher_local_rank=launcher_local_rank,
            device_count=dc,
            pin_to_ray_gpu_id=True,
        )
        torch.cuda.set_device(int(lr))
        # physical identity of the device the pin actually selected
        idx = torch.cuda.current_device()
        try:
            uuid = torch.cuda.get_device_properties(idx).uuid
        except Exception:
            uuid = None
        # also resolve via nvidia-ml against the physical (CVD-stripped) ordering
        return {
            "rank": rank,
            "host": socket.gethostname(),
            "ray_gpu_ids": gpu_ids,
            "CUDA_VISIBLE_DEVICES": cvd,
            "noset": noset,
            "device_count": dc,
            "resolved_LOCAL_RANK": lr,
            "current_device": idx,
            "phys_uuid": str(uuid),
        }


def main():
    ray.init(address=os.environ.get("RAY_ADDRESS", "auto"))
    num_gpus = int(os.environ.get("PROBE_NUM_GPUS", "4"))
    from ray.util.placement_group import placement_group, placement_group_table
    from skyrl_train.utils import get_ray_pg_ready_with_timeout
    from skyrl_train.utils.utils import get_reordered_bundle_indices
    from ray.util.placement_group import PlacementGroupSchedulingStrategy

    bundles = [{"GPU": 1, "CPU": 1} for _ in range(num_gpus)]
    pg = placement_group(bundles, strategy="PACK")
    get_ray_pg_ready_with_timeout(pg, timeout=300)

    reordered = get_reordered_bundle_indices(pg)
    print("REORDERED_BUNDLE_INDICES:", reordered, flush=True)

    actors = []
    for rank in range(num_gpus):
        bidx = reordered[rank] if reordered else rank
        a = Probe.options(
            scheduling_strategy=PlacementGroupSchedulingStrategy(
                placement_group=pg, placement_group_bundle_index=bidx
            )
        ).remote()
        actors.append((rank, a))

    rows = ray.get([a.report.remote(rank % num_gpus, rank) for rank, a in actors])
    rows.sort(key=lambda r: r["rank"])
    print("\n==== PER-ACTOR DEVICE ASSIGNMENT ====", flush=True)
    for r in rows:
        print(r, flush=True)

    seen = {}
    collision = False
    for r in rows:
        key = (r["host"], r["phys_uuid"])
        if key in seen:
            collision = True
            print(f"COLLISION: rank {r['rank']} and rank {seen[key]} both on {key}", flush=True)
        seen[key] = r["rank"]

    distinct = len(seen)
    print(f"\nDISTINCT_PHYSICAL_GPUS={distinct} / {num_gpus}", flush=True)
    print("PROBE_RESULT=" + ("FAIL_STACKING" if collision else "PASS_DISTINCT"), flush=True)


if __name__ == "__main__":
    main()
