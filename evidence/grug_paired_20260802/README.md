# Grug eager/grouped gate evidence

This directory preserves the launchers used for the 2026-08-02 gate on the
measurement commit `d81f6364ab66947cdf520a3a42a274b586e830da`. It is evidence,
not product code.

The exact measurement source is the parent of this evidence-only commit. It is
based on MarinSkyRL #276 at
`b37cd1cb4027c0a11d705734554c739e5f9f67f7`, with #249 held common at
`7276dee1f7d9c94d4925bf91a1eff07d0d86295f`. The immutable runtime image is:

```text
ghcr.io/marin-community/marinskyrl@sha256:188eb430485f12182f483a7ee1c2c50191898b5a91e0fa6fea9ef183c4b947a6
```

## What ran

`run_dev_preflight_d81f636.sh` ran three one-microbatch-per-rank arms, in this
order, inside one reserved eight-H100 pod:

1. eager without attribution hooks;
2. eager with attribution hooks;
3. native grouped with attribution hooks.

It then invoked the pinned verifier. The verifier stopped at `paired topology
differs`. The predeclared stop rule prohibited the 32-H100 timing pair. The
development allocation was released.

The three signed results are under:

```text
s3://marin-us-east-02a/iris/grug-training-perf-gap/20260802/paired-d81f636/preflight/
```

`launch_preflight_failure_readback_d81f636.sh` submitted the successful
CPU-only independent readback job:

```text
/romain/grug-preflight-failure-readback-d81f636-r2-20260802
```

That readback rehashed all three results. It confirmed that the same physical
GPU set was reused, while ranks 2 and 3 exchanged GPUs. It also found route-load
differences on all ranks and 46 representative-gradient tolerance violations.
The eager instrumentation oracle passed.

## Reproduction

Run from a Marin checkout so Iris can load the local object-store declarations.
Set `MSRL_ROOT`, `MARIN_ROOT`, and `READBACK_FILE` in the launchers for the
local checkout. The committed values preserve the executed commands. Do not
print object-store or registry credentials.

The build launcher records the exact image-build submission. The following
commands reserve and release the exact image through Marin's normal dev-GPU
helper:

```bash
export DEV_GPU_TASK_IMAGE='ghcr.io/marin-community/marinskyrl@sha256:188eb430485f12182f483a7ee1c2c50191898b5a91e0fa6fea9ef183c4b947a6'
uv run --project "$MARIN_ROOT" "$MSRL_ROOT/evidence/grug_paired_20260802/dev_gpu_exact_image.py" \
  --config "$MARIN_ROOT/lib/iris/config/cw-rno2a.yaml" \
  --name <unique-session-name> allocate

# Copy run_dev_preflight_d81f636.sh into the reserved pod and execute it once.

uv run --project "$MARIN_ROOT" "$MSRL_ROOT/evidence/grug_paired_20260802/dev_gpu_exact_image.py" \
  --config "$MARIN_ROOT/lib/iris/config/cw-rno2a.yaml" \
  --name <unique-session-name> release
```

Do not rerun the production pair from this evidence. First fix or explain the
route and gradient mismatch, then pass a newly predeclared eight-H100 gate.
