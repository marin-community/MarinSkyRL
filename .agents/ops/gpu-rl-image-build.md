# GPU-RL image — deploy boundary, build contract, pins

Facts for building and deploying `ghcr.io/marin-community/marinskyrl`. The
`build-gpu-rl-image-iris` skill points here; this file owns the constants.

## Deploy boundary

| edited path | reaches the cluster how |
|---|---|
| `skyrl-train/**`, including `ppo_base_config.yaml` | **image rebuild required** |
| `cloud/iris/**` — launcher, `run_rl.py`, `configs/*.yaml` | `/app` workspace bundle, no rebuild |

Why: `cloud/iris/run_rl.py` runs the entrypoint with `cwd=/opt/skyrl/skyrl-train`, and `python -m`
puts the CWD first on `sys.path`, ahead of `PYTHONPATH`. `config_dir` in
`skyrl_train/entrypoints/main_base.py` is derived from `__file__`, so Hydra loads the baked
`ppo_base_config.yaml`.

Failure signature when a `trainer.*` key outruns the image — fires ~8 minutes in, after the Ray
head is up:

```
Could not override 'trainer.<name>'.
Key '<name>' is not in struct        full_key: trainer.<name>
```

Task 0 exits 1; the remaining ranks are reaped on `Job exceeded max_task_failures`, so the wasted
allocation is ~5× the time to failure.

## Build contract

`docker/build_gpu_rl_kaniko.sh` hard-fails with `: "${VAR:?}"` on each of these before building:

| variable | required when |
|---|---|
| `GITSHA` | always |
| `DOCKER_USER_ID` | always |
| `GHCR_TOKEN` | always — GitHub PAT with `write:packages`, not a Docker Hub `dckr_pat_…` |
| `PREBUILT_WHEEL_ARTIFACT_URI` | prebuilt-wheelhouse path; SHA-verified `s3://` artifact |
| `PREBUILT_WHEEL_ARTIFACT_SHA256` | prebuilt-wheelhouse path |

`GHCR_IMAGE_REPOSITORY` defaults to `ghcr.io/marin-community/marinskyrl` in the script, and
`KANIKO_CACHE_REPOSITORY` derives from it as `<repository>/cache`. Neither needs an env var. Set
them only to push somewhere else on purpose.

The script is the authority — read its top rather than trusting this table.

`--no-sync` is required on the `iris job run` line. The task image is `ubuntu:22.04`, which has no
`uv`, so without it iris runs its own setup phase and the job dies at `[iris setup] step 1/2` with
`uv: command not found`. It does not suppress the workspace bundle: `/app` still receives the repo,
which is what `DOCKER_CONTEXT=/app` needs.

Blackwell (`TORCH_CUDA_ARCH_LIST` including `10.0`, for the GB200 nodes on cw-us-east-08a) requires
`WHEEL_SOURCE=wheel-builder`. The published prebuilt wheels are compiled at `8.0;9.0`, and the
script folds the Dockerfile's arch list into the expected wheel MANIFEST and `cmp`s it against the
artifact, so a prebuilt-wheelhouse build fails the comparison instead of shipping wheels with no
`sm_100` kernels. That path pays the full nvcc compile.

A `wheel-builder` build also pushes the wheel stage as `<repository>:wheels-<gitsha>`, so the
compiled vLLM-fork and flash-attn wheels survive the job. Cut a new prebuilt-wheelhouse artifact
from that tag rather than repeating the compile: `crane export <repository>:wheels-<gitsha> - |
tar -x wheels/`, then tar `wheels/` (containing both `.whl` files and `MANIFEST`), upload to the
`s3://` path below, and record the new URI and SHA-256 here. The `MANIFEST` inside already carries
the `TORCH_CUDA_ARCH_LIST` the wheels were built for, which is what the prebuilt path compares.

`PREBUILT_WHEEL_ARTIFACT_URI` / `_SHA256` are the one pair with no home in the repo, so record them
here. Current values, used by the `ac9c5f39`/`f2b44d4a`/`90072ada` builds:

```
PREBUILT_WHEEL_ARTIFACT_URI=s3://marin-us-east-02a/iris/grug-vllm-wheels/4b55591306c9-torch211-cu128-cp312-fb6ff59.tar.gz
PREBUILT_WHEEL_ARTIFACT_SHA256=d247a8865c56ea2756512fcb0b102f8127b2020bdfbfb92d76dbd1386d514417
```

The digest is written out in full here on purpose. It was recorded truncated once, and the next
rebuild had to recover the remaining characters from an earlier launch before it could start.

The wheel tarball is keyed on the vLLM native-donor commit (`4b55591306c9`) plus the torch/CUDA/py
triple, so it changes only when one of those baked pins moves — at which point a new tarball must be
built before the image can be. A rebuild agent that cannot find this value has to recover it from a
prior launch or an unmerged branch, which has already cost one build cycle.

## Pins

- Deployed digests live in `cloud/iris/launch_rl_iris.py` (`DEFAULT_RL_DOCKER_IMAGE`,
  `DEFAULT_RL_MEGATRON_DOCKER_IMAGE`). Single source of truth; do not copy a digest elsewhere.
- The launcher picks the variant from `trainer.strategy`. `strategy: megatron` needs the megatron
  variant; a plain `gpu-rl-*` image dies at driver init with `No module named 'megatron'`.
- **Baked pins are declared in `docker/Dockerfile.gpu-rl` and nowhere else.**
  `build_gpu_rl_kaniko.sh` reads `HARBOR_COMMIT`, `VLLM_FORK_COMMIT`,
  `VLLM_NATIVE_DONOR_COMMIT`, `FLASH_ATTN_VERSION` and `TORCH_VERSION` from it and echoes each as
  `[pin] NAME=value`. An env override that disagrees with the Dockerfile is a **hard error**, and a
  missing declaration is a hard error. To change a pin, edit the Dockerfile and commit it.
- Rebuild is also required for: vLLM fork commit, flash-attn version, torch/CUDA base,
  `skyrl-train/uv.lock`, the torchtitan `ep` extra, the rl-stage apt set.

## Grug FSDP2 constraints

- Keep `accelerate>=1.14,<2`. Accelerate 1.11 forwards Transformers 5's
  `_is_hf_initialized` marker into `torch.nn.Parameter` construction under
  `init_empty_weights`, which breaks FSDP2 meta loading. Accelerate 1.14 strips and
  restores the marker.
- Ray imports `skyrl_train.worker_setup` before assigning actor CUDA masks. Keep it and
  `skyrl_train.__init__` free of Torch, Transformers, and Ray imports so CUDA cannot
  initialize against the driver's device view. Jobs 927538 and 930208 showed that an
  actor-constructor policy reset does not prevent uvloop from creating the concurrency
  loop first. `RolloutCoordinator` repeats the idempotent hook because Ray cloudpickles
  the class by value; keep its Terminal Bench import inside the actor constructor.
- Weight-sync rank offsets are per logical engine, not per vLLM data-parallel actor.
  Rendezvous ports use 20000–29999 rather than the Linux ephemeral client range and
  must be unique across logical engines in a job.
- Grug router logits and query-bias observations remain FP32. The persistent query-bias
  buffer is replicated by FSDP2 and transferred as its own mixed-dtype weight-sync
  chunk; fused weight sync is unsupported.

## Known gaps

- A fuller rewrite of the build skill sits on the unmerged branch
  `origin/romain-dev/vllm-fork-build-docs-20260724`. It predates the deploy boundary above.
- APEX in the image is not built with `--cpp_ext --cuda_ext`, which forces
  `gradient_accumulation_fusion: false` in megatron configs.
