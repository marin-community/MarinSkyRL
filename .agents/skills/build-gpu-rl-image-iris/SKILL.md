---
name: build-gpu-rl-image-iris
description: >-
  Build and push the linux/amd64 GPU-RL image from MarinSkyRL as a CPU-only Iris
  Kaniko job. Covers parameterized GHCR destinations, write:packages credentials,
  the strict prebuilt-wheel artifact fast path, cache selection, monitoring, digest
  verification, and the separate shared-default release step.
---

# Build the GPU-RL image on Iris

The canonical implementation is `docker/build_gpu_rl_kaniko.sh` with
`docker/Dockerfile.gpu-rl`. The Iris job only needs CPU: use 48 CPU, 512 GB RAM,
and 400 GB disk. The image is `linux/amd64`; no live GPU is required to build it.

Kaniko is load-bearing here. CoreWeave pods do not grant the mount privileges
needed by BuildKit. The script starts in Ubuntu, overlays Kaniko's executor with
`crane export`, and builds from the Iris-synced `/app` workspace.

## Safety and credentials

- Set `GHCR_IMAGE_REPOSITORY` explicitly. The script has no shared destination
  default.
- Run a push preflight against the exact target repository before allocating the
  build node.
- Use a GitHub token with `write:packages`. The default `gh auth token` is not
  necessarily sufficient. Prefer an already configured Docker credential or a
  purpose-specific `GHCR_TOKEN`.
- Never print the token. The script consumes it before enabling shell tracing.
- For scratch builds, use a personal repository, `PUSH_FLOATING=0`, and
  `KANIKO_CACHE=0` unless a writable personal cache repository has been proven.
- Do not move `DEFAULT_RL_DOCKER_IMAGE` for a scratch build. A shared release and
  its digest update are a separate, deliberate operation.

## Wheel source

Use `WHEEL_SOURCE=prebuilt-wheelhouse` whenever a matching wheel artifact exists.
Set `PREBUILT_WHEEL_ARTIFACT_URI` to an archive containing:

- `wheels/MANIFEST`
- exactly one `vllm-*.whl`
- exactly one `flash_attn-*.whl`

The launcher derives the expected ABI and source pins from the Dockerfile,
compares the manifest, and exits on any mismatch or fetch failure. The Dockerfile
checks the manifest again before installing the wheels. This path never falls
through to a source compile.

`prebuilt-wheelhouse` is the script and Dockerfile default. It requires
`PREBUILT_WHEEL_ARTIFACT_URI` and fails before installing build dependencies when
the URI is missing. Use `WHEEL_SOURCE=wheel-builder` only when a multi-hour
source rebuild is explicitly intended.

## Push preflight

Before a build, push a tiny disposable OCI image or manifest to a uniquely named
tag in `GHCR_IMAGE_REPOSITORY`, then resolve its digest. This proves that the
credential can create and write the package. Do not use another organization's
namespace as a permission probe.

Package creation and Kubernetes pull access are separate checks. New GHCR
packages are private by default. Before scheduling a GPU validation job, verify
that the Iris namespace's existing image-pull identity can fetch the manifest.
If the package is intentionally made public, verify an anonymous manifest fetch
too. Do not modify a shared service account or image-pull secret for a scratch
release.

## Canonical Iris launch

The example below uses the prebuilt-wheel fast path and disables Kaniko cache.
Choose a unique personal prefix and retain it for every disposable resource.

```bash
cd /path/to/MarinSkyRL
export KUBECONFIG=/path/to/coreweave-iris-gpu-kubeconfig
IRIS=/path/to/iris
GITSHA=$(git rev-parse --short HEAD)
BUILD_B64=$(base64 -w0 docker/build_gpu_rl_kaniko.sh)
SCRATCH_PREFIX="YOUR_USER-dev"

# Populate these without echoing their values.
DOCKER_USER_ID="YOUR_GHCR_USER"
GHCR_TOKEN="YOUR_WRITE_PACKAGES_TOKEN"
GHCR_IMAGE_REPOSITORY="ghcr.io/YOUR_GHCR_USER/${SCRATCH_PREFIX}-gpu-rl"
WHEEL_URI="s3://marin-us-east-02a/iris/grug-vllm-wheels/4b55591306c9-torch211-cu128-cp312-fb6ff59.tar.gz"

$IRIS --cluster=cw-us-east-02a job run \
  --task-image docker.io/library/ubuntu:22.04 \
  --no-sync --enable-extra-resources \
  --cpu 48 --memory 512GB --disk 400GB \
  --job-name ${SCRATCH_PREFIX}-gpurl-kaniko-$GITSHA \
  --max-retries 0 --timeout 18000 \
  -e DOCKER_USER_ID "$DOCKER_USER_ID" \
  -e GHCR_TOKEN "$GHCR_TOKEN" \
  -e GHCR_IMAGE_REPOSITORY "$GHCR_IMAGE_REPOSITORY" \
  -e GITSHA "$GITSHA" \
  -e TAG_PREFIX ${SCRATCH_PREFIX}-grug \
  -e PUSH_FLOATING 0 \
  -e KANIKO_CACHE 0 \
  -e WHEEL_SOURCE prebuilt-wheelhouse \
  -e PREBUILT_WHEEL_ARTIFACT_URI "$WHEEL_URI" \
  -e BUILD_B64 "$BUILD_B64" \
  --no-wait \
  -- bash -lc 'echo "$BUILD_B64" | base64 -d > /tmp/build.sh && exec bash /tmp/build.sh'
```

On macOS, replace `base64 -w0` with `base64 -i ... | tr -d '\n'`.

If cache use is intentional, replace `KANIKO_CACHE=0` with:

```bash
-e KANIKO_CACHE 1 \
-e KANIKO_CACHE_REPOSITORY "ghcr.io/YOUR_GHCR_USER/${SCRATCH_PREFIX}-gpu-rl-cache"
```

The floating tag is not moved unless `PUSH_FLOATING=1`. Never set
`SINGLE_SNAPSHOT=1`: it creates a very large layer that can build and push
successfully but fail to pull through CoreWeave's egress.

## Monitor and verify

Poll Iris lifecycle state, not a log substring. A pod can be OOM-reaped without
a useful terminal log line.

```bash
$IRIS --cluster=cw-us-east-02a job summary /USER/YOUR_USER-dev-gpurl-kaniko-GITSHA
$IRIS --cluster=cw-us-east-02a job logs /USER/YOUR_USER-dev-gpurl-kaniko-GITSHA \
  --since-ms SUBMISSION_TIME_MS --no-tail
```

A successful job proves that the Dockerfile's import and Grug registry checks
passed and that Kaniko completed its push. Still verify all of the following:

- resolve the immutable tag to its `sha256` digest;
- inspect the manifest and confirm the largest layer is below 8 GB;
- preflight a pull through the same mechanism Iris will use;
- launch validation by immutable digest, never a floating tag.

The runtime image is consumed via the launcher's `--docker_image` override. Only
a separately approved shared release should update
`DEFAULT_RL_DOCKER_IMAGE` and its provenance comment.

## When a rebuild is necessary

Rebuild for compiled or baked inputs: the vLLM fork commit, flash-attn, torch or
CUDA ABI, `skyrl-train/uv.lock`, the `ep`/Megatron extras, Harbor pin, or runtime
apt packages. A MarinSkyRL-only source change can normally use the Iris-synced
`/app` workspace as the deploy vector; it does not require recompiling vLLM.

Do not stop, restart, or reconfigure the Iris cluster. Any cleanup must be scoped
to the disposable build job and personal image tags created for the task.
