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
- The wheel fetch uses the `FSSPEC_S3` configuration and S3 credentials injected
  into Iris task pods. Do not forward host AWS credentials; they override the
  working in-cluster endpoint configuration.
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

The launcher derives the expected ABI and native-donor pin from the Dockerfile,
verifies an operator-supplied archive SHA-256, compares the manifest, and exits
on any mismatch or fetch failure. The Dockerfile checks the manifest again,
then installs the exact `VLLM_FORK_COMMIT` Git source with the local vLLM wheel
as its precompiled native donor. The source commit and native donor are recorded
separately in the image. This path never falls through to a native source
compile or an upstream nightly wheel.

`prebuilt-wheelhouse` is the script and Dockerfile default. It requires
`PREBUILT_WHEEL_ARTIFACT_URI` and `PREBUILT_WHEEL_ARTIFACT_SHA256`; missing or
malformed values fail before installing build dependencies. Use
`WHEEL_SOURCE=wheel-builder` only when a multi-hour source rebuild is explicitly
intended.

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
IRIS=/path/to/iris
GITSHA=$(git rev-parse --short HEAD)
BUILD_B64=$(base64 -w0 docker/build_gpu_rl_kaniko.sh)
SCRATCH_PREFIX="YOUR_USER-dev"

# Populate these without echoing their values.
DOCKER_USER_ID="YOUR_GHCR_USER"
GHCR_TOKEN="YOUR_WRITE_PACKAGES_TOKEN"
GHCR_IMAGE_REPOSITORY="ghcr.io/YOUR_GHCR_USER/${SCRATCH_PREFIX}-gpu-rl"
WHEEL_URI="s3://marin-us-east-02a/iris/grug-vllm-wheels/4b55591306c9-torch211-cu128-cp312-fb6ff59.tar.gz"
WHEEL_SHA256="d247a8865c56ea2756512fcb0b102f8127b2020bdfbfb92d76dbd1386d514417"

$IRIS --cluster=cw-us-east-02a job run \
  --task-image docker.io/library/ubuntu:22.04 \
  --no-sync --enable-extra-resources \
  --cpu 48 --memory 512GB --disk 400GB \
  --priority batch --no-preemptible \
  --job-name ${SCRATCH_PREFIX}-gpurl-kaniko-$GITSHA \
  --max-retries 0 --timeout 18000 \
  -e DOCKER_USER_ID "$DOCKER_USER_ID" \
  -e GHCR_TOKEN "$GHCR_TOKEN" \
  -e GHCR_IMAGE_REPOSITORY "$GHCR_IMAGE_REPOSITORY" \
  -e GITSHA "$GITSHA" \
  -e TAG_PREFIX ${SCRATCH_PREFIX}-vllm \
  -e PUSH_FLOATING 0 \
  -e KANIKO_CACHE 0 \
  -e WHEEL_SOURCE prebuilt-wheelhouse \
  -e PREBUILT_WHEEL_ARTIFACT_URI "$WHEEL_URI" \
  -e PREBUILT_WHEEL_ARTIFACT_SHA256 "$WHEEL_SHA256" \
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

A successful job proves that the Dockerfile's dependency and import checks
passed and that Kaniko completed its push. Still verify all of the following:

- resolve the immutable tag to its `sha256` digest;
- inspect the manifest and confirm the largest layer is below 8 GB;
- preflight a pull through the same mechanism Iris will use;
- launch validation by immutable digest, never a floating tag.

The runtime image is consumed via the launcher's `--docker_image` override. Only
a separately approved shared release should update
`DEFAULT_RL_DOCKER_IMAGE` and its provenance comment.

## When a rebuild is necessary

Rebuild the image for any baked input: the vLLM source commit,
`skyrl-train/uv.lock`, the `ep`/Megatron extras, Harbor pin, or runtime apt
packages. Rebuild the native wheels only when a compiled input changes: vLLM
native source, flash-attn, torch or CUDA ABI, Python ABI, architecture, or
compiler inputs. A Python-only vLLM change may reuse a donor only after a
reviewed donor-versus-source diff proves that no native input changed.

A MarinSkyRL-only source change can normally use the Iris-synced `/app`
workspace as the deploy vector. Certification runs must still prove that the
synced source is exactly the recorded commit, or disable source sync and import
the baked tree.

Do not stop, restart, or reconfigure the Iris cluster. Any cleanup must be scoped
to the disposable build job and personal image tags created for the task.
