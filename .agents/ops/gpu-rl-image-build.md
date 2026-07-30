# GPU-RL image build operations

This runbook owns mutable facts for building and deploying
`ghcr.io/marin-community/marinskyrl`. The build script and Dockerfiles remain the executable source
of truth; inspect them before each build.

## Operator environment

Work from the repository root discovered with Git. Initialize Iris and Kubernetes access from
`coreweave.md`. Resolve the registry username from the authenticated GitHub account and obtain the
registry token at runtime; never store or print it.

## Image variants

Build standard and Megatron variants from the same committed source on each required architecture.

| host architecture | build cluster | placement | tag forms |
|---|---|---|---|
| amd64 | `cw-rno2a` | GPU-free request on an H100 worker | `gpu-rl-<sha>`, `gpu-rl-megatron-<sha>` |
| arm64 | `cw-us-east-08a` | request one GB200 to force a Grace host | `gpu-rl-<sha>-arm64`, `gpu-rl-megatron-<sha>-arm64` |

Current build resources:

```text
amd64: --cpu 48 --memory 512GB --disk 400GB --priority interactive --max-retries 0 --timeout 18000
arm64: --gpu GB200x1 --cpu 96 --memory 640GB --disk 500GB --priority interactive --max-retries 3 --timeout 28800
```

Recheck live cluster configuration and capacity before submission. The arm64 GPU request is a host
architecture constraint; kaniko itself does not use the GPU.

## Rebuild boundary

Rebuild for changes to baked source/configuration under `skyrl-train`, frozen dependencies, Harbor,
the vLLM fork, CUDA extensions, torch/CUDA, or system packages. Launcher-only configuration under
`cloud/iris` reaches jobs through the workspace bundle and ordinarily does not require an image.

The Dockerfiles declare baked pins. Change and commit the Dockerfiles; do not override a disagreeing
pin at build time. Compare Harbor dependency metadata between old and new pins before building,
because its post-lock install can alter the resolved environment.

## Build contract

`docker/build_gpu_rl_kaniko.sh` requires:

| variable | condition |
|---|---|
| `GITSHA` | always; full committed MarinSkyRL revision |
| `DOCKER_USER_ID` | always |
| `GHCR_TOKEN` | always; token must push packages |
| `PREBUILT_WHEEL_ARTIFACT_URI` and `PREBUILT_WHEEL_ARTIFACT_SHA256` | only for `prebuilt-wheelhouse` |

Architecture is derived from the build host. It selects the Dockerfile, crane/kaniko platform,
wheel platform, cache repository, and arm64 tag suffix. Do not override those as an architecture
selection mechanism.

Use `WHEEL_SOURCE=wheel-builder` for arm64. On amd64, use `prebuilt-wheelhouse` only when the operator
supplies an artifact URI and digest whose manifest matches the Dockerfile; otherwise use
`wheel-builder`. That mode preserves the compiled wheel stage under `wheels-<sha><arch-suffix>` and
reuses the registry cache on retries.

## Submit

Run from the clean committed build worktree:

```bash
GITSHA=$(git rev-parse HEAD)
DOCKER_USER_ID=$(gh api user --jq .login)
GHCR_TOKEN=$(gh auth token)
BUILD_B64=$(base64 < docker/build_gpu_rl_kaniko.sh | tr -d '\n')
```

Every job uses `docker.io/library/ubuntu:22.04`, `--no-sync`, `--enable-extra-resources`,
`--no-wait`, the architecture's resource line above, and:

```bash
-e BUILD_B64 "$BUILD_B64" \
-e GITSHA "$GITSHA" \
-e DOCKER_USER_ID "$DOCKER_USER_ID" \
-e GHCR_TOKEN "$GHCR_TOKEN" \
-e WHEEL_SOURCE wheel-builder \
-- bash -lc 'echo "$BUILD_B64" | base64 -d > /tmp/build.sh && exec bash /tmp/build.sh'
```

For Megatron add:

```bash
-e INSTALL_MEGATRON 1 -e TAG_PREFIX gpu-rl-megatron
```

Name jobs uniquely by variant, architecture, and short revision. Do not pass object-store credentials;
task pods receive the correct identity through cluster secret injection.

`--no-sync` is required because the Ubuntu task image has no `uv`. The workspace bundle still
places the build context at `/app`.

## Registry constraints

- Target clusters may lack GHCR pull secrets, so the package must support anonymous pulls.
- Keep every compressed image layer below the current operational ceiling of approximately 8 GB.
- Deployed standard and Megatron digests are pinned in `cloud/iris/launch_rl_iris.py`.

The `build-gpu-rl-image-iris` skill owns monitoring, image validation, and deployment procedure.
