---
name: build-gpu-rl-image-iris
description: Build, push, validate, and deploy-pin MarinSkyRL GPU-RL container variants as Iris jobs using the repository's kaniko driver. Use when rebuilding an RL runtime after MarinSkyRL source, Harbor, the frozen dependency lock, CUDA extensions, system packages, or backend extras change.
---

# Build GPU-RL images on Iris

Build GPU-RL images from a clean MarinSkyRL source revision, monitor every Iris build to a terminal state, validate
the published manifests, and update the launcher digest pins. Treat the repository as the source of truth; do not
patch a remote builder or preserve mutable build facts in this skill.

## Load the live build state

Before changing or submitting anything, read these files completely:

1. `AGENTS.md` for repository policy.
2. `.agents/ops/gpu-rl-image-build.md` for current clusters, resource requests, credential sources, registry
   constraints, and launch template.
3. `docker/build_gpu_rl_kaniko.sh` for the required-variable contract and architecture selection.
4. The selected `docker/Dockerfile.gpu-rl*` for baked dependency pins and build assertions.
5. `cloud/iris/launch_rl_iris.py` for the deployed standard and Megatron digest pins.

The script and Dockerfiles outrank prose when they disagree. Correct stale ops documentation in the same change.

## Establish the build set

- Inventory the GPU-RL variants consumed by the launcher. Build the standard and Megatron variants together so a
  strategy switch cannot cross a MarinSkyRL or Harbor boundary.
- Resolve every requested host architecture from current cluster state. Let the build script derive its Dockerfile,
  kaniko platform, cache namespace, and tag suffix from the build host.
- Build each requested architecture on a native host. Do not cross-build CUDA images from a laptop.
- Use the user's requested cluster and priority when supplied. Otherwise use the current placements in the ops file.

## Prepare immutable source

1. Fetch the default branches of MarinSkyRL and Harbor.
2. Work from a clean MarinSkyRL worktree based on the latest default branch.
3. Resolve the Harbor commit to bake at execution time. Compare its dependency manifests with the currently pinned
   Harbor revision before changing the Dockerfiles.
4. Update every architecture-specific Dockerfile to the same Harbor revision when the change applies to all variants.
5. Commit baked-pin changes before building. Use the commit hash in the immutable image tags so the tag identifies
   the exact build context.
6. Confirm the Iris bundle contains only intended tracked and unignored files and stays within the controller limit.

Never identify a build with a commit that predates uncommitted Dockerfile or source changes.

## Reconcile and submit

1. Read the top of `docker/build_gpu_rl_kaniko.sh` immediately before submission and satisfy every required variable.
2. Select the wheel source from the build script's architecture defaults and any validated artifact recorded in the
   ops file. Never claim a prebuilt artifact is available without its URI, digest, and compatible manifest.
3. Obtain registry credentials at execution time. Pass them as job environment variables without printing, encoding
   into documentation, or persisting them in logs.
4. Bundle the current build script into each Iris job exactly as described by the current ops launch template.
5. Submit one job per architecture and variant with explicit job names, cluster, priority, resources, retry policy,
   and timeout.
6. Keep the standard and Megatron tags distinct. Do not move floating tags unless the user explicitly requests it.

Use kaniko through the repository driver. Do not substitute a builder that requires privileges unavailable to the
selected cluster.

## Monitor to terminal state

- Poll authoritative Iris lifecycle state with the helper documented in
  `.agents/ops/iris-operator-scripts.md`. Do not infer completion from a log string.
- Keep Iris and cluster calls synchronous. Poll separate submitted jobs in a bounded loop and give the user concise
  progress updates at least once per minute while actively waiting.
- On failure, capture the job summary, complete time-window logs, pod termination reason, and relevant Kubernetes
  events before deciding whether to retry.
- Retry only when the failure class and current policy allow it. Preserve cache state; never delete a shared cache to
  recover a single build.
- Never restart or stop a shared cluster.

A successful job must reach the push step after all Dockerfile assertions pass. A disappeared pod or quiet log is not
success.

## Validate published images

For every built tag:

1. Resolve and record the immutable digest from the architecture-specific tag.
2. Inspect the raw manifest and verify the platform matches the intended host architecture.
3. Verify the layer count and maximum layer size against the current operational ceiling in the ops file.
4. Pull the manifest anonymously to confirm clusters without registry credentials can access it.
5. Confirm the build logs contain the expected standard or Megatron assertion set and the resolved MarinSkyRL and
   Harbor revisions.

Do not deploy-pin a tag that passes build assertions but fails platform, layer, visibility, or anonymous-access
validation.

## Update deployment pins

1. Update the standard and Megatron digest constants in `cloud/iris/launch_rl_iris.py` together for each supported
   architecture boundary represented there.
2. Keep provenance comments concise: source revision, Harbor revision, build jobs, variant relationship, and validation
   evidence. Move extended build history to the ops file.
3. Run the launcher tests, repository lint, and required review pass.
4. Commit, push, and open or update the PR using the `commit` and `writing-style` skills.
5. Watch CI and review activity through merge. Merge only with user authorization and green required checks.

Return the build job IDs, tags, digests, platforms, validation results, launcher PR, and merge commit.

## Rebuild boundary

Resolve the rebuild boundary from the current ops runbook and runtime entrypoint before deciding.
