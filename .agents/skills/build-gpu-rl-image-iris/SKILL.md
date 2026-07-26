---
name: build-gpu-rl-image-iris
description: >-
  Build + push the gpu-rl Docker image (`ghcr.io/open-thoughts/openthoughts-agent:gpu-rl`) — the RL runtime for
  CoreWeave H100 (torch 2.11.0+cu128 + the from-source vLLM fork + flash-attn 2.8.3 + skyrl-train/torchtitan EP) —
  IN the CoreWeave `cw-us-east-02a` cluster, AS AN IRIS JOB USING KANIKO. Self-contained to THIS repo
  (MarinSkyRL): the RL env is `uv sync --frozen` against `skyrl-train/{pyproject.toml,uv.lock}` (the lock is the
  single source of truth — no constraint file). Covers WHY kaniko not buildkit (the cluster denies
  CAP_SYS_ADMIN/bind-mounts + runs gVisor), the crane-export-over-ubuntu recipe, the load-bearing resource/flag
  settings (512GB node, per-instruction layers not `--single-snapshot`, `--cache`, ghcr GitHub-PAT creds), the
  Dockerfile gotchas baked into the build (uv `--index-strategy unsafe-best-match`, `python -m pip wheel`,
  torchtitan+tyro via the `ep` extra), the build-asserts-as-validation, capturing the digest + bumping
  `DEFAULT_RL_DOCKER_IMAGE`, monitoring the build job, and WHEN a rebuild is actually required vs a live
  `/app` source sync. Use when asked to build / rebuild / push the gpu-rl image, after bumping the vLLM-fork
  commit / flash-attn / torch-CUDA / a baked dep. The Mac CANNOT build it (arm64 + needs linux/amd64 + ~512GB).
  Reference (this repo): docker/Dockerfile.gpu-rl + docker/build_gpu_rl_kaniko.sh (canonical build),
  cloud/iris/launch_rl_iris.py (canonical launcher, `DEFAULT_RL_DOCKER_IMAGE` digest pin).
---

# build-gpu-rl-image-iris

The **gpu-rl image** is the CoreWeave RL runtime — `linux/amd64`, **from-source CUDA**: the vLLM fork (cutlass
GEMM kernels) + flash-attn 2.8.3 (`flash_attn_2_cuda`) compiled with `nvcc` against torch 2.11.0+cu128 / cp312
/ x86_64, plus skyrl-train editable + torchtitan EP + harbor. It is consumed by the RL launcher
(digest-pinned in `cloud/iris/launch_rl_iris.py:DEFAULT_RL_DOCKER_IMAGE`). This skill is the **how-to to BUILD +
PUSH it**, and is self-contained to this repo — everything you need is hardcoded below.

> **Local clone = ground truth.** ALL Dockerfile / build-script edits go in the local Mac checkout of THIS repo
> → commit → (push). **The iris job bundles the LOCAL workspace to `/app`** via `git ls-files --cached --others
> --exclude-standard` (respects `.gitignore`; reads WORKING-TREE content, so uncommitted *tracked* edits ARE
> included — you do NOT have to commit/push before a build), so a local edit takes effect on the next build
> immediately. Never hand-edit on a remote / leave divergent state / patch-by-rsync.

## 0. Where the build runs, and why it can't run elsewhere

- **The Mac CANNOT build it.** It's arm64; the image is `linux/amd64`-only (CoreWeave H100 + the x86 CUDA
  build). Under QEMU + Docker Desktop the amd64 nvcc compiles are impractically slow and RAM-bound
  (`MAX_JOBS=8 × ~5 GB/job ≈ 40 GB` just for the compile; wheel-packaging needs much more — see §4). Don't.
- **There is NO in-cluster image-build primitive in iris.** `iris build` = LOCAL `docker buildx` + `docker
  login`. So to build amd64 in the cluster you run the build **AS AN IRIS JOB** — a CPU-only iris task that
  does the Docker build itself.
- **The build needs no GPU** (nvcc compiles on a CPU builder). It's a CPU/memory/disk job.

## 0.5 Ops preamble (hardcoded — no external docs needed)

Everything the build steps below rely on, inlined so this skill stands alone:

- **Build from the local Mac**, `cd /Users/benjaminfeuer/Documents/MarinSkyRL` first. Use the **otagent-env**
  full paths (symlinks fail in the sandbox):
  - python: `/Users/benjaminfeuer/miniconda3/envs/otagent/bin/python` (ships iris + a WORKING `kubernetes`).
  - iris: `/Users/benjaminfeuer/miniconda3/envs/otagent/bin/iris` (the cw-capable binary). **Do NOT use the
    marin `.venv/bin/iris`** — its `kubernetes` import is broken and it cannot drive CoreWeave.
- **`export KUBECONFIG=~/.kube/coreweave-iris-gpu`** — HARD prereq in the SAME shell for every CoreWeave call.
  The Mac's default kubeconfig points at a different cluster, so without this export you get misleading
  "0 pods / not found". **All `iris`/`kubectl` calls SYNCHRONOUS — never backgrounded.**
- **Cluster `cw-us-east-02a`.** No ssh / no login node — the iris SDK drives it over the controller tunnel.
- **Build job is CPU-only:** `--cpu 48 --memory 512GB --disk 400GB` (256GB OOMs at wheel-packaging; keep
  `--cpu 48` — not 64 — for daemonset gang-fit).
- **Node hardware context:** each node is 8×H100-80GB + InfiniBand; the build uses no GPU.
- **ghcr push creds:** `GHCR_TOKEN=$(gh auth token)` — a **GitHub PAT** (user `penfever`, scope
  `write:packages`). This is passed to the job as `DOCKER_TOKEN`. **It is NOT the Docker-Hub `DOCKER_TOKEN`
  (`dckr_pat_…`) in secrets.env — that one is WRONG for ghcr.io.** Any other secrets are sourced via
  `source "$DC_AGENT_SECRET_ENV"`; never paste a token VALUE (or a base64 of one) into a doc or command line.

## 1. Mechanism: KANIKO, not BuildKit (decisive)

The build job uses **kaniko** (`gcr.io/kaniko-project/executor`), NOT buildkit/`buildctl`:

- **BuildKit FAILS on this cluster.** `buildkitd`'s executor needs **bind mounts / `CAP_SYS_ADMIN`**. The
  cluster DENIES that: `--container-profile container_profile_privileged` is silently resolved to **DEFAULT
  caps** for a non-admin key (privileged "requires admin"), so the pod gets only `SYS_PTRACE`, no privileged.
  The nodes also run **gVisor** (`RuntimeClass gvisor`). Every buildkit attempt died
  `failed to mount …: permission denied` (both native + overlayfs snapshotters).
- **KANIKO snapshots its OWN container rootfs** — no bind mounts, no privileged, runs as root under these
  restrictions. This is the standard k8s no-daemon build answer, and it WORKS here.

## 2. How the iris job is shaped (the crane-export trick)

- The iris pod command is a **HARDCODED `["bash","-lc", script]`** — you can't set an arbitrary entrypoint, and
  **kaniko's image is distroless (no bash)**, so kaniko CANNOT be the task image directly.
- So: use **`docker.io/library/ubuntu:22.04`** as the task image (it has bash), then inside the script
  **`crane export` the kaniko executor rootfs over `/`** and run `/kaniko/executor`. (crane is fetched in the
  script; the export overlays kaniko's `/kaniko/…` onto the ubuntu rootfs.)
- **Context = the iris-synced `/app` bundle** (this repo). The build's Dockerfile is `docker/Dockerfile.gpu-rl`
  and the kaniko driver `docker/build_gpu_rl_kaniko.sh`; context is the repo root.
- **`WHEEL_SOURCE`: the FAST `prebuilt-wheelhouse` path is the DEFAULT; `wheel-builder` is the SLOW fallback.**
  The `rl` stage's `prebuilt-wheelhouse` COPYs `docker/wheelhouse/*.whl` and installs them → **ZERO nvcc**
  (build = minutes, dominated by the rl-stage `uv sync`). `wheel-builder` instead compiles the vLLM-fork +
  flash-attn wheels inline (the ~3 h nvcc cost; §4).
  - **The wheels do NOT ride the iris bundle.** `docker/wheelhouse/*.whl` is gitignored, AND even force-staged
    the ~900 MB of wheels blow the **hard 25 MB bundle cap** (`iris` `MAX_BUNDLE_SIZE_BYTES`; no per-file skip,
    no override). Only `docker/wheelhouse/.keep` is tracked.
  - **FAST PATH (no nvcc) — fetch the prebuilt wheels in-build.** `build_gpu_rl_kaniko.sh` step 3.5 curls the 2
    wheels + MANIFEST from the **public HF dataset `laion/gpu-rl-build-wheels`** (vLLM-fork + flash-attn are
    open-source) into `/app/docker/wheelhouse/` BEFORE `kaniko/executor` runs (kaniko reads the context dir
    live, so `COPY docker/wheelhouse/` then finds them), and **fails fast if a wheel is missing** (never
    silently falls back to a compile). This is the script default (`WHEEL_SOURCE=prebuilt-wheelhouse`).
    - **`--skip-unused-stages` is LOAD-BEARING on this path.** kaniko builds EVERY Dockerfile stage by default
      (unlike BuildKit, which prunes), so WITHOUT this flag the `wheel-builder` (nvcc) stage compiles even when
      the `rl` stage takes its wheels from `prebuilt-wheelhouse`. WITH it, the unreferenced `wheel-builder`
      stage is pruned → ZERO nvcc. `build_gpu_rl_kaniko.sh` passes it. (Symptom if missing: "Building wheel for
      flash-attn … still running" despite the prebuilt path.) **The wheel filenames the script fetches pin the
      vLLM-fork commit + flash-attn version** (`flash_attn-2.8.3-…` and `vllm-…g76259c63a…`) — they MUST match
      the Dockerfile's `{VLLM_FORK_COMMIT, FLASH_ATTN_VERSION, TORCH_VERSION, cu128, cp312}`. A SKYRL-source or
      harbor-only bump does not touch the wheels, so the same wheels stay ABI-correct.
  - **SLOW FALLBACK — `wheel-builder`** (`-e WHEEL_SOURCE wheel-builder`): use ONLY when no prebuilt wheels are
    available to fetch. Pays the ~3 h nvcc compile.
- **`--destination …:gpu-rl-<gitsha>`** (immutable pinned tag). kaniko's `--destination` push is the TERMINAL
  step → a successful job = a pushed image (§6). The floating `:gpu-rl` tag is **NOT moved by default**
  (`PUSH_FLOATING=0`); consumers pin the digest of the `:gpu-rl-<gitsha>` tag.

### ghcr push creds (the load-bearing gotcha)
- **`GHCR_TOKEN=$(gh auth token)` (GitHub PAT, `write:packages`)** — passed as `-e DOCKER_TOKEN`. The Docker Hub
  `dckr_pat_…` is WRONG for ghcr.io.
- kaniko reads `$DOCKER_CONFIG/config.json`. `build_gpu_rl_kaniko.sh` **writes that config AFTER the
  crane-export overlay** (kaniko's image ships its own `/kaniko` dir and would clobber a config written before
  the overlay) and sets `export DOCKER_CONFIG=/kaniko/.docker`, contents
  `{"auths":{"ghcr.io":{"auth":"<base64 of penfever:$DOCKER_TOKEN>"}}}`. The script disables shell tracing
  until AFTER the token is consumed, so the PAT never lands in the R2-persisted finelog.

## 3. The launch command (verbatim shape)

```bash
cd /Users/benjaminfeuer/Documents/MarinSkyRL
source "${DC_AGENT_SECRET_ENV:?set DC_AGENT_SECRET_ENV to the secrets file first}"
export KUBECONFIG=~/.kube/coreweave-iris-gpu                       # HARD prereq (§0.5)
IRIS=/Users/benjaminfeuer/miniconda3/envs/otagent/bin/iris        # the cw-capable (otagent-env) iris binary
GHCR_TOKEN=$(gh auth token)                                       # GitHub PAT, NOT the Docker Hub DOCKER_TOKEN
GITSHA=$(git rev-parse --short HEAD)                              # REQUIRED — build_gpu_rl_kaniko.sh has `: "${GITSHA:?}"` (the immutable :gpu-rl-<gitsha> tag)
B64=$(base64 -i docker/build_gpu_rl_kaniko.sh | tr -d '\n')       # the kaniko build script, base64'd in

$IRIS --cluster=cw-us-east-02a job run \
  --task-image docker.io/library/ubuntu:22.04 --no-sync --enable-extra-resources \
  --cpu 48 --memory 512GB --disk 400GB \
  --job-name gpurl-kaniko-$GITSHA \
  --max-retries 0 --timeout 18000 \
  -e DOCKER_USER_ID penfever -e DOCKER_TOKEN "$GHCR_TOKEN" -e BUILD_B64 "$B64" \
  -e GITSHA "$GITSHA" --no-wait \
  -- bash -lc 'echo "$BUILD_B64" | base64 -d > /tmp/build.sh && exec bash /tmp/build.sh'
```
> **`-e GITSHA` is MANDATORY** (`build_gpu_rl_kaniko.sh` hard-requires it via `: "${GITSHA:?}"` for the immutable
> tag; the job dies immediately without it). **`WHEEL_SOURCE` and `SINGLE_SNAPSHOT` need no `-e`** — the script
> defaults are already the ones you want (`prebuilt-wheelhouse` fast path; `SINGLE_SNAPSHOT=0` = per-instruction
> layers). Pass `-e WHEEL_SOURCE wheel-builder` ONLY to force the slow nvcc compile when no prebuilt wheels
> exist. For the **megatron variant**, add `-e INSTALL_MEGATRON 1 -e TAG_PREFIX gpu-rl-megatron`.
>
> **Do NOT set `SINGLE_SNAPSHOT=1`.** That collapses the image to ONE ~16.6 GB layer that **CANNOT be pulled**
> over the CoreWeave→ghcr egress (containerd EOFs the single-blob GET mid-download → every pod
> `ImagePullBackOff`, the gang never starts). The default `SINGLE_SNAPSHOT=0` gives per-instruction layers
> (max ~3.5 GB, each retryable). **The build SUCCEEDS + pushes either way — the un-pullable-ness only shows at
> LAUNCH**, so a green build with `SINGLE_SNAPSHOT=1` still dooms the grid. The Dockerfile ALSO re-layers the
> `uv sync` (§4) so no single layer exceeds ~4 GB. **ALWAYS verify post-build:** `docker buildx imagetools
> inspect …:gpu-rl-<gitsha> --raw` → max layer <8 GB, layers >20.

`build_gpu_rl_kaniko.sh` does, in order: fetch crane → `crane export` the kaniko executor over `/` → write
`$DOCKER_CONFIG/config.json` with the ghcr `penfever:$DOCKER_TOKEN` auth → (prebuilt path) curl the wheels from
`laion/gpu-rl-build-wheels` into `/app/docker/wheelhouse/` → run `/kaniko/executor --context dir:///app
--dockerfile docker/Dockerfile.gpu-rl --build-arg WHEEL_SOURCE=… --skip-unused-stages --compressed-caching=false
--cache=true --cache-repo=ghcr.io/open-thoughts/openthoughts-agent/cache --destination
…:gpu-rl-<gitsha>` (plus `--destination …:gpu-rl` only if `PUSH_FLOATING=1`). `--no-sync` here means only the
*kaniko* job's own sync; the context is still the iris-synced repo.

## 4. The critical resource + kaniko flag settings

- **512 GB node — NOT 256 GB.** A `mem256GB` run **DIES SILENTLY** (pod reaped, no error log) at the
  wheel-packaging / snapshot step → OOM. Use **`--cpu 48 --memory 512GB --disk 400GB`**. (`--cpu 48` not 64 is
  the daemonset-overhead gang-fit rule; for a 1-node build it's less critical, but keep 48.)
- **Per-instruction layers (default), not `--single-snapshot`.** The Dockerfile deliberately splits the single
  heavy `uv sync --frozen` into a chain of identical syncs that defer the heavy CUDA wheels one bucket at a time
  via `--no-install-package` (each package installed by the FIRST sync that stops excluding it; the FINAL sync
  excludes nothing → the closure is EXACTLY the lock, zero drift). This keeps every layer under ~4 GB so the
  image is pullable. kaniko runs with `--compressed-caching=false` to keep multi-layer snapshotting within the
  memory budget.
- **`--cache=true --cache-repo=ghcr.io/open-thoughts/openthoughts-agent/cache`** — reuses layers across
  rebuilds. On the prebuilt-wheelhouse path there is no nvcc RUN to cache, so the win is small; on the
  `wheel-builder` path note that per-layer cache reuse and `--single-snapshot` are mutually defeating — prefer
  the prebuilt fast path over relying on a cache HIT to save the 3 h.
- **Build time:** prebuilt-wheelhouse ≈ minutes (~25 min end-to-end, dominated by the rl-stage `uv sync` + the
  build asserts + the push). `wheel-builder` ≈ 3 h (flash-attn ~30–40 min, vLLM cutlass ~90 min, then
  packaging). `--timeout 18000` (5 h) gives headroom for either.

## 5. The Dockerfile gotchas baked into the build (would fail on ANY host)

These are FIXED in `docker/Dockerfile.gpu-rl` already — listed so you recognize them if a regression
reintroduces one:

1. **uv first-index guard.** A `uv pip install --extra-index-url <pytorch-cu128> … "setuptools>=77"` makes
   setuptools UNSATISFIABLE (uv pins to the first index). FIX (in the Dockerfile): **`--index-strategy
   unsafe-best-match`** on every `uv pip install`/`uv sync` that mixes the pytorch cu128 index with PyPI (the
   `wheel-builder` venv install uses it).
2. **`uv pip wheel` does not exist.** uv has NO `wheel` subcommand. The `wheel-builder` stage uses
   **`python -m pip wheel … --no-cache-dir --no-build-isolation`**, NOT `uv pip wheel`.
3. **torchtitan + tyro ride the lock's `ep` extra.** MarinSkyRL's `apply_ep` imports
   `torchtitan.distributed.expert_parallel.ExpertParallel` for EP>1 (the 30B-A3B MoE arms). `torchtitan/
   __init__.py` is EAGER (`→ quantization/float8.py → config/manager.py → import tyro`), and `tyro` is NOT a
   transitive of torch/transformers. skyrl-train's `ep` extra (in `pyproject.toml` + `uv.lock`) declares BOTH
   torchtitan AND tyro, so **`uv sync --frozen --extra ep`** installs them together — there is NO separate
   `--no-deps torchtitan` + tyro-first step (that was the old OT-Agent-Dockerfile pattern). The build's
   `from torchtitan.distributed.expert_parallel import ExpertParallel` assert proves it resolved.

> **Reproducibility.** The RL env is built purely from `uv sync --frozen` against
> `skyrl-train/{pyproject.toml,uv.lock}`; the lock pins the entire transitive set (incl.
> `nvidia-nccl-cu12`/torch/ray) by construction — no float is possible, so the class of the old
> gpu-rl-00220aac NCCL-drift regression is structurally prevented. There is **NO `rl_env_constraints.txt`** and
> **NO `UV_CONSTRAINT` step**. The only post-sync steps are `--no-deps` swaps of the compiled vLLM-fork +
> flash-attn wheels and a `git+` harbor install (harbor is dependency-agnostic — litellm/daytona, no
> torch/vllm/ray pins — so it cannot perturb the synced stack).

4. **IB VERBS USERSPACE for NET/IB (the `Could not find libnccl-net.so` red herring).** Symptom: cross-node
   NCCL rides **`NET/Socket : Using [0]enp157s0np0:…`** (slow TCP-over-ethernet) instead of **`NET/IB … GPU
   Direct RDMA`**, and the log shows `NET/Plugin: Could not find libnccl-net.so`. **That `libnccl-net.so` line
   is BENIGN and a RED HERRING** — on CoreWeave's Mellanox IB, NCCL's *built-in* IB transport needs NO external
   OFI/`libnccl-net.so` plugin (that plugin is an EFA/AWS thing). **Real root cause:** NCCL's built-in IB
   `dlopen`s **`libibverbs.so.1` + the `libmlx5` provider at runtime**; if the IB verbs userspace is absent from
   the image, NCCL **silently disables IB** and falls back to NET/Socket — no error, just the socket line. The
   CoreWeave pods DO expose the IB devices (`/dev/infiniband/*`, `/sys/class/infiniband/mlx5_0..8`, ports
   `ACTIVE`) — the gap is purely the missing userspace, NOT pod/config (no `rdma/*` resource or hostNetwork
   change) and NOT `NCCL_IB_DISABLE`/`NCCL_IB_HCA` (leave both UNSET). **Diagnose in-pod:** `ibv_devices` →
   "command not found" + `ldconfig -p | grep libibverbs` empty = the gap confirmed. **FIX (baked):** the
   rl-stage apt-get installs **`rdma-core ibverbs-providers libibverbs1 librdmacm1 ibverbs-utils`**. (The
   `NCCL_SOCKET_IFNAME=enp157s0np0` pin in the grid configs governs only the socket + bootstrap channel — it is
   COMPLEMENTARY to IB, does not force socket; keep it.) Expected post-fix signal (`NCCL_DEBUG=INFO`):
   `NET/IB : Using [0]mlx5_0:1/IB` + `GPU Direct RDMA Enabled`.

## 6. Build-asserts ARE the validation (a successful build = a working EP>1 stack)

The `rl` stage RUNs hard import asserts at build time. Because kaniko fails the build if any RUN exits non-zero,
**a SUCCESSFUL build PROVES the stack works**, and the `--destination` push is the terminal step, so
**success = pushed**:
- `import flash_attn, flash_attn_2_cuda` — the CUDA extension EXISTS (from the compiled wheel).
- `import torch, transformers, vllm, skyrl_train, flash_attn, flash_attn_2_cuda`.
- `import boto3, smart_open, s3fs` + a `boto3.client('s3', …)` ctor — the S3/rendezvous client stack (comes
  entirely through the frozen lock; the ctor is where a botocore skew would surface).
- `from torchtitan.distributed.expert_parallel import ExpertParallel` — the **EP>1 MoE unblock**. If this
  prints `… import OK`, `apply_ep` resolves `ExpertParallel` and the CoreWeave EP=8 RL jobs can launch.
- (megatron variant only) `import megatron.core, megatron.bridge, transformer_engine`.

You do NOT need a separate post-build smoke for these — they're inside the build.

## 7. Monitoring the build job

> **⚠ State-poll, NEVER a log-string watch.** Poll the authoritative iris lifecycle state; a clean kill /
> OOM-reap emits no terminal log line, so a log-string watch can hang forever on a dead job.

```bash
PY=/Users/benjaminfeuer/miniconda3/envs/otagent/bin/python   # otagent env: ships iris + a WORKING kubernetes
IRIS=/Users/benjaminfeuer/miniconda3/envs/otagent/bin/iris
export KUBECONFIG=~/.kube/coreweave-iris-gpu                 # HARD prereq for every cw call (§0.5)
$PY scripts/iris/watch_job_state.py /benjaminfeuer/gpurl-kaniko-<gitsha> --interval 60      # watch until terminal
# richest single call (state + error + exit + finished_at):
$IRIS --cluster=cw-us-east-02a job summary /benjaminfeuer/gpurl-kaniko-<gitsha> --json
# full log by time-window (NOT --tail, which under-samples): --since-ms <submitted_at_ms> --no-tail
$IRIS --cluster=cw-us-east-02a job logs /benjaminfeuer/gpurl-kaniko-<gitsha> --since-ms <submit_ms> --no-tail
```
All `iris`/`kubectl` calls **synchronous** — never background them.

## 8. Capture the digest + bump the launcher (the terminal step)

After the build SUCCEEDS (state SUCCEEDED, push done), capture the immutable digest and pin it:

```bash
# resolve the :gpu-rl-<gitsha> tag's sha256 (use the immutable tag, NEVER the floating :gpu-rl):
docker buildx imagetools inspect ghcr.io/open-thoughts/openthoughts-agent:gpu-rl-<gitsha>
crane manifest --platform linux/amd64 ghcr.io/open-thoughts/openthoughts-agent:gpu-rl-<gitsha>   # then digest it
```
Then **bump the matching pin in `cloud/iris/launch_rl_iris.py`** to the new
`ghcr.io/open-thoughts/openthoughts-agent@sha256:<digest>` and update the provenance comment: a
`gpu-rl-<gitsha>` digest goes in `DEFAULT_RL_DOCKER_IMAGE`, a `gpu-rl-megatron-<gitsha>` digest in
`DEFAULT_RL_MEGATRON_DOCKER_IMAGE`. The launcher selects between them from the config's
`trainer.strategy`, so **bump BOTH in the same commit** — leaving one behind makes a strategy switch
silently cross a harbor-version boundary. **Pin the DIGEST, never the floating `:gpu-rl` tag** (it
stale-caches under `imagePullPolicy: IfNotPresent`). Commit the launcher bump locally.

**Last known-good digest (currently pinned on main):**
`sha256:e8b48241b548da570a319ff421e72787692ee87dae2408c99a2d0c6794186177` (gpu-rl-e03896b7 — baked harbor
`f4a6b1a0`; PULLABLE, 48 layers max 3.46 GB; build asserts green). This pin **predates the merged harbor-bridge
change** — harbor is baked into the image, so an image bump is the durable follow-up before dropping the runtime
`--harbor-ref` override.

## 9. WHEN a rebuild is actually required (vs a runtime checkout)

Rebuild the image ONLY for a change to something the image **BAKES / COMPILES**:
- **vLLM-fork commit** (`VLLM_FORK_COMMIT` ARG) — compiled from source; a runtime checkout can't replace a CUDA
  build.
- **flash-attn version** (`FLASH_ATTN_VERSION`) — compiled.
- **torch / CUDA** (`TORCH_VERSION` / `BASE_IMAGE` cu128) — the whole ABI / wheel cache-key.
- **a baked dep change** — anything in `skyrl-train/uv.lock` (bump the lock first, then rebuild: `uv sync
  --frozen` reproduces the new set by construction), the torchtitan/tyro/`ep` extra, harbor (`HARBOR_COMMIT`),
  or the rl-stage apt set.

Do **NOT** rebuild for a **MarinSkyRL source** change — the launcher syncs the local workspace to `/app`
(first on `PYTHONPATH`), so `skyrl_train.*` + `examples.terminal_bench.*` are picked up live from `/app` at
launch, no image rebuild. The compiled vLLM-fork is
the only thing that genuinely forces a rebuild for a code change.

> If the build host WERE a real x86 box (not the cluster), the documented fast path is the wheel-cache:
> `docker/build_wheels.sh` (compile the wheels once → `docker/wheelhouse/`) then a `docker buildx` build with
> the default `WHEEL_SOURCE=prebuilt-wheelhouse` (no nvcc). On the Mac neither is usable (arm64); in the cluster
> the kaniko wheel-fetch from `laion/gpu-rl-build-wheels` is the equivalent no-nvcc win. See
> `docker/README.gpu-rl-wheelcache.md`.

## 10. Standing constraints

- **Never `iris cluster restart`/stop/bounce** (kills every running job). The build job is its own iris job;
  `iris job kill /benjaminfeuer/gpurl-kaniko-<gitsha>` is job-scoped (with permission).
- **`--cache-repo` is shared** — don't delete the populated cache repo.

---

## Cross-reference
- **Consumes this image:** the iris RL launcher (`cloud/iris/launch_rl_iris.py`; bump its
  `DEFAULT_RL_DOCKER_IMAGE` digest after a build).
- **Dockerfile internals + the wheel cache (for a real x86 host):** `docker/README.gpu-rl-wheelcache.md`,
  `docker/Dockerfile.gpu-rl` (header comments), `docker/build_wheels.sh`.
- **Build primitive:** `scripts/iris/watch_job_state.py` (the state-poll monitor).
- (If you also have OT-Agent access, cross-ref `.claude/ops/iris/coreweave_gpu_ops.md` — NOT a dependency;
  everything needed is inlined above.)
