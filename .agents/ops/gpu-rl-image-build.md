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
`s3://` path below, and record the new URI and SHA-256 here.

**Rewrite line 1 of the MANIFEST before uploading.** The wheel-builder stage writes
`VLLM_FORK_COMMIT=${VLLM_FORK_COMMIT}` — the fork commit it compiled. The prebuilt path expects that
line to carry `VLLM_NATIVE_DONOR_COMMIT` instead, in both `docker/Dockerfile.gpu-rl` and the expected
copy in `build_gpu_rl_kaniko.sh`. A MANIFEST copied out verbatim therefore fails every prebuilt build
about 35 s in with `/tmp/expected-wheel-manifest … MANIFEST differ: char 18, line 1`. The artifact
recorded below has the unrewritten line and cannot be used until it is re-cut. Every other line is
correct as written, including the `TORCH_CUDA_ARCH_LIST` the wheels were built for.

While the cache repo still holds the wheel layers, `WHEEL_SOURCE=wheel-builder` is the working
fallback and costs no nvcc time — the whole stage is a cache hit and the build reaches the rl stage
in about seven minutes.

`PREBUILT_WHEEL_ARTIFACT_URI` / `_SHA256` are the one pair with no home in the repo, so record them
here. The recorded artifact, cut from `wheels-2b14abd3` on 2026-07-28 (951,075,128 bytes), is the
one carrying the unrewritten MANIFEST line above, so **no prebuilt build can currently use it**:

```
PREBUILT_WHEEL_ARTIFACT_URI=s3://marin-us-east-02a/iris/grug-vllm-wheels/4b55591306c9-torch211-cu129-cp312-2b14abd3.tar.gz
PREBUILT_WHEEL_ARTIFACT_SHA256=0147c62621c92d60d7407255bec9790fcddbb311edb99a209b0ee33fa037a6bb
```

Its wheels are otherwise the right ones — they carry `sm_100`, so once the MANIFEST is re-cut this
is the first set that can build a Blackwell image without paying the nvcc compile. The superseded
cu128 pair (`...cu128-cp312-fb6ff59.tar.gz`, sha `d247a886...d514417`) cannot satisfy the Dockerfile
at all: its MANIFEST reads `CUDA=12.8` and `TORCH_CUDA_ARCH_LIST=8.0;9.0`, both of which the
expected-manifest comparison now rejects.

Cut the artifact with an **iris job on cw-us-east-02a**, never from a laptop. The wheel-builder image
is ~20 GB and the bucket is in that region, so a local `crane export` pulls tens of GB down and pushes
back over the internet. Do NOT pass `AWS_ACCESS_KEY_ID` / `AWS_SECRET_ACCESS_KEY` to that job: the pod
already receives correct CoreWeave credentials and `AWS_ENDPOINT_URL` from the `iris-task-env` Secret
via `envFrom`, explicit container `env` overrides `envFrom`, and forwarding the launch host's pair
retargets the upload at real AWS S3 — it fails with `The access key ID you provided does not exist in
our records`. Let fsspec's default discovery find the injected pair.

The digest is written out in full here on purpose. It was recorded truncated once, and the next
rebuild had to recover the remaining characters from an earlier launch before it could start.

The wheel tarball is keyed on the vLLM native-donor commit (`4b55591306c9`) plus the torch/CUDA/py
triple, so it changes only when one of those baked pins moves — at which point a new tarball must be
built before the image can be. A rebuild agent that cannot find this value has to recover it from a
prior launch or an unmerged branch, which has already cost one build cycle.

## arm64 / GB200

`cw-us-east-08a` is **arm64**. All 808 GPUs sit behind Grace hosts; the only amd64 nodes there are
4 CPU-only ones with no GPUs.

```
GPUs by host arch: {'amd64': 0, 'arm64': 808}
arm64  gb200-4x           x202   144 cores, 955 GB RAM, GB200 compute_cap 10.0
amd64  cd-gp-i64-erapids  x4     CPU only
```

The shipped gpu-rl image is a single-platform `linux/amd64` manifest with `linux_x86_64` wheels, so
it cannot run there at all. `sm_100` kernels are necessary but not sufficient — the host CPU is the
binding constraint, and there is no amd64 GPU capacity to fall back to.

`docker/Dockerfile.gpu-rl-arm64` is the counterpart, built by the same
`docker/build_gpu_rl_kaniko.sh`. kaniko builds for the architecture it runs on, so the image
architecture follows the build host and there is no cross-compilation: an iris job on
`cw-us-east-08a` lands on aarch64 with 144 cores and 955 GB, against 48 cores and 512 GB on the
amd64 builder. Use `KUBECONFIG=~/.kube/coreweave-iris`, the only CoreWeave kubeconfig.

Launch shape that worked. `--gpu` is what forces an arm64 node — the build needs no GPU, and a
CPU-only request lands on one of the four amd64 nodes instead. Ask for one GPU, not four: the
cluster runs near full, and a whole-node request is both harder to schedule and preempted sooner.

```
--task-image docker.io/library/ubuntu:22.04 --no-sync --enable-extra-resources
--gpu GB200x1 --cpu 96 --memory 640GB --disk 500GB --priority batch --max-retries 3 --timeout 28800
-e INSTALL_MEGATRON 1  -e TAG_PREFIX gpu-rl-megatron
```

The Dockerfile, the `-arm64` tag suffix, the kaniko cache repo and `WHEEL_SOURCE=wheel-builder` all
default from `uname -m` in the script; an env var still overrides each, but nothing has to be passed.
The suffix exists because every tag is derived from the git sha and the same commit builds both
images; without it an arm64 build overwrites the amd64 tag, including the `wheels-<gitsha>` tag whose
wheels are not interchangeable. All four follow the build host rather than an operator flag because
getting one wrong is silent: the wrong Dockerfile bakes a `linux_x86_64` wheel MANIFEST, and a
missing suffix destroys a shipped tag.

`--max-retries 3` is preemption insurance, not a blind retry: at batch priority on a full cluster a
build is preempted within minutes, and kaniko resumes from the cache repo rather than from zero.
A first probe run was preempted 88 seconds in by a production job and reported only `Pod not found`;
the real reason is in `kubectl get events`, as `EvictedDueToPreempted`.

### What the aarch64 dependency survey actually found

The earlier survey in this file said TransformerEngine had no aarch64 wheel anywhere and needed a
source build. That was wrong, and the correction matters because it removes the only hard compile:

| dependency | aarch64 status (checked 2026-07-28) |
|---|---|
| torch, torchvision | `manylinux_2_28_aarch64` at cu129 |
| triton, ray, tensordict, flashinfer-jit-cache, vLLM 0.23.0 | aarch64 wheels in the lock |
| megatron-core 0.18.0 | `cp312-manylinux_2_28_aarch64` on PyPI |
| megatron-bridge 0.5.0, nvidia-modelopt, flash-linear-attention | `py3-none-any` |
| transformer_engine 2.11.0 | `py3-none-any` (pure-Python framework layer) |
| **transformer_engine_cu12 2.11.0** | **`py3-none-manylinux_2_28_aarch64` on PyPI** — the CUDA core library |
| **transformer_engine_torch 2.11.0** | **`cp312-linux_aarch64` in the erictang000 release**, beside the x86_64 one |
| mamba-ssm, causal-conv1d | `cp312-linux_aarch64` in the erictang000 releases |
| deep-ep | x86_64 only, and not in either image — see below |

No source build is required for any of it.

**Survey the pinned version, never the latest one.** This is what produced the wrong conclusion, and
it will produce another one otherwise. TransformerEngine's latest release at the time of the survey
was 2.17.0, which publishes x86_64 only; the version this image pins is 2.11.0, which publishes
both arches. A package's newest release says nothing about the platform coverage of the release you
actually install, and the direction of the error is not predictable — a project can add an
architecture later or drop one, so "latest has it" is no safer an inference than "latest lacks it".
Resolve the pin first, then query that exact version:
`curl -s https://pypi.org/pypi/<name>/<pinned-version>/json | jq -r '.urls[].filename'`, and for a
GitHub-hosted wheel list the release assets rather than assuming the tag matches upstream's matrix.

Two further reasons the obvious sources understate what exists. The lock records only the x86_64
files for most of this subtree, because those packages are reachable only under an `x86_64` marker
and uv prunes wheels to the platforms where a package is reachable — so the lock is evidence about
the amd64 resolution, not about what the index holds. And some of these wheels are not on an index
at all: `transformer-engine-torch`, `mamba-ssm` and `causal-conv1d` come from erictang000 GitHub
releases, which exist precisely because upstream does not build against torch 2.11, so upstream's
platform matrix is the wrong thing to check for them.

**deep-ep is not a gap.** It is declared only in skyrl-train's `deepep` extra, and the Dockerfiles
sync `--extra vllm --extra ep` (plus `megatron`) — never `deepep`. The amd64 megatron image in
production therefore does not contain deep-ep either. `deep_ep` is imported only by
`skyrl_train/distributed/deepep.py`, lazily from the FSDP2 strategy; `megatron_strategy.py`, which
owns the `bridge.save_hf_weights` call, does not reference it. Its absence on aarch64 is not a
regression.

### How Megatron is installed on aarch64

Every requirement in the `megatron` extra is gated on `platform_machine == 'x86_64'`, and
`[tool.uv.sources]` pins transformer-engine-torch, mamba-ssm and causal-conv1d to x86_64 wheel URLs.
So `uv sync --extra megatron` SUCCEEDS on aarch64 and installs nothing; the failure surfaces much
later at `import transformer_engine`. Measured: the frozen lock resolves to 236 packages on
aarch64 against 332 on x86_64, and the 96-package difference is exactly the Megatron subtree.

Relaxing those markers means regenerating `skyrl-train/uv.lock`, which is an input to the amd64
image that production is pinned to. `Dockerfile.gpu-rl-arm64` therefore installs the stack
explicitly with `uv pip install` from the aarch64 artifacts above, and asserts afterwards that torch
and torchvision are still `2.11.0+cu129` / `0.26.0+cu129`. On the probe run that install left torch,
transformers 5.8.1, vllm 0.23.0, ray 2.51.1 and accelerate 1.14.0 untouched.

The Megatron subtree is consequently not lock-pinned on the arm64 image. The versions that matter
are ARGs in the Dockerfile.

**`uv pip install` applies no project configuration, including
`[tool.uv] override-dependencies`.** `uv sync` gives the amd64 image that list for free; the explicit
aarch64 install does not get it, and the whole list goes missing at once, not just the entry you
notice. The one that fails the build is `nvidia-resiliency-ext ; sys_platform == 'never'` — without
it, megatron-bridge's `megatron-core[dev]` requires nvidia-resiliency-ext 0.6.0, which publishes
`manylinux_2_39` wheels only and cannot install on this glibc 2.35 base:

```
Because nvidia-resiliency-ext==0.6.0 has no wheels with a matching platform tag
(e.g., `manylinux_2_35_aarch64`) ... your requirements are unsatisfiable
hint: Wheels are available on: `manylinux_2_39_aarch64`, `manylinux_2_39_x86_64`
```

Note the hint names an aarch64 wheel, so this reads as an architecture gap and is not one — both
architectures are published, and both are too new for the base image. The Dockerfile extracts the
override list out of `skyrl-train/pyproject.toml` at build time and passes it as `--overrides` rather
than restating it, so the two cannot drift; the rest of the list is what stops the megatron install
from moving the boto3/fsspec/flashinfer versions the lock resolved.

### Two ordering traps

`import transformer_engine` dlopens its core library at import time and fails with
`OSError: libcudart.so.12: cannot open shared object file` unless the CUDA runtime is already
loaded. `import megatron.core, megatron.bridge, transformer_engine` passes because megatron imports
torch first. A standalone `import transformer_engine.pytorch` in a non-CUDA image does not. The
arm64 build assert imports torch first for this reason.

`crane export` defaults to `linux/amd64` whatever the host, so the kaniko executor has to be
exported with an explicit `--platform`, and the crane release asset itself is
`go-containerregistry_Linux_arm64.tar.gz`. Both are now derived from `uname -m` in
`build_gpu_rl_kaniko.sh`.

## The image must be PUBLIC on ghcr, because two of three clusters cannot authenticate

Image pull secrets in namespace `iris`, measured 2026-07-28:

| cluster | ghcr pull secret |
|---|---|
| cw-us-east-02a | `ghcr-helw150` |
| cw-rno2a | **none** |
| cw-us-east-08a | **none** |

Only east-02a can pull a private or internal package. rno2a and east-08a pull anonymously, so a
non-public image fails there with `ImagePullBackOff` while working perfectly on east-02a — an
asymmetry that reads as an image problem when it is a visibility problem. The digest is fine; the
node simply cannot fetch it.

This was invisible until the registry moved. `ghcr.io/open-thoughts/openthoughts-agent` was public,
so rno2a never needed a secret and nothing recorded that it depended on that. The first build pushed
to `ghcr.io/marin-community/marinskyrl` created the package as **internal** — GitHub's default for a
new org package — and every RL arm on rno2a sat in `ImagePullBackOff` for two and a half hours.

**After creating a new package, check its visibility and set it public.** The REST API cannot do
this; `PATCH /orgs/.../packages/container/<name>` returns 404 whatever the payload. It is UI-only:

```
github.com/orgs/marin-community/packages/container/<name>/settings -> Change visibility -> Public
```

Verify with an ANONYMOUS pull, not an authenticated one — your own `gh` credentials will succeed
against an internal package and tell you nothing about what a cluster node sees:

```bash
DOCKER_CONFIG=$(mktemp -d) crane manifest --platform linux/amd64 <repository>@sha256:<digest>
```

Recovery needs no intervention once visibility flips. Kubelet's backoff caps around five minutes;
24 pods went from `ImagePullBackOff` to 23 running in five minutes with no relaunch. Do not kill and
resubmit arms that are merely waiting on a backoff cycle.

Copying `ghcr-helw150` into the other clusters would also work and is the wrong fix: it clones one
person's credential across clusters and leaves the next new package broken in the same way.

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
- **A harbor bump can move the rl closure, because the harbor install is a post-hoc `uv pip
  install` over the frozen lock.** Whatever harbor requires wins over what the lock resolved. Read
  the dependency diff between the old and new harbor pin before building. At `74d76ecb`, harbor
  moved `datasets` and `claude-agent-sdk` from core dependencies into `datasets` and `analysis`
  extras; the Dockerfile names both extras so the closure stays where it was. Dropping `datasets`
  would silently downgrade it from the release harbor used to pull to the lock's 4.0.0, whose
  `fsspec[http]<=2025.3.0` cap the lock's fsspec 2026.4.0 violates. The plain image's
  exactly-one-conflict gate catches that; `INSTALL_MEGATRON=1` does not, so a megatron-only build
  would have shipped it.

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
