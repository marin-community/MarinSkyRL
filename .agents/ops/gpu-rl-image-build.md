# GPU-RL image — deploy boundary, build contract, pins

Facts for building and deploying `ghcr.io/open-thoughts/openthoughts-agent`. The
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
| `GHCR_IMAGE_REPOSITORY` | always |
| `DOCKER_USER_ID` | always |
| `GHCR_TOKEN` | always — GitHub PAT with `write:packages`, not a Docker Hub `dckr_pat_…` |
| `KANIKO_CACHE_REPOSITORY` | `KANIKO_CACHE=1`; shared, do not repoint or delete |
| `PREBUILT_WHEEL_ARTIFACT_URI` | prebuilt-wheelhouse path; SHA-verified `s3://` artifact |
| `PREBUILT_WHEEL_ARTIFACT_SHA256` | prebuilt-wheelhouse path |

The script is the authority — read its top rather than trusting this table.

## Pins

- Deployed digests live in `cloud/iris/launch_rl_iris.py` (`DEFAULT_RL_DOCKER_IMAGE`,
  `DEFAULT_RL_MEGATRON_DOCKER_IMAGE`). Single source of truth; do not copy a digest elsewhere.
- The launcher picks the variant from `trainer.strategy`. `strategy: megatron` needs the megatron
  variant; a plain `gpu-rl-*` image dies at driver init with `No module named 'megatron'`.
- `HARBOR_COMMIT` defaults in `docker/Dockerfile.gpu-rl` and `docker/build_gpu_rl_kaniko.sh` must
  match what production bakes. They are `772e20f7` as of 2026-07-27. A stale default reverts
  harbor while the build log reports no change.
- Rebuild is also required for: vLLM fork commit, flash-attn version, torch/CUDA base,
  `skyrl-train/uv.lock`, the torchtitan `ep` extra, the rl-stage apt set.

## Known gaps

- A fuller rewrite of the build skill sits on the unmerged branch
  `origin/romain-dev/vllm-fork-build-docs-20260724`. It predates the deploy boundary above.
- APEX in the image is not built with `--cpp_ext --cuda_ext`, which forces
  `gradient_accumulation_fusion: false` in megatron configs.
