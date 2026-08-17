# Typed runtime configuration and environment ownership

Issues [#52](https://github.com/marin-community/MarinSkyRL/issues/52) and
[#305](https://github.com/marin-community/MarinSkyRL/issues/305) describe the same boundary problem from two directions.
Training behavior still reads Marin-owned `SKYRL_*` variables, while launchers, YAML files, worker setup, shell scripts,
and container recipes retain 309 grandfathered environment definitions across 292 location/name keys.

This migration removes environment variables as a MarinSkyRL training-control surface. Typed configuration travels with
the resolved Hydra config and is consumed directly by the component that owns the behavior. `EnvVarManager` remains the
only projection boundary for values that must be strings in a child process environment, such as NCCL, vLLM, CUDA,
credentials, scheduler metadata, and runtime-library paths.

## Configuration contract

Add typed fields under the component that consumes each value. Defaults preserve current production behavior.

| Owner | Typed setting | Replaces |
| --- | --- | --- |
| `trainer.distributed` | placement-group and collective timeouts | `SKYRL_RAY_PG_TIMEOUT_IN_S`, `SKYRL_WORKER_NCCL_TIMEOUT_IN_S` |
| `trainer.policy.host_memory_monitor` | enabled state and sample interval | `SKYRL_POLICY_HOST_RAM_MONITOR*` |
| `trainer.model_load_retry` | retry count and bounded backoff | `SKYRL_HF_LOAD_MAX_RETRIES`, `SKYRL_HF_LOAD_BACKOFF_BASE`, `SKYRL_HF_LOAD_BACKOFF_CAP` |
| `trainer.progress` | output mode and throttle intervals | `SKYRL_PROGRESS_*` |
| `trainer.algorithm` | TIS splice and LCS alert threshold | `SKYRL_TIS_SPLICE`, its two aliases, `SKYRL_TIS_LCS_ALERT_THRESHOLD` |
| `generator` | R3 transport, dispatch timeout, coordinator workers | `SKYRL_R3_RESIDENT`, `SKYRL_R3_DECENTRAL`, `SKYRL_DISPATCH_PUT_TIMEOUT_S`, `SKYRL_COORDINATOR_EXECUTOR_WORKERS` |
| `generator` | fused weight transfer | `SKYRL_FUSE_WEIGHTS`; the existing `generator.fuse_weights` field is authoritative |
| policy model config | GatedDeltaNet implementation | `SKYRL_GDN_FLASHQLA`; use an enum instead of paired mask/enable booleans |
| FSDP policy config | streamed expert-loader row budget | `SKYRL_EP_LOADER_CHUNK_ROWS` |
| runtime launch config | NUMA policy and source/export paths | `SKYRL_ENABLE_NUMA_AFFINITY`, `SKYRL_PYTHONPATH_EXPORT`, `SKYRL_EXPORT_PATH` |

Hydra overrides remain the command-line override mechanism. Existing Iris convenience flags emit Hydra overrides into the
resolved config; they no longer emit `SKYRL_*` variables. Raw top-level `extra_env` is not a supported route for changing
MarinSkyRL behavior.

Configuration validation runs before Ray actor creation. Enums reject unknown modes, durations and counts must be positive,
and incompatible settings fail with their typed field names. Runtime code does not fall back to ambient `SKYRL_*` values.

## Delete settled toggles

The following variables select a default-on correctness path or a representation that has already replaced the historical
path. Preserve the behavior and delete the switch:

- `SKYRL_FORWARD_DISPATCH_FIX`
- `SKYRL_WEIGHTSYNC_DRAIN_BARRIER`
- `SKYRL_CP_REQUIRE_RIGHT_ALIGN`
- `SKYRL_W13_RELOAD_BRACKET`
- `SKYRL_QWEN3_5_VLM_UNWRAP`
- `SKYRL_GDN_MASK_FLA`; the supported default is the architecture-selected PyTorch path
- `SKYRL_R3_TENSOR_CAPTURE`; routed-expert evidence uses the contiguous narrow array carrier whenever it exists

`SKYRL_TITO_FULL` is also removed because `trainer.algorithm.tito_full` already owns the setting. The two deprecated TIS
splice aliases are removed instead of becoming typed compatibility fields. This hard fork does not retain environment
compatibility shims.

## Environment contract

Move the registry shared by `EnvVarManager` and the repository checker to a lightweight module in the root package. Each
entry records:

- one owner;
- `CONFIG`, `DERIVED`, `EXTERNAL`, or `SECRET` source;
- driver, task-runtime, Ray-worker, inference-worker, build, or process-local scope;
- the permitted writer boundary.

Marin-owned config and derived values may be written only by `EnvVarManager`. Python callers ask for a named projection and
merge the returned mapping into a process, Ray runtime, or Iris task request. They do not assign literal environment keys.

Some values cannot originate in Hydra:

- Iris and Ray identity, rendezvous, rank, device, and scheduler variables are derived at their process boundary.
- credentials are read from the selected secrets source and projected only into declared scopes;
- CUDA, NCCL, vLLM, W&B, Hugging Face, telemetry, locale, and build-tool variables remain external interfaces;
- Docker build arguments and shell-only build settings remain declarative build boundaries.

These exceptions still require registry entries. The checker permits only the writer kind declared by the entry and rejects
unregistered names, a second owner, or a managed value written outside `EnvVarManager`. A name that genuinely has multiple
architecture-specific build sites must opt into that cardinality in its registry entry.

Remove `infra/env_var_legacy_definitions.json`. There is no grandfathered count after this migration; every surviving site is
either a manager projection or an explicit registered boundary.

## Migration sequence

1. Add the typed fields and validation. Thread values through constructors and method calls that already carry the resolved
   config. Replace import-time environment constants with instance values.
2. Make settled correctness and representation paths unconditional. Remove environment-only branches and tests that exist
   only to switch back to historical behavior.
3. Replace Iris `extra_env` blocks with typed config and manager-owned third-party projections. Cluster network defaults remain
   launcher-derived and enter the task through the manager.
4. Move Python environment writes into manager projection functions. Register the remaining secret, scheduler, third-party,
   process-local, shell, and Docker boundaries.
5. Delete the legacy baseline and make the checker compare every detected definition with the registry contract.

The work lands as one PR so no intermediate commit accepts both an ambient variable and a typed setting with ambiguous
precedence. Commits may separate typed consumers, unconditional behavior, and contract enforcement for review.

## Behavioral tests

- Resolve every maintained Iris YAML and compare the relevant typed runtime values with the current launch behavior.
- Assert that Iris convenience flags change Hydra overrides and do not add `SKYRL_*` task variables.
- Exercise config-to-driver, task, Ray-worker, and inference-worker projection for registered third-party settings.
- Exercise timeout, R3 transport, host-memory monitoring, model-load retry, progress, TIS, fused transfer, GDN, and expert-loader
  behavior through typed inputs.
- Preserve regression coverage for forward dispatch, weight-sync drain, CP alignment, w13 reload, VLM unwrapping, and routed
  expert array transport without environment setup.
- Reject direct managed writes, unregistered names, wrong writer kinds, and duplicate ownership in the repository checker.
- Assert that the legacy baseline file is absent and the production tree has no runtime read of a Marin-owned `SKYRL_*`
  control.

The ordinary root CPU suite is the merge gate. GPU tests are required only if implementation changes a numerical or
collective path instead of deleting the switch around the already-tested default path.
