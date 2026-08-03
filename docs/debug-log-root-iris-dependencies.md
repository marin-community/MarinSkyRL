# Debugging log for root Iris dependencies

Restore the repository-root Iris CLI and RL watcher on macOS without changing the Linux production
dependency closure.

## Initial status

After a frozen root sync on macOS, `uv run iris --help` failed first because `s3fs` was absent and,
when supplied as an overlay, failed again because `marin-iris` imported `INTRA_CLUSTER_CIDRS` from
`marin-finelog 0.2.10`. The RL watcher likewise failed at module import because `boto3` and
`botocore` were absent.

## Hypothesis 1

The Linux markers in `override-dependencies` replace the base boto and S3 requirements on every
platform, then evaluate false on macOS and remove those packages from the solution.

## Changes to make

Move the Linux-only boto and S3 pins to `constraint-dependencies`. Constraints narrow matching Linux
requirements without replacing the base requirements on macOS.

## Results

Before the change, the synced macOS environment contained none of `boto3`, `botocore`, or `s3fs`.
The refreshed macOS environment installed `boto3 1.43.48`, `botocore 1.43.48`, `s3transfer
0.19.1`, `aiobotocore 3.9.0`, and `s3fs 2026.4.0`. Imports for boto3, botocore, and s3fs all
succeeded, as did the Iris and RL watcher help entrypoints. The Linux lock resolves the constrained
compatible set: boto3/botocore 1.42.97, s3transfer 0.16.1, aiobotocore 3.7.0, and s3fs 2026.4.0.

## Hypothesis 1a

The existing Linux pins describe a compatible boto stack and can move unchanged from overrides to
constraints.

## Results

Refuted. Real constraint solving exposed that `aiobotocore 3.7.0` requires
`botocore>=1.42.90,<1.43.1`, while the old override forcibly installed `botocore 1.43.48`. The
resolver had been prevented from enforcing the dependency's own compatibility bound. An isolated
install verified `boto3 1.42.97`, `botocore 1.42.97`, `s3transfer 0.16.1`, `aiobotocore 3.7.0`, and
`s3fs 2026.4.0` as a mutually compatible set, so the Linux constraints use those versions.

## Hypothesis 1b

The boto and S3 family are the only cross-platform requirements incorrectly placed behind a Linux
override.

## Results

Refuted by a real Iris submission after the initial fix. CLI import and `--help` succeeded, but
constructing the RPC client failed with `ModuleNotFoundError: opentelemetry.metrics` before job
submission. `opentelemetry-api` had the same Linux-only override shape and was therefore absent on
macOS. Replacing the old `1.44.0` Linux override with a real constraint or removing it was also
refuted: Megatron Core declares `opentelemetry-api<1.34`, while the pinned Iris dependency pyqwest
requires `>=1.39`, making the universal Megatron environment unsatisfiable without the existing
override. The Linux override remains as a documented compatibility exception, paired with an
unversioned non-Linux override so macOS retains the transitive package instead of deleting it. The
macOS CI smoke test constructs `connectrpc._client_sync.SyncClient` so it exercises this lazy import
boundary.

## Hypothesis 2

The pinned Iris release declares `marin-finelog>=0.2.10` but imports a deployment symbol absent from
0.2.10. The resolver legally selected that incompatible minimum.

## Changes to make

Pin the published compatible build, `marin-finelog==0.2.19.dev30724030287`, and smoke-test the Iris
CLI in both the Linux launcher job and the macOS install job. A stable `0.2.11` is not published on
the configured package index, so expressing that apparent floor makes the lock unsatisfiable unless
prereleases are enabled globally.

## Results

The working Marin operator environment uses its local `marin-finelog 0.2.11`. The published
`0.2.19.dev30724030287` wheel was separately installed in an isolated environment and its
`finelog.deploy.config.INTRA_CLUSTER_CIDRS` import succeeded. The refreshed frozen macOS environment
then ran `uv run iris --help` successfully with that published build.

## Future work

- [ ] Tighten the lower bound in the published `marin-iris` package so downstream repositories do
  not need to restate this compatibility floor.
- [ ] Reconcile the OpenTelemetry bounds in Iris/pyqwest and Megatron Core, then replace the remaining
  platform override with ordinary dependency resolution.
