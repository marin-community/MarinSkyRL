# Debugging log for PR 380 Hydra storage overrides

Make lifecycle-managed Iris RL storage paths valid SkyRL Hydra overrides.

## Initial status

An Iris smoke test at MarinSkyRL commit `1a8edb55` used Qwen3-0.6B, 16 GSM8K
training rows, four validation rows, and one H100x8 node. The job installed the
frozen runtime, staged the dataset, started Ray, and translated the RL config.
Hydra then failed before the first optimizer step with `mismatched input '='
expecting <EOF>`.

The failing command contained launcher-generated overrides such as
`++trainer.ckpt_path=s3://marin-us-east-02a/tmp/ttl=14d/.../checkpoints`.
Hydra's override parser rejects that unquoted scalar because `=` is grammar
syntax. Quoting the URI or escaping it as `ttl\=14d` parses to the original URI.

## Hypothesis 1

The launcher passes lifecycle paths directly as Hydra scalar values. Encoding
string values with Hydra's quoted-string syntax will preserve the object-store
URI while making `ttl=14d` unambiguous to the parser.

## Changes to make

Add a behavior-level regression that builds the task command, extracts the
storage overrides, parses them with Hydra's real override parser, and checks
that Hydra resolves the original storage paths. Then apply one string-value
encoder to every launcher-owned storage override.

## Results

The regression failed at the direct launcher, typed launcher, and terminal
export boundaries with Hydra's `mismatched input '=' expecting <EOF>` error.
All three passed after reusing the config translator's Hydra string formatter
for launcher-owned storage overrides. The committed launcher, typed job,
context budget, export, and training-driver test selection passed all 95 tests.
Lint review then found that the context-budget artifact selector needed to
decode the quoted trials path; a regression now verifies that the artifact
still lands beside the lifecycle-managed traces.

## Smoke follow-up

The first follow-up smoke reached the GPU task with the quoted overrides, but
the task wrapper launched `python -m cloud.iris.training_driver` from `/app`.
That small bootstrap bundle preceded the immutable checkout at
`/app/marinskyrl`, so Python loaded an older `training_driver.py` that could not
import the checkout's `storage_policy` module.

The task shell keeps the controller and its imports in `/app` for
bundle-integrity validation, then starts only the training driver from the
selected checkout. Regressions verify the controller working directory,
bootstrap source-path order, and driver working directory. The focused launcher
and task-runtime selection passed all 74 tests.

## Hypothesis 2

The successful trainer process remained inside `ray.shutdown()` for more than
five minutes because it was trying to disconnect from the Ray cluster owned by
the surrounding Iris task runtime. That prevented the synchronous launcher from
observing training success and submitting its required terminal export.

## Changes to make

Mark the Ray cluster as Iris-owned in the training-driver environment. Attached
SkyRL entrypoints leave that cluster connected and unregister Ray's redundant
process-exit shutdown hook; the Iris task runtime remains the single owner of
bounded `ray stop --force` teardown.

## Results

The ownership and local-cluster fallback regressions pass. The next live smoke
returned from the training loop immediately, but Python then entered Ray's C++
global destructors and aborted with `corrupted size vs. prev_size`. The attached
driver now closes process telemetry and flushes logs before exiting without
running those process-global destructors; the task runtime remains responsible
for stopping the cluster.

The following smoke reached that clean exit, then spent more than five minutes
in the task runtime's best-effort final Ray log upload. Final diagnostic uploads
now have a 60-second bound and may finish partially; the periodic uploader still
preserves session material throughout the run, while diagnostics can no longer
block cluster teardown and terminal export indefinitely.

The first run with that bound remained inside the SkyRL process rather than
reaching task-runtime teardown. Loguru's queue-drain helper was waiting on the
large Ray/NCCL log backlog immediately before the process-only exit. The exit
path was changed to emit its handoff message, flush the standard streams
directly, and exit without the redundant queue drain.

Ray also replaces the driver standard streams, and their explicit `flush()`
calls blocked before the process exit in the next smoke. The handoff log was
already visible remotely, so the external-owner path now invokes `os._exit(0)`
immediately after it, without touching Ray's wrapped streams.

## Future work

- [ ] Rerun the Qwen3-0.6B GSM8K smoke test through checkpoint creation and
  synchronous terminal export.
