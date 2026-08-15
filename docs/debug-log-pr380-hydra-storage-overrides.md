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

The task shell now changes to the selected runtime checkout and puts that
checkout ahead of the bootstrap bundle on `PYTHONPATH`. A regression executes
the shell with a probe controller and verifies both the working directory and
source-path order. The launcher and runtime-bundle selection passed all 59
focused tests.

## Future work

- [ ] Rerun the Qwen3-0.6B GSM8K smoke test through checkpoint creation and
  synchronous terminal export.
