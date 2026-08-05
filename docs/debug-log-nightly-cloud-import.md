# Debugging log for nightly cloud import

Restore the scheduled `gsm8k-h100` run after GitHub Actions run
`31000958888` failed before training.

## Initial status

The Iris job reached `cw-rno2a`, acquired one H100, loaded the pinned GPU-RL
environment, and prepared the GSM8K dataset. Importing
`skyrl_train.entrypoints.main_base` then failed with
`ModuleNotFoundError: No module named 'cloud.iris.env_vars'`.

The first failing scheduled run used `c078f48`, which introduced
`skyrl_train.env_vars` as a forwarding module for `cloud.iris.env_vars`. The
previous scheduled run at `82443a2` passed.

## Hypothesis 1

The nightly source overlay omits the checkout root from `PYTHONPATH`. The job
changes into `skyrl-train/` and exposes that directory plus its copied
`skyrl-gym/` package. `skyrl_train` therefore resolves from the checkout, but
the new sibling `cloud/` package does not. The Iris workspace bundle contains
`cloud/iris/env_vars.py`, so bundle trimming is not the source of the missing
module.

## Changes to make

Reproduce the import with the nightly's current source path, then repeat it with
the checkout root included. Do not change production code for this step.

## Results

With the process working directory set to `skyrl-train/`, an isolated Python
process reproduced the failure when `PYTHONPATH` contained only
`skyrl-train/skyrl-gym` and `skyrl-train`. Adding the checkout root made the
same import succeed and resolve `EnvVarScope.DRIVER`.

The hypothesis is confirmed. Add a regression test around the executable
nightly script before changing its source setup.

## Hypothesis 2

Adding the checkout root to the nightly's `PYTHONPATH` will expose the shared
launcher modules without changing the pinned GPU-RL environment or the copied
gym package.

## Changes to make

Run the nightly shell script against fake GPU and Python process boundaries.
At the trainer launch boundary, import `skyrl_train.env_vars` with Python site
packages disabled. Add the checkout root to the source paths and repeat the
test.

## Results

Before the production change, the regression test failed at the trainer launch
boundary with `ModuleNotFoundError: No module named 'cloud'`. After adding the
checkout root to `PYTHONPATH`, the same test completed the trainer import and
the nightly script exited successfully.

## Future work

- [x] Make the nightly source overlay include the checkout root.
