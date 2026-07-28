# Debugging log for nightly GSM8K H100 run 30354057133

Diagnose the scheduled run, separate Iris failures from trainer regressions, and make the
nightly exercise a reproducible GPU runtime.

## Initial status

GitHub marked the `gsm8k-h100` job cancelled after its 90-minute limit. The failure reporter
captured no failed-step logs.

## Hypothesis 1

The known cw-rno2a federation admission or log-forwarding fault prevented the task from
starting.

## Changes to make

None. Inspect the Actions job log and compare it with the preceding scheduled run.

## Results

Refuted as the only cause. cw-rno2a admitted the handoff, but the log stream immediately
replayed the preceding run's task output: the embedded timestamps, Ray session path, PID,
and `libcudart.so.13` traceback all came from 2026-07-27. No 2026-07-28 task output appeared,
and Iris never reported a terminal state before GitHub cancelled the step. Reusing the fixed
job name made consecutive runs share stale task/log state.

## Hypothesis 2

The replayed `libcudart.so.13` traceback is an independent trainer-runtime regression that
will recur once a uniquely named job starts.

## Changes to make

Compare the run's frozen dependency closure with the validated GPU-RL image contract.

## Results

Confirmed. The failed run installed stock `vllm==0.23.0` from PyPI into a fresh environment
with torch 2.11.0+cu128. The vLLM extension requires `libcudart.so.13`, while the lock
contains CUDA 12 runtime libraries. The production GPU-RL image replaces that stock wheel
with the Marin vLLM fork and validated native objects compiled against the image's torch/CUDA
closure. Pulling the digest-pinned production image and using its environment avoids the
unsupported mixed ABI.

## Hypothesis 3

The workflow can use the deployed GPU-RL environment while still exercising the checked-out
trainer and committed Grug parity fixture.

## Changes to make

- Derive the task image from `DEFAULT_RL_DOCKER_IMAGE`, the repository's existing source of
  truth.
- Run with `/opt/openthoughts/envs/rl/bin/python` and put the checked-out trainer and gym
  sources first on `PYTHONPATH`.
- Make the Grug parity contract directly executable without installing pytest into the
  production runtime.
- Give each Actions attempt a unique Iris job name and leave headroom between the Iris and
  GitHub timeouts.

## Results

Implemented. The workflow resolves the current production image digest through
`DEFAULT_RL_DOCKER_IMAGE`, gives each Actions attempt a unique Iris name, and leaves ten
minutes between the 90-minute Iris limit and the Actions limit. The task imports torch and
vLLM before training, runs the checked-out source on the image environment, and executes the
same Grug parity function used by pytest.

Local validation passed for the shell syntax, Ruff, the image-digest lookup, and all 17
nightly gate tests. The H100 parity test collected and skipped on the CPU host. The Marin
lint-review command could not start its reviewer lanes because the reviewer account had
reached its monthly spend limit.

The full branch workflow
[30398993649](https://github.com/marin-community/MarinSkyRL/actions/runs/30398993649)
then passed on cw-rno2a. It completed all 30 training steps in 717 seconds, the metric gate
accepted the final step, and the H100 Grug parity check matched the committed fixture. Iris
reported one succeeded task with exit 0, zero failures, and zero preemptions.

## Future work

- [x] Confirm the fixed workflow on cw-rno2a with a full manual dispatch.
- [ ] Fix Iris federation so a reused canonical job path cannot replay a previous task's logs
      or wait forever after that task is already terminal (marin-community/marin#7654).
