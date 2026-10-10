# Unified verification

SkyRL task sessions use [Verifyit](https://github.com/marin-community/marin/tree/main/lib/verifyit),
Marin's shared verifier package, and retain source-specific response extraction and reward policy.
The `marin-verifyit` PyPI release is selected by the root and standalone lockfiles.
SkyRL pins math-verify for reproducible answer parsing.

## Install and reproduce

The task session package requires Python 3.12. From the repository root:

```bash
uv sync --project skyrl-gym --frozen --extra dev
uv run --project skyrl-gym --frozen python tools/verifyit/replay.py \
  --output /tmp/skyrl-verifier-replay.json
uv run --project skyrl-gym --frozen pytest skyrl-gym/tests/
```

The replay compares the native and verifyit paths for MCQA, Reasoning Gym, and math.
It uses fixed candidate responses and a local HTTP judge.
It retains each input and result, including a scored zero.
A mismatch causes a nonzero exit code.
This evidence does not establish all-route parity or reproduce archived model scores.

CPU runtime fixtures execute trusted programs through local subprocesses.
They do not establish container isolation.
Python execution uses a stateful interpreter; SQL uses a read-only database snapshot.
Lean boundary tests replace the external compiler process and retain the real command path.
A real Lean compiler and task image require separate validation.

## Select the verifier

Set `verifyit_enabled: true` in the task configuration:

```yaml
environment:
  task_sessions:
    reasoning_gym:
      verifyit_enabled: true
    nemotron_ultra:
      verifyit_enabled: true
```

The source importer stores configuration and reference inputs in the private verifier payload.
The rollout engine selects a direct TaskSession factory from the task's named interaction.
There is no Gym environment or registry.

The flag defaults to false. Reasoning Gym, IFEval, and SQL retain native/verifyit selection.
Nemotron retains the switch for math, Python-tool answer grading, tool calls, calendar,
format checks, MCQA, structured outputs, instructions, judge profiles, and reasoning tasks.
Code, ARC, chemistry, and Lean use shared execution or grading paths without that switch.
Code execution uses verifyit's exact output comparator.
GSM8K, AIME, MCQ, and search use shared primitives directly.

GenRM remains group grading in the trainer.
Dataset preparation stores its private group-grader specification in trainer metadata.
The [group grader](../skyrl-train/skyrl_train/rollouts/genrm_grading.py) reads `verifyit_enabled` from those parameters.

## Execution and failures

Python, SQL, and Lean sessions use the Shellbox machine in TaskSpec.
The [task configuration](../skyrl-train/skyrl_train/config/task_session_config/default.yaml)
declares the Python build context and per-agent machines.
Serialized build files use base64 content.
Lean requires a project with its toolchain and dependencies in the selected image.
The default Lean image reuses the repository's pinned NeMo runtime image and its `/lean4/my_project` project.
It does not start the NeMo HTTP service.

Judge routes require the configured endpoint, model, and credential.
See the [acceptance configuration](../cloud/iris/configs/nemotron_ultra_rlvr_acceptance.yaml)
and [judge client](../skyrl-gym/skyrl_gym/envs/nemotron_ultra/judge.py).
Keep judge credentials and private reference answers outside the candidate machine.

An incorrect candidate can receive a valid zero or negative grade.
A failed or unavailable verifier has no verdict and excludes the rollout from loss and group baselines.
An explicit skipped grade retains trainable model tokens.

[The route inventory](../tools/verifyit/route-inventory.json) records the 37 source routes and their pinned historical locations.
Its configuration strings refer to that revision. Use the task configuration above for current settings.
See [canonical task rollouts](../skyrl-train/docs/tutorials/task_rollouts.rst) for current execution and training policies.
