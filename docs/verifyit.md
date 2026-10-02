# Unified verification

SkyRL clients call verifyit's existing verifier modes while retaining task-specific response extraction and framework reward reporting. The dependency is pinned to published commit `c57d19b9f77c2d8fc2ec6f4919ac351e0b8e0c7e` in the project metadata. No local campaign checkout or unpublished wheel is needed. SkyRL uses math-verify 0.9.0, upgraded from 0.8.0 to satisfy the unified dependency. Math parsing or equivalence behavior can change with this upgrade; the 2026-10-01 campaign snapshot used math-verify 0.8.0. The offline comparisons use 0.9.0 on both paths.

## Install and reproduce

SkyRL Gym now requires Python >=3.11 (previously >=3.10), matching verifyit’s minimum. The root launcher remains Python 3.12. The commands below select Python 3.12.

From this checkout, with Git and [uv](https://docs.astral.sh/uv/) installed:

```bash
uv venv --python 3.12 .venv-verifiers
uv pip install --python .venv-verifiers/bin/python -e './skyrl-gym[dev]'
.venv-verifiers/bin/python tools/verifyit/replay.py --output /tmp/skyrl-verifier-replay.json
.venv-verifiers/bin/python -m pytest skyrl-gym/tests/test_verifyit_reasoning_mcqa.py
```

The replay uses checked-in response fixtures. It invokes both the original and cutover MCQA scorer and both original and cutover Reasoning Gym environments, and both math scoring entrypoints with a local HTTP judge fixture. The output retains each input and both results, including scored zero. A mismatch exits unsuccessfully. Math wrong-answer fallback receives the same fixed non-equivalence response from a local HTTP server on both paths; no model inference is involved. This is a scoring roundtrip, without model inference or a live judge. The fixtures are synthetic; they do not reproduce archived model-run scores.

For a frozen gym installation, run `uv sync --project skyrl-gym --frozen --extra dev --python 3.12`, then `uv run --project skyrl-gym --frozen python tools/verifyit/replay.py --output /tmp/skyrl-verifier-replay.json`. The gym and root locks include the exact published verifyit revision and math-verify 0.9.0.

The normal launcher installation uses the root project's CPU or GPU profile described in the README. The smaller installation above exercises verifiers without installing a training runtime. Code and Lean verification additionally require the configured sandbox runtime. Judge routes require their configured provider and credentials; they cannot be exercised through the offline fixtures.

The Gym CI job starts the real [NeMo Skills local sandbox](https://github.com/NVIDIA-NeMo/Skills/blob/bcf059af55c20a89f797724598f9908d126153e6/nemo_skills/code_execution/local_sandbox/local_sandbox_server.py) at revision `bcf059af55c20a89f797724598f9908d126153e6`, verifies its SHA256, and installs Flask 3.1.2, IPython 9.6.0, psutil 7.1.0, NumPy 2.2.6 and pandas 2.3.0 in a separate test environment. The service requires Linux resource limits; its setup, bounded health check and process-group cleanup are in [cpu_ci.yaml](../.github/workflows/cpu_ci.yaml). This supplies execution for code integration tests rather than substituting precomputed rewards.

## Enable verifyit

For MCQ, AIME, GSM8K and environments that retain their original scorer, pass `verifyit_enabled: true` in the environment configuration:

```python
import skyrl_gym
from omegaconf import OmegaConf

environment = skyrl_gym.make(
    "reasoning_gym",
    env_config=OmegaConf.create({"verifyit_enabled": True}),
    extras={"reward_model": {"ground_truth": {
        "task": "simple_equations",
        "entry": {"answer": "42", "metadata": {"source_dataset": "simple_equations"}},
    }}},
)
print(environment.step("Answer: 42"))
```

Set the option to `false` or omit it to run the original Reasoning Gym, IFEval, SQL, LiveCodeBench or Nemotron scorer. The [launcher acceptance configuration](../cloud/iris/configs/nemotron_ultra_rlvr_acceptance.yaml) shows the deployed sandbox host/port and judge `base_url`, `model`, and `api_key_env` settings. Set `environment.skyrl_gym.nemotron_ultra.verifyit_enabled: true` alongside those fields; the [trajectory runner](../skyrl-train/skyrl_train/trajectory_runners/skyrl_gym.py) passes each environment configuration to its constructor and propagates the option to GenRM. Other environments use `environment.skyrl_gym.<environment_name>.verifyit_enabled: true`.

Code and Lean use the [SandboxClient protocol](../skyrl-gym/skyrl_gym/envs/nemotron_ultra/sandbox.py): point the configured host/port to a running NeMo Skills sandbox with the benchmark’s Python dependencies or Lean project/toolchain. The acceptance configuration’s cluster hostname is an example deployment, not a public service. Judge settings are consumed by [OpenAIJudge](../skyrl-gym/skyrl_gym/envs/nemotron_ultra/judge.py); set the named environment variable locally with your provider credential before running judge routes. Direct source APIs expose `verifyit_enabled=True` where applicable; the checked-in replay demonstrates MCQA's switch.

MCQ, AIME and GSM8K also preserve their original scoring when the option is omitted or false. AIME retains its extraction and optimization reward policy; GSM8K retains its configured strict, flexible or final-line extraction. Enabled non-strict AIME canonicalizes finite exact constants under a bounded worker, then uses Exact comparison. Colon ratios are translated to fractions. Parsed names and expressions containing variables retain literal text comparison; this does not grant symbolic equivalence. Multiple answers, undefined references and parsing failures are rejected conservatively; strict-box mode retains literal comparison. Invalid references or worker failures produce an error verdict and AIME reward -1. The enabled paths send prepared candidates to verifyit. The following command tests correct and wrong responses on both paths, including package import and default scoring with verifyit unavailable:

```bash
uv run --project skyrl-gym --locked --extra dev python -m pytest skyrl-gym/tests/test_mcq.py skyrl-gym/tests/test_aime.py skyrl-gym/tests/test_gsm8k.py
```

LiveCodeBench and Nemotron code generation use verifyit for output comparison and combining test verdicts. Their adapters manage sandbox sessions and transform wire values; they do not calculate correctness or partial credit. Source-native grading remains the default. Configure the sandbox host/port and set `verifyit_enabled: true` to use the unified grading path. Nonzero Nemotron reasoning-format penalties remain environment reward shaping.

Verification failures return minimum reward and retain framework verification/error information. A wrong candidate scoring zero is distinct from an invalid reference or unavailable verifier. Intentional corrections can change scores on malformed inputs; ordinary valid inputs should preserve source behavior.

## Coverage and limits

[The route inventory](../tools/verifyit/route-inventory.json) lists the 37 routes in the maintained packages and their original source locations. The replay command above produces fresh, local evidence for representative routes; it does not establish all-route parity.

Exact, numeric, schema, instruction, code and judge clients reuse existing verifyit modes. No new verifier template is introduced. Source-specific setup, external services and sandbox requirements remain part of each benchmark's contract.
