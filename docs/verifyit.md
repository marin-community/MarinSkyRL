# Unified verification

SkyRL clients call verifyit's existing verifier modes while retaining task-specific response extraction and framework reward reporting. The dependency is pinned to a published source commit in the project metadata. No local campaign checkout or unpublished wheel is needed. SkyRL uses math-verify 0.9.0, upgraded from 0.8.0 to satisfy the unified dependency. Math parsing or equivalence behavior can change with this upgrade; the 2026-10-01 campaign snapshot used math-verify 0.8.0. The offline comparisons use 0.9.0 on both paths.

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

For a frozen gym installation, run `uv sync --project skyrl-gym --frozen --extra dev`, then `uv run --project skyrl-gym --frozen python tools/verifyit/replay.py --output /tmp/skyrl-verifier-replay.json`. The gym and root locks include the exact published verifyit revision and math-verify 0.9.0.

The normal launcher installation uses the root project's CPU or GPU profile described in the README. The smaller installation above exercises verifiers without installing a training runtime. Code and Lean verification additionally require the configured sandbox runtime. Judge routes require their configured provider and credentials; they cannot be exercised through the offline fixtures.

The Gym CI job starts the real [NeMo Skills local sandbox](https://github.com/NVIDIA-NeMo/Skills/blob/bcf059af55c20a89f797724598f9908d126153e6/nemo_skills/code_execution/local_sandbox/local_sandbox_server.py) at revision `bcf059af55c20a89f797724598f9908d126153e6`, verifies its SHA256, and installs Flask 3.1.2, IPython 9.6.0, psutil 7.1.0, NumPy 2.2.6 and pandas 2.3.0 in a separate test environment. The service requires Linux resource limits; its setup, bounded health check and process-group cleanup are in [cpu_ci.yaml](../.github/workflows/cpu_ci.yaml). This supplies execution for code integration tests rather than substituting precomputed rewards.

## Enable verifyit

For environments that retain their original scorer, pass `verifyit_enabled: true` in the environment configuration:

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

Code and Lean use the [SandboxClient protocol](../skyrl-gym/skyrl_gym/envs/nemotron_ultra/sandbox.py): point the configured host/port to a running NeMo Skills sandbox with the benchmark’s Python dependencies or Lean project/toolchain. The acceptance configuration’s cluster hostname is an example deployment, not a public service. Judge settings are consumed by [OpenAIJudge](../skyrl-gym/skyrl_gym/envs/nemotron_ultra/judge.py); set the named environment variable locally with your provider credential before running judge routes. Direct source APIs expose `verifyit_enabled=True` where applicable; the checked-in replay demonstrates MCQA's switch. The remaining dormant STEM judge source API accepts `verifyit_enabled=True`. GeneralReactTask, coder1 and its exclusive math/QA verifiers were retired upstream.

GSM8K, AIME, MCQ, search exact match, ARC grid comparison and chemistry numeric comparison call the unified primitives directly. These clients do not have an original-path switch; compare them against the pinned source revision linked in the route inventory when investigating a difference.

Verification failures return minimum reward and retain framework verification/error information. A wrong candidate scoring zero is distinct from an invalid reference or unavailable verifier. Intentional corrections can change scores on malformed inputs; ordinary valid inputs should preserve source behavior.

## Coverage and limits

[The route inventory](../tools/verifyit/route-inventory.json) lists all 38 currently included routes and their original source locations. The 2026-10-01 campaign snapshot validated 30 routes with real traces and 15 with source fixtures at source revision `91c7a60`. Those historical counts do not establish current all-route parity; the replay command above produces fresh, local evidence for its representative routes.

Upstream revision `8b4b6924704432df896143e2785b30e5f944a441` retired coder1, GeneralReactTask and seven previously included math/QA routes, resolving [MarinSkyRL #880](https://github.com/marin-community/MarinSkyRL/issues/880). The current 38-route inventory retains the historical 45-route membership and records those seven retirements separately. The [retirement guide](coder1-retirement.md) describes supported migration paths.

Exact, numeric, schema, instruction, code, judge and retained source-runtime clients reuse existing verifyit modes. No new verifier template is introduced. Source-specific setup, external services and sandbox requirements remain part of each benchmark's contract.

## Archival agent packaging

The remaining dormant STEM judge integration changes its source scorer, without reviving standalone `skyrl-agent` packaging or training. Upstream [903c3a9](https://github.com/marin-community/MarinSkyRL/commit/903c3a9) repairs standalone dependency locking tracked by [#884](https://github.com/marin-community/MarinSkyRL/issues/884); this branch includes that change. Isolated frozen exports for the base, VERL and Tinker dependency profiles pass, while a full legacy training runtime has not been exercised here. The STEM judge requires its original provider configuration; its entrypoint and opt-in argument are recorded in the route inventory. The retired QA example is no longer available on current main.
