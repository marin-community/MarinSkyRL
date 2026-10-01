# Unified verification

SkyRL clients call verifyit's existing verifier modes while retaining task-specific response extraction and framework reward reporting. The dependency is pinned to a published source commit in the project metadata. No local campaign checkout or unpublished wheel is needed. This branch upgrades SkyRL’s math-verify pin from 0.8.0 to 0.9.0 to satisfy the unified dependency. Math parsing or equivalence behavior can change with this upgrade; the earlier campaign results used the previous source pin. Fresh original/cutover comparisons here use 0.9.0 on both paths.

## Install and reproduce

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

## Enable a cutover

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

Code and Lean use the [SandboxClient protocol](../skyrl-gym/skyrl_gym/envs/nemotron_ultra/sandbox.py): point the configured host/port to a running NeMo Skills sandbox with the benchmark’s Python dependencies or Lean project/toolchain. The acceptance configuration’s cluster hostname is an example deployment, not a public service. Judge settings are consumed by [OpenAIJudge](../skyrl-gym/skyrl_gym/envs/nemotron_ultra/judge.py); set the named environment variable locally with your provider credential before running judge routes. Direct source APIs expose `verifyit_enabled=True` where applicable; the checked-in replay demonstrates MCQA's switch. Dormant `GeneralReactTask` math routes accept the option in the instance; coder1 continues using its original implementation.

GSM8K, AIME, MCQ, search exact match, ARC grid comparison and chemistry numeric comparison call the unified primitives directly. These clients do not have an original-path switch; compare them against the pinned source revision linked in the route inventory when investigating a difference.

Verification failures return minimum reward and retain framework verification/error information. A wrong candidate scoring zero is distinct from an invalid reference or unavailable verifier. Intentional corrections can change scores on malformed inputs; ordinary valid inputs should preserve source behavior.

## Coverage and limits

[The route inventory](../tools/verifyit/route-inventory.json) lists all 45 included routes and their original source locations. The preceding campaign validated 30 routes with real traces and 15 with source fixtures. Those historical counts do not imply that all 45 have been rerun on this latest-main branch; the replay command above produces fresh, local evidence for its representative routes.

The 46th inventoried route, coder1, is excluded pending deprecation. Its arbitrary same-interpreter Python test contract does not cleanly translate to isolated execution. [MarinSkyRL #880](https://github.com/marin-community/MarinSkyRL/issues/880) tracks deprecation. This change preserves its existing source behavior and omits its partial adapter.

Exact, numeric, schema, instruction, code, judge and retained source-runtime clients reuse existing verifyit modes. No new verifier template is introduced. Source-specific setup, external services and sandbox requirements remain part of each benchmark's contract.

## Dormant verifier source APIs

The legacy standalone `skyrl-agent` packaging/runtime is not revived by this change. [MarinSkyRL #884](https://github.com/marin-community/MarinSkyRL/issues/884) tracks its preexisting broken local trainer dependency. Its archival manifest and lock remain unchanged. Use the supported root/gym environment and load the source scorer directly when assessing a dormant route. For example, the QA scorer imports LiteLLM even for exact matching:

```bash
uv pip install --python .venv-verifiers/bin/python litellm
.venv-verifiers/bin/python - <<'PYTHON'
import importlib.util
from pathlib import Path
path = Path("skyrl-agent/skyrl_agent/tasks/verifiers/qa.py")
spec = importlib.util.spec_from_file_location("dormant_qa", path)
qa = importlib.util.module_from_spec(spec)
spec.loader.exec_module(qa)
record = {"target": "Paris"}
print(qa.compute_score_em("Paris", record))
print(qa.compute_score_em("Paris", record, verifyit_enabled=True))
PYTHON
```

This loads the actual checked-in scorer without starting the archival agent harness. Its judge variants additionally need the original provider configuration. The source inventory links each dormant entrypoint; historical fixture evidence for these APIs is distinct from a supported legacy training runtime.
