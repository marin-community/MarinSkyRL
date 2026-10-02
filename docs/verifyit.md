# Unified verification

SkyRL clients call verifyit's existing verifier modes while retaining task-specific response extraction and framework reward reporting. The dependency is pinned to published commit `b08a5ee8d94fcb4a2134562aee94ff9715dadd14` in the project metadata. No local campaign checkout or unpublished wheel is needed. SkyRL uses math-verify 0.9.0, upgraded from 0.8.0 to satisfy the unified dependency. Math parsing or equivalence behavior can change with this upgrade; the 2026-10-01 campaign snapshot used math-verify 0.8.0. The offline comparisons use 0.9.0 on both paths.

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

Enabled GenRM grades completed comparison cohorts through core Judge and Schema inside a bounded Script worker. At least two responses are required. Verification rewards use [0, 1]; native optimization rewards retain the source's adjusted rating scale (base bounds −1.5–7.5) and length bonuses/penalties. The original source declares 1–5 bounds even though tie adjustments and shaping can exceed them. Malformed provider cohorts become error verdicts with optimization reward zero before shaping. Invalid trusted cohort data is reported separately as `invalid_task`; the enabled boundary conservatively masks all affected GenRM rows in that batch, including any earlier graded group, while preserving unrelated environments. Provider protocol and trusted-task failures never receive the source's default score of three. The archived environment traces contain pending-cohort placeholders, so the local controlled HTTP tests establish post-cohort behavior without claiming final archived judge-score parity.

The following offline tests serve controlled judge responses over local HTTP, compare original and enabled cohort rewards, and check malformed provider and child-task failures without credentials:

```bash
uv run --project skyrl-gym --locked --extra dev python -m pytest skyrl-gym/tests/test_genrm_verifyit.py
```

Code and source-native Lean use the [SandboxClient protocol](../skyrl-gym/skyrl_gym/envs/nemotron_ultra/sandbox.py): point the configured host/port to a running NeMo Skills sandbox with the benchmark’s Python dependencies or Lean project/toolchain. The acceptance configuration’s cluster hostname is an example deployment, not a public service. Judge settings are consumed by [OpenAIJudge](../skyrl-gym/skyrl_gym/envs/nemotron_ultra/judge.py); set the named environment variable locally with your provider credential before running judge routes. Direct source APIs expose `verifyit_enabled=True` where applicable; the checked-in replay demonstrates MCQA's switch.

MCQ, AIME and GSM8K also preserve their original scoring when the option is omitted or false. AIME retains its extraction and optimization reward policy; GSM8K retains its configured strict, flexible or final-line extraction. Enabled non-strict AIME canonicalizes finite exact constants under a bounded worker, then uses Exact comparison. Colon ratios are translated to fractions. Parsed names and expressions containing variables retain literal text comparison; this does not grant symbolic equivalence. Multiple answers, undefined references and parsing failures are rejected conservatively; strict-box mode retains literal comparison. Invalid references or worker failures produce an error verdict and AIME reward -1. The enabled paths send prepared candidates to verifyit. The following command tests correct and wrong responses on both paths, including package import and default scoring with verifyit unavailable:

```bash
uv run --project skyrl-gym --locked --extra dev python -m pytest skyrl-gym/tests/test_mcq.py skyrl-gym/tests/test_aime.py skyrl-gym/tests/test_gsm8k.py
```

LiveCodeBench and Nemotron code generation use verifyit for output comparison and combining test verdicts. Their adapters manage sandbox sessions and transform wire values; they do not calculate correctness or partial credit. Source-native grading remains the default. Configure the sandbox host/port and set `verifyit_enabled: true` to use the unified grading path. Nonzero Nemotron reasoning-format penalties remain environment reward shaping.

Verification failures return minimum reward and retain framework verification/error information. A wrong candidate scoring zero is distinct from an invalid reference or unavailable verifier. Intentional corrections can change scores on malformed inputs; ordinary valid inputs should preserve source behavior.

## Coverage and limits

[The route inventory](../tools/verifyit/route-inventory.json) lists the 37 routes in the maintained packages and their original source locations. The replay command above produces fresh, local evidence for representative routes; it does not establish all-route parity.

Exact, numeric, schema, instruction, code and judge clients reuse existing verifyit modes. No new verifier template is introduced. Source-specific setup, external services and sandbox requirements remain part of each benchmark's contract.


### Audited Lean cutover

The opt-in `math_formal_lean_refinement_agent` requires the audited runtime shipped in
[tools/lean_runtime](../skyrl-gym/tools/lean_runtime). An ordinary sandbox completion flag cannot
establish that a proof matches the task. From this checkout:

```bash
docker build --platform linux/amd64 -t skyrl-lean-audit skyrl-gym/tools/lean_runtime
docker run --rm --init --name skyrl-lean-audit --cpus 1 --memory 4g --pids-limit 128 \
  -p 127.0.0.1:6000:6000 skyrl-lean-audit
```

Set `environment.skyrl_gym.nemotron_ultra.verifyit_enabled: true` and its sandbox host/port to
`127.0.0.1:6000` when the trainer runs on the same machine. Stop the service with
`docker stop skyrl-lean-audit`. Keep the original path on its original sandbox; the audited
service rejects unaudited requests. It handles requests serially, with one 30-second deadline
for task compilation, candidate compilation and inspection. Candidate and task Lean run under
separate unprivileged users. HTTP access must remain restricted to trusted callers.

The runtime pins Lean 4.12.0 and Mathlib `809c3fb3b5c8f5d7dace56e200b426187516535a`.
The archived producer's historical toolchain revision is unknown; these pins implement the
published Nemotron profile. The 4 GiB runtime limit is required for the Mathlib audit; it does
not describe the historical producer's resources.

The immutable task statement determines the expected theorem type. A separate inspector reads
only the candidate module's declarations, rechecks them with Lean's kernel against trusted
imports, and checks a witness against that expected type. It rejects requested-name collisions
with trusted declarations. Ordinary `propext`, `Classical.choice` and `Quot.sound` dependencies
are allowed; `sorryAx` and candidate-added axioms are rejected. Candidate stdout cannot supply
audit evidence. Existing verifyit JSON Schema primitives and the shared ALL reducer own the
score; the runtime only supplies compiler and inspection results. A compiler rejection scores
zero and retains correction feedback; missing or truncated audit evidence reports an error
with minimum optimization reward. A trusted declaration that cannot compile under the pinned
libraries is an invalid task; unavailable tooling and audit timeouts remain infrastructure errors.
Unsupported candidate declaration kinds fail closed.

Reasoning Gym cutovers validate the original serialized trusted record before normalization. Duplicate JSON keys and nonfinite values produce minimum-reward error verdicts even for blank candidates. Omitting `verifyit_enabled` preserves source parsing and grading. Both the `reasoning_gym` environment and Nemotron `reasoning_gym_simple_agent` delegate scores to verifyit's existing ReasoningGym mode; dataset scoring uses reasoning-gym 0.1.25.

Search and SearchCode keep source grading when the option is omitted. Enabled Search preserves last-answer-tag extraction, punctuation/article/whitespace normalization and exact alternative matching through Schema, Exact and shared reducers. Enabled SearchCode forwards its final history to the existing numeric-answer verifier. Malformed trusted references produce minimum-reward errors; tool calls and retrieval remain framework operations. The final-step contracts can be exercised offline with `uv run --project skyrl-gym --frozen python -m pytest skyrl-gym/tests/test_verifyit_search.py`.

### Indirect prompt injection

The opt-in `indirect_prompt_injection_simple_agent` uses the original NeMo Gym
resource server pinned to `7a19900a114f8c349c9fac031b016575e39cfa36`.
Configure `verifyit_enabled: true` and
`ipi_resources_url: http://127.0.0.1:18765`. The server requires Python >=3.13.14;
SkyRL communicates over HTTP and retains its existing Python requirement. From
the pinned NeMo checkout, start the original resource app:

```bash
uv sync --frozen --no-dev --python 3.13
uv run --no-sync python - <<'PY'
import uvicorn
from omegaconf import OmegaConf
from nemo_gym.config_types import BaseServerConfig
from nemo_gym.server_utils import ServerClient
from resources_servers.indirect_prompt_injection.app import IPIResourcesServer, IPIResourcesServerConfig
server = IPIResourcesServer(
    config=IPIResourcesServerConfig(host="127.0.0.1", port=18765, entrypoint="app.py", name="ipi"),
    server_client=ServerClient(head_server_config=BaseServerConfig(host="127.0.0.1", port=18766), global_config_dict=OmegaConf.create({})),
)
uvicorn.run(server.setup_webserver(), host="127.0.0.1", port=18765)
PY
```

Rollouts must supply structured assistant tool calls and completion reasons.
The client seeds isolated cookie sessions and forwards declared tools to the
original service. Schema owns required-tool, attacker-discriminator and
truncation checks. Unknown verification types use all attacker argument keys, matching the source
fallback. Nonfinite, null or container-valued trusted discriminators are invalid tasks. Ordinary tool
arguments preserve source string normalization and null handling. Each session
has a 30-second grading deadline and a separate five-second disposal request.
NeMo's `/verify` is the only endpoint that actually removes IPI state, so cleanup
invokes its native computation but ignores its reward. Cleanup errors remain
visible and cannot produce credit. Without the option, the route retains its
original unimplemented behavior.


Tool-comparison routes, including the SWE pivot alias, snapshot their input records before
preparing schema and numeric contracts. With `verifyit_enabled: true`,
`verifyit_tool_comparison_policy: nemotron_strict_typed_arguments_v1` names the established
strict opt-in policy: duplicate JSON keys fail closed, integer and floating types remain
distinct, and absent tool calls become an empty list while other falsey values retain their
types. Verifyit's Schema/Numeric modes and ALL reducer determine correctness. Framework
metadata records the effective policy and SHA-256 hashes of the input records; trusted
reference contents are not newly exposed in diagnostics. Errors retain policy and stage
provenance. Unrepresentable numeric candidates and excessively nested candidate JSON
score zero; malformed trusted action types and nested trusted JSON remain invalid tasks. Omitting the opt-in still uses
the original source scorer. The policy retains the documented stricter behavior for
boolean/integer confusion and malformed JSON; it is not a source-parity claim for those inputs.
