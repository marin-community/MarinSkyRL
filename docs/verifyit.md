# Unified verification

Set `verifyit_enabled: true` to use verifyit's existing modes. Omit the option or set it to
`false` to retain native scoring. SkyRL owns tool interactions, sandbox services, source
response preparation and framework reward reporting. Dependency revisions are pinned in
[the root manifest](../pyproject.toml), [the Gym manifest](../skyrl-gym/pyproject.toml) and their
lockfiles; no campaign checkout is required.

## Install and enable

Gym requires Python >=3.11; the root launcher uses Python 3.12. For a frozen Gym install:

```bash
uv sync --project skyrl-gym --frozen --extra dev --python 3.12
uv run --project skyrl-gym --frozen python tools/verifyit/replay.py --output /tmp/skyrl-verifier-replay.json
```

The replay uses synthetic local fixtures and a fixed HTTP judge. It checks representative
source/enabled outcomes without model inference; it does not establish all-route parity.
For launcher and training dependencies, follow the [README](../README.md).

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

Training uses `environment.skyrl_gym.<environment_name>.verifyit_enabled: true`.
For Nemotron, set `environment.skyrl_gym.nemotron_ultra.verifyit_enabled: true`; the
trajectory runner also forwards it to GenRM. The [launcher configuration](../cloud/iris/configs/nemotron_ultra_rlvr_acceptance.yaml)
shows sandbox host/port and judge `base_url`, `model`, and `api_key_env` fields. Set the
credential environment variable locally. Its cluster hostname requires your deployment.

Invalid references and infrastructure failures receive minimum reward with an error
status. A wrong candidate scoring zero remains a completed verdict. Public diagnostics
retain policy names and input hashes; protected receipts may contain references and
provider prose. Source-native diagnostics remain unchanged. Math parsing uses
math-verify 0.9.0; the dependency upgrade from 0.8.0 can change equivalence behavior.

## Preparation and limits

- MCQ uses `skyrl_mcq_first_box_v1`; Ultra MCQA uses `skyrl_ultra_mcqa_source_v1`.
  Invalid options, modes, references and regexes are task errors, including for blank
  candidates; regex timeouts are infrastructure errors. This fixes native malformed-regex
  fallback. Their worker budget is ten seconds.
- AIME/GSM8K use `verifyit_timeout` (default ten seconds). AIME retains its source answer
  window and strict-box or Minerva policy. Non-strict AIME supports finite constants,
  colon ratios and flat numeric tuples; tuple order and spelling remain significant.
  Nested/singleton tuples and unsupported parsed forms are invalid references. Strict-box
  mode distinguishes a missing box from an empty box. GSM8K retains first-marker,
  last-number and completed final-line policies; multi-turn format bonuses remain shaping.
- Reasoning Gym and IFEval use existing modes. Reasoning Gym rejects duplicate JSON keys
  and nonfinite trusted records before normalization. Search retains last-answer-tag and
  QA normalization; SearchCode forwards final history to its numeric-answer verifier.
- Structured outputs use `nemotron_structured_output_source_v1` for source JSON/XML/CSV
  parsing and schema-directed coercion. Invalid schemas are task errors even when the
  candidate cannot be prepared. Tool comparison uses `nemotron_strict_typed_arguments_v1`:
  integer/float/boolean distinctions are preserved, duplicate keys fail closed and
  malformed candidate JSON scores zero. These strict cases can differ from native scores.
- Seeded SQL uses `skyrl_seeded_round6_multiset_columns_readonly_finite_100000_reference_first_v1`:
  six-decimal rounding, numeric int/float equivalence, duplicate rows, column count and
  optional order. Original and perturbed databases must match. WITHOUT ROWID tasks are
  unsupported. Legacy SQL uses `skyrl_legacy_set_numeric_readonly_finite_100000_reference_first_v1`;
  empty results ignore column count and framework scores remain -1/0/1. Both policies
  admit all references first, require read-only queries and cap finite results at 100000
  rows. Final grading has a 90-second worker budget; interactive tools use five seconds
  per call. Native episode limits remain separate.

## Judge and code services

Judge profiles use `verifyit_judge_total_timeout_seconds` (default 120).
`verifyit_judge_profile_policies` defaults to `response: nemotron_final_answer_v1`,
`abstention: nemotron_articles_punctuation_case_v1`, `rubric: nemotron_yes_unless_no_v1`,
`labels: source_alias_lines_no_contradictions_v1`, and `composition: source_v1`.
Alternatives are `literal_v1` for response/abstention, `binary_only_v1` for rubric,
`bracketed_only_no_contradictions_v1` for labels, and `mean_v1`/`product_v1` for composition.
Unknown controls are invalid tasks. Contradictory completed labels fail as infrastructure
errors; native parsing considers only the final line. Jailbreak fractional rewards are
preserved. Math/judge uses `nemotron_math_judge_source_v1`: source extraction and reference
admission precede core Math, then symmetric Judge when needed. A first negative judge
skips the second request while retaining both components in the denominator.

GenRM requires at least two responses and uses core paired Judge scores. Its
`verifyit_timeout_seconds` defaults to 120; `verifyit_score_json_policy` selects
`strict_single_object_v1` or `source_last_object_v1`, and `verifyit_peer_policy` selects
`source_valid_peers_v1` or `require_all_peers_v1`. Length bonuses remain trainer shaping.

Code and native Lean require a [SandboxClient](../skyrl-gym/skyrl_gym/envs/nemotron_ultra/sandbox.py)
service with the task's Python dependencies or Lean toolchain. Code uses `lcb_source_v1`
for source test normalization and fenced-code extraction; core primitives compare outputs
and combine test verdicts. `code_verifier.total_timeout_seconds` must be finite and positive
(default 300). Queueing and parent serialization consume that budget, though serialization
is not interruptible. Parent session cleanup uses a separate ten-second HTTP timeout;
this is not a strict end-to-end wall-clock bound. Cleanup failure prevents credit.

## Audited Lean service

The enabled `math_formal_lean_refinement_agent` requires [this runtime](../skyrl-gym/tools/lean_runtime):

```bash
docker build --platform linux/amd64 -t skyrl-lean-audit skyrl-gym/tools/lean_runtime
docker run --rm --init --name skyrl-lean-audit --cpus 1 --memory 4g --pids-limit 128 -p 127.0.0.1:6000:6000 skyrl-lean-audit
```

Configure sandbox host/port `127.0.0.1:6000` for a local trainer; stop with
`docker stop skyrl-lean-audit`. Restrict HTTP access to trusted callers. The runtime pins
Lean 4.12.0 and Mathlib `809c3fb3b5c8f5d7dace56e200b426187516535a`, and handles requests
serially with a 30-second deadline. Separate unprivileged users compile task/candidate
modules. Kernel inspection checks the immutable theorem type and rejects `sorryAx`, new
axioms and trusted-name collisions; `propext`, `Classical.choice` and `Quot.sound` are allowed.
Compiler rejection scores zero; missing audit evidence and unavailable tools fail closed.

## Indirect prompt injection service

`indirect_prompt_injection_simple_agent` requires NeMo Gym resource revision
`7a19900a114f8c349c9fac031b016575e39cfa36`, Python >=3.13.14, and
`ipi_resources_url: http://127.0.0.1:18765`. From that checkout:

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

Supply structured assistant tool calls and completion reasons. Each isolated session has
30 seconds for grading and a separate five-second disposal request. Cleanup calls NeMo's
`/verify` only to remove state and ignores its reward. Cleanup errors prevent credit.
Nonfinite/null/container trusted discriminators are invalid tasks. Unknown verification
types use the source fallback of all attacker keys. The disabled route remains unimplemented.
