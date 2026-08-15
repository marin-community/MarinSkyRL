# Trajectory runner migration

## Decision

Replace the trainer-facing generator interface with a trajectory-runner interface. A trajectory runner owns the
interaction needed to acquire a rollout from a task. It does not represent the inference engine, verifier, reward
policy, or training objective.

Remove the Verifiers integration. No Iris configuration, launcher, recorded Marin run, or maintained test selects
its entrypoint. Its standalone shell script is the only checked-in caller, and its GPU tests are disabled because
the integration no longer imports against Verifiers main.

This change preserves rollout tokens, rewards, masks, metrics, training behavior, and the Hydra `generator.*`
configuration namespace. The namespace currently combines inference topology, sampling, shaping, and retention
across the launcher and trainer. Splitting those settings belongs with the typed verifier and reward contracts in
the next change.

## Current boundary

`GeneratorInterface` accepts a batch of prompts and task metadata, drives a harness or environment, and returns a
normalized token-level batch to the trainer. Concrete implementations also select model transport and output
granularity through inheritance:

- `SkyRLGymGenerator` performs direct inference-engine calls and emits whole trajectories.
- `StepWiseGenerator` repeats the SkyRL-Gym loop to emit one training sample per environment step.
- `SkyRLGymHTTPGenerator` repeats the SkyRL-Gym loop over an OpenAI-compatible endpoint.
- `TerminalBenchGenerator` drives Harbor and translates persisted trials.
- `MiniSweAgentGenerator` drives Mini-SWE-Agent and translates its transcript.
- `VerifiersGenerator` loads an external Verifiers environment that performs generation and grading.

Terminal Bench, Mini-SWE-Agent, and the fully asynchronous HTTP runner live under unrelated example or integration
directories. The Iris launcher names the Terminal Bench example module directly, so an example path is part of the
production launch contract.

## Target boundary

`TrajectoryRunner` is the trainer-facing protocol:

```python
class TrajectoryRunner(ABC):
    async def run(self, request: TrajectoryRequestBatch) -> TrajectoryBatch: ...
    async def startup(self) -> None: ...
    async def shutdown(self) -> None: ...
```

`TrajectoryRequestBatch` contains prompts, task metadata, sampling parameters, trajectory identities, and training
phase. `TrajectoryBatch` retains the current normalized fields until the verifier-contract change replaces its
parallel optional lists with structured records.

The runner owns:

- harness or environment lifecycle;
- model requests needed by that harness;
- transcript acquisition;
- conversion of the transcript into token-aligned trajectories; and
- propagation of harness failures without losing evidence.

The runner does not own the semantics of correctness, optimization reward, or training disposition. Existing code
that performs those operations remains behavior-identical in this migration and moves behind typed protocols in
the next change.

## Composition points

### Model client

Introduce a model-client protocol used by SkyRL-Gym. The direct client delegates to `InferenceEngineClient`; the
HTTP client uses the OpenAI-compatible endpoint. Both return the existing inference-engine response record. The
HTTP client records whether token IDs came from the server or were reconstructed by the runner.

Model transport is selected when constructing a runner. It is no longer represented by a SkyRL-Gym subclass.
Other runners may reuse the client without inheriting SkyRL-Gym behavior.

### Sample projection

Introduce a sample-projection protocol with whole-trajectory and step-wise implementations. The SkyRL-Gym runner
composes a rollout collector with a projection. The default whole-trajectory collector returns one completed agent
loop per trajectory; the step-wise collector returns one structured record per environment transition. Their paired
projections convert those records into the normalized batch expected by the trainer. These reusable pieces live in
`trajectory_runners/collectors.py` and `trajectory_runners/projections.py`, with runner-specific adapters beside the
runner that owns them.

Projection is selected when constructing a runner. Step-wise output is no longer represented by a runner subclass.
Future harnesses can emit step-wise training samples by supplying the same step records to the projection.

### Shared finalization

`TrajectoryRunner.run()` retains the existing shared finalization order:

1. run the harness;
2. apply runner-independent trajectory reward shaping;
3. validate and report token/logprob alignment; and
4. retain the normalized trajectory sample.

The initial migration changes names and ownership without changing finalization behavior.

## Package layout

Runtime implementations move under `skyrl_train.trajectory_runners`:

```text
trajectory_runners/
  base.py
  types.py
  model_clients.py
  projections.py
  skyrl_gym.py
  harbor/
    runner.py
    dataset.py
    configuration.py
    rollout_dispatcher.py
    ...
  mini_swe/
    runner.py
    environment.py
    ...
```

Shared reward shaping and retention remain beside the runner boundary for this migration. Typed verifier, reward,
and disposition contracts will move those policies to packages named for their responsibilities.

Production entrypoints move under `skyrl_train.entrypoints`:

```text
entrypoints/
  main_base.py
  terminal_bench.py
  terminal_bench_generate.py
  terminal_bench_teacher_logits.py
  mini_swe.py
  fully_async.py
```

Iris configuration files and local run scripts will name these modules. Example directories may retain tutorials,
sample configuration, and shell commands, but no runtime runner or production entrypoint.

The Terminal Bench Hydra config group moves into `skyrl_train.config`. This removes the extra Hydra search path from
local commands and ensures the installed wheel contains the same configuration used by Iris.

## Verifiers removal

Delete:

- `integrations/verifiers/verifiers_generator.py`;
- its entrypoint, preparation and installation scripts, README, and run script; and
- the disabled Verifiers GPU test.

No compatibility import remains. MarinSkyRL is a hard fork and repository policy requires updating callers instead
of maintaining dead aliases.

This does not remove the concept of an externally maintained verifier. Harbor and other trajectory runners may
vendor or call external verifiers. The deferred verifier contract defines how every runner reports those results.

## Behavior gates

Focused CPU tests must establish:

- `TrajectoryRunner.run()` preserves shaping, alignment metrics, and retention order;
- whole-trajectory and step-wise projections preserve the current token, mask, reward, trajectory-ID, and final-step
  behavior;
- direct and HTTP model clients return normalized response text, token IDs, stop reasons, and logprobs with explicit
  token provenance;
- the Terminal Bench entrypoint constructs the same runner and trainer class for colocated and fully asynchronous
  configurations;
- `RolloutDispatcher` preserves the runner lifecycle and returns the underlying `TrajectoryBatch` unchanged; and
- every checked-in Iris RL configuration resolves its remapped entrypoint and Hydra config groups from the installed
  package.

Existing SkyRL-Gym, Terminal Bench, Mini-SWE-Agent, async staleness, trajectory shaping, trajectory retention, and
launcher configuration tests remain in the CPU gate. The Verifiers GPU tests are removed with the dead integration.

## Initial migration boundaries

The initial migration does not:

- change verifier semantics or reward numerics;
- change error classification, masking, baseline inclusion, or retry policy;
- change inference-engine placement, weight synchronization, or sampling defaults;
- rename the `generator.*` Hydra namespace; or
- fix the AIME outcome and length-instrumentation defects.

The deferred contract migration introduces `RolloutEvidence`, `VerificationResult`, `RewardResult`, and
`TrainingDisposition` without changing numerical output. AIME remains outside this structural change so its
instrumentation and evaluation-budget fixes can be validated against the completed contracts.
