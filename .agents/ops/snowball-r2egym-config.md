# Snowball R2E-Gym recipe configuration rationale

This document explains coupled settings in `cloud/iris/configs/snowball_r2egym_arm_a.yaml`. Read
current values from that file. The file also carries a `probe:` overlay; the sections below say
which reasoning applies to an arm and which to a probe.

## What a recipe adds to an ordinary RL config

Four blocks beyond the usual sections:

- `base:` names another recipe, relative to this one, that this file extends. The base is loaded
  first and this file is deep-merged over it, so a variant recipe carries only its deltas and
  cannot drift from what it extends in a key nobody re-read. A null deletes a base key, as in
  `probe:`. `snowball_r2egym_migsmoke.yaml` is arm A this way.
- `backend:` names the sandbox backend and everything that distinguishes one from another. Only the
  environment type, and Daytona's automatic snapshot flag, reach Hydra. The rest describes bridges,
  worker fleets, proxies and key files, which the launcher exports as environment variables.
- `mode:` selects `arm` or `probe`. A probe is an eval-only pass over one task tree at step zero.
- `probe:` overlays the fields a probe changes. It uses the same section names as the rest of the
  file, so it reads as a diff; a null value deletes the arm's key rather than setting it to null.

`cloud/iris/recipe_preflight.py` refuses to launch when any of the couplings below is broken. The
training driver runs it for every config carrying a `backend:` block before the run claims a GPU, so
a broken recipe fails at launch rather than seventeen minutes in. Run it directly to see every check, the rendered environment block and, with `--print-args`, the full
argument list:

```bash
python -m cloud.iris.recipe_preflight --recipe snowball_r2egym_arm_a --num-nodes 40
```

## Backend

- Only one Hydra key distinguishes the backends, so declaring the environment type by hand while a
  backend block is present is an error: one value with two homes is how a setting silently reverts.
- Apptainer needs a bridge URL that has no Hydra key anywhere, and a CPU worker fleet submitted
  separately on another cluster. That fleet needs its harbor source path exported or it exits within
  seconds, and it must supply at least as many seats as the recipe asks for concurrent trials.
- Daytona needs automatic snapshots on, or harbor falls back to per-sandbox declarative builds that
  the eval org rejects and the wave dies at start. It needs an API key referenced by file path, never
  by value, and a SOCKS route because compute nodes have no egress.
- Leave the Daytona region unset. Setting it region-qualifies snapshot names, which then stop
  matching the ones that were prebuilt.
- Each backend declares the exception names that must be masked so its own infrastructure failures do
  not score zero. Preflight asserts they are present rather than appending them, because appending
  would silently reorder a list that a frozen run reproduces exactly.

## Settings that change results and have no Hydra key

Four environment variables and one path change what the agent does while leaving the rendered Hydra
arguments identical. Two runs that differ only here are not comparable, so the recipe carries them:

- The harbor client connect timeout. Too low and connection failures become masked samples rather
  than retried ones, which reads as a throughput collapse rather than an error.
- The tmux batch execution timeout margin, on apptainer.
- The Terminus history contract, which decides whether prior think spans stay in the history fed back
  to the model. Dropping them teaches the model to stop thinking.
- The harbor overlay directory and its commit, which the launcher prepends to the interpreter path.
  An overlay that disagrees with the installed package changes agent behaviour invisibly.

Node counts, engine counts and the tracking project are derived from the same recipe fields as the
Hydra arguments and exported alongside them, so preflight can assert the two agree. Hand-cloned
sbatch files have disagreed with their own configs on all three.

## Geometry

- The requested node count must equal policy nodes plus generator nodes, plus reference nodes when
  the reference is not colocated with the policy.
- FSDP size must equal the role's node count times its GPUs per node.
- An engine's data-parallel group times its tensor-parallel size must fit inside one node. A group
  split across nodes hangs at startup with no error.
- Expert parallelism on the generator must equal data parallelism times tensor parallelism.
- Tensor parallelism must divide the model's attention-head count, or vLLM wedges at engine init
  with no launcher-side signal.

## Context budget

Declare the request window, the per-turn output allowance and the turn cap. Every length argument is
derived from those three; declaring a derived one directly is rejected upstream. When the request
window exceeds the model's positional limit, widen it explicitly through the engine's HF overrides,
which render as one opaque passthrough rather than one Hydra key per field.

## Timeouts and error policy

- Set the verifier timeout from the observed tail of verifier work under full load, not from the
  median. A budget that is comfortable at low concurrency turns a measurable fraction of samples per
  step into false zeros when every seat is busy.
- Keep preserve-on-timeout off. Harbor's own default is on, so silence here is the unsafe value, and
  the combination with a tight verifier budget is what manufactures the false zeros.
- Rollout detail collection must stay on or the truncated-importance-sampling and rollout-logprob
  objectives fail their capability check.
- Database registration stays off; registration is a manual step.

## Probes

A probe runs one step, evaluates before training, resumes nothing and writes no checkpoints. Two
couplings are not preferences:

- Truncated importance sampling must be off. A group masked entirely by exceptions otherwise raises
  a missing-rollout-logprobs error and kills the job mid-probe.
- The evaluation batch size follows from the task count divided by the coordinator count, so a probe
  of a differently sized tree needs it recomputed rather than inherited.

Leave the reference model path out of the probe overlay's rendered arguments. SkyRL's base config
already defaults it to the policy path, so the checkpoint under test arrives once, as a launch
argument.

## Seats

Concurrent trials divided by coordinators gives the trials each coordinator drives. Past roughly
sixty-six the coordinator becomes the bottleneck rather than the seats. Exceeding it is allowed only
with an entry in the recipe's acknowledgements block, which preflight prints at launch.

## Reproducing a frozen run

`cloud/iris/tests/test_snowball_r2egym_recipe.py` renders this recipe and compares it, argument by
argument, against a frozen launcher configuration checked in under the tests' fixtures. Comparison is
by meaning: both sides are parsed with Hydra's own override parser, which folds repeated keys to
their last value and canonicalises number, list and dict spellings. Run names, checkpoint and export
paths and the artifact-store mount are normalised to placeholders because they encode the job name
and a hash of the artifact-store image path.
