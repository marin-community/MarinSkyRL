# SWE tasks with Shellbox

SWE-Gym and SWE-Bench use the shared rollout engine in Marin. The model
changes repository files through the `shell` tool. The grader applies the Git
patch in a fresh copy of the task image, then runs the private evaluation script.

## Prepare task Parquet

The frozen root dependencies include the shared task packages.
After [installation](../../docs/getting-started/installation.rst), run these commands from `MarinSkyRL/skyrl-train`.
Materialize the source rows:

```bash
uv run --no-sync --project .. examples/mini_swe_agent/preprocess_swegym.py \
  --train_revision TRAIN_DATASET_COMMIT \
  --eval_revision EVAL_DATASET_COMMIT \
  --image_manifest /path/to/instance-images.json \
  --max_turns 20 \
  --command_timeout 180 \
  --output_dir ~/data/swe_gym_subset
```

Replace the two revision values with pinned dataset commit IDs. The converter
reads `SumanthRH/SWE-Gym-Subset` for training and
`SumanthRH/SWE-bench_Verified` for evaluation. Each output row contains one
serialized `LoweredTaskSpec` in the `lowered_task_spec` column. The evaluation script stays in
the private verifier fields.

The image manifest maps every selected instance ID to a prebuilt image reference with its SHA-256 digest.
Prepare those images before conversion. Image tags and task-specific builds cause rejection.
The converter does not build or resolve images. Task commands use `/testbed`.
The task environment permits network access and uses the environment variables
in `preprocess_swegym.py`. The grader timeout is 3600 seconds.

## Run training

The Docker factory requires a Docker daemon and Skopeo on each rollout worker.
It resolves registry images into `trajectory_runner.image_cache` and reuses them
for task and verifier machines. Set `trajectory_runner.skopeo` to the executable
path if Skopeo is not on `PATH`.
List the task images from the materialized files:

```bash
uv run --no-sync --project .. python - <<'PY'
from pathlib import Path
from datasets import Dataset
from rolloutengine.spec import LoweredTaskSpec

directory = Path("~/data/swe_gym_subset").expanduser()
images = {
    LoweredTaskSpec.model_validate_json(row["lowered_task_spec"]).task.environment_requirements.docker_image
    for name in ("train.parquet", "validation.parquet")
    for row in Dataset.from_parquet(str(directory / name))
}
for image in sorted(images):
    print(image)
PY
```

```bash
bash examples/mini_swe_agent/run_mini_swe_8B.sh
# For the two-node example:
bash examples/mini_swe_agent/run_mini_swe_30B.sh
```

The scripts use `skyrl_train.entrypoints.taskcompendium`.
Edit `DATA_DIR` and `CKPT_PATH` in the script to select task Parquet and checkpoint directories.
The two-node example requires a Ray cluster with eight GPUs per node.
See the [cluster setup](../../docs/getting-started/installation.rst#initialize-ray-cluster).
The materialized session settings control execution limits.
The preparation command above selects 20 turns and a 180-second command timeout for the 8B example.
For the 30B example, prepare the data with `--max_turns 50`.
Each tool turn has a separate 600-second timeout. The cumulative turn deadline is 3600 seconds.
Final verification has a separate 3600-second deadline. Each cleanup action has a 30-second limit.
Select turn and command limits with the preparation flags.
Change the other deadlines in `preprocess_swegym.py` before materialization.
The launch-time `generator.max_turns` setting does not override a materialized task.

To change task setup, add commands to `environment_requirements.setup_commands` during materialization.
The verifier's `environment_requirements` declares setup for its fresh grading machine.
The task saves the initial Git revision in `refs/taskcompendium/base` before inference.
The grader collects all changes against that revision, including agent commits and new files.
It transfers the patch as a file to the grading machine.

A successful evaluation command gives reward `1`. A failed patch application or
evaluation command gives reward `0`. An evaluation timeout has no grade and
excludes the rollout from training. An environment setup failure aborts the
prompt group. The engine closes the agent and grading machines after execution.

The common engine retains exact model token IDs and masks tool observations.
Training can use one sample per trajectory or one sample per model turn through
`trainer.step_wise_training`.
