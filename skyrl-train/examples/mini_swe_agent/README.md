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
  --output_dir ~/data/swe_gym_subset
```

Replace the two revision values with pinned dataset commit IDs. The converter
reads `SumanthRH/SWE-Gym-Subset` for training and
`SumanthRH/SWE-bench_Verified` for evaluation. Each output row contains one
serialized `TaskSpec` in the `task_spec` column. The evaluation script stays in
the private verifier fields.

The converter uses each row's `image_name`, when present. Otherwise it derives
the image name from the dataset and instance ID. Task commands use `/testbed`.
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
from taskcompendium.models import TaskSpec

directory = Path("~/data/swe_gym_subset").expanduser()
images = {
    TaskSpec.model_validate_json(row["task_spec"]).environment.image.reference
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
The scripts set the command timeout to 180
seconds. `generator.max_turns` controls the model turn limit.

To change task setup, add commands to the task's `environment.setup` during
materialization. The engine applies the same setup to the fresh grading machine.
The grader collects staged and unstaged changes, including new files, through
`git add -A` and `git diff --cached --binary`. It transfers the patch as a file,
so the patch does not consume shell argument space.

A successful evaluation command gives reward `1`. A failed patch application or
evaluation command gives reward `0`. An evaluation timeout has no grade and
excludes the rollout from training. An environment setup failure aborts the
prompt group. The engine closes the agent and grading machines after execution.

The common engine retains exact model token IDs and masks tool observations.
Training can use one sample per trajectory or one sample per model turn through
`trainer.step_wise_training`.
