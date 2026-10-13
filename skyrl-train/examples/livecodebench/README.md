# Code task preparation

Prepare code tasks with a prebuilt Python image. From `skyrl-train`:

```bash
uv run --project .. examples/livecodebench/lcb_dataset.py \
  --dataset_dir /path/to/raw-code-datasets \
  --local_dir /path/to/prepared-code-tasks \
  --python_image 'registry/python@sha256:IMAGE_DIGEST'
```

The image reference must contain a SHA-256 digest. The output Parquet files contain
bare TaskSpecs. Public prompts do not contain private reference tests.
The task declares the image. The run config selects the Shellbox backend and execution limits.

Set `DATA_DIR` in `run_lcb.sh` to the output directory. The script selects the
`lcb` session and sends the prepared tasks through RolloutEngine.
The script requests 1 CPU and 8192 MB for each task machine.
