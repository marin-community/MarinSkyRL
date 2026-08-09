# Debugging log for the runtime-bundled parquet extractor

Hugging Face dataset IDs passed through `resolve_rl_train_data` must remain usable after the Iris launcher copies its runtime bundle into a task workspace.

## Reported failure

`rl_data.py` invoked `python -m cloud.iris.extract_tasks_from_parquet` for a Hugging Face dataset ID. Before this fix, the runtime manifest included `rl_data.py` and `hf_datasets.py` but omitted the invoked module and its local `tasks_parquet` dependency.

## Hypothesis 1

An isolated Python process rooted in the generated runtime workspace cannot load the parquet extractor because its source closure is absent from the bundle.

## Experiment

The regression test builds the real runtime workspace, disables site packages, supplies stubs for optional third-party imports, and invokes the extractor's `--help` entrypoint. This exercises module loading without downloading or reading a dataset.

## Results

The isolated entrypoint failed with `No module named cloud.iris.extract_tasks_from_parquet`, matching the task failure. Adding `extract_tasks_from_parquet.py` and `tasks_parquet.py` to the runtime manifest made the same entrypoint load successfully with site packages disabled. The test therefore covers the invoked module and its local import rather than relying on the launcher's development environment.
