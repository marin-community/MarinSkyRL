# Nemotron task preparation

Prepare source rows before training. From the repository root:

```bash
uv run python -m marinskyrl.nemotron_preparation \
  --input /path/to/source.parquet --output /path/to/tasks.parquet \
  --source my_blend --config /path/to/graders.json \
  --environments skyrl-train/examples/nemotron/environments.json \
  --terminal-tasks /path/to/harbor-tasks.parquet
```

The environment file declares the Python tool image and Lean workspace.
The grader file contains verifier parameters and configured judge identities.
Keep credentials outside that file.
Omit `--terminal-tasks` when the source contains no SWE rows.
The output contains bare TaskSpecs with separate training labels.

The input is a Parquet file from the Nemotron Ultra extraction tools in `infra/rl_data`.
Each row contains chat messages in `prompt` and private source metadata in `extra_info.nemotron_ultra`.
That metadata contains `blend`, `agent`, `route`, `record_json`, and `request_json`.
SWE rows use `route=terminal_bench` and supply `terminal_bench_instance_id`.
`--source` supplies the dataset identifier when a row has no `data_source`.

For graders that require a judge, `graders.json` can contain:

```json
{
  "grading": "verify",
  "judges": {
    "general": {
      "model": "JUDGE_MODEL",
      "base_url": "https://JUDGE_HOST/v1",
      "api_key_env": "JUDGE_API_KEY"
    }
  }
}
```

Replace the model and endpoint with the selected judge service.
Supply `JUDGE_API_KEY` to the worker environment.
Add a `safety` entry with the same fields when the task requires a safety judge.
Use an empty object for tasks that require no extra verifier parameters.
GenRM tasks require a separate comparison judge in `genrm.judge`:

```json
{
  "grading": "verify",
  "genrm": {
    "judge": {
      "model": "GENRM_MODEL",
      "base_url": "https://GENRM_HOST/v1",
      "api_key_env": "GENRM_API_KEY"
    }
  }
}
```

The GenRM scorer applies its standard reward and length-penalty settings when those fields are absent.

For SWE rows, prepare supported single-stage Harbor packages first:

```bash
uv run python -m marinskyrl.harbor_preparation \
  --input /path/to/harbor-packages --output-dir /path/to/prepared-harbor \
  --execution-output /path/to/harbor-execution.json
```

The command prints the TaskSpec artifact path for `--terminal-tasks`.
For the `path`/`task_binary` Parquet artifact from the SWE extraction tools, use:

```bash
uv run python -m marinskyrl.harbor_preparation \
  --archive-input /path/to/swe-artifact/tasks.parquet \
  --output-dir /path/to/prepared-harbor \
  --execution-output /path/to/harbor-execution.json
```

Packages must declare prebuilt images and separate verifier environments.
Preparation rejects archives that require image builds or shared verifier mode.
The execution file contains overrides under `environment.task_sessions.sources`.
Merge its execution settings into the run config at `environment.task_sessions.sources`.
The Iris Daytona credential check uses the session selections in this merged file.
Keep the `shellbox` source selection for Harbor tasks in a Nemotron run.
Nemotron SWE tasks retain the Harbor source used by those settings.
Other tasks use the selected `nemotron_ultra` session and the run's machine settings.
The supplied Iris configs are geometry templates. Their launch inputs must supply prepared TaskSpec files.
