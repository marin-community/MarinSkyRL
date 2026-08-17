# Debugging log for watcher trials URI resolution

Make the RL watcher inventory the trace prefix submitted by the launcher instead of reconstructing a retired path.

## Initial status

The watcher records `s3://marin-us-east-02a/iris/<job>/trace_jobs` for an agentic run whose submitted Hydra
override points to a populated lifecycle-managed `tmp/ttl=14d/.../trace_jobs` prefix. Its trace inventory reports
zero available trials.

## Hypothesis 1

`iris_trials_uri` searches the JSON-rendered entrypoint with a pattern that requires an unquoted S3 URI in the
same string as the option. The launcher submits a quoted Hydra override, and Iris may preserve argv as separate
JSON strings.

## Changes to make

Reproduce the quoted Hydra and split-argv forms. Verify both the object-store inventory prefix and persisted bundle
manifest use the submitted trials directory.

## Results

All three regressions fail before the fix. Both submitted forms resolve to the retired
`s3://marin-us-east-02a/iris/run/trace_jobs` fallback, and trace inventory lists that prefix.

## Hypothesis 2

Parsing the decoded string leaves preserves quoted Hydra assignments while retaining argv order for a separate
`--trials-dir` value.

## Changes to make

Resolve the first submitted S3 trials directory from decoded entrypoint strings. Accept quoted Hydra values,
hyphenated and underscored launcher options, and split argv. Use the legacy convention only when none is present.

## Results

The focused regressions pass. Quoted Hydra and split-argv submissions both resolve to
`s3://bucket/tmp/ttl=14d/run/trace_jobs`; trace inventory lists that prefix and the bundle manifest records it.
The complete Iris launcher suite passes 322 tests.
