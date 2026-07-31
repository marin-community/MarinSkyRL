# Debugging log for S3 model paths

Prevent object-store model paths from consuming a GPU gang before failing at tokenizer load.

## Initial status

The launcher accepts `s3://`, `gs://`, and `gcs://` model paths, omits model prestaging for them, and
passes the URI unchanged to `AutoTokenizer.from_pretrained`. The controller comment says another
path handles these URIs, but no such path exists.

## Hypothesis 1

The existing S3 warm-sync path cannot resolve a direct object-store model path because it builds a
Hugging Face cache snapshot keyed by a repo ID and intentionally leaves the configured model path
unchanged. Rejecting cloud URIs during normalization prevents the expensive fail-late behavior.

## Changes to make

Add a regression test for each accepted cloud scheme, reject those paths before launch, document the
supported repo-ID plus warm-mirror form, and make direct controller staging reject the same input.

## Results

The regression failed before the implementation for all three cloud schemes: normalization returned
success instead of raising, which allowed the unsupported URI into the task command. After the fix,
the launcher rejects each scheme during normalization and the controller rejects direct staging
before importing model-download dependencies. Launcher tests and mechanical lint pass.

## Future work

- None.
