# Iris priority-band compatibility

## Reported behavior

The RL launcher raises `AttributeError` before submission because the installed Iris protobuf exposes `PRIORITY_BAND_PRODUCTION`, `PRIORITY_BAND_INTERACTIVE`, and `PRIORITY_BAND_BATCH`, but not `PRIORITY_BAND_UNSPECIFIED`.

## Hypothesis

Python evaluates the default argument to `dict.get` eagerly. The launcher therefore reads `job_pb2.PRIORITY_BAND_UNSPECIFIED` even though argparse limits `--priority` to a key that is present in the mapping.

## Reproduction

The regression supplies the installed protobuf surface with only the three supported enum members and resolves every parser-accepted priority. Before the fix, collection fails because the exhaustive resolver does not exist.

## Verification

The focused launcher-default regression passes for every parser-accepted priority against a protobuf surface with no unspecified member.
