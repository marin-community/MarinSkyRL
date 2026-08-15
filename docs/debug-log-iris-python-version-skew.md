# Debugging log for Iris Python version skew

Make multi-node Iris runtime selection deterministic and reject incompatible gang members before Ray join.

## Initial status

The root project accepts any Python 3.12 patch release and has no uv interpreter pin. Mixed Iris placements
selected Python 3.12.13 and 3.12.14, then failed inside Ray's exact version check on every gang attempt.
The rendezvous document contains no runtime identity, so the worker cannot diagnose skew before `ray start`.

## Hypothesis 1

A root `.python-version` fixed at 3.12.13 makes every `uv sync --project` select the same managed interpreter.
Publishing Python and Ray versions in the rendezvous lets a worker fail before invoking Ray with both runtime
identities in the error.

## Changes to make

Pin Python 3.12.13 at the project root. Add rendezvous version validation and behavior tests for matching and
mismatched Python versions.

## Results

The root pin resolves Python 3.12.13. Rendezvous tests confirm that the head publishes its node, Python version,
and Ray version; matching workers pass validation; and Python or Ray skew raises before `ray start` with both
nodes and conflicting versions in the error. The focused bootstrap, rendezvous, and spill-policy suite passes.

## Future work

- [ ] None identified.
