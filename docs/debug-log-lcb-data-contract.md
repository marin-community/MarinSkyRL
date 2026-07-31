# Debugging log for LiveCodeBench data contracts

Trace a prepared LiveCodeBench row from source normalization through trainer pass-through and runtime scoring.

## Initial hypotheses

The new ingestion path writes `reward_model.ground_truth`, while the runtime environment and legacy example still read or write `reward_spec.ground_truth`.

The LCB contract is schema-only in name but does not validate a schema: it JSON-encodes any list or mapping and reports every response as correct. APPS and verifiable-code rows also use different test-spec shapes, so JSON encoding alone cannot give the runtime one executable representation.

## Reproduction plan

Add public contract tests that normalize both source shapes and distinguish a known-good response from a known-bad response with the runtime verifier. Update environment tests to pass the new `reward_model` field. At the ingestion boundary, require both code sources to preflight their canonical solution and record two-sided verification.

## Results

The regression tests failed before the implementation:

- `LCBEnv` rejected rows that carried `reward_model.ground_truth`.
- The contract accepted a deliberately incorrect response because its correctness predicate always returned `True`.
- Neither code-source adapter called `validate_example`, so ingestion could publish unusable verifier inputs.

The implementation now converts both supported source schemas into the test-case list consumed by the
LiveCodeBench runtime, executes the canonical and deliberately incorrect solutions during source preflight,
and passes the standard `reward_model` field through the example dataset and environment. Malformed runtime
ground truth produces a zero reward with verifier-error metadata instead of terminating a rollout.

Focused contract, environment, and ingestion tests pass. The complete `skyrl-gym` suite passes under its
CI Python version, and the complete `infra/tests` CI command passes.
