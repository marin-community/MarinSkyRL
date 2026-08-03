# Debugging log for Harbor error classification

Prevent Harbor infrastructure failures from becoming reward-zero agent outcomes when error names evolve.

## Failure mechanism

MarinSkyRL classifies the string in a persisted Harbor `ExceptionInfo` against three locally maintained sets.
`ContextManagementInfrastructureError` is absent, so configurations with `default_error_treatment: zero` turn
Harbor context-management failures into policy training signal.

Harbor's lightweight configuration package publishes the semantic category for every stable trial-outcome
exception, so the duplicated lists are not an appropriate ownership boundary.

## Hypothesis

Making the published taxonomy the default source of truth, while retaining campaign lists as explicit
overrides, will mask the new infrastructure error and expose future unknown names before applying a fallback.

## Decision

The immutable `harbor-config` wheel is a direct dependency. Persisted exception names use its public taxonomy;
campaign lists remain explicit overrides, and an unknown category is logged as an error before its configured
fallback is applied.

## Evidence

The regression contract verifies that a persisted `ContextManagementInfrastructureError` name maps to `mask`
even when the unknown-error fallback is `zero`. It also covers Harbor's agent and passthrough categories,
campaign-override precedence, and error-level reporting for unknown names.

The frozen lock retains the existing dependency graph. Its Harbor wheel entry records the release URL and
the independently verified SHA-256 digest.
