# Debugging log for Harbor error classification

Prevent Harbor infrastructure failures from becoming reward-zero agent outcomes when error names evolve.

## Initial status

MarinSkyRL classifies the string in a persisted Harbor `ExceptionInfo` against three locally maintained sets.
`ContextManagementInfrastructureError` is absent, so configurations with `default_error_treatment: zero` turn
Harbor context-management failures into policy training signal.

Harbor release `harbor-config-01a904ab5a1e3a6ad6ad4f96cc39e82242e4ff8c` publishes the semantic category
for every stable trial-outcome exception. The downloaded wheel matches its published SHA-256 digest.

## Hypothesis

Making the published taxonomy the default source of truth, while retaining campaign lists as explicit
overrides, will mask the new infrastructure error and expose future unknown names before applying a fallback.

## Changes to make

Pin the immutable `harbor-config` wheel, classify persisted exception names through its public API, empty the
duplicated schema defaults, and cover infrastructure, agent, passthrough, override, and unknown-category paths.

## Results

The taxonomy tests fail at collection before the classifier exists. With the immutable wheel and classifier
in place, the serialized context-management failure maps to `mask` even when the configured unknown-error
fallback is `zero`. Agent and passthrough categories follow Harbor's decisions, campaign overrides take
precedence, and an unknown name emits an error before applying the configured fallback.

The frozen lock retains the existing dependency graph. Its Harbor wheel entry records the release URL and
the independently verified SHA-256 digest.
