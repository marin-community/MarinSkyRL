# Debugging log for unpinned Marin dependencies

Remove independent version constraints from MarinSkyRL's `marin-*` dependencies and verify the resolved Iris client
against the current Marin-owned cluster configuration.

## Initial status

MarinSkyRL pinned Iris `0.2.70` while `_resolve_cluster_config_default` selected YAML from a separate Marin checkout.
The current `cw-rno2a.yaml` contains `kubernetes_provider.cache_max_age` and `user_budget_defaults`, which Iris `0.2.70`
rejects.

## Hypothesis 1

Removing every `marin-*` constraint and allowing prereleases will let one lock resolution select a coherent set of
published Marin packages.

## Changes to make

Remove exact constraints from Finelog, Finelog Server, Iris, Iris Native, and both Rigging requirements. Configure uv
to allow prereleases because the published Marin packages use development versions, then refresh those lock entries.

## Results

The lock resolves Iris `0.2.83`, Iris Native `0.1.6`, Finelog and Finelog Server `0.2.28`, and Rigging `0.2.83`. Iris
`0.2.83` successfully loads the current `cw-rno2a.yaml` and exposes both new fields.

## Hypothesis 2

The current Iris wheel removed client symbols from the `iris.client` package initializer but retains them in the
documented implementation module used by Iris itself.

## Changes to make

Import `IrisClient` and `JobFailedError` from `iris.client.client` in launcher code and its test.

## Results

The initial launcher-suite collection failed in five modules because the old re-exports no longer exist. Iris `0.2.83`
imports these symbols from `iris.client.client` throughout its CLI and testing packages. After updating the imports, all
315 launcher tests pass. A configuration-contract test now covers both the unconstrained Marin runtime dependency policy
and the two cluster schema fields that exposed the mismatch.

## Future work

- [ ] Evaluate resolving the Marin package family from the selected launch checkout instead of advancing it during
  ordinary lock refreshes.
