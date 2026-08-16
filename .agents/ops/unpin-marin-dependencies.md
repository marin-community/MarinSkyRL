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

The refreshed lock resolves one coherent set of Marin packages. The resolved Iris package successfully loads the current
`cw-rno2a.yaml` and exposes both new fields.

## Hypothesis 2

The resolved Iris wheel removed client symbols from the `iris.client` package initializer but retains them in the module
used by Iris itself.

## Changes to make

Import `IrisClient` and `JobFailedError` from `iris.client.client` in launcher code and its test.

## Results

The initial launcher-suite collection failed because the old re-exports no longer exist. Iris imports these symbols from
`iris.client.client` throughout its CLI and testing packages. After updating the imports, the launcher suite passes.

## Future work

- [ ] Evaluate resolving the Marin package family from the selected launch checkout instead of advancing it during
  ordinary lock refreshes.
