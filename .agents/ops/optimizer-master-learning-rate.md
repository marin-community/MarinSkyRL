# Debugging log for optimizer master learning rate

Make `optimizer_config.lr` the default learning rate for every optimizer parameter group while preserving explicit
per-route learning-rate overrides.

## Initial status

MuonH defaults its plain-Adam route to `6e-4` even when the run config sets another master learning rate. The hero arms set
`lr=1e-5` and no `adam_lr`, so routers, attention gates, embeddings, and norms trained at 60 times the configured rate. The
hybrid Muon optimizer has the same contract defect: its Muon route defaults to `0.02` while AdamW inherits the master rate.

## Hypothesis 1

Direct construction tests with no route-specific override will expose both hidden defaults.

## Changes to make

Add regressions that require every MuonH and hybrid Muon parameter group to inherit `optimizer_config.lr` when
`optimizer_kwargs` contains no learning-rate override.

## Results

Confirmed. MuonH produces parameter-group rates `{1e-5, 6e-4}` and hybrid Muon produces `{8e-6, 0.02}` when neither
configuration contains a route-specific learning-rate override.

## Hypothesis 2

Resolving missing composite-route rates from `optimizer_config.lr`, combined with a strategy-level allowlist of the master
rate and explicit `*_lr` values, will remove both hidden defaults and reject future optimizer groups that introduce one.

## Changes to make

Make the MuonH Adam and hybrid Muon routes inherit the master rate. Validate every FSDP optimizer group before scheduler
construction, and reject the duplicate `optimizer_kwargs.lr` spelling so the master setting has one canonical location.

## Results

Confirmed. Default MuonH and hybrid Muon groups inherit the master rate; existing explicit route overrides remain valid;
undeclared group rates and `optimizer_kwargs.lr` fail before scheduler construction. The distributed CPU suite passes with
101 tests passed and one skipped.

## Future work

- [ ] Re-run the affected hero recipe with one master learning rate after the code fix lands.
