# Compose Iris RL recipes

Source recipes in `cloud/iris/configs/` support Hydra's `defaults` list. Inherit a
recipe and put `_self_` last so local values override the parent:

```yaml
defaults:
  - snowball_mopd_ultra_32k
  - _self_
trainer:
  max_steps: 4
  algorithm:
    off_policy_correction: tis
  rollout_buffer:
    max_staleness_steps: 1
    batch_policy: rolling
    max_in_flight: 96
```

The launcher composes the source recipe before deriving context budgets,
teacher placement and the trainer configuration. Hydra searches the source
file's directory first, then the bundled `cloud/iris/configs/` directory. Use
`load_rl_recipe` from `cloud.iris.rl_config_translation` when inspecting a
recipe's inherited fields; loading the YAML alone returns only its overrides.

A launch document can select a bundled recipe inside `skyrl`:

```yaml
skyrl:
  defaults: [snowball_mopd_ultra_async_32k_smoke, _self_]
  trainer:
    max_steps: 2
```

Use bundled recipe names at this boundary; local sibling files are not forwarded
with the launch document. Its resolved
launch document contains the composed values, so task execution and reloading
that document do not need the source recipe's local directory.

Hydra merges nested mappings. Disable an inherited sampling policy with
`data.sampling.kind: null`; an empty mapping does not clear inherited fields.
Algorithm groups selected through `config_groups` still compose separately
into the trainer's base configuration; the composed recipe's explicit values
override the group values.
