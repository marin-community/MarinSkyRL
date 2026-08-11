# Debugging log for Megatron router replay configuration

Prevent Megatron training from silently accepting MoE router replay while discarding the captured routes.

## Initial status

`RayPPOTrainer.convert_to_training_input` reads `trainer.policy.fsdp_config.moe_router_replay` for every
training strategy. Megatron's forward path does not consume `rollout_routed_experts`, so the configuration
captures and transports routes without replaying them. The DCP validator also treats the FSDP replay flag as
active under Megatron and rejects an otherwise legal rollout configuration.

## Hypothesis 1

Router replay needs a training-strategy capability check at configuration validation and at the trainer boundary.
The DCP guard should only consider training-side replay active for a strategy that implements it.

## Changes to make

Add regression coverage for unsupported Megatron replay and for Megatron DCP with the irrelevant FSDP replay
flag. Centralize the strategy check, use it when the trainer decides whether to transport routed experts, and
make unsupported configurations fail before resources are allocated.

## Results

The two regression tests failed before the production change: Megatron config validation accepted router replay,
and the DCP validator rejected Megatron when only the unused FSDP replay flag was set. After centralizing the
strategy capability check, both regressions pass. Router replay now fails at config validation for Megatron and
at batch construction if validation is bypassed; DCP only treats the flag as active for FSDP/FSDP2 training.

Targeted result: 27 passed in `test_megatron_config.py` and `test_dcp_config.py`. The full root CPU gate
completed with 1,347 passed and 21 skipped.

## Future work

- [ ] Implement router replay in Megatron before allowing this configuration.
