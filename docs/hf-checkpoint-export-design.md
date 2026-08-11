# Hugging Face checkpoint export design

Hugging Face conversion is a model operation, not a training run. The export entrypoint therefore reconstructs
only the policy model at the checkpoint's recorded node and GPU geometry, restores model tensors, invokes the
strategy-owned converter, validates the result, and optionally publishes it.

The exporter deliberately does not construct datasets, rollout engines, generators, tracking, reference or
critic models, optimizers, schedulers, profilers, or weight-sync clients. Its frozen runtime profiles similarly
omit vLLM, Harbor, Daytona, and telemetry extras. FSDP2, DeepSpeed, and Megatron still use their production model
builders and conversion implementations so checkpoint and Hugging Face formats retain one owner.

An export is valid only when the requested checkpoint contains a completed trainer-state marker for the exact
step and the policy checkpoint subtree exists. Model-only loading does not read optimizer, scheduler, RNG, or
callback state. The output must exist at the canonical export path before publication begins; conversion or
publication failures propagate and leave the request rerunnable.

Iris uses the shared launcher to preserve one task protocol, runtime identity, and source-bundle contract for
training and conversion. The launcher selects an explicit checkpoint-export runtime mode and suppresses every
training-only default or resource. The in-task driver then branches immediately into its separate conversion
pipeline, before training configuration, context budgets, data resolution, ingress, or rollout environment setup.

Before using a strategy for a production model, validate one small accelerator export at the checkpoint's saved
parallel geometry. CPU tests enforce command routing, dependency selection, model-only loading, exact-step
validation, output-before-publication ordering, and committed runtime-bundle identity.
