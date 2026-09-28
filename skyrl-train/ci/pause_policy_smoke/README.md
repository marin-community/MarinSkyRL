These deterministic addition prompts use the GSM8K reward schema for a short
asynchronous weight-sync smoke. They require no sandbox provider or external
dataset download. The training split has 32 distinct prompts so rollout batches
can overlap an optimizer step at positive staleness.
