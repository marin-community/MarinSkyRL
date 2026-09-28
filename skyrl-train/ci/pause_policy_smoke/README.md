The `cloud/iris/configs/vllm_pause_policy_smoke.yaml` recipe uses these
deterministic addition prompts with the GSM8K reward schema. It requires no
sandbox provider or external dataset download. The training split has 32
distinct prompts so rollout batches can overlap an optimizer step at positive
staleness.
