The `cloud/iris/configs/vllm_pause_policy_smoke.yaml` recipe uses these
deterministic addition prompts with the GSM8K reward schema. It requires no
sandbox provider or external dataset download. The training split has 32
distinct prompts so rollout batches can overlap an optimizer step at positive
staleness.

`run_stream_probe.sh` runs the focused vLLM GPU regression on an Iris GPU task.
It checks a live streamed chat completion across a real weight broadcast in
both abort and keep modes.
