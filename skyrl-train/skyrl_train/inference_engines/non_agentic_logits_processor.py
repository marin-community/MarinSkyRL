"""Opt-in vLLM V1 adapter; raw model logprobs are retained before forcing."""

from vllm.v1.sample.logits_processor import AdapterLogitsProcessor

from skyrl_train.trajectory_runners.non_agentic_interventions import InterventionState, TokenIntervention


class NonAgenticTokenProcessor(AdapterLogitsProcessor):
    def __init__(self, vllm_config, device, is_pin_memory):
        if vllm_config.model_config.logprobs_mode != "raw_logprobs":
            raise ValueError("Token interventions require native pre-processor raw_logprobs")
        super().__init__(vllm_config, device, is_pin_memory)

    def is_argmax_invariant(self) -> bool:
        return False

    def new_req_logits_processor(self, params):
        config = (params.extra_args or {}).get("non_agentic_intervention")
        if config is None:
            return None
        state = InterventionState(TokenIntervention(**config))

        def process(output_ids, logits):
            forced = state.advance(output_ids)
            if forced is not None:
                if forced >= logits.shape[-1]:
                    raise ValueError("Forced token is outside the native vocabulary")
                # vLLM's raw_logprobs are computed before this adapter mutates
                # logits. This distribution forces an action; it must not be
                # used as a behavior likelihood or enter the policy loss.
                logits.fill_(float("-inf"))
                logits[forced] = 0.0
            return logits

        return process
