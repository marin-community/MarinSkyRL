from skyrl_train.inference_engines.vllm.utils import pop_vllm_wrapper_kwargs


def test_pop_vllm_wrapper_kwargs_normalizes_aliases_and_preserves_engine_options():
    engine_kwargs = {
        "enable_auto_tools": 1,
        "tool_parser": "json",
        "other": "keep",
    }
    wrapper_kwargs = pop_vllm_wrapper_kwargs(engine_kwargs)

    assert wrapper_kwargs == {"enable_auto_tools": True, "tool_parser": "json"}
    assert engine_kwargs == {"other": "keep"}

    engine_kwargs = {"enable_auto_tool_choice": 0, "tool_call_parser": "proto"}
    wrapper_kwargs = pop_vllm_wrapper_kwargs(engine_kwargs)

    assert wrapper_kwargs == {"enable_auto_tools": False, "tool_parser": "proto"}
    assert engine_kwargs == {}
