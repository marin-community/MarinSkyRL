"""The K16 probe's engine constructor must name its logits processor the way vLLM loads it."""

import importlib
import sys
import types

import pytest


@pytest.fixture
def native_probe(monkeypatch):
    """Import the probe without vLLM: only the names its import chain touches are stubbed."""

    class Placeholder:
        def __init__(self, *args, **kwargs):
            pass

    def stub(name, **attributes):
        module = types.ModuleType(name)
        module.__path__ = []
        module.__dict__.update(attributes)
        monkeypatch.setitem(sys.modules, name, module)

    stub("vllm", SamplingParams=Placeholder)
    stub("vllm.engine")
    stub("vllm.engine.arg_utils", AsyncEngineArgs=Placeholder)
    stub("vllm.inputs", TokensPrompt=dict)
    stub("vllm.v1")
    stub("vllm.v1.engine")
    stub("vllm.v1.engine.async_llm", AsyncLLM=Placeholder)
    stub("vllm.v1.sample")
    stub("vllm.v1.sample.logits_processor", AdapterLogitsProcessor=Placeholder, LogitsProcessors=Placeholder)
    stub("vllm.v1.sample.logits_processor.interface", BatchUpdate=Placeholder)
    stub("vllm.v1.sample.metadata", SamplingMetadata=Placeholder)
    stub("vllm.v1.sample.sampler", Sampler=Placeholder)
    for name in [m for m in sys.modules if m.startswith("skyrl_train.entrypoints.non_agentic_")]:
        monkeypatch.delitem(sys.modules, name)
    monkeypatch.delitem(sys.modules, "skyrl_train.inference_engines.non_agentic_logits_processor", raising=False)
    return importlib.import_module("skyrl_train.entrypoints.non_agentic_native_probe")


def test_engine_logits_processor_is_loadable_by_vllm(native_probe):
    arguments = native_probe.engine_arguments({"model_uri": "s3://bucket/model"})
    assert arguments["logprobs_mode"] == "raw_logprobs"
    processors = arguments["logits_processors"]
    assert len(processors) == 1 and isinstance(processors[0], str)
    # vLLM's _load_logitsprocs_by_fqcns: `module_path, qualname = logitproc.split(":")`
    module_path, qualname = processors[0].split(":")
    loaded = getattr(importlib.import_module(module_path), qualname)
    assert loaded is native_probe.TimedNonAgenticTokenProcessor
    assert issubclass(loaded, native_probe.NonAgenticTokenProcessor)
