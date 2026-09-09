"""Exercise the probe's actual client boundary without loading a model."""

from types import SimpleNamespace

import pytest
from omegaconf import OmegaConf

from skyrl_train.entrypoints.non_agentic_native_probe import NativeModelClient, all_worker_memory, runner_configuration


@pytest.mark.asyncio
async def test_native_client_retains_raw_tokens_likelihoods_and_length_reason(tmp_path):
    cfg = runner_configuration("force_close", SimpleNamespace(eos_token_id=128001))

    class EngineBoundary:
        async def generate(self, prompt, sampling, request_id):
            assert prompt["prompt_token_ids"] == [2, 3]
            assert sampling.max_tokens == 4096 and sampling.stop_token_ids == [128001, 128009]
            assert sampling.extra_args["probe_request_identity"] == request_id == "force_close-00"
            yield SimpleNamespace(
                finished=True,
                outputs=[
                    SimpleNamespace(
                        token_ids=[11, 128003],
                        text="native text",
                        logprobs=[{11: SimpleNamespace(logprob=-0.5)}, {128003: SimpleNamespace(logprob=-12.0)}],
                        finish_reason="length",
                        stop_reason=None,
                    )
                ],
            )

    client = NativeModelClient(
        EngineBoundary(), OmegaConf.to_container(cfg.generator.sampling_params, resolve=True), "force_close", tmp_path
    )
    result = await client.generate({"prompt_token_ids": [[2, 3]], "sampling_params": None})
    assert result["response_ids"] == [[11, 128003]]
    assert result["response_logprobs"] == [[-0.5, -12.0]]
    assert result["stop_reasons"] == ["length"]
    assert client.records[0]["raw_selected_token_logprobs"] == [-0.5, -12.0]
    assert client.records[0]["stop_reason"] is None


@pytest.mark.asyncio
async def test_memory_readback_retains_every_native_core_and_rejects_missing_core():
    class CoreBoundary:
        core_engines = [bytes([index]) for index in range(8)]

        async def _call_utility_async(self, method, callback, timeout, args, kwargs, *, engine):
            index = engine[0]
            assert method == "collective_rpc"
            return [{"pid": index + 1, "data_parallel_rank": index, "gpu_uuid": f"gpu-{index}"}]

    engine = SimpleNamespace(
        engine_core=CoreBoundary(), vllm_config=SimpleNamespace(parallel_config=SimpleNamespace(data_parallel_size=8))
    )
    rows = await all_worker_memory(engine)
    assert [row["core_identity_hex"] for row in rows] == [bytes([index]).hex() for index in range(8)]
    assert [row["data_parallel_rank"] for row in rows] == list(range(8))
    engine.engine_core.core_engines = engine.engine_core.core_engines[:-1]
    with pytest.raises(AssertionError):
        await all_worker_memory(engine)
