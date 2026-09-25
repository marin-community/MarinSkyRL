from unittest.mock import AsyncMock, MagicMock

from omegaconf import OmegaConf
import pytest

from skyrl_train.entrypoints import fully_async
from skyrl_train.trajectory_runners.model_clients import DirectModelClient


@pytest.mark.asyncio
async def test_trajectory_runner_uses_direct_model_client_without_http(monkeypatch):
    cfg = OmegaConf.create(
        {
            "generator": {
                "enable_http_endpoint": False,
                "use_conversation_multi_turn": True,
            },
            "environment": {"skyrl_gym": {}},
        }
    )
    inference_engine_client = MagicMock()
    inference_engine_client.generate = AsyncMock(
        return_value={
            "responses": ["answer"],
            "response_ids": [[3]],
            "stop_reasons": ["stop"],
            "response_logprobs": None,
            "prompt_logprobs": None,
        }
    )
    tokenizer = MagicMock()
    runner = MagicMock(custom_chat_template="template")
    create_runner = MagicMock(return_value=runner)
    monkeypatch.setattr(fully_async, "SkyRLGymTrajectoryRunner", create_runner)

    result = fully_async.AsyncPPOExp.get_trajectory_runner(MagicMock(), cfg, tokenizer, inference_engine_client)

    assert result is runner
    model_client = create_runner.call_args.kwargs["model_client"]
    assert isinstance(model_client, DirectModelClient)
    output = await model_client.generate({"prompt_token_ids": [[1, 2]]})
    assert output["response_ids"] == [[3]]
    assert output["token_provenance"] == "engine"
