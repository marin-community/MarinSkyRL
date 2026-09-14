from unittest.mock import MagicMock

from omegaconf import OmegaConf

from skyrl_train.entrypoints import fully_async


def test_trajectory_runner_uses_resolved_served_model_name(monkeypatch):
    cfg = OmegaConf.create(
        {
            "generator": {
                "enable_http_endpoint": True,
                "use_conversation_multi_turn": True,
                "http_endpoint_host": "127.0.0.1",
                "http_endpoint_port": 8000,
            },
            "environment": {"skyrl_gym": {}},
        }
    )
    inference_engine_client = MagicMock(model_name="served-policy")
    tokenizer = MagicMock()
    model_client = MagicMock()
    runner = MagicMock(custom_chat_template="template")
    create_model_client = MagicMock(return_value=model_client)
    create_runner = MagicMock(return_value=runner)
    monkeypatch.setattr(fully_async, "OpenAIHTTPModelClient", create_model_client)
    monkeypatch.setattr(fully_async, "SkyRLGymTrajectoryRunner", create_runner)

    result = fully_async.AsyncPPOExp.get_trajectory_runner(MagicMock(), cfg, tokenizer, inference_engine_client)

    assert result is runner
    create_model_client.assert_called_once_with(
        base_url="http://127.0.0.1:8000",
        model_name="served-policy",
        tokenizer=tokenizer,
    )
    assert create_runner.call_args.kwargs["model_client"] is model_client
