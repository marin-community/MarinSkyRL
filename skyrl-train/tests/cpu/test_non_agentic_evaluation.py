import copy

import pytest
from omegaconf import OmegaConf
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import WhitespaceSplit
from transformers import PreTrainedTokenizerFast

from skyrl_train.config.utils import get_default_config
from skyrl_train.non_agentic_evaluation import evaluate_endpoints, request_endpoint
from skyrl_train.non_agentic_evaluation_metrics import endpoint_contract_metrics
from skyrl_train.trajectory_runners.skyrl_gym import SkyRLGymTrajectoryRunner
from skyrl_train.trajectory_runners.types import BatchMetadata, TokenProvenance


@pytest.mark.parametrize(
    "phase,mode,parser",
    [
        ("train", "common_off", "post-thinking-native-v1"),
        ("eval", "invalid", "post-thinking-native-v1"),
        ("eval", "common_off", None),
    ],
)
def test_endpoint_cannot_disable_intervention_on_a_training_or_wrong_protocol_request(phase, mode, parser):
    with pytest.raises(ValueError):
        request_endpoint({"batch_metadata": BatchMetadata(25, phase), "non_agentic_evaluation_endpoint": mode}, parser)


def setup_runner(*, intervention=False, shaped=False, parser=True):
    words = ["[UNK]", "<|start_think|>", "<|end_think|>", "<|eot_id|>", "####", "41", "42", "question"]
    decoder = Tokenizer(WordLevel(dict(zip(words, range(len(words)), strict=True)), unk_token="[UNK]"))
    decoder.pre_tokenizer = WhitespaceSplit()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=decoder,
        unk_token="[UNK]",
        eos_token="<|eot_id|>",
        additional_special_tokens=["<|start_think|>", "<|end_think|>"],
    )
    tokenizer.chat_template = "{% for m in messages %}{{ m['content'] }} <|eot_id|> {% endfor %}{% if add_generation_prompt %}<|start_think|>{% endif %}"
    cfg = get_default_config()
    cfg.generator.non_agentic_parser_protocol = "post-thinking-native-v1" if parser else None
    cfg.generator.max_turns = 1
    cfg.generator.batched = False
    cfg.generator.use_conversation_multi_turn = True
    cfg.generator.sampling_params.logprobs = 0
    cfg.generator.trajectory_retention.enabled = False
    cfg.environment.skyrl_gym.max_env_workers = 0
    tokens = tokenizer.encode("#### 41 <|end_think|> #### 42 <|eot_id|>", add_special_tokens=False)
    calls = []

    class Client:
        async def generate(self, request):
            calls.append(copy.deepcopy(request))
            return {
                "responses": [tokenizer.decode(tokens, skip_special_tokens=True)],
                "response_ids": [list(tokens)],
                "response_logprobs": [[-0.5] * len(tokens)],
                "prompt_logprobs": None,
                "stop_reasons": ["stop"],
                "token_provenance": TokenProvenance.ENGINE,
            }

    if intervention:
        cfg.generator.non_agentic_intervention = dict(
            protocol="non-agentic-token-intervention-v1",
            kind="force_close",
            thinking_end_id=2,
            eos_id=3,
            force_close_after=2,
        )
        cfg.generator.engine_init_kwargs = {
            "logits_processors": [
                "skyrl_train.inference_engines.non_agentic_logits_processor.NonAgenticTokenProcessor"
            ],
            "logprobs_mode": "raw_logprobs",
        }
    if shaped:
        cfg.generator.trajectory_reward_shaping.enabled = True
        cfg.generator.trajectory_reward_shaping.overlong.l_max = len(tokens) + 2
        cfg.generator.trajectory_reward_shaping.overlong.l_cache = 4
    runner = SkyRLGymTrajectoryRunner(cfg.generator, cfg.environment.skyrl_gym, None, tokenizer, model_client=Client())
    extras = {
        "reward_spec": {"ground_truth": "42"},
        "reward_model": {"ground_truth": "42"},
        "extra_info": {"contract": "gsm8k-first-hash-v1"},
        "data_source": "gsm",
    }
    request = {
        "prompts": [[{"role": "user", "content": "question"}]],
        "env_classes": ["gsm8k"],
        "env_extras": [extras],
        "sampling_params": None,
        "trajectory_ids": None,
        "batch_metadata": BatchMetadata(25, "eval"),
    }
    return runner, tokenizer, request, calls


@pytest.mark.asyncio
async def test_actual_runner_request_scope_preserves_package_after_common_off():
    runner, tokenizer, request, calls = setup_runner(intervention=True)
    configured = runner.token_intervention
    for mode in ["package_on", "common_off", "package_on"]:
        batch = await runner.run({**request, "non_agentic_evaluation_endpoint": mode}, disable_tqdm=True)
        forced = mode == "package_on"
        assert ("non_agentic_intervention" in (calls[-1]["sampling_params"] or {}).get("extra_args", {})) == forced
        contract = batch["non_agentic_contract"][0]
        assert contract["evaluation_endpoint"] == mode
        assert ("intervention" in contract) == forced
        assert batch["loss_masks"][0][2] == int(not forced)
        assert batch["rollout_logprobs"][0] == [-0.5] * len(batch["response_ids"][0])
        assert (
            endpoint_contract_metrics(
                request["env_classes"], request["env_extras"], batch, tokenizer, "post-thinking-native-v1"
            )["eval/all/contract_completed"]
            == 1
        )
        assert runner.token_intervention is configured


@pytest.mark.asyncio
async def test_shaped_reward_does_not_change_correctness_and_corrupted_verifier_rejects():
    runner, tokenizer, request, _ = setup_runner(shaped=True)
    batch = await runner.run(request, disable_tqdm=True)
    before = copy.deepcopy(batch)
    metrics = endpoint_contract_metrics(
        request["env_classes"], request["env_extras"], batch, tokenizer, "post-thinking-native-v1"
    )
    assert metrics["eval/all/score_contract"] == 0.5
    assert metrics["eval/all/corrected_verifier_reward"] == metrics["eval/all/contract_completed"] == 1
    assert metrics["eval/all/legacy_full_text_reward_diagnostic"] == 0
    assert batch == before
    batch["non_agentic_contract"][0]["verifier_reward"] = 0.5
    with pytest.raises(ValueError, match="token-delimited"):
        endpoint_contract_metrics(
            request["env_classes"], request["env_extras"], batch, tokenizer, "post-thinking-native-v1"
        )


@pytest.mark.asyncio
async def test_legacy_endpoint_dispatch_keeps_original_full_text_contract():
    runner, tokenizer, request, _ = setup_runner(parser=False)
    batch = await runner.run(request, disable_tqdm=True)
    metrics = endpoint_contract_metrics(request["env_classes"], request["env_extras"], batch, tokenizer, None)
    assert metrics["eval/all/contract_completed"] == 0
    assert "eval/all/corrected_verifier_reward" not in metrics and "non_agentic_contract" not in batch


@pytest.mark.asyncio
@pytest.mark.parametrize("intervention,shaped", [(False, False), (True, False), (False, True)])
async def test_real_evaluation_persists_separate_endpoint_token_reward_and_mask_evidence(
    tmp_path, intervention, shaped
):
    import json
    from skyrl_train.evaluate import evaluate

    runner, tokenizer, request, calls = setup_runner(intervention=intervention, shaped=shaped)
    cfg = get_default_config()
    cfg.generator = copy.deepcopy(runner.trajectory_runner_cfg)
    cfg.generator.eval_sampling_params.logprobs = 0
    cfg.generator.non_agentic_eval_endpoints = ["common_off", "package_on"] if intervention else ["package_on"]
    cfg.trainer.dump_eval_results = True
    cfg.trainer.completion = {"request_fingerprint": "a" * 64}
    cfg.trainer.export_path = str(tmp_path)

    class Loader:
        def __len__(self):
            return 1

        def __iter__(self):
            return iter(
                [
                    [
                        {
                            "prompt": request["prompts"][0],
                            "uid": "q1",
                            "env_class": "gsm8k",
                            "env_extras": request["env_extras"][0],
                        }
                    ]
                ]
            )

    original_config = copy.deepcopy(cfg)
    metrics = await evaluate_endpoints(
        evaluate,
        cfg=cfg,
        policy_version=25,
        eval_dataloader=Loader(),
        trajectory_runner=runner,
        global_step=25,
        tokenizer=tokenizer,
    )
    assert cfg == original_config
    assert metrics["eval/package_on/all/contract_completed"] == 1
    assert len(calls) == (2 if intervention else 1)
    for mode in cfg.generator.non_agentic_eval_endpoints:
        path = tmp_path / "dumped_evals/global_step_25_evals" / mode
        row = json.loads((path / "gsm.jsonl").read_text())
        assert row["non_agentic_contract"]["evaluation_endpoint"] == mode
        assert row["non_agentic_contract"]["metric_protocol"] == "post-thinking-native-metrics-v1"
        assert row["non_agentic_contract"]["verifier_reward"] == 1
        assert sum(row["score"]) == (0.5 if shaped else 1)
        assert row["behavior_logprobs"] == [-0.5] * len(row["response_ids"])
        assert row["policy_action_mask"][2] == int(not (intervention and mode == "package_on"))
        aggregate = json.loads((path / "aggregated_results.jsonl").read_text())
        assert aggregate["eval/all/contract_completed"] == 1
        assert aggregate["eval/all/score_contract"] == (0.5 if shaped else 1)
        metadata = json.loads((path / "endpoint_metadata.json").read_text())
        assert metadata["endpoint"] == mode and metadata["dump_namespace"] == mode
        assert metadata["global_step"] == metadata["policy_version"] == 25
        assert metadata["parser_protocol"] == "post-thinking-native-v1"
        assert metadata["request_fingerprint"] == cfg.trainer.completion.request_fingerprint
    assert (tmp_path / "dumped_evals/global_step_25_evals/common_off").exists() == intervention


@pytest.mark.asyncio
@pytest.mark.parametrize("setting", [{"eval_mode": "background"}, {"eval_on_installed_weights": True}])
async def test_quality_endpoints_reject_background_or_stale_installed_weights(setting):
    cfg = OmegaConf.create(
        {
            "trainer": {"step_wise_training": False, "fully_async": setting},
            "generator": {
                "non_agentic_parser_protocol": "post-thinking-native-v1",
                "non_agentic_eval_endpoints": ["package_on"],
            },
        }
    )

    async def forbidden(**kwargs):
        pytest.fail("Unqualified endpoint generator was invoked")

    with pytest.raises(ValueError, match="freshly published"):
        await evaluate_endpoints(forbidden, cfg=cfg)


@pytest.mark.parametrize(
    "env,answer,expected",
    [("aime", "41", -1.0), ("reasoning_gym", "42 reasoning", 2 / 12), ("reasoning_gym", "42", 1.0)],
)
@pytest.mark.parametrize("stop", ["stop", "length"])
def test_corrected_endpoint_preserves_signed_fractional_and_completed_channels(env, answer, expected, stop):
    import json
    from dataclasses import asdict
    from skyrl_gym.envs.thinking_contract import score_thinking_contract

    _, tokenizer, _, _ = setup_runner()
    tokenizer.add_tokens(["Answer:", "reasoning"])
    tokens = tokenizer.encode("<|end_think|> Answer: " + answer + " <|eot_id|>", add_special_tokens=False)
    gold = (
        "42"
        if env == "aime"
        else json.dumps(
            dict(
                task="chain_sum", entry=dict(question="41 + 1", answer="42", metadata=dict(source_dataset="chain_sum"))
            )
        )
    )
    receipt = asdict(
        score_thinking_contract(
            env_class=env,
            ground_truth=gold,
            native_response=tokenizer.decode(tokens, skip_special_tokens=True),
            prompt_tokens=[1],
            response_tokens=tokens,
            stop_reason=stop,
            decoder=tokenizer.backend_tokenizer,
        )
    )
    receipt["metric_protocol"] = "post-thinking-native-metrics-v1"
    batch = {
        "response_ids": [tokens],
        "prompt_token_ids": [[1]],
        "rewards": [[expected]],
        "unshaped_rewards": [expected],
        "stop_reasons": [stop],
        "non_agentic_contract": [receipt],
    }
    extra = {
        "reward_model": {"ground_truth": gold},
        "reward_spec": {"ground_truth": gold},
        "data_source": env,
        "extra_info": {
            "contract": "aime-last-answer-300-v1" if env == "aime" else "reasoning-gym-0.1.25-last-answer-v1"
        },
    }
    metrics = endpoint_contract_metrics([env], [extra], batch, tokenizer, "post-thinking-native-v1")
    assert metrics["eval/all/score_contract"] == metrics["eval/all/corrected_verifier_reward"] == expected
    assert metrics["eval/all/contract_correct"] == int(expected == 1)
    assert metrics["eval/all/contract_completed"] == int(expected == 1 and stop == "stop")


@pytest.mark.asyncio
async def test_endpoint_refuses_stale_installed_version_before_generation(tmp_path):
    cfg = get_default_config()
    cfg.generator.non_agentic_parser_protocol = "post-thinking-native-v1"
    cfg.generator.non_agentic_eval_endpoints = ["package_on"]
    cfg.trainer.export_path = str(tmp_path)
    cfg.trainer.completion = {"request_fingerprint": "a" * 64}

    async def forbidden(**kwargs):
        pytest.fail("Stale endpoint generated responses")

    with pytest.raises(ValueError, match="observed installed version"):
        await evaluate_endpoints(forbidden, cfg=cfg, policy_version=24, global_step=25)
    assert list(tmp_path.iterdir()) == []
