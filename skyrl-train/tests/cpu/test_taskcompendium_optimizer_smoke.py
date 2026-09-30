"""One CPU policy update from a mixed TaskCompendium rollout."""

import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Thread
from unittest.mock import AsyncMock

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf
from taskcompendium.grading import exact_answer
from taskcompendium.lowering import HarborEnvironmentConfig, lower_to_harbor
from taskcompendium.models import AnswerType, ConversationInput, EnvironmentRequirements, Source, TaskSpec, TextMessage
from taskcompendium.submission import PlainText
from tokenizers import Tokenizer, models, pre_tokenizers
from transformers import PreTrainedTokenizerFast

from skyrl_train.dataset.preprocess import convert_prompts_responses_to_batch_tensors
from skyrl_train.trajectory_runners.taskcompendium import (
    NativeTaskCompendiumRunner,
    TaskCompendiumHarborRunner,
    TaskCompendiumTaskDataset,
    TaskCompendiumTrajectoryRouter,
)
from skyrl_train.trajectory_runners.types import TrajectoryID
from skyrl_train.utils.advantage_estimators import compute_advantages_and_returns
from skyrl_train.utils.algorithm_registry import AdvantageEstimator
from skyrl_train.utils.policy_losses import LossScaling, compute_policy_objective, ppo_policy_loss
from skyrl_train.config.utils import get_default_config


@pytest.mark.integrations
@pytest.mark.asyncio
async def test_mixed_taskcompendium_rollout_updates_cpu_policy(
    tmp_path: Path, trusted_workplace_checkout: Path, workplace_import, workplace_source_row: bytes
):
    pytest.importorskip("harbor")
    chat = TaskSpec(
        id="chat",
        context=ConversationInput(events=(TextMessage(role="user", content="Reply with blue."),)),
        verifier=exact_answer("blue"),
        source=Source(dataset="test", revision="revision", row="chat", importer_revision="importer"),
        environment_requirements=EnvironmentRequirements(),
        answer_type=AnswerType.TEXT,
    )
    plain = PlainText(id="plain")
    lower_to_harbor(chat, plain, HarborEnvironmentConfig(), tmp_path / "chat")
    workplace, convention, binding = workplace_import
    lower_to_harbor(
        workplace,
        convention,
        binding,
        tmp_path / "workplace",
        trusted_provider_sources={"workplace": trusted_workplace_checkout},
    )
    source_row = json.loads(workplace_source_row)
    gold = source_row["ground_truth"][0]
    requests = []

    class Endpoint(BaseHTTPRequestHandler):
        def log_message(self, *_args):
            pass

        def do_POST(self):
            requests.append(json.loads(self.rfile.read(int(self.headers["Content-Length"]))))
            message = (
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call-1",
                            "type": "function",
                            "function": {"name": gold["name"], "arguments": gold["arguments"]},
                        }
                    ],
                }
                if len(requests) == 1
                else {"role": "assistant", "content": "Done."}
            )
            body = json.dumps({"choices": [{"message": message}]}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    backend = Tokenizer(models.WordLevel({"[UNK]": 0, "[EOS]": 1}, unk_token="[UNK]"))
    backend.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend, unk_token="[UNK]", eos_token="[EOS]", pad_token="[UNK]"
    )
    tokenizer.chat_template = (
        "{% for message in messages %}{{ message['role'] }}: {{ message['content'] or '' }}"
        "{% if message.get('tool_calls') %}{{ message['tool_calls'] }}{% endif %} [EOS] {% endfor %}"
        "{% if add_generation_prompt %}assistant: {% endif %}"
    )
    client = AsyncMock()
    client.generate.return_value = {
        "responses": ["red"],
        "response_ids": [[0, 1]],
        "stop_reasons": ["stop"],
    }
    cfg = OmegaConf.create({"sampling_params": {"temperature": 1.0}, "chat_template": None, "chat_template_kwargs": {}})
    server = ThreadingHTTPServer(("127.0.0.1", 0), Endpoint)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        dataset = TaskCompendiumTaskDataset(
            [str(tmp_path / "chat"), str(tmp_path / "workplace")],
            api_base=f"http://127.0.0.1:{server.server_port}/v1",
            model_name="fixture",
        )
        router = TaskCompendiumTrajectoryRouter(
            native_runner=NativeTaskCompendiumRunner(cfg, tokenizer, client),
            harbor_runner=TaskCompendiumHarborRunner(
                tokenizer, tmp_path / "trials", concurrency=1, max_turns=3, timeout=30
            ),
            require_rollout_logprobs=False,
            tis_lcs_alert_threshold=0.005,
        )
        batch = {
            "prompts": [row["prompt"] for row in dataset],
            "env_classes": [row["env_class"] for row in dataset],
            "env_extras": [row["env_extras"] for row in dataset],
            "sampling_params": {"temperature": 1.0},
            "trajectory_ids": [TrajectoryID(row["uid"], 0) for row in dataset],
            "batch_metadata": None,
        }
        await router.startup()
        try:
            rollout = await router.run(batch)
        finally:
            await router.shutdown()
    finally:
        server.shutdown()
        server.server_close()
        thread.join()

    assert rollout["rewards"] == [0.0, 1.0]
    assert len(requests) == 2
    reward_tokens = [
        [0.0] * (len(tokens) - 1) + [reward]
        for tokens, reward in zip(rollout["response_ids"], rollout["rewards"], strict=True)
    ]
    sequences, _attention, response_mask, rewards, loss_mask, rollout_logprobs, *_ = (
        convert_prompts_responses_to_batch_tensors(
            tokenizer,
            rollout["prompt_token_ids"],
            rollout["response_ids"],
            reward_tokens,
            rollout["loss_masks"],
        )
    )
    assert rollout_logprobs is None
    assert loss_mask.sum(dim=1).gt(0).all()

    training_config = get_default_config()
    OmegaConf.update(training_config, "trainer.algorithm.max_seq_len", sequences.shape[1], force_add=True)
    algorithm = training_config.trainer.algorithm
    algorithm.policy_loss_type = "regular"
    algorithm.loss_reduction = "token_mean"
    algorithm.use_tis = False
    algorithm.use_kl_loss = False
    advantages, _ = compute_advantages_and_returns(
        token_level_rewards=rewards,
        response_mask=response_mask,
        index=np.array(["chat", "workplace"]),
        adv_estimator=AdvantageEstimator.REINFORCE_PP,
        config=algorithm,
        values=None,
        gamma=1.0,
        lambd=1.0,
        grpo_norm_by_std=True,
    )
    assert advantages[0].lt(0).any() and advantages[1].gt(0).any()

    torch.manual_seed(0)
    model = torch.nn.Sequential(torch.nn.Embedding(tokenizer.vocab_size, 8), torch.nn.Linear(8, tokenizer.vocab_size))
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2)
    before = [parameter.detach().clone() for parameter in model.parameters()]
    logits = model(sequences[:, :-1])
    response_length = response_mask.shape[1]
    log_probs = logits.log_softmax(dim=-1).gather(-1, sequences[:, 1:].unsqueeze(-1)).squeeze(-1)
    action_log_probs = log_probs[:, -response_length:]
    objective = compute_policy_objective(
        action_log_probs=action_log_probs,
        old_action_log_probs=action_log_probs.detach(),
        base_action_log_probs=None,
        advantages=advantages,
        loss_mask=loss_mask,
        rollout_logprobs=None,
        response_span_tags=None,
        token_entropy=torch.zeros_like(action_log_probs),
        config=algorithm,
        policy_loss_fn=ppo_policy_loss,
        accumulation_steps=1,
        scaling=LossScaling.CALLER,
    )
    assert torch.isfinite(objective.optimization_loss)
    objective.optimization_loss.backward()
    optimizer.step()

    assert any(not torch.equal(old, current) for old, current in zip(before, model.parameters(), strict=True))
