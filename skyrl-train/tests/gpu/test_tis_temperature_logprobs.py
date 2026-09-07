"""Opt-in single-GPU parity check with an offline, randomly initialized model.

Run in the pinned vLLM environment:
    pytest skyrl-train/tests/gpu/test_tis_temperature_logprobs.py
"""

import torch
import pytest
from omegaconf import OmegaConf
from transformers import GPT2Config, GPT2LMHeadModel
from vllm import LLM, SamplingParams

from skyrl_train.config.tis import configure_tis_sampling
from skyrl_train.inference_engines.utils import get_vllm_sampling_params
from skyrl_train.inference_engines.vllm.utils import pop_openai_kwargs


@pytest.fixture(scope="module")
def model_and_engine(tmp_path_factory):
    torch.manual_seed(42)
    model = GPT2LMHeadModel(
        GPT2Config(
            vocab_size=64,
            n_positions=32,
            n_embd=32,
            n_layer=2,
            n_head=2,
            bos_token_id=1,
            eos_token_id=2,
            resid_pdrop=0.0,
            embd_pdrop=0.0,
            attn_pdrop=0.0,
        )
    ).eval()
    path = tmp_path_factory.mktemp("tis-model")
    model.save_pretrained(path)
    generator = OmegaConf.create({"sampling_params": {"temperature": 1.0}, "engine_init_kwargs": {}})
    configure_tis_sampling(generator)
    options = OmegaConf.to_container(generator.engine_init_kwargs)
    pop_openai_kwargs(options)
    # CUDA may already be initialized by the test process; workers must start fresh.
    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.setenv("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
        engine = LLM(
            model=str(path),
            skip_tokenizer_init=True,
            dtype="float32",
            enforce_eager=True,
            max_model_len=32,
            gpu_memory_utilization=0.2,
            **options,
        )
        yield model, engine


@pytest.mark.parametrize("temperature", [0.7, 1.2])
def test_same_weights_temperature_scaled_tis_ratios_are_one(model_and_engine, temperature):
    model, engine = model_and_engine
    generator = OmegaConf.create(
        {
            "sampling_params": {
                "temperature": temperature,
                "max_generate_length": 4,
                "logprobs": 0,
                "ignore_eos": True,
                "top_p": 1.0,
                "top_k": -1,
                "min_p": 0.0,
            },
            "engine_init_kwargs": {},
        }
    )
    configure_tis_sampling(generator)
    sampling = SamplingParams(**get_vllm_sampling_params(generator.sampling_params))
    prompts = [[1, 4, 7], [1, 9, 6]]
    outputs = engine.generate([{"prompt_token_ids": prompt} for prompt in prompts], sampling, use_tqdm=False)
    for prompt, output in zip(prompts, outputs, strict=True):
        generated = output.outputs[0]
        tokens = torch.tensor([prompt + list(generated.token_ids)])
        with torch.no_grad():
            logits = model(tokens).logits[0, len(prompt) - 1 : -1].float()
        sampled_ids = torch.tensor(generated.token_ids).unsqueeze(-1)
        trainer = (logits / temperature).log_softmax(-1).gather(-1, sampled_ids).squeeze(-1)
        serving = torch.tensor(
            [position[token].logprob for position, token in zip(generated.logprobs, generated.token_ids, strict=True)]
        )
        ratios = (trainer - serving).exp()
        torch.testing.assert_close(ratios, torch.ones_like(ratios), rtol=1e-3, atol=1e-3)
        # The former raw convention must be distinguishable on the same samples.
        raw = logits.log_softmax(-1).gather(-1, sampled_ids).squeeze(-1)
        assert (trainer - raw).abs().max() > 1e-3
