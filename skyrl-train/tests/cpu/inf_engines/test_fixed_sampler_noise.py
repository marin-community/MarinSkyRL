import torch
import pytest

from skyrl_train.inference_engines.vllm.fixed_sampler_noise import FixedSamplerNoise


def test_noise_uses_initial_weights_after_multiple_fresh_syncs_and_restores_exact_clean_values():
    original = torch.arange(1, 33, dtype=torch.float32).reshape(4, 8)
    parameter = torch.nn.Parameter(original.clone())
    noise = FixedSamplerNoise([("model.weight", parameter)], 0.05, 17)
    delta = parameter.detach().clone() - original
    assert delta.abs().max() > 0
    identity = noise.evidence()
    for fresh in (original * 2, original * 3):
        with torch.no_grad():
            parameter.copy_(fresh)
        noise.after_sync()
        torch.testing.assert_close(parameter, fresh + delta, atol=1e-5, rtol=0)
        noise.set_enabled(False)
        torch.testing.assert_close(parameter, fresh, atol=0, rtol=0)
        noise.set_enabled(True)
        torch.testing.assert_close(parameter, fresh + delta, atol=1e-5, rtol=0)
        assert noise.evidence() == identity


def test_same_initial_model_and_seed_give_same_delta_in_independent_sampler_replicas():
    values = torch.randn(9, 7, generator=torch.Generator().manual_seed(42))
    a, b = torch.nn.Parameter(values.clone()), torch.nn.Parameter(values.clone())
    FixedSamplerNoise([("model.weight", a)], 0.05, 2)
    FixedSamplerNoise([("model.weight", b)], 0.05, 2)
    torch.testing.assert_close(a, b, atol=0, rtol=0)


@pytest.mark.parametrize("scale", [0.0, 0.05])
def test_clean_current_evaluation_restores_frozen_bf16_sampler_and_its_original_base(scale):
    initial = torch.arange(1, 17, dtype=torch.bfloat16).reshape(4, 4)
    parameter = torch.nn.Parameter(initial.clone())
    noise = FixedSamplerNoise([("model.weight", parameter)], scale, 2)
    frozen = parameter.detach().clone()
    identity = noise.evidence()

    noise.snapshot_for_evaluation()
    with torch.no_grad():
        parameter.copy_(initial * 2)
    noise.after_sync()
    noise.set_enabled(False)
    torch.testing.assert_close(parameter, initial * 2, atol=0, rtol=0)

    noise.restore_after_evaluation()
    torch.testing.assert_close(parameter, frozen, atol=0, rtol=0)
    noise.set_enabled(False)
    torch.testing.assert_close(parameter, initial, atol=0, rtol=0)
    noise.set_enabled(True)
    torch.testing.assert_close(parameter, frozen, atol=0, rtol=0)
    assert noise.evidence() == identity
