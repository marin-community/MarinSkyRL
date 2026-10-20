from datetime import timedelta
from importlib.metadata import version
from pathlib import Path

import torch
import torch.distributed as dist
from megatron.bridge import AutoBridge
from megatron.core import parallel_state as mpu
from megatron.core.distributed import DistributedDataParallelConfig
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed

import skyrl_train
import skyrl_train.models.grug_megatron_bridge  # noqa: F401 - registers the Grug bridge
from skyrl_train.distributed.megatron.grug_muonh import MegatronGrugMuonH
from skyrl_train.distributed.megatron.optimizer import get_megatron_optimizer
from skyrl_train.models.grug_moe import GrugMoeConfig
from skyrl_train.workers.megatron.bridge_provider_config import configure_bridge_provider
from tests.gpu.utils import run_distributed_with_deadline


def run_clipping_rank(rank, world_size, rendezvous):
    assert Path(skyrl_train.__file__).resolve().parent == Path(__file__).resolve().parents[3] / "skyrl_train"
    torch.cuda.set_device(rank)
    dist.init_process_group(
        "nccl", init_method=rendezvous, rank=rank, world_size=world_size, timeout=timedelta(seconds=120)
    )
    try:
        mpu.initialize_model_parallel()
        model_parallel_cuda_manual_seed(31)
        config = GrugMoeConfig(
            vocab_size=128,
            hidden_size=64,
            intermediate_size=64,
            shared_expert_intermediate_size=64,
            num_hidden_layers=2,
            num_local_experts=2,
            num_experts_per_tok=1,
            num_attention_heads=2,
            num_key_value_heads=1,
            head_dim=32,
            max_position_embeddings=32,
            sliding_window=16,
            qk_mult=1.5703274004183786,
            architectures=["GrugMoeForCausalLM"],
        )
        for max_grad_norm in (1.0, 0.0):
            provider = AutoBridge.from_hf_config(config).to_megatron_provider(load_weights=False)
            configure_bridge_provider(
                provider,
                tensor_model_parallel_size=1,
                pipeline_model_parallel_size=1,
                context_parallel_size=1,
                expert_model_parallel_size=1,
                expert_tensor_parallel_size=1,
                bf16=True,
                attention_backend="fused",
            )
            provider.seq_length = 16
            provider.finalize()
            models = provider.provide_distributed_model(
                ddp_config=DistributedDataParallelConfig(
                    use_distributed_optimizer=False, grad_reduce_in_fp32=True, overlap_grad_reduce=False
                )
            )
            optimizer = get_megatron_optimizer(
                models,
                {"optimizer": "MuonH", "lr": 1e-4, "weight_decay": 0.0, "max_grad_norm": max_grad_norm},
                {},
            )
            for model in models:
                model.zero_grad_buffer()
            # A linear objective gives every trainable Grug parameter a known, nonzero gradient.
            loss = sum(
                (parameter.float() * (index + 1) / 16).sum()
                for model in models
                for index, parameter in enumerate(model.parameters())
                if parameter.requires_grad
            )
            loss.backward()
            for model in models:
                model.finish_grad_sync()
            assert not optimizer.prepare_grads()
            children = optimizer.chained_optimizers
            before = [
                (child, parameter, parameter.grad.clone()) for child in children for parameter in child.get_parameters()
            ]
            original_norm = torch.linalg.vector_norm(torch.cat([gradient.flatten() for _, _, gradient in before]))
            assert original_norm > 1
            torch.testing.assert_close(torch.tensor(float(optimizer.get_grad_norm()), device="cuda"), original_norm)
            success, reported_norm, _ = optimizer.step()
            assert success
            torch.testing.assert_close(torch.tensor(float(reported_norm), device="cuda"), original_norm)
            factor = max_grad_norm / original_norm if max_grad_norm else torch.ones_like(original_norm)
            recovered = []
            routes = set()
            for child, parameter, gradient in before:
                state = child.optimizer.state[parameter]
                if isinstance(child.optimizer, MegatronGrugMuonH) and "momentum_buffer" in state:
                    consumed_gradient = state["momentum_buffer"]
                    routes.add("muonh")
                else:
                    consumed_gradient = state["exp_avg"] / (1 - child.config.adam_beta1)
                    routes.add("adamh" if isinstance(child.optimizer, MegatronGrugMuonH) else "adam")
                torch.testing.assert_close(consumed_gradient, gradient * factor, rtol=1e-5, atol=1e-7)
                recovered.append(consumed_gradient.flatten())
            assert {"muonh", "adam"} <= routes
            clipped_norm = torch.linalg.vector_norm(torch.cat(recovered))
            expected_norm = torch.tensor(max_grad_norm, device="cuda") if max_grad_norm else original_norm
            torch.testing.assert_close(clipped_norm, expected_norm, rtol=1e-5, atol=1e-6)
            print(
                f"Grug clipping max_grad_norm={max_grad_norm} original_norm={original_norm.item():.8g} "
                f"consumed_norm={clipped_norm.item():.8g} routes={sorted(routes)}",
                flush=True,
            )
            del optimizer, models
    finally:
        mpu.destroy_model_parallel()
        dist.destroy_process_group()


def test_grug_muonh_and_adam_share_global_clipping(tmp_path):
    assert torch.cuda.device_count() >= 1, "Run on an allocation with one GPU"
    print(
        f"GPU={torch.cuda.get_device_name()} torch={torch.__version__} megatron-core={version('megatron-core')}",
        flush=True,
    )
    run_distributed_with_deadline(run_clipping_rank, (1, (tmp_path / "clipping").as_uri()), 1)
