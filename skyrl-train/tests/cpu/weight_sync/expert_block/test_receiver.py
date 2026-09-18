"""The receiver refuses models it cannot write into in place, and syncs after its parameters were reallocated."""

from types import SimpleNamespace

import pytest
import torch

from skyrl_train.weight_sync.expert_block.receiver import ExpertBlockReceiver
from skyrl_train.weight_sync.expert_block.stream import InstallReport


class RoutedExperts(torch.nn.Module):
    def __init__(self, backend="TRITON"):
        super().__init__()
        self.w13_weight = torch.nn.Parameter(torch.zeros(2, 4, 3, dtype=torch.bfloat16), requires_grad=False)
        self.w2_weight = torch.nn.Parameter(torch.zeros(2, 3, 2, dtype=torch.bfloat16), requires_grad=False)
        self.quant_method = SimpleNamespace(unquantized_backend=SimpleNamespace(name=backend))

    def _map_global_expert_id_to_local_expert_id(self, expert):
        return expert if expert < 2 else -1


class PaddedVocab(torch.nn.Module):
    """The attributes of vLLM's VocabParallelEmbedding that the receiver reads."""

    def __init__(self, rows=5, padded=8, dim=3, tp_size=1, start=0):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(padded, dim, dtype=torch.bfloat16), requires_grad=False)
        self.org_vocab_size = rows
        self.num_embeddings_padded = padded
        self.tp_size = tp_size
        self.shard_indices = SimpleNamespace(org_vocab_start_index=start)


class Model(torch.nn.Module):
    def __init__(self, layers=1, backend="TRITON", vocab=None):
        super().__init__()
        self.model = torch.nn.Module()
        if vocab is not None:
            self.model.embed_tokens = vocab
        self.model.layers = torch.nn.ModuleList()
        for _ in range(layers):
            layer = torch.nn.Module()
            layer.mlp = torch.nn.Module()
            layer.mlp.experts = torch.nn.Module()
            layer.mlp.experts.routed_experts = RoutedExperts(backend)
            layer.mlp.router = torch.nn.Module()
            layer.mlp.router.weight = torch.nn.Parameter(torch.zeros(4, 3), requires_grad=False)
            self.model.layers.append(layer)


def vllm_config(model_type="grug_moe", quantization=None, tp=1, eplb=False, layers=1):
    hf = SimpleNamespace(
        model_type=model_type, num_experts=4, hidden_size=3, moe_intermediate_size=2, num_hidden_layers=layers
    )
    return SimpleNamespace(
        model_config=SimpleNamespace(hf_config=hf, quantization=quantization),
        parallel_config=SimpleNamespace(tensor_parallel_size=tp, pipeline_parallel_size=1, enable_eplb=eplb),
    )


def receiver(config=None, model=None):
    return ExpertBlockReceiver(
        config or vllm_config(),
        torch.device("cpu"),
        model or Model(),
        ep_rank=0,
        ep_size=2,
        pp_rank=0,
        pp_size=1,
        gpu_uuid="GPU-0",
    )


def test_inventory_reports_the_dense_parameters_and_serving_map():
    report = receiver().inventory()
    assert report["ep_rank"] == 0 and report["expert_parallel_size"] == 2
    assert (report["pp_rank"], report["pp_size"], report["layers"]) == (0, 1, [0])
    assert list(report["dense"]) == ["model.layers.0.mlp.router.weight"]
    assert report["model"]["num_hidden_layers"] == 1


def test_padded_vocabulary_tensors_report_their_hf_rows():
    report = receiver(model=Model(vocab=PaddedVocab(rows=5, padded=8))).inventory()
    assert report["dense"]["model.embed_tokens.weight"] == [[5, 3], "bfloat16"]


@pytest.mark.parametrize(
    "config,model,message",
    [
        (None, Model(vocab=PaddedVocab(tp_size=2)), "must run TP=1"),
        (None, Model(vocab=PaddedVocab(start=4)), "must run TP=1"),
        (vllm_config(model_type="qwen3_moe"), None, "supports grug_moe"),
        (vllm_config(quantization="fp8"), None, "unquantised"),
        (vllm_config(tp=2), None, "TP=1"),
        (vllm_config(eplb=True), None, "no EPLB"),
        (None, Model(backend="FLASHINFER_CUTLASS"), "requires the TRITON MoE backend"),
        (vllm_config(layers=2), None, "Found 1 routed-expert layers, expected 2"),
    ],
)
def test_models_this_transport_cannot_write_into_are_refused(config, model, message):
    with pytest.raises(ValueError, match=message):
        receiver(config, model).inventory()


def test_a_sync_after_parameters_were_reallocated_is_refused():
    model = Model()
    worker = receiver(model=model)
    worker.inventory()
    worker.stream = SimpleNamespace(run=lambda version: InstallReport(4, version, 2, 96, 0.01))
    assert worker.receive_weights({"version": 2})["version"] == 2
    model.model.layers[0].mlp.router.weight.data = torch.zeros(4, 3)
    with pytest.raises(RuntimeError, match="storage changed"):
        worker.receive_weights({"version": 3})
