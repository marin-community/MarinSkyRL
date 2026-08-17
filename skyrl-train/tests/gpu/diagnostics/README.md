# Backend numerics diagnostics

These operator-run tests localize FSDP2 and Megatron training divergence. They are deliberately outside
`gpu_ci` and use filenames without the `test_` prefix, so ordinary pytest discovery and PR CI do not run them.
Every command writes `discrepancies.csv` and `results.json` below `SKYRL_NUMERICS_ARTIFACT_DIR`.

Run all three diagnostics on one four-GPU node with the production Megatron runtime:

```bash
sbatch tests/gpu/diagnostics/run_backend_numerics_jupiter.sbatch
```

T1 runs the same serialized toy-MoE batch through an fp32 oracle, the production TorchTitan EP wrapper at
EP=4, and Megatron Core's production gradient-norm helper. Its per-tensor output includes raw-gradient cosine,
norm ratio, a fitted scale and scale-removed residual, and the cosine and norm ratio of a simulated first Adam
step. T2 checks eager, compiled, chunked, inference-only, TP=1, and TP=2 log-probability variants against an
fp64 `log_softmax` oracle. T3 reports per-tensor gradient cosines and norm ratios for the SkyRL for-loop and
grouped-MM experts, Megatron Core's Transformer Engine grouped experts, Hugging Face FlashAttention2, and
Transformer Engine attention.

The checked-in Jupiter batch file combines the validated Jupiter GPU container with a frozen Megatron Python
environment resolved from this checkout's `uv.lock`. The container supplies the cluster CUDA runtime; the
frozen environment supplies the exact Torch, Megatron Core, and Transformer Engine closure under test. The job
records the checkout revision, GPU topology, imported package versions, commands, and test output alongside the
numerical artifacts.
