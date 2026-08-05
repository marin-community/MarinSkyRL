# Grug FP32 combine paired H100 measurement

Raw packet SHA-256: `60e882a908643ae487b6f2f2a1c0c979a6a3d10bb9dccd97c40601c98e02526e`.

The primary metric is warmed full sparse-block forward plus backward time. The verdict was frozen as estimate-only; no practical materiality threshold was invented after seeing results.

| GPU | parent ms | candidate ms | candidate-parent ms | ratio |
| ---: | ---: | ---: | ---: | ---: |
| 0 | 13.978896 | 14.481008 | +0.502112 | 1.035919 |
| 1 | 13.955856 | 14.496912 | +0.541056 | 1.038769 |
| 2 | 13.995952 | 14.494256 | +0.498303 | 1.035603 |
| 3 | 13.961344 | 14.502992 | +0.541648 | 1.038796 |
| 4 | 13.992480 | 14.513088 | +0.520608 | 1.037206 |
| 5 | 13.967488 | 14.501280 | +0.533792 | 1.038217 |
| 6 | 14.001856 | 14.543184 | +0.541328 | 1.038661 |
| 7 | 13.986448 | 14.532112 | +0.545664 | 1.039014 |

## Compact paired summary

- Full block candidate-parent median delta: `+0.537424` ms; per-GPU range `+0.498303` to `+0.545664` ms; median ratio `1.038439`.
- Complete combine boundary candidate-parent median delta: `+0.531640` ms; per-GPU range `+0.523488` to `+0.543248` ms; median ratio `1.340920`.
- Full-block incremental peak allocated HBM delta: median `+0` bytes; range `+0` to `+0` bytes.
- Combine-boundary incremental peak allocated HBM delta: median `+251592704` bytes; range `+251592704` to `+251592704` bytes.
- Estimate-only direction: `candidate_slower_on_all_eight_gpus`.

## Correctness and limits

The fixed fixture distinguishes the parent's BF16 running-sum error. Candidate eager and grouped outputs match the independent FP32 slot-wise contract exactly, their combine-relevant gradients pass the frozen tolerance, and the FP64 fixed-order reduction reports the candidate's reduced local accumulation error.

This is a local, fixed-route sparse-block result. It does not change the failed 32-H100 action-output gate, prove distributed semantic equivalence, measure end-to-end MFU, identify attention as causal, or put `fbb1fc8` on MarinSkyRL #276.
