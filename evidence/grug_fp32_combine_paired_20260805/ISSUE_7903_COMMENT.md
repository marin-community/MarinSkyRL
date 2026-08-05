🤖 Bounded paired cost update: local candidate `fbb1fc8378601e0346d00d186809f10d1ad0360d` satisfies the fixed-route FP32-combine contract, but it was slower than #276 parent `0c213586b5491b8046ca7780e965c4b26dc6a2a2` on every one of eight paired H100s and used more HBM at the affected combine boundary. This was predeclared as estimate-only, so I am not assigning a practical-regression threshold after seeing the result.

## Local correctness

The arms used byte-identical BF16 state, hidden inputs, fixed expert IDs, combine weights, and cotangent. Expert-output times weight multiplication remained BF16; only the running-sum dtype differed. Independent fixed-slot-order FP32 and FP64 reductions consumed the path's BF16 weighted summands rather than calling either product combine implementation.

- Parent eager and grouped outputs each differed from the FP32 reference in `4,096 / 524,288` elements, with maximum absolute error `0.015625`.
- Candidate eager and grouped outputs were bitwise equal to the FP32 reference and to the BF16-cast FP64 reference: zero output failures and zero maximum error.
- Against the independent uncast FP64 accumulator, the returned BF16 output's maximum/mean error fell from parent `0.01171875 / 0.000091552734375` to candidate `0.00390625 / 0.000030517578125`.
- Candidate eager and grouped outputs were bitwise equal. All combine-weight, hidden, expert-gate, up, and down gradient checks passed the frozen `rtol=0.08, atol=0.0001` rule with zero failures. The eager/grouped hidden-gradient maximum difference was `2.384185791015625e-7`; the other four gradient targets were bitwise equal.
- The first four required gates passed at their stated scope, including exact candidate output/gradient checks and discrimination of the parent's known BF16 running-sum error. Post-run review found that the fifth gate's "pre-cast" label overstated its implementation: its true boolean compared the returned BF16 product output with the independent uncast FP64 accumulator. The product's internal pre-final-cast FP32 accumulator was not captured, so I am not making a direct product pre-cast claim. The frozen packet and reader are unchanged.

The correctness fixture was 4,096 tokens, hidden 128, expert intermediate 64, five experts, top-4.

## Paired H100 measurement

The measured fixture used the real expert shape: 8,192 tokens, hidden 2,560, expert intermediate 1,280, 256 experts, top-4. Routes were fixed outside the router and perfectly balanced at 128 rows per expert. Each physical GPU ran both arms twice in alternating order; each fresh process performed five warmups and twenty measured iterations, for 40 samples per arm/GPU. CUDA events enclosed forward and backward with synchronized completion. Setup and gradient clearing were outside the timed region.

For HBM accounting, the frozen protocol emptied the allocator cache after warmup. The first retained iteration in each process therefore included post-empty-cache allocations and was visibly slower than the other 19. All samples were retained. Each arm had two such high samples in its 40-sample per-GPU pool, so neither determined the reported median; the median came from the stable post-allocation iterations.

The primary boundary called the hash-pinned product `_forward_grouped` with routing fixed: route sorting, routed-input construction, grouped expert projections, BF16 weighting, combine, and backward. It does not include the router. The product has no standalone combine callable, so the isolated combine boundary was a driver implementation of the same affected operations.

| GPU | Parent full F+B ms | Candidate full F+B ms | Delta ms | Ratio |
| ---: | ---: | ---: | ---: | ---: |
| 0 | 13.978896 | 14.481008 | +0.502112 | 1.035919 |
| 1 | 13.955856 | 14.496912 | +0.541056 | 1.038769 |
| 2 | 13.995952 | 14.494256 | +0.498303 | 1.035603 |
| 3 | 13.961344 | 14.502992 | +0.541648 | 1.038796 |
| 4 | 13.992480 | 14.513088 | +0.520608 | 1.037206 |
| 5 | 13.967488 | 14.501280 | +0.533792 | 1.038217 |
| 6 | 14.001856 | 14.543184 | +0.541328 | 1.038661 |
| 7 | 13.986448 | 14.532112 | +0.545664 | 1.039014 |

Across the eight paired GPU medians:

- Full-block forward + backward: candidate-parent median `+0.537424` ms, range `+0.498303` to `+0.545664` ms; median ratio `1.038439`, range `1.035603` to `1.039014`.
- Full-block forward alone: median `+0.212600` ms; ratio `1.046230`. Backward alone: median `+0.310968` ms; ratio `1.033177`.
- Complete affected combine forward + backward: median `+0.531640` ms, range `+0.523488` to `+0.543248` ms; median ratio `1.340920`, range `1.335353` to `1.348955`.
- Combine forward alone: median `+0.217800` ms; ratio `1.219420`. Backward alone: median `+0.313688` ms; ratio `1.553185`.

HBM baselines, peaks, and peak-minus-baseline were identical across repetitions and GPUs within each arm:

| Boundary and arm | Baseline bytes | Peak bytes | Incremental peak bytes |
| --- | ---: | ---: | ---: |
| Full block, parent | 5,118,757,888 | 10,819,433,472 | 5,700,675,584 |
| Full block, candidate | 5,118,757,888 | 10,819,433,472 | 5,700,675,584 |
| Complete combine, parent | 5,622,931,456 | 6,168,322,048 | 545,390,592 |
| Complete combine, candidate | 5,622,931,456 | 6,419,914,752 | 796,983,296 |

The complete-combine incremental peak therefore increased by exactly `251,592,704` bytes on every GPU, a ratio of `1.461307`. The full-block peak delta was zero because a larger part of that boundary set both peaks; this does not show that FP32 accumulation itself has no memory cost.

## Pins, safety, and readback

Iris supplied one whole `h100-8x` node in `cw-rno2a` at interactive priority: eight H100 80GB devices, 32 CPU, 128 GiB memory, and 200 GiB ephemeral storage. The pinned runtime used Torch `2.11.0+cu129`, CUDA 12.9, driver 595.71.05, image digest `sha256:9af9a3d38f57c2ed8dfe1d6f6657a9f4a00c582ec06a5ac2af8fcddbe51da03c`, and package-inventory SHA-256 `8df26c4fa6f5bd79d25f22a9fd31bda8d4627af8083a975d1124f1b6dda84a77`.

Two dependency-preflight attempts exposed missing `ray` and then a lazy `torchtitan` import. Neither wrote a result JSON and performance did not start. The runtime was completed with the project's exact pins; only the runtime inventory and freeze timestamp changed. The fixture, numerical rules, finite schedule, driver, reader, and estimate-only verdict did not.

The successful run recorded 93 one-second memory samples. Maximum cgroup use was `25,553,707,008` bytes, below the frozen 100 GiB stop point; minimum host available memory was `2,063,631,482,880` bytes, above the 256 GiB stop point. The remote host had no swap. No worker ran on the local host. After copying and verifying the packet, the Iris holder was terminated and its Kubernetes pod was absent.

The standard-library reader passed 344 assertions and recomputed the paired summary. A local readback regenerated the JSON summary, Markdown summary, and reader stdout byte-for-byte from the copied raw packet. Raw packet SHA-256: `60e882a908643ae487b6f2f2a1c0c979a6a3d10bb9dccd97c40601c98e02526e`.

[Human-readable report, frozen protocol, raw packet, reader, and compact summaries](https://github.com/marin-community/MarinSkyRL/tree/40d365d661f3acc411a78b8801000a4eae964512/evidence/grug_fp32_combine_paired_20260805).

## Disposition and limits

This result does not change the controlling distributed evidence. The latest instrumented 32-H100 gate still failed `1 / 12,288` action log probabilities at rank 22, microbatch 54, model token 8010. Its last matching sampled boundary was layer-2 decoder input and its first differing sampled boundary was layer-2 post-attention hidden; the first sampled route mismatch appeared only at layer 23. The bounded causal sparse-block probe reported `no_causal_internal_divergence`. That probe neither exonerates grouped MoE across runs nor identifies attention as causal.

No live replay or distributed run was launched for this measurement. It proves only the candidate's local fixed-route accumulation contract and estimates its block-level cost on this shape. `fbb1fc8` remains off MarinSkyRL #276, the failed action-output gate remains failed, and #7903 remains unresolved.
