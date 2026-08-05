# Grug FP32 combine: bounded paired H100 evidence

## Outcome

The local FP32-accumulation candidate does what its small fixed-route contract
requires, but it costs time and combine-boundary HBM at the real Grug expert
shape. The predeclared verdict was estimate-only, so this report makes no claim
about practical materiality.

On every one of eight H100s, candidate `fbb1fc8378601e0346d00d186809f10d1ad0360d`
was slower than parent `0c213586b5491b8046ca7780e965c4b26dc6a2a2` for
the warmed fixed-route full-block forward plus backward boundary. The paired
median increase was `0.537424` ms, or a median ratio of `1.038439`; the eight
per-GPU increases ranged from `0.498303` to `0.545664` ms. The complete affected
combine boundary increased by a paired median `0.531640` ms, or a median ratio
of `1.340920`.

The candidate's complete combine boundary allocated `251,592,704` more peak
bytes than the parent after the same baseline. The measured full-block peak was
identical because a larger part of that boundary determined the peak in both
arms. That zero full-block delta is not evidence that FP32 accumulation has no
memory cost.

This is local fixed-route block evidence only. It does not change the failed
32-H100 action-output gate, prove distributed equivalence, measure end-to-end
MFU, identify attention as causal, resolve Marin #7903, or put `fbb1fc8` on
MarinSkyRL #276.

## Correctness result

The parent and candidate ran in fresh processes with byte-identical BF16 state,
hidden inputs, fixed expert IDs, combine weights, and cotangent. The expert
output times combine-weight multiplication remained BF16 in both arms. Only the
subsequent running-sum dtype differed. The fixture was fixed before numerical
results at 4,096 tokens, hidden size 128, expert intermediate size 64, five
experts, and top-4 routing.

The independent references reduce the path's BF16 weighted summands in fixed
slot order. One uses FP32 to express the candidate contract; the other uses
FP64 to benchmark returned-output accuracy against a higher-precision
fixed-order accumulator. They do not call either product combine implementation.

- Parent eager and grouped outputs each differed from the FP32 reference in
  `4,096 / 524,288` elements, with maximum absolute error `0.015625`.
- Candidate eager and grouped outputs were bitwise equal to the independent
  FP32 reference: zero failures and zero maximum error. They were also bitwise
  equal after casting the independent FP64 result to BF16.
- Against the independent uncast FP64 accumulator, the returned BF16 output's
  maximum error was `0.01171875` for the parent and `0.00390625` for the
  candidate; mean error was `0.000091552734375` and `0.000030517578125`,
  respectively.
- Candidate eager and grouped outputs were bitwise equal. Combine-weight,
  expert gate, up, and down gradients were bitwise equal between those paths;
  the hidden-gradient maximum difference was `2.384185791015625e-7`.
- For both candidate paths, all five combine-relevant gradient targets passed
  the frozen `rtol=0.08, atol=0.0001` FP32 and FP64-reference checks with zero
  tolerance failures. The output rule was exact: `rtol=0, atol=0`.

Post-run review found one protocol-label deviation. The fifth required gate says
"candidate maximum pre-cast error," but the product path retained only its
returned BF16 output. The recorded `fp64_pre_cast_accuracy.actual` values compare
that returned output with the independent uncast FP64 accumulator; the product's
internal pre-final-cast FP32 accumulator was not captured. Thus the recorded
`candidate_reduces_fp64_error=true` supports the narrower returned-output result
above, not a direct product pre-cast measurement. The first four required gates
passed at their stated scope, and the fixture discriminated the known parent
error. The generated `SUMMARY.md` phrase "reduced local accumulation error"
must likewise be read at returned-output scope. The frozen protocol, raw packet,
reader, and generated summaries remain unchanged so their executed identities
and byte-for-byte readback are preserved.

## Paired timing result

The performance fixture used 8,192 tokens, hidden size 2,560, expert
intermediate size 1,280, 256 experts, and top-4 routing. Every expert received
exactly 128 routed rows. Each physical GPU ran both arms twice in alternating
order. Each fresh process performed five warmups and twenty measured iterations,
giving 40 timing samples per arm per GPU. CUDA events enclosed forward and
backward, and the final backward event was synchronized. Setup and gradient
clearing stayed outside the timed regions.

The frozen HBM procedure emptied the allocator cache after those five warmups.
The first retained iteration in each process therefore included post-empty-cache
allocations and was visibly slower than its other 19 iterations. All samples
were retained. Each arm had two such high samples in its 40-sample per-GPU pool,
so neither determined the reported median; the median came from the stable
post-allocation iterations.

The full-block boundary is `GrugMoeSparseMoeBlock._forward_grouped` with the
router held outside the measurement. It includes route sorting, routed-input
construction, grouped expert projections, BF16 weighting, combine accumulation,
and backward. The combine boundary is the complete affected section: weight
flatten/index selection/cast, BF16 expert-output multiplication, output
allocation, `index_add` accumulation, the candidate casts, and backward.
The primary full-block boundary calls the hash-pinned product module directly.
Because the product does not expose combine as a standalone callable, the
driver implements the isolated combine boundary with those same affected
operations.

The parent and candidate columns below are medians across the eight per-GPU arm
medians. Delta and ratio are the median of the eight paired per-GPU comparisons;
their ranges are also across those eight GPUs.

| Boundary | Parent ms | Candidate ms | Paired delta ms (range) | Paired ratio (range) |
| --- | ---: | ---: | ---: | ---: |
| Full-block forward | 4.592880 | 4.811432 | +0.212600 (+0.192288 to +0.242560) | 1.046230 (1.041689 to 1.052800) |
| Full-block backward | 9.382608 | 9.687456 | +0.310968 (+0.300448 to +0.321200) | 1.033177 (1.032006 to 1.034152) |
| Full-block forward + backward | 13.982672 | 14.502136 | +0.537424 (+0.498303 to +0.545664) | 1.038439 (1.035603 to 1.039014) |
| Combine forward | 0.992272 | 1.210104 | +0.217800 (+0.210432 to +0.229696) | 1.219420 (1.211750 to 1.232264) |
| Combine backward | 0.567208 | 0.880680 | +0.313688 (+0.312336 to +0.314592) | 1.553185 (1.549717 to 1.554640) |
| Combine forward + backward | 1.559472 | 2.090952 | +0.531640 (+0.523488 to +0.543248) | 1.340920 (1.335353 to 1.348955) |

The primary per-GPU values are preserved in `SUMMARY.md`; every per-iteration
timing sample is in `raw_packet.json`.

## Allocated HBM

After warmup and gradient clearing, each process emptied the allocator cache,
synchronized, recorded its allocated baseline, reset peak statistics, ran the
measured iterations, and recorded peak minus baseline. These byte counts were
identical across all repetitions and GPUs within each arm.

| Boundary and arm | Baseline bytes | Peak bytes | Incremental peak bytes |
| --- | ---: | ---: | ---: |
| Full block, parent | 5,118,757,888 | 10,819,433,472 | 5,700,675,584 |
| Full block, candidate | 5,118,757,888 | 10,819,433,472 | 5,700,675,584 |
| Complete combine, parent | 5,622,931,456 | 6,168,322,048 | 545,390,592 |
| Complete combine, candidate | 5,622,931,456 | 6,419,914,752 | 796,983,296 |

Thus the paired full-block incremental-peak delta was exactly zero on all eight
GPUs. The complete-combine delta was exactly `+251,592,704` bytes on every GPU,
a ratio of `1.461307`.

## Runtime, safety, and readback

Iris provided one whole `h100-8x` node in `cw-rno2a` at interactive priority:
eight H100 80GB devices, 32 CPU cores, a 128 GiB cgroup limit, and 200 GiB of
ephemeral storage. The pinned image digest was
`sha256:9af9a3d38f57c2ed8dfe1d6f6657a9f4a00c582ec06a5ac2af8fcddbe51da03c`.
The runtime used Python 3.12.13, Torch 2.11.0+cu129, CUDA 12.9, NVIDIA driver
595.71.05, and the exact package inventory whose SHA-256 is
`8df26c4fa6f5bd79d25f22a9fd31bda8d4627af8083a975d1124f1b6dda84a77`.

The first two preflight launches exposed missing runtime imports. The first
stopped before importing Grug or constructing tensors because `ray` was absent.
The second reached the lazy grouped-kernel import and stopped because
`torchtitan` was absent; it emitted no result JSON and performance never began.
The runtime was completed with the project's exact pins before its inventory
was frozen. No fixture, numerical rule, schedule, driver, reader, or verdict
changed. `PREFLIGHT.md` and the two small stderr records preserve this history.

During the successful run, 93 one-second memory samples recorded a maximum
cgroup use of `25,553,707,008` bytes, below the frozen 100 GiB stop point. The
minimum host `MemAvailable` was `2,063,631,482,880` bytes, above the frozen
256 GiB stop point. The remote host had no swap, so the conditional swap rule
did not apply. No measurement worker ran on the local host; after copy and
readback it still had about 8.3 GiB available RAM and 6.0 GiB of 8.0 GiB swap
free.

The independent standard-library reader passed 344 assertions. It checked the
revision, module, protocol, driver, runtime, resource, fixture, schedule,
per-GPU pairing, source, sample-count, route-balance, HBM-accounting, safety,
and raw-packet-digest contracts, then recomputed the paired summary. A second
local invocation regenerated `summary.json`, `SUMMARY.md`, and the reader's
stdout byte-for-byte from the copied raw packet. Iris then reported the holder
job killed by the user, Kubernetes reported its pod absent, and the local
holder process exited.

## Frozen identity and artifact hashes

- Protocol SHA-256: `5b9ae45ff0b9e9b14c81f503cbca095cbf8fa25e2a36e3ea39b507ac703bdbf3`
- Driver SHA-256: `c862518172dd623f3bd5f7d138aaa128361e492958f431639e73d1099e5a2ac8`
- Reader SHA-256: `5f83e9030295d4219c9180a52147082bbbd14247c562217f84105456cc48cc4e`
- Parent module SHA-256: `b1e63368996530dd8fa678ec3b482a1bd63007c0d69901cd13c6a4e42c294d50`
- Candidate module SHA-256: `2dd51d842afe6f57fe00337c21945923da68664d9c23678633c578c27504ba93`
- Raw packet SHA-256: `60e882a908643ae487b6f2f2a1c0c979a6a3d10bb9dccd97c40601c98e02526e`
- Compact JSON summary SHA-256: `d2199f2f4bbd95cd0d1db0503192b36db3df47df2fa8130ea24119aee7dfc7dd`
- Compact Markdown summary SHA-256: `1941341978a687f0a627bae57a3af437b75c5629e9812a61bba2fedf93cdebb5`

The source files and frozen hashes are evidence-only. No product code was
changed or placed on #276 by this measurement.
