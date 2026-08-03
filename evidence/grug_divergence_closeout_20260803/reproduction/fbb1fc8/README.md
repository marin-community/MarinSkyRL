# FP32 combine candidate closeout packet

This directory preserves the exact local candidate patch and execution/readback
programs used for the final `fbb1fc8` correctness cycle. It is evidence only;
the candidate was not pushed to MarinSkyRL PR #276.

## Result

- Candidate: `fbb1fc8378601e0346d00d186809f10d1ad0360d`, based on PR #276 head
  `0c213586b5491b8046ca7780e965c4b26dc6a2a2`.
- Harness commit: `2dd905e29597b848f912dd5cdaf2cebdfbf3d0c2`.
- Image:
  `ghcr.io/marin-community/marinskyrl@sha256:bf136bb210ded34b07b3583de7272f2c534df195bca556debbb38c0664640770`.
- Preflight artifact:
  `s3://marin-us-east-02a/iris/grug-training-perf-gap/20260803/divergence-closeout-fbb1fc8/preflight-paired-s1.json`.
  Its frozen readback passed.
- Headline artifact:
  `s3://marin-us-east-02a/iris/grug-training-perf-gap/20260803/divergence-closeout-fbb1fc8/headline-paired-s1.json`.
- The sole executed full pair failed: 3 of 12,288 representative action log
  probabilities exceeded the frozen tolerance. CE and all 5,184 representative
  gradient comparisons passed. The observed `14.22793536158026` timing ratio is
  diagnostic only.
- The final artifact retained no paired route traces, so it does not identify
  the operation responsible for the three remaining failures.

The final writer job was
`/romain/grug-paired-eager-grouped-fbb1fc8-s1-rno-20260803`; the out-of-process
reader was `/romain/grug-candidate-headline-readback-fbb1fc8-s1-20260803`.
The identical East submission never started a task and was stopped before the
approved RNO replacement.

## Executed composition

The launchers archived the harness commit, overlaid only the candidate
`grug_moe.py`, and verified the following files before submission and again in
the task:

```text
b46a8d3e2c0516032b8ca9466b047b911f0ec50d1a527df393878c2522049404  skyrl-train/scripts/grug_fixed_replay_benchmark.py
c6a954f2cb69996efcfa68fdbac4e43b63955f1f01c5539bf6ed41b1aa7d15b1  skyrl-train/skyrl_train/workers/fsdp/fsdp_worker.py
2dd51d842afe6f57fe00337c21945923da68664d9c23678633c578c27504ba93  skyrl-train/skyrl_train/models/grug_moe.py
f2e8484de7d5566f7e39d6d7e8ef3c03744960a53dd9a39c20a9dfba4db6d2ba  cloud/iris/start_rl_iris_controller.py
```

`0001-rl-Accumulate-Grug-expert-outputs-in-FP32.patch` is the exact candidate
commit as a mail patch. The JSON files are the exact frozen-reader summaries.

## Archived-file SHA-256

```text
eebb383c6f2c13ddb205c1cad7c12a577661bf336ce2b3960dd60df0ab18dd5a  0001-rl-Accumulate-Grug-expert-outputs-in-FP32.patch
528843e16ff683f34b7c3ca6e5a19b117f915b84945a07330fe648d1d81910fe  headline.summary.json
8b03c052b8a2a878ca3f11516f81794ad4f85b1ecc747a58e2beefb9be57e86a  launch_candidate_headline_fbb1fc8.sh
dbbd9ad3634606ed54181c7ede7e9e27e1d7cedb020f5716ce59740a04f4cebd  launch_candidate_headline_fbb1fc8_rno_24cpu_1200g.sh
62ea714133e926f48a7a5d17159459ada477e4fb170aeb1bc56ec25e4c9922b6  launch_candidate_headline_readback_fbb1fc8.sh
4a533668e5fa4373a9c42e1b630822a947a93c21fdb14c1eaf5c86ee4bf865b8  launch_candidate_preflight_fbb1fc8.sh
a1594296a783a9e92fd421241a07a79806a54ca5822f81979f32a54e504472df  launch_candidate_preflight_readback_fbb1fc8.sh
54961e907e85f3db2c895670df6c8f31beef0d4c0669dd68ecfdd0c721b0b8bb  launch_grug_fp32_image_fbb1fc8.sh
d34e1aa9e11bbc3ca3667e2dafffc7450774c2aa621aa8cded07f932a185eac1  launch_grug_fp32_image_reuse_wheels_fbb1fc8.sh
996a4578c2012daa057156cd87abe3b010dfa19f1e3591fd51e6c0c5c8dc2de7  monitor_candidate_headline_fbb1fc8_rno.sh
30473c5f17d668cc2362172d36d3d6e8ccf764683b4c49d3dd920e1e11d06994  preflight.summary.json
a96fca9dce7c952b9f6859460c61fdbad4ff2af96d867a8ca640413b133910a1  run_candidate_headline_fbb1fc8.sh
ec7cdd28c708ceebc20ccdf5482c375b4a3ecedeac67f97547febb7817d8c322  run_candidate_preflight_fbb1fc8.sh
eb4bbc981ed9d67cae1f23001515df1d66d7dc2ef56a2a0c507d58c1aafd9d6e  verify_candidate_headline_artifact.py
4348e5d1beaede1a870fc4991c8717bc9fde6c5fcb72d8bceb88f091e340eb60  verify_candidate_preflight_artifact.py
```

Complete local logs were not committed because of their size. Their hashes are:

```text
791d27c97b46d575db1ee6da1dd0f9d9628576b5182c41a248993d0f06a65bd9  frozen reader Iris log
fa7d2d0042e3e258bb4f8b9ec426ede916d6bcb0536d91f59a64f1a3b7058d7c  final host monitor
44f8757c0afa2e3d664df43bd2bc0f4169219b8ae5f21e16d00ff53f54496c82  final writer Iris tail
```
