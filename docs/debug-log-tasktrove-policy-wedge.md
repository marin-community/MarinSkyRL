# Debugging log for the TaskTrove policy wedge

Determine why long-running TaskTrove policy jobs stop making progress while every process remains alive, and
restore a measured upper bound from loss of progress to job teardown.

This is the evidence ledger for the investigation. The incident escalation preserves the raw chronology,
including claims later withdrawn. This file keeps only observations with an artifact or repeatable result.
Candidate mechanisms remain hypotheses until an experiment distinguishes them.

## Status

The failure is reproduced across the TaskTrove fleet, but its cause is open. The affected jobs remain allocated
for hours after trainer progress stops. The runtime protections added in PR #260 therefore do not establish the
intended liveness bound for this production failure, even though the single-node fault contract added in PR #268
terminates a smaller synthetic phase divergence.

The strongest live capture is job 1170543. Its six-node artifact is stored at
`/e/data1/datasets/playground/ot-baf/_wedge_forensics_1170543/` and contains three synchronized all-rank samples.
The source escalation is
`2026-08-02_escalation-tasktrove-wedge-root-cause-fsdp2-prefetch-vs-ep-alltoall-deadlock.md`; its filename is
historical and does not describe an established cause.

## Investigation entries

### 2026-08-05: inherited blocking wait defeats the ProcessGroupNCCL deadline

Jupiter job 1242876 ran the four-node EP4/FSDP4 permanent-divergence contract against the production GH200
runtime. The controller injected the TaskTrove worker environment's `NCCL_BLOCKING_WAIT=1` together with
`TORCH_NCCL_ASYNC_ERROR_HANDLING=1`. All 16 ranks completed three healthy EP/FSDP warmup rounds and reported
a 60-second process-group timeout. All ranks then entered the deliberate fault: rank 0 entered its inter-node
FSDP all-gather, the other ranks entered EP all-to-all, and the 12 ranks in unaffected EP groups completed.

The four blocked ranks remained alive until the controller's independent 120-second fault deadline. PyTorch
emitted neither a collective-timeout detection nor a watchdog abort. The controller killed and reaped the
Slurm step, and pytest failed after 379.38 seconds including environment startup. This is a deterministic
reproduction of the liveness-contract failure: an inherited blocking-wait setting turns a permanent subgroup
non-arrival into a live gang past the configured ProcessGroupNCCL bound.

This experiment does not identify what initiates the first subgroup stall in a natural training run. It
isolates the separate MarinSkyRL defect that allows such a stall to remain wedged indefinitely. The matching
green experiment must use the production worker bootstrap, prove that it removes all blocking-wait aliases
before importing torch, and show the same fault terminating under ProcessGroupNCCL's asynchronous watchdog.

Jupiter job 1243041 ran that matching experiment after the worker-bootstrap fix. All 16 ranks again completed
the warmup and entered the same fault, but every readiness record reported `blocking_wait=None`. The four
blocked collectives timed out after 60.07 to 60.10 seconds, all ranks received the diagnostic dump signal, and
torchrun exited nonzero before the controller deadline. The enclosing pytest passed in 341.41 seconds and the
batch job completed normally. The red/green pair establishes that clearing inherited blocking-wait state at
the production worker boundary restores the configured liveness bound for this failure shape.

### 2026-08-05: NUMA is refuted and phase records isolate one FSDP subgroup

The seven jobs in the August 4 fleet all logged strict GPU-local CPU binding before process-group
initialization and later reproduced the wedge. This refutes NUMA placement as a sufficient fix for this
failure. It does not invalidate the separate host-memory placement contract added in PR #287.

The diagnostic baseline job 1211183 recorded a coherent final region at global step 7, microbatch 838.
All 16 ranks entered backward with identical EP, FSDP, and WORLD sequence numbers. Twelve ranks completed
backward; ranks 1, 5, 9, and 13 did not. Those four ranks share EP coordinate 1 and form one complete
inter-node FSDP subgroup. The other three FSDP subgroups advanced by 240 EP and 196 FSDP operations and
exited the step. This observation refutes whole-world schedule divergence for that instance and localizes
loss of progress to one complete FSDP subgroup. It does not distinguish a subgroup transport stall from a
later subgroup-local operation.

The no-packing and no-reshard bisect arms also reached a final `backward_enter` on every rank and timed out
at the eight-hour allocation boundary. They completed one more global step than the paired baseline, which
is not enough evidence that either option changes the hazard. Neither option prevented the wedge.

Every August 4 worker also inherited `NCCL_BLOCKING_WAIT=1` alongside
`TORCH_NCCL_ASYNC_ERROR_HANDLING=1`. The next controlled experiment runs the existing four-node permanent
phase-divergence contract with that exact conflict. The test must first fail by exceeding its independent
deadline, then pass only after the production worker bootstrap removes the incompatible blocking-wait
settings before torch is imported.

### 2026-08-02: replace Python-frame inference with native evidence

The paired `py-spy --native` capture refuted the mask-construction hotspot explanation. Two threads with
different visible Python frames stopped at the same unresolved native offsets and consumed similar CPU. The
investigation returned to the last directly observed boundary: every sampled rank stopped making trainer
progress while remaining alive in the same unresolved native location.

Source inspection also found that OpenThoughts-Agent's Jupiter environment sets
`TORCH_NCCL_BLOCKING_WAIT_TIMEOUT_MS=1800000` with a comment describing a 30-minute NCCL watchdog timeout.
PyTorch 2.9 does not read that variable. `TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=1800`, set next to it, bounds a stalled
watchdog heartbeat and is not a collective-operation timeout. MarinSkyRL's separate process-group `timeout=`
path is the only source-level evidence for the intended operation bound; its resolved live value remains to be
captured.

## Evidence standard

Use four states in this log:

- **Established:** a named artifact, source inspection, or repeatable experiment directly supports the claim.
- **Refuted:** a counterexample or more discriminating measurement contradicts the claim.
- **Preliminary:** an observed pattern lacks enough measurements or controls to support a mechanism.
- **Open:** the available evidence does not distinguish the alternatives.

A Python frame names the last visible Python call, not necessarily the native work currently executing. CPU
utilization, process state, elapsed time, and missing logs are signals to combine with direct stacks or counters;
none identifies a root cause alone.

## Established observations

### The jobs lose trainer progress without process death

The TaskTrove arms stop banking steps while the scheduler continues to report the jobs as running. Live ranks
remain present for 3 to 11 hours after the intended 30-minute collective bound. Restarting from an earlier
checkpoint can pass the previously observed failure point. The failure is therefore not a deterministic function
of checkpoint step or a single replayed row under the observed execution schedule.

This does not rule out data-dependent routing or shape effects. Asynchronous sampling, batch construction, and
rank timing can change after a restart even when the checkpoint is unchanged.

### The captured Python frames do not locate the native stall

In job 1170543, `py-spy --native` sampled one thread whose visible Python frame was TorchTitan
`_token_dispatch` and another whose visible frame was Transformers `find_packed_sequence_indices`. Both native
stacks ended at the same two library-relative offsets:

| thread | visible Python frame | unresolved native offsets |
| --- | --- | --- |
| 2334506 | `_token_dispatch` | `...f4048`, `...74ef4` |
| 2335115 | `find_packed_sequence_indices` | `...f4048`, `...74ef4` |

The high address bits differ because the processes map the library at different ASLR bases. The capture supports
one conclusion: the two Python frames lead to the same unresolved native location. It does not identify that
native function. Symbolization against the exact image libraries remains required.

### CPU utilization does not separate computation from waiting

The live environment contains `NCCL_BLOCKING_WAIT=1`. PyTorch 2.9 accepts this as a legacy alias for
`TORCH_NCCL_BLOCKING_WAIT`; its documented effect is to make `wait()` block until the process-group operation
completes or reaches its process-group timeout. The two sampled threads consumed similar CPU:

- the thread showing `find_packed_sequence_indices`: 62.4%, 04:28:17 cumulative CPU time;
- the thread showing `_token_dispatch`: 60.4%, 04:19:37 cumulative CPU time.

Those measurements cannot distinguish active mask construction from a collective wait. Process state `R` and
rising CPU time are not progress evidence under this runtime mode.

### Colocation and memory pressure are absent in the captured wedge

One sampled policy node had four GPU UUIDs and four policy PIDs, one PID per GPU. Each process used about 24 GiB
of a 95 GiB device, leaving about 70 GiB free. No inference process was present. This refutes policy/inference
colocation and GPU-memory pressure as causes of job 1170543's wedge.

### The production liveness bound did not hold

The live capture contains `NCCL_BLOCKING_WAIT=1`, `TORCH_NCCL_ASYNC_ERROR_HANDLING=1`,
`TORCH_NCCL_DUMP_ON_TIMEOUT=1`, and a 20,000-event flight-recorder buffer. No timeout marker, NCCL error, or
flight-recorder dump was found after ranks remained stuck beyond 1,800 seconds.

`TORCH_NCCL_BLOCKING_WAIT_TIMEOUT_MS=1800000` also appears in the captured environment, but PyTorch 2.9 neither
documents nor reads that variable. OpenThoughts-Agent sets it in `hpc/hpc.py`, but it is not evidence that a
1,800-second timeout reached the live process groups. MarinSkyRL separately passes
`SKYRL_WORKER_NCCL_TIMEOUT_IN_S` to WORLD initialization and applies the same value to the FSDP and EP
subgroups. The next capture must record the resolved MarinSkyRL value and subgroup timeout from the running
image before concluding that the expected timeout reached the implicated group.

PR #260's heartbeat monitor only terminates a process if the ProcessGroupNCCL watchdog itself stops
heartbeating. A healthy watchdog can continue heartbeating while an operation makes no useful progress. The
absence of a heartbeat abort therefore does not show that the monitor was disabled.

## Refuted explanations

### The visible `find_packed_sequence_indices` frame is a CPU hotspot

Refuted by the paired native samples. The mask-frame and token-dispatch-frame threads end at the same native
offsets and consume indistinguishable CPU. The Python frame and CPU counter do not show that either thread is
executing mask construction.

### A split between visible EP and FSDP Python frames proves collective-order divergence

Refuted as a diagnostic rule. A visible Python-frame split does not establish the currently executing native
operation. Earlier samples also observed similar frame distributions before a job was known to be wedged.
Collective sequence evidence or symbolized native stacks are required.

### High CPU or process state `R` proves forward progress

Refuted by hours of unchanged trainer state and the paired blocking-wait samples. CPU consumption can accompany
a wait in this configuration.

### The previously passed timeout tests cover the production failure

Refuted by the live jobs. PR #268 exercises four ranks on one node, warms EP2 and FSDP2 communicators, and then
splits ranks between collectives. PR #272 exercises healthy and delayed traffic in the production EP4/FSDP4
geometry on four nodes but never withholds a rank permanently. Neither combines the production model, 16-rank
multi-node geometry, long runtime, and permanent loss of progress. A pass in either suite does not imply that a
TaskTrove job will tear down.

## Preliminary patterns

### Restarts can pass the earlier failure point

Checkpoint resumes have progressed past the step or data position associated with an earlier wedge. This weighs
against a fixed bad row or deterministic checkpoint-state failure. Record the exact checkpoint, sampler state,
batch identity, global step, microbatch, and rank-local routing choices before treating this as a data-independent
failure.

### Time to wedge appears clustered

Multiple arms wedged after similar wall-clock durations. Exact launch time, trainer-stop time, completed policy
steps, processed tokens, collective counts, and restart ancestry have not yet been normalized into one table.
Until that table exists, elapsed time is a correlation and not evidence for a timer, leak, or cumulative resource
threshold.

## Open hypotheses and discriminating experiments

| Hypothesis | Evidence that would support it | Evidence that would refute it | Next experiment |
| --- | --- | --- | --- |
| A ProcessGroupNCCL operation is stuck at the shared native offsets. | Symbolized stacks place affected ranks in the same NCCL, CUDA, or ProcessGroupNCCL function. | The offsets resolve outside the distributed stack, or ranks resolve to different native functions. | Preserve `/proc/<pid>/maps`, exact image libraries, build IDs, and full native stacks; symbolize the two known offsets before changing code. |
| The implicated subgroup lacks the intended 1,800-second operation timeout. | Runtime inspection shows a different or unset timeout on the exact EP/FSDP group. | The live group reports 1,800 seconds and remains stuck beyond it. | Add or use a read-only diagnostic that logs WORLD, EP, and FSDP timeout values after mesh creation; capture `SKYRL_WORKER_NCCL_TIMEOUT_IN_S` from every policy rank. |
| Production reaches a ProcessGroupNCCL state that the single-node PR #268 harness does not cover. | A 16-rank, four-node permanent-divergence test wedges under the captured environment while the four-rank test terminates. | The 16-rank test terminates under the same deadline and environment. | Extend the PR #272 geometry with a disposable, permanently withheld EP or FSDP participant after successful warmup. Keep an independent controller deadline and bounded reap. |
| The full model produces a sequence divergence absent from synthetic collective tests. | PR #274 records the first mismatched EP/FSDP sequence or the first rank that stops crossing a phase boundary. | All ranks retain matching subgroup sequences and stop at the same phase. | Rebuild with PR #274, enable `--collective-phase-diagnostics`, and retain per-rank records through the first stalled interval. |
| Failure risk follows cumulative work rather than wall time. | Wedges cluster by policy step, microbatch count, token count, or collective sequence after normalizing restarts. | Wedges cluster by elapsed time while cumulative work varies materially. | Build a fleet table from durable logs and checkpoints with both wall-clock and work counters. Do not compare raw scheduler age across restarted jobs. |
| Failure risk follows elapsed time rather than cumulative work. | Wedges cluster by elapsed trainer runtime while steps, tokens, and collective counts differ. | Restarted and fresh jobs wedge after similar cumulative work at different elapsed times. | Run matched fresh and resumed arms with PR #274 diagnostics and record time-to-wedge as a survival observation, not a single endpoint. |
| FSDP2 backward prefetch is necessary for the wedge. | A controlled arm with prefetch disabled survives materially longer or converts the native/sequence signature while a matched control wedges. | Both arms reproduce the same signature and hazard. | Run one paired arm only after native symbolization and phase records confirm that FSDP participation is relevant. |

## Immediate evidence requests while jobs remain live

These operations are read-only and preserve information that disappears when the jobs are killed:

1. Preserve `/proc/<pid>/maps`, executable and shared-library build IDs, and full native stacks for at least two
   ranks with different visible Python frames. Resolve `...f4048` and `...74ef4` against the copied libraries.
2. Capture the complete distributed environment from every policy rank, including
   `SKYRL_WORKER_NCCL_TIMEOUT_IN_S`, `TORCH_NCCL_*`, `TORCH_FR_*`, `NCCL_*`, PyTorch version, CUDA version, NCCL
   version, hostname, global rank, local rank, and mesh coordinate.
3. Record whether a manual flight-recorder trigger pipe exists. PyTorch 2.9 supports
   `TORCH_NCCL_DEBUG_INFO_PIPE_FILE`; if the running image configured one, trigger it and preserve every rank's
   dump. Do not infer anything from its absence because MarinSkyRL does not currently set this variable.
4. Take synchronized GPU-kernel samples or traces across all policy ranks. Record the kernel name and stream for
   the unresolved native wait instead of mapping it back from Python.
5. Preserve exact launch time, last trainer-progress time, last completed global step and microbatch, checkpoint
   lineage, and processed-token counters for every arm before restart or cancellation.

## Next code contracts

1. Add the 16-rank, four-node permanent-nonarrival experiment described above. This is the smallest missing
   bridge between PR #268 and PR #272.
2. Make the effective WORLD and device-mesh timeouts observable after process-group construction. A configured
   environment value is not sufficient evidence.
3. Add an independent phase-progress deadline outside the collective-running worker only if the multi-node
   production-mode test confirms that ProcessGroupNCCL cannot enforce the bound. The watchdog must capture
   diagnostics before it converts loss of progress into process death.
4. Run PR #274's phase diagnostics in a production-sized arm before changing collective ordering or disabling
   FSDP2 prefetch.

## References

- [PyTorch 2.9 ProcessGroupNCCL environment variables](https://docs.pytorch.org/docs/2.9/torch_nccl_environment_variables.html)
- [PyTorch 2.9 distributed process-group timeout contract](https://docs.pytorch.org/docs/2.9/distributed.html)
- [PyTorch 2.9 ProcessGroupNCCL source](https://github.com/pytorch/pytorch/blob/v2.9.0/torch/csrc/distributed/c10d/ProcessGroupNCCL.hpp)
- `docs/debug-log-collective-timeout-contract.md`
- `docs/debug-log-nccl-communicator-timeout.md`
- `docs/debug-log-warmed-collective-divergence.md`
- `docs/debug-log-collective-phase-diagnostics.md`
- `skyrl-train/tests/gpu/fault_injection/nccl_collective_contract.py`
- `skyrl-train/tests/gpu/fault_injection/multi_node_ep_fsdp_worker.py`
- `skyrl-train/tests/gpu/fault_injection/multi_node_phase_divergence_worker.py`

## Update protocol

Append an entry only after recording the artifact, command, or experiment revision. Move a hypothesis to
**Established** or **Refuted** only when a discriminating observation warrants it. Preserve corrections in git
history; do not rewrite a failed hypothesis into a successful one. Record negative results because they narrow
the search space.
