# Workload-owned trajectory-runner execution

## Context

MarinSkyRL constructs a `HarborTrajectoryRunner` before it selects a trainer. `FullyAsyncRayPPOTrainer` may then
replace that runner with `RolloutDispatcher`; `RayPPOTrainer` never performs the replacement. The global
`rollout.fanout.enabled` setting therefore affects fully asynchronous training and is silently ignored by synchronous
training.

This distinction is unrelated to the `TrajectoryRunner` contract. Synchronous and fully asynchronous Terminal-Bench
training use the same Harbor runner and inference HTTP endpoint. They differ in request shape:

- synchronous training calls `run()` once with multiple reward groups, with each group identified by a shared
  `TrajectoryID.instance_id`;
- fully asynchronous generation calls `run()` once per reward group and later assembles completed groups at the
  training barrier.

The current dispatcher assumes every call contains one group. It sends the complete request to one coordinator
without inspecting trajectory identities. Enabling it unchanged for synchronous training would send the entire
multi-group batch to one coordinator.

The synchronous path also keeps the Harbor orchestrator, LiteLLM request preparation, result gathering, and
token/logprob/reward processing in the `skyrl_entrypoint` process. Uvicorn runs on another thread in that process.
The two event loops share a GIL and CPU allocation. Two synchronous 96-trial Harbor runs reported 77 and 69
five-second TCP-connect timeouts against the local endpoint during context-exhaustion waves. The endpoint was healthy
outside the bursts; event-loop lag and serialization timing were not recorded, so process contention remains a
hypothesis. See the [incident record](https://echo.oa.dev/wiki/308) and the
[cross-project design record](https://echo.oa.dev/wiki/309).

## Decision

Trajectory-runner construction owns execution placement. Trainer selection does not participate in the decision and
cannot replace a constructed runner.

Execution placement is resolved once during setup from a serializable runner specification and a declared workload.
It does not switch in response to individual request sizes or runtime latency. The resulting object implements the
existing `TrajectoryRunner` lifecycle and is passed unchanged to either trainer.

Production Harbor workloads use process-isolated execution. Explicit development and test workloads may construct an
in-process Harbor runner. This is a deterministic policy, not a concurrency threshold. Other runner types declare the
execution modes they support; their policy can change independently of trainer cadence.

```python
@dataclass(frozen=True)
class TrajectoryWorkload:
    environment: ExecutionEnvironment
    phase: TrainingPhase
    expected_concurrent_trials: int


class TrajectoryRunnerFactory(Protocol):
    kind: TrajectoryRunnerKind

    def build(self) -> TrajectoryRunner: ...


def build_trajectory_runner(
    factory: TrajectoryRunnerFactory,
    workload: TrajectoryWorkload,
    resources: ProcessPoolResources,
) -> TrajectoryRunner: ...
```

The factory contains only serializable construction inputs. A live Harbor runner contains locks, event loops, and an
orchestrator and must not be serialized into coordinator actors.

`rollout.fanout.enabled` is removed. Pool size, CPU allocation, executor size, and coordinator RPC timeout remain
typed resource settings. These settings tune the selected process pool; they do not select execution placement.

### Process-isolated runner

The process-isolated implementation is a `TrajectoryRunner`, not a trainer feature or a Harbor-specific call-site
substitution. It owns coordinator construction, startup, evaluation sessions, request dispatch, result assembly,
retention, and shutdown.

For every request it:

1. validates that every row has a `TrajectoryID` and that all row-oriented request fields have equal length;
2. partitions row indices by `TrajectoryID.instance_id`, preserving first-seen group order;
3. sends each complete group to one coordinator, without splitting repetitions across processes;
4. awaits all group results under the existing coordinator RPC watchdog;
5. validates one output for every requested `TrajectoryID` and restores the original row order;
6. aggregates batch metrics and group capture steps through the existing trajectory-batch collation contract; and
7. runs shared finalization and retention once on the reconstructed request and output.

Reward shaping that depends on sibling trials remains coordinator-local because every reward group is atomic. GRPO
therefore retains exactly `k` trials per group, and ragged algorithms retain their configured group floor. Request
rows need not be contiguous: reconstruction uses trajectory identity, not concatenation order.

A coordinator failure remains a failure of the caller's `run()` operation. The wrapper adds the affected group
identity to the error and does not manufacture a successful group or retry a passthrough classification. The RPC
watchdog does not cancel remote work; shutdown remains responsible for bounded cleanup.

Fully asynchronous generation already sends one group per call, so partitioning produces one partition. Synchronous
generation sends multiple groups and gains parallel coordinator use without trainer-specific code.

### HTTP bridge observations

The HTTP bridge extends the canonical snapshot-and-callback metrics contract introduced in
[#460](https://github.com/marin-community/MarinSkyRL/pull/460). It does not configure a second telemetry runtime,
publish directly from the Uvicorn thread, or mutate trainer metrics from that thread.

A thread-safe accumulator in the bridge is the sole producer of `HTTPBridgeStatsSnapshot`. The snapshot carries
cumulative histograms and an interval view with `PEEK` and `RESET` semantics matching `InferenceStatsSnapshot`:

- `event_loop_lag_seconds`, measured from scheduled versus actual wake time on the Uvicorn loop;
- `response_bytes`, measured from the rendered non-streaming body and emitted streaming chunks; and
- `json_serialization_seconds`, measured around rendering the response body that is returned to the client.

Metric labels are bounded to endpoint, transport mode, and status class. Model names, session IDs, trajectory IDs,
task IDs, and exception text are excluded.

The inference stats callback consumes one typed inference snapshot containing engine and bridge observations. At a
training-step boundary, it reads `RESET` and projects count, mean, p95, and maximum bridge values into
`trainer.all_metrics` under `inference_bridge/*`. Console, W&B, and other configured trackers continue to receive
that flat payload through the trainer. The callback's periodic `PEEK` read publishes complete labelled histograms
through the existing Rigging/Finelog sink. Both projections consume the same snapshot; neither reads bridge internals.

## Enforcement

CPU tests cover behavior through the runner and metrics interfaces:

- the same production Harbor workload selects process isolation for synchronous and fully asynchronous trainers;
- development and test workloads can construct the in-process runner without a production configuration escape
  hatch;
- an interleaved multi-group request is distributed by group, never splits repetitions, and returns rows in request
  order;
- one-group requests preserve current fully asynchronous behavior, evaluation lifecycle, reward shaping, metrics, and
  retention;
- malformed or incomplete coordinator output fails before a training batch is returned;
- bridge interval reads reset only interval state, while periodic reads preserve the next step's values;
- tracker and typed-telemetry projections receive the same bridge snapshot and remain independent on sink failure; and
- dependency-boundary tests reject direct Rigging publishers and tracker mutation from the HTTP endpoint.

A bounded local load test sends 96 concurrent requests through the real Uvicorn endpoint and verifies response
accounting and connection completion. It is a regression signal for the reported workload, not proof that Python can
never experience scheduling delay. Production follow-up confirms that a synchronous multi-group Harbor batch uses all
configured coordinators and exports bridge metrics.

## Alternatives and limits

Keeping the gate in `FullyAsyncRayPPOTrainer` preserves the current failure: execution placement changes when model
placement changes. Selecting isolation from a request-size threshold makes restarts and evaluation sessions depend on
incidental batching and is rejected. A bridge-specific metrics exporter would recreate the multiple-owner metrics
paths removed by #460.

Moving Uvicorn into a separate process would isolate its GIL from the trainer. This design defers that change until
the bridge measurements identify contention in the Uvicorn process after Harbor processing is isolated. The typed
snapshot remains valid if the bridge later moves behind an RPC boundary.

Process isolation adds Ray actors, serialization, CPU reservations, and coordinator lifecycle failures. Keeping each
reward group intact bounds request fragmentation, while configurable coordinator count and CPU allocation prevent the
pool from multiplying Harbor or Daytona concurrency.

## Rollout

Land this specification before its implementation. The implementation then replaces trainer-owned fan-out with the
workload-owned builder, generalizes the dispatcher into the process-isolated runner, removes the obsolete enable gate,
and extends the existing inference metrics snapshot and callback.

The migration preserves Harbor retry and exception taxonomy, reward and baseline semantics, trainer admission rules,
inference routing, and weight synchronization. In-process construction remains available to CPU tests and explicit
development entrypoints. Rollback restores the previous runner factory and dispatcher gate as one change; the new
observations do not affect training decisions.
