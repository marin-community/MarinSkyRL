# Watching an RL run on Grafana

The trainer is instrumented. `skyrl_train/telemetry.py` publishes what each step measured, each
rollout engine publishes its own vLLM metrics, and the Ray head publishes the raylet's. All of it
goes to the cluster's finelog, which forwards to the `marin` hub, which is what Grafana queries.
Nothing is scraped from your logs and nothing is written to disk.

Both trainers use that same contract, so one dashboard covers synchronous and fully asynchronous
runs: **RL Post-training**, at <https://grafana.oa.dev/d/marin-rl-runs>.

## Finding it

From <https://grafana.oa.dev/>, open **Dashboards** in the left sidebar and pick **RL Post-training**
out of the `Marin` folder. Home, Runs and the two Inference dashboards also link to it from their
top bar. Home lists recent RL runs at the bottom as well; clicking one opens the view already
framed on that run.

## Launching so that your run shows up

Publishing turns itself on only when `SKYRL_TELEMETRY_ENDPOINT`, `SKYRL_RUN_ID` and
`SKYRL_EXECUTION_UID` are set in the training process. Nothing sets them outside an Iris cluster,
so a laptop run publishes nothing and never says so.

**From marin.** Nothing to do. A step built with `marin.rl.skyrl` launches through the packaged
launcher, whose pod entrypoint resolves those variables before starting the trainer:

```bash
uv run python -m experiments.post_training.iceball_micro --version 2026.09.10 --stage rl --run
```

Your run id is the step's own `<step_name>-<version>`, so a run is findable from either side.

**From MarinSkyRL.** Enter through the task entrypoint, which resolves the same variables from the
task's Iris context and starts Ray with the metrics port the collector scrapes:

```bash
iris --cluster marin job run --target-cluster cw-rno2a --gpu H100x1 -- \
  python cloud/iris/task_runtime.py --run-id my-experiment-2026.09.10 -- \
  python -m skyrl_train.entrypoints.main_base <hydra args>
```

Pick a run id you will recognise next week; the picker shows it verbatim beside everyone else's.
Without `--run-id` it defaults to the Iris job id, which reads like `/runner/…-34231183130-1` and
has to be percent-encoded into a URL.

**What does not work** is calling the trainer directly:

```bash
iris ... job run -- python -m skyrl_train.entrypoints.main_base ...   # publishes nothing
```

That skips the resolver, so the variables stay unset and the run trains normally while reporting
nothing. If you cannot use `task_runtime.py`, the resolver alone covers everything except the two
Ray panels: `python -m cloud.iris.telemetry_env -- python -m skyrl_train.entrypoints.main_base …`.

Confirm it took by grepping your job's log:

```
[task-runtime] [telemetry] http://finelog-…:10001/v1/telemetry run_id=<your run id>
```

If that line is absent, nothing was published, and no amount of looking at Grafana will help.

## Opening one run

The picker only offers runs that reported inside the dashboard's time window, so yesterday's run is
invisible on a one-hour window. Widen the window first, or address the run directly:

```
https://grafana.oa.dev/d/marin-rl-runs?var-cluster=cw-rno2a&var-run=<run id>&from=now-24h&to=now
```

`var-cluster` is the cluster the run executed on, not the federation hub. Percent-encode a run id
that begins with a slash, or the parameter truncates there and selects nothing:
`var-run=%2Frunner%2Fmarinskyrl-…`.

## Reading it

The rows answer five questions, in order down the page: is it alive and moving, where did the wall
clock go, why is generation slow, is it learning, and how is Ray's object store holding up.

**Start with the producer census** when something looks wrong. It lists every service, role and
metric source that stamped your run id, and a healthy synchronous run shows four rows: trainer,
trainer with `metric_source=vllm`, controller, and controller with `metric_source=ray`. A missing
row is direct evidence that a producer never started -- most often the Ray one, which means the
launch did not go through `task_runtime.py`.

Four panels can be legitimately empty, for two unrelated reasons, and their titles say which.
Rollout buffer occupancy and off-policy staleness are asynchronous-only: a synchronous run is
on-policy by construction and has no buffer. The two Ray panels need a launch that starts Ray
itself, which is a launch-path question rather than a synchronous-versus-asynchronous one.

Two properties of the data mislead people. Work counters are deltas, so they sum; gauges such as
`policy_step` are snapshots, so they do not. And the engine's metrics arrive under the same service
as the trainer's, told apart only by `metric_source`.

## Asynchronous runs

A fully asynchronous run fills every panel here, plus the rollout buffer and staleness ones.

There is also a deeper asynchronous view, **Async RL Training** (`/d/marin-async-rl`), with about
sixty panels. **It is not merged and not on grafana.oa.dev.** It lives on
`atqamar/async-non-agentic-rl-v2-dashboard` in marin, alongside the asynchronous instrumentation on
a branch of this repository. Until both land, use the dashboard above.

## The nightly

The scheduled nightly publishes under
`nightly-gsm8k-h100-<strategy>-<date>-<github run id>-<attempt>`. Its GitHub job summary links to
its own view and lists which signals arrived, so a blank panel is already explained there -- and
saying so never fails the run.
