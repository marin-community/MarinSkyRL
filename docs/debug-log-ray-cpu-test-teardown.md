# Debugging log for Ray CPU-test teardown

Ensure a failing local CPU test exits within a bounded time and does not leave a Ray session running.

## Initial status

The session-scoped autouse fixture in `skyrl-train/tests/cpu/conftest.py` starts Ray before every CPU-test run and calls
`ray.shutdown()` after the session. The reported macOS failure completed in isolation and Linux CI passed, but a failing
full suite remained alive while `gcs_server.out` reported Ray's 30-second graceful actor-shutdown timeout.

## Hypothesis 1

Starting Ray for tests that do not use it unnecessarily couples every test failure to Ray's local-cluster shutdown path.
Only tests that execute Ray work should own a Ray fixture; pure CPU tests should not create a Ray session.

## Changes to make

Identify every CPU test that uses a real Ray runtime and determine whether an explicit fixture can limit the cluster
lifetime without changing those tests' behavior.

## Results

Confirmed. The stale context-parallel baseline fails without using Ray, yet the autouse fixture starts a local cluster
before it. The fixture also affects local-only registry tests by making them create named Ray actors.

## Hypothesis 2

The persistent registry actor and Ray's automatic uv runtime environment make teardown both slow and failure-prone.
Under `uv run --frozen pytest`, Ray 2.51.1 packages the 19 MiB trainer checkout and creates a separate 134-package uv
environment for each local Ray session. `test_registry_reset_after_ray_shutdown` restarts Ray with a named actor alive;
the second worker environment consumed the remaining `/tmp` space during reproduction. Ray's reported
`Graceful shutdown timeout (30000ms) exceeded` comes from `GcsActorManager::DestroyActor`, which waits for an actor to
exit before force-killing it.

## Changes to make

Disable Ray's uv runtime hook for CPU tests so workers use the locked test environment, make the Ray fixture explicit,
and force-kill the two named registry actors before shutting down their local session. Use function scope by default and
a lazily requested module scope for actor-heavy tests that need session reuse.

## Results

The regression test initially failed because the autouse fixture initialized Ray for a pure test. Ray then tried to
upload a 1.1 GiB core dump left by the timed reproduction, demonstrating that the implicit uv working-directory upload
also couples unrelated checkout contents to test startup.

After the first fixture change, the full CPU suite completed with its five pre-existing failures in 437.61 seconds and
returned control without `ray stop`. It exposed one new failure: module-scoped sessions left a cached registry actor
handle pointing into the previous cluster. Function scope is required for isolated Ray tests; actor-heavy tests can
share a lazily requested module fixture. Teardown must invalidate registry actor handles after force-killing the named
actors.

A clean regression run with the five unrelated, pre-existing failures deselected completed with 763 passing tests and
17 skipped tests in 412.18 seconds. Pytest returned normally after the explicitly scoped Ray fixtures shut down.
The stale context-parallel baseline failure itself now reports and returns in 0.38 seconds without starting Ray.

## Future work

- [ ] Confirm macOS no longer enters Ray's graceful actor shutdown path.
