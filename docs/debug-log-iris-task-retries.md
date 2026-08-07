# Debugging log for Iris task retries

Verify whether the launcher's `--max-retries` value reaches both Iris failure budgets and restore the contract used by the Iris CLI.

## Initial status

The launcher passes `max_retries_failure` to `IrisClient.submit` but omits `max_task_failures`. Iris defaults the omitted job-wide budget to zero, so the first failed task can terminally fail the job before its per-task retry is admitted.

## Hypothesis

Passing the same configured value to both submit fields makes `--max-retries` a coherent application-failure budget, matching `iris job run`.

## Changes to make

Capture a real normalized launch's submit arguments and require both budgets to equal the CLI value. Add the omitted keyword without changing preemption classification or retry policy.

## Results

Confirmed. Before the source change, the recorded submission carried `max_retries_failure=3` and no `max_task_failures` key. After adding the missing argument, the same normalized launch passed three to both budgets. The full Iris launcher suite passed with 254 tests.
