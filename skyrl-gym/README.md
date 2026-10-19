# SkyRL task sessions

`skyrl-gym` supplies task sessions and graders for MarinSkyRL.

The [StepHarness interface](https://github.com/marin-community/marin/blob/main/lib/rolloutengine/src/rolloutengine/harness/step.py) runs episodes that define `prepare`, `advance`, and `close`. SkyRL's episode adapter records per-turn rewards and grades. Marin's rollout engine calls the model and records exact tokens. Shellbox supplies machines for executable tasks.

Pure answer tasks do not create a machine. Python, SQL, Lean, and OpenEnv tasks
use the requirements in `TaskSpec` and machine selection in `TaskRolloutSpec.task_machine`. Private reference answers stay on
the rollout worker.

From the repository root:

```bash
uv sync --project skyrl-gym --frozen --extra dev
uv run --project skyrl-gym --frozen pytest skyrl-gym/tests/
```

The CPU tests execute trusted programs through local subprocesses. These fixtures do not provide container isolation.

See [custom task sessions](../skyrl-train/docs/tutorials/new_env.rst) and [canonical task rollouts](../skyrl-train/docs/tutorials/task_rollouts.rst).
