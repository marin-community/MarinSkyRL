"""Resolve the telemetry endpoint and run identity for one Iris task.

`skyrl_train.telemetry` reads `SKYRL_TELEMETRY_ENDPOINT`, `SKYRL_RUN_ID` and
`SKYRL_EXECUTION_UID` and stays inert when they are unset. This module is the writer.
Resolution happens inside the task, not on the launch host: the log server is registered
in the cluster's endpoint registry, so only an in-cluster client resolves it to an address
a pod can reach, and the run identity has to be the one Iris itself would stamp.
"""

from __future__ import annotations

import logging
import os
import sys

from iris.client.client import get_iris_ctx
from iris.cluster.client.job_info import get_job_info
from iris.cluster.endpoints import LOG_SERVER_ENDPOINT_NAME, TELEMETRY_ENDPOINT_PATH

from cloud.iris.env_vars import EXECUTION_UID_ENV, RUN_ID_ENV, TELEMETRY_ENDPOINT_ENV

logger = logging.getLogger(__name__)


def execution_uid(task_id: str, attempt_id: int, attempt_uid: str | None = None) -> str:
    """Spell one task attempt the way Iris's own producers spell it.

    The pinned `marin-iris` builds this from the task id and attempt number alone. Newer Iris
    prefers an attempt uid when the runtime sets one, so this reads the uid first and falls back
    to the same formula: the string matches whichever version is installed, rather than matching
    the pin today and silently diverging when it is bumped.
    """
    if attempt_uid:
        return f"iris:{attempt_uid}"
    return f"iris:{task_id}:attempt:{attempt_id}"


def telemetry_environment(*, run_id: str | None = None) -> dict[str, str]:
    """Return the telemetry variables for this task, or nothing outside a cluster.

    Args:
        run_id: Experiment identity to join rows on. Precedence is this argument, then an
            inherited SKYRL_RUN_ID, then the Iris job id. The initiator names the run when
            it has a name for it; deriving from the job is the fallback, not the rule.
    """
    try:
        job_info = get_job_info()
        ctx = get_iris_ctx()
        if job_info is None or ctx is None or ctx.client is None:
            logger.debug("no in-cluster Iris context; MarinSkyRL telemetry stays inert")
            return {}
        endpoint = ctx.client.resolve_endpoint(LOG_SERVER_ENDPOINT_NAME).rstrip("/") + TELEMETRY_ENDPOINT_PATH
        return {
            TELEMETRY_ENDPOINT_ENV: endpoint,
            RUN_ID_ENV: run_id or os.environ.get(RUN_ID_ENV) or str(job_info.job_id),
            EXECUTION_UID_ENV: execution_uid(
                str(job_info.task_id), job_info.attempt_id, getattr(job_info, "attempt_uid", None)
            ),
        }
    except Exception:
        logger.warning("could not resolve the MarinSkyRL telemetry environment", exc_info=True)
        return {}


def _main(argv: list[str]) -> None:
    if len(argv) >= 3 and argv[1] == "--":
        os.execvpe(argv[2], argv[2:], {**os.environ, **telemetry_environment()})
    raise SystemExit(f"usage: {argv[0]} -- COMMAND [ARG ...]")


if __name__ == "__main__":
    _main(sys.argv)
