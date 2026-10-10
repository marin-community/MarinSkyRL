"""Read existing native benchmark results without issuing another training call."""

from functools import partial
from itertools import count
import json
from types import SimpleNamespace

import ray


def capture_step(policy, batch, *, native_step, output, client_factory, locate, update_indices):
    update = next(update_indices)
    references = []

    def capture_request(*args, **kwargs):
        request = policy.async_run_ray_method(*args, **kwargs)
        references.append(request)
        return request

    status = native_step(SimpleNamespace(async_run_ray_method=capture_request), batch)
    assert len(references) == 1, "The benchmark helper must issue one training request"
    results = ray.get(references[0])
    row = {
        "update": update,
        "timing_scope": "Existing per-rank CPU dispatch wall phases; overlapping GPU work is not additive.",
        "results": [
            {"result_index": index, "phases": result.metadata["qualification_phases"]}
            for index, result in enumerate(results)
        ],
    }
    if update == 0:
        row["runtime_settings"] = ray.get(
            policy.async_run_ray_method("pass_through", "qualification_performance_settings")
        )
    bucket, prefix = locate(output)
    client_factory().put_object(
        Bucket=bucket,
        Key=f"{prefix}/diagnostic-phases/update-{update}.json",
        Body=json.dumps(row, default=str).encode(),
    )
    return status


def install(helper, output, client_factory, locate):
    helper._train_step = partial(
        capture_step,
        native_step=helper._train_step,
        output=output,
        client_factory=client_factory,
        locate=locate,
        update_indices=count(),
    )

