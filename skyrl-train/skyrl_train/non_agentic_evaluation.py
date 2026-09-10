"""Request-scoped package and common-off evaluation for the K16 parser contract."""

from copy import deepcopy

from omegaconf import open_dict
from skyrl_gym.envs.thinking_contract import THINKING_CONTRACT_VERSION

METRIC_VERSION = "post-thinking-native-metrics-v1"
ENDPOINTS = ("common_off", "package_on")


def request_endpoint(request, parser_protocol):
    endpoint = request.get("non_agentic_evaluation_endpoint")
    if endpoint is None:
        return None
    metadata = request.get("batch_metadata")
    if endpoint not in ENDPOINTS or metadata is None or metadata.training_phase != "eval":
        raise ValueError("Non-agentic endpoints require a named evaluation request")
    if parser_protocol != THINKING_CONTRACT_VERSION:
        raise ValueError("Non-agentic endpoints require the versioned post-thinking parser")
    return endpoint


async def evaluate_endpoints(evaluate_fn, *, cfg, **kwargs):
    """Evaluate installed weights sequentially without changing the training config.

    Token-intervention arms generate both views. Shaping-only arms generate one
    package view; its token-identical common-off correctness is an explicit alias
    in the experiment receipt, not a second generated dump or raw reward claim.
    """
    modes = list(cfg.generator.get("non_agentic_eval_endpoints", []))
    if not modes:
        return await evaluate_fn(cfg=cfg, **kwargs)
    if cfg.trainer.step_wise_training or cfg.generator.non_agentic_parser_protocol != THINKING_CONTRACT_VERSION:
        raise ValueError("Endpoint evaluations require the single-turn post-thinking path")
    async_cfg = cfg.trainer.get("fully_async", {})
    if async_cfg.get("eval_mode", "blocking") != "blocking" or async_cfg.get("eval_on_installed_weights", False):
        raise ValueError("Quality endpoints require blocking evaluation of freshly published weights")
    intervention = cfg.generator.get("non_agentic_intervention")
    expected = list(ENDPOINTS) if intervention is not None else ["package_on"]
    if modes != expected:
        raise ValueError("Endpoint list differs from the declared generation intervention")
    namespace = kwargs.get("dump_namespace")
    combined = {}
    for endpoint in modes:
        endpoint_cfg = deepcopy(cfg)
        with open_dict(endpoint_cfg.generator):
            endpoint_cfg.generator.non_agentic_evaluation_endpoint = endpoint
        arguments = {**kwargs, "dump_namespace": f"{namespace}_{endpoint}" if namespace else endpoint}
        metrics = await evaluate_fn(cfg=endpoint_cfg, **arguments)
        if endpoint == "package_on":
            combined.update(metrics)
        combined.update({f"eval/{endpoint}/{key.removeprefix('eval/')}": value for key, value in metrics.items()})
    return combined
