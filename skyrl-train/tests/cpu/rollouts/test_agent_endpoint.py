import time

import pytest
from iris.client.client import IrisClient, IrisContext, iris_ctx_scope
from iris.cluster.client.job_info import JobInfo, get_job_info, set_job_info
from iris.cluster.types import PROXY_TIMEOUT_METADATA_KEY, EndpointAccess, JobName
from iris.rpc import controller_pb2

from skyrl_train.rollouts.agent_endpoint import registered_agent_endpoint


@pytest.mark.asyncio
@pytest.mark.parametrize("token_lifetime", [600, 1])
async def test_agent_endpoint_removes_registration_after_trial_or_token_failure(token_lifetime):
    endpoints = {}

    class Cluster:
        def register_endpoint(self, *, name, address, task_attempt, metadata, access):
            # Iris consumes this wire metadata; an omitted value imposes its 120-second default.
            assert float(metadata[PROXY_TIMEOUT_METADATA_KEY]) == 1800
            endpoints[name] = (address, task_attempt, access)
            return name

        def unregister_endpoint(self, endpoint_id):
            del endpoints[endpoint_id]

        def mint_endpoint_token(self, endpoint_name, ttl=None):
            assert endpoints[endpoint_name][2] == EndpointAccess.ENDPOINT_ACCESS_LINK
            response = controller_pb2.Controller.MintEndpointTokenResponse(
                capability_url=f"https://ingress.test/proxy/t/scoped/{endpoint_name.strip('/')}"
            )
            response.expires_at.epoch_ms = int((time.time() + token_lifetime) * 1000)
            return response

    info = JobInfo(task_id=JobName.from_wire("/fixture/agent/0"), advertise_host="10.0.0.2")
    previous_info = get_job_info()
    set_job_info(info)
    try:
        with iris_ctx_scope(IrisContext.from_job_info(info, IrisClient(Cluster()))):
            async with registered_agent_endpoint(8080, request_timeout=1800) as endpoint:
                assert endpoints[endpoint.name] == (
                    "http://10.0.0.2:8080",
                    info.task_attempt,
                    EndpointAccess.ENDPOINT_ACCESS_LINK,
                )
                if token_lifetime == 1:
                    with pytest.raises(ValueError, match="shorter than the configured agent attempt"):
                        await endpoint.api_base(required_lifetime=300)
                else:
                    url = await endpoint.api_base(required_lifetime=300)
                    assert url == f"https://ingress.test/proxy/t/scoped/{endpoint.name.strip('/')}/v1"
            assert endpoints == {}
    finally:
        set_job_info(previous_info)
