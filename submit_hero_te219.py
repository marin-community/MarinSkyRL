"""Submit an exact task-owned Hero qualification plan on the standard Iris runtime."""

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

from iris.cli.connect import open_iris_client
from iris.cluster.constraints import Constraint, ConstraintOp
from iris.cluster.platforms.k8s.coreweave_topology import gpu_gang_coscheduling_level
from iris.cluster.types import CoschedulingConfig, Entrypoint, EnvironmentSpec, ResourceSpec, gpu_device
from iris.rpc import job_pb2

from task_runtime_identity import SOURCE_MANIFEST, manifest_identity, verify_source_manifest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--submit', action='store_true')
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    subprocess.run(['git', 'diff', '--exit-code', 'HEAD', '--'], cwd=root, check=True)
    revision = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip()
    digest, _ = manifest_identity(root)
    source_digest = hashlib.sha256((root / SOURCE_MANIFEST).read_bytes()).hexdigest()
    verify_source_manifest(root, revision, source_digest)
    plan = json.loads(args.plan.read_text())
    env = plan['environment']
    env.update(HERO_SOURCE_REVISION=revision, HERO_RUNTIME_BUNDLE_SHA256=digest,
               HERO_SOURCE_MANIFEST_SHA256=source_digest)
    variant = env['HERO_VARIANT']
    gpus = 8 if variant == 'H100' else 4
    nodes = int(env['HERO_POLICY_NODES']) + int(env['HERO_SERVING_NODES'])
    pp, ep, cp = [int(env[f'HERO_{key}']) for key in ('PP', 'EP', 'CP')]
    world = int(env['HERO_POLICY_NODES']) * gpus
    assert 48 % pp == 0 and world % (pp * cp) == 0 and (world // pp) % ep == 0
    assert 16 % (world // (pp * cp)) == 0
    assert 384 % (int(env['HERO_SERVING_NODES']) * gpus) == 0
    level = gpu_gang_coscheduling_level(variant, gpus, nodes)
    plan.update(source_revision=revision, runtime_bundle_sha256=digest,
                source_manifest_sha256=source_digest, nodes=nodes, gpus_per_node=gpus,
                coscheduling=level, priority='interactive', retries_failure=0)
    print(json.dumps(plan, indent=2, sort_keys=True), flush=True)
    if not args.submit:
        return
    with open_iris_client(cluster_name=plan['cluster'], workspace=root) as client:
        job = client.submit(
            name=env['HERO_RUN_NAME'],
            entrypoint=Entrypoint.from_command('bash', 'hero_te219_task.sh'),
            resources=ResourceSpec(cpu=32, memory=f"{plan['host_memory_gib']}GB", disk='300GB',
                                   device=gpu_device(variant, gpus)),
            environment=EnvironmentSpec(env_vars=env, setup_scripts=[]),
            replicas=nodes,
            constraints=[Constraint.create(key='nvlink.domain', op=ConstraintOp.EQ,
                                           value=plan['nvlink_domain'])] if plan.get('nvlink_domain') else None,
            coscheduling=CoschedulingConfig(group_by=level),
            task_image=plan['task_image'],
            priority_band=job_pb2.PRIORITY_BAND_INTERACTIVE,
            max_retries_failure=0, max_task_failures=0, max_retries_preemption=3,
        )
        print('JOB_ID=' + str(job.job_id), flush=True)


if __name__ == '__main__':
    main()
