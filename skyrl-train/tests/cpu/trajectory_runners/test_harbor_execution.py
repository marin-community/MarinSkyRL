from omegaconf import OmegaConf

from skyrl_train.trajectory_runners.harbor.execution import configured_concurrent_trials, per_worker_limits


def test_workers_share_harbor_limits_while_evaluation_keeps_the_pool_concurrency():
    config = OmegaConf.create(
        {"harbor": {"n_concurrent_trials": 32}, "environment": {"kwargs": {"connection_pool_maxsize": 10}}}
    )

    per_worker = per_worker_limits(config, 3)

    assert per_worker.harbor.n_concurrent_trials == 10
    assert per_worker.environment.kwargs.connection_pool_maxsize == 3
    assert per_worker_limits(config, 64).harbor.n_concurrent_trials == 1
    assert configured_concurrent_trials(config) == 32
