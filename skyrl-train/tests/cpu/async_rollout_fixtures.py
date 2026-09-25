"""Build the production admission reader around lightweight trainer fixtures."""

from skyrl_train.async_rollout_buffer import AsynchronousRolloutBuffer


async def _noop(*args):
    pass


def reader_for(trainer, queues):
    staleness_manager = getattr(trainer, "_staleness_manager", None)
    return AsynchronousRolloutBuffer(
        queues,
        mini_batch_size=trainer.mini_batch_size,
        admission_policy=trainer._group_admission_policy,
        selection_policy=trainer._group_selection_policy,
        max_candidate_groups=trainer._dynamic_sampling_max_candidate_groups,
        max_sample_batches=getattr(trainer, "_dynamic_sampling_max_sample_batches", 0),
        current_step=lambda: trainer.global_step,
        consumed_uids=trainer.data_tracker.get_consumed_uids_in_epoch,
        step_time_history=trainer._step_time_history,
        stall_timeout=trainer.group_admission_stall_timeout,
        on_admitted=trainer._submit_admitted_groups_for_teacher_scoring,
        on_discarded=staleness_manager.on_rollouts_discarded if staleness_manager is not None else _noop,
        retain=_noop,
        metrics=lambda: trainer.all_metrics,
    )
