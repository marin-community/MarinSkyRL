"""CPU-only lifecycle participants for trainer evaluation tests."""


class EvaluationLifecycle:
    def __init__(self):
        self.events = []

    async def pause_and_drain(self):
        self.events.append("drained")

    def resume(self):
        self.events.append("resumed")

    def set_trajectory_sink(self, sink):
        pass

    async def start_eval_session(self, **kwargs):
        self.events.append("started")

    async def stop_eval_session(self):
        self.events.append("stopped")
