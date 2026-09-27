import json
import os
import resource
import sys
import time

from loguru import logger
import torch
import torch.distributed


def _peak_rss_bytes() -> int:
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return peak if sys.platform == "darwin" else peak * 1024


class Profiler:
    """Capture one forward micro-batch on the selected ranks."""

    def __init__(self, config):
        self.saved = False
        self.running = False
        self.captured = False
        self.prof = None
        if not config.enable:
            return
        self.save_path = config.save_path
        self.capture_update_index = config.capture_update_index
        self.capture_mini_batch_index = config.capture_mini_batch_index
        self.export_chrome_trace = config.export_chrome_trace
        self.current_update_index = -1
        self.rank = torch.distributed.get_rank()
        if self.rank in config.ranks:
            logger.info(f"[Profiler] Profiler init for rank {self.rank}")
            activities = [torch.profiler.ProfilerActivity.CPU]
            if torch.cuda.is_available():
                activities.append(torch.profiler.ProfilerActivity.CUDA)
            self.prof = torch.profiler.profile(
                activities=activities,
                record_shapes=config.record_shapes,
                with_stack=config.with_stack,
            )

    def begin_update(self) -> None:
        if self.prof is not None and not self.saved:
            self.current_update_index += 1

    def start_mini_batch(self, index: int) -> None:
        if (
            self.prof is None
            or self.saved
            or self.current_update_index != self.capture_update_index
            or index != self.capture_mini_batch_index
        ):
            return
        logger.info(
            f"[Profiler] Capturing first forward micro-batch of update "
            f"{self.current_update_index}, mini-batch {index} on rank {self.rank}"
        )
        self.capture_started = time.perf_counter()
        self.peak_rss_bytes_before_capture = _peak_rss_bytes()
        self.prof.start()
        self.running = True

    def stop_capture(self) -> None:
        if not self.running:
            return
        stop_started = time.perf_counter()
        self.prof.stop()
        self.stop_wall_seconds = time.perf_counter() - stop_started
        self.running = False
        self.captured = True
        self.capture_wall_seconds = time.perf_counter() - self.capture_started
        self.peak_rss_bytes_after_capture = _peak_rss_bytes()

    def save(self):
        if not self.captured or self.saved:
            return
        export_started = time.perf_counter()
        os.makedirs(self.save_path, exist_ok=True)
        table_path = os.path.join(self.save_path, f"prof_rank_{self.rank}.txt")
        sort_key = "self_cuda_time_total" if torch.cuda.is_available() else "self_cpu_time_total"
        with open(table_path, "w", encoding="utf-8") as table_file:
            table_file.write(self.prof.key_averages().table(sort_by=sort_key, row_limit=50))
        logger.info(f"[Profiler] Saved operator table to {table_path}")
        trace_bytes = 0
        if self.export_chrome_trace:
            trace_path = os.path.join(self.save_path, f"prof_rank_{self.rank}.json")
            self.prof.export_chrome_trace(trace_path)
            trace_bytes = os.path.getsize(trace_path)
            logger.info(f"[Profiler] Saved Chrome trace to {trace_path}")
        self.prof = None
        self.saved = True
        metadata = {
            "capture_scope": "first_forward_micro_batch",
            "capture_update_index": self.capture_update_index,
            "capture_mini_batch_index": self.capture_mini_batch_index,
            "capture_wall_seconds": self.capture_wall_seconds,
            "stop_wall_seconds": self.stop_wall_seconds,
            "export_wall_seconds": time.perf_counter() - export_started,
            "peak_rss_bytes_before_capture": self.peak_rss_bytes_before_capture,
            "peak_rss_bytes_after_capture": self.peak_rss_bytes_after_capture,
            "peak_rss_bytes_after_export": _peak_rss_bytes(),
            "table_bytes": os.path.getsize(table_path),
            "trace_bytes": trace_bytes,
        }
        metadata_path = os.path.join(self.save_path, f"prof_rank_{self.rank}_metadata.json")
        with open(metadata_path, "w", encoding="utf-8") as metadata_file:
            json.dump(metadata, metadata_file, indent=2)


class CudaTimer:
    def __init__(self, device):
        self.device = device

        self.start_event = torch.cuda.Event(enable_timing=True)
        self.end_event = torch.cuda.Event(enable_timing=True)

    def __enter__(self):
        self.start_event.record()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_event.record()
        torch.cuda.synchronize(self.device)
        self.elapsed_time = self.start_event.elapsed_time(self.end_event)  # Calculate the elapsed time
