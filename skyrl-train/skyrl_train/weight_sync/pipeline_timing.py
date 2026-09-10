"""CUDA event intervals on one device; host scheduling is not overlap evidence."""

import math

import torch


def merged_intervals(intervals):
    merged = []
    for start, end in sorted(intervals):
        if not math.isfinite(start) or not math.isfinite(end) or start < 0 or end < start:
            raise ValueError("CUDA timing interval must be ordered and nonnegative")
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return merged


def overlap_seconds(left, right):
    left, right = merged_intervals(left), merged_intervals(right)
    total, i, j = 0.0, 0, 0
    while i < len(left) and j < len(right):
        total += max(0.0, min(left[i][1], right[j][1]) - max(left[i][0], right[j][0]))
        if left[i][1] < right[j][1]:
            i += 1
        else:
            j += 1
    return total


class CudaPipelineTiming:
    def __init__(self, device):
        self.device = device
        self.origin = torch.cuda.Event(enable_timing=True)
        self.origin.record(torch.cuda.current_stream(device))
        self.events = []
        self.finished = False

    def start(self, stage, stream):
        if self.finished:
            raise RuntimeError("CUDA timing catalogue is already complete")
        stream.wait_event(self.origin)
        begin = torch.cuda.Event(enable_timing=True)
        begin.record(stream)
        return stage, begin

    def end(self, token, stream):
        if self.finished:
            raise RuntimeError("CUDA timing catalogue is already complete")
        stage, begin = token
        end = torch.cuda.Event(enable_timing=True)
        end.record(stream)
        self.events.append((stage, begin, end))

    def finish(self):
        if self.finished:
            raise RuntimeError("CUDA timing catalogue is already complete")
        intervals = {}
        for stage, begin, end in self.events:
            end.synchronize()
            if not end.query():
                raise RuntimeError("CUDA stage event is not complete")
            intervals.setdefault(stage, []).append(
                [self.origin.elapsed_time(begin) / 1000, self.origin.elapsed_time(end) / 1000]
            )
        unions = {stage: merged_intervals(rows) for stage, rows in intervals.items()}
        self.finished = True
        self.events.clear()
        return {
            "schema": "cuda-pipeline-stages-v1",
            "clock": "one device, CUDA events relative to an origin event awaited by every stage stream",
            "units": "seconds",
            "intervals": intervals,
            "stage_seconds": {stage: sum(end - start for start, end in rows) for stage, rows in unions.items()},
            "export_send_overlap_seconds": overlap_seconds(intervals.get("export", []), intervals.get("nccl_send", [])),
            "receive_load_overlap_seconds": overlap_seconds(
                intervals.get("nccl_receive", []), intervals.get("load", [])
            ),
            "events_complete": True,
            "scope": "install only; GPU stream stage spans, not isolated kernel durations; unions and intersections on one device; no replay events",
        }
