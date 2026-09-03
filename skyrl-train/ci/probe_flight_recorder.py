"""O5 — which flight recorder does a CUDA torch actually expose, and does ours see NCCL?

`_dump_fr_trace_json` and `_dump_nccl_trace_json` are NOT a rename of each other. They are two
recorders: the generic FlightRecorder<c10::Event> (FlightRecorder.hpp:329) and
ProcessGroupNCCL's own FlightRecorder<at::cuda::CUDAEvent> (ProcessGroupNCCL.hpp:1520).
flight_recorder_summary.py reads the first; NCCL writes to the second. On a CPU wheel the NCCL
symbol is absent entirely, which is what made the wrong one look correct.

This runs on ONE GPU and prints the answer.
"""
import json, os, sys
import torch
import torch.distributed as dist

c = torch._C._distributed_c10d
print("torch", torch.__version__, "cuda", torch.cuda.is_available())
print("dump symbols:", sorted(n for n in dir(c) if "dump" in n.lower()))

os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
os.environ.setdefault("MASTER_PORT", "29591")
os.environ.setdefault("TORCH_NCCL_ENABLE_TIMING", "1")
os.environ.setdefault("TORCH_NCCL_TRACE_BUFFER_SIZE", "20000")
dist.init_process_group("nccl", rank=0, world_size=1)
t = torch.ones(1024, device="cuda")
for _ in range(8):
    dist.all_reduce(t)
torch.cuda.synchronize()

for name in ("_dump_fr_trace_json", "_dump_nccl_trace_json"):
    fn = getattr(c, name, None)
    if fn is None:
        print(f"{name:26s} ABSENT")
        continue
    p = json.loads(fn(True, False))
    e = p.get("entries") or ()
    timed = sum(1 for x in e if x.get("duration_ms") is not None)
    print(f"{name:26s} entries={len(e):4d}  with duration_ms={timed:4d}  keys={sorted(p)[:4]}")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from skyrl_train.distributed import flight_recorder_summary as frs  # noqa: E402
tr = frs._dump_trace()
n = len(((tr or {}).get("entries")) or ())
print(f"\nOUR _dump_trace() -> entries={n}")
print("VERDICT:", "READS NCCL — instrument works" if n else
      "READS THE WRONG (EMPTY) RECORDER — this is O5's root cause")
dist.destroy_process_group()
