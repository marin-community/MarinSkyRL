"""Illustrative nested-divergence probe, not a known bug-6111901 reproducer."""

import ctypes as c
import importlib.metadata
import hashlib
import json
import os
from pathlib import Path
import sys

import numpy as np
import torch

source = r'''
extern "C" __global__ void nested(const int* in, int* out, int n, int seed) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n) {
        int x = in[tid], v = x + seed;
        if ((tid & 31) < 16) {
            if (x > 0) {
                v = v * 3 + 7;
                if (x & 1) {
                    v = v - 11;
                    if (x & 4) { v = v * 2 + 1; }
                    v = v + 13;
                }
                v = v - 17;
            }
            v = v + 19;
        }
        out[tid] = v * 2 - x;
    }
}
'''
root = Path(sys.prefix)/'lib/python3.12/site-packages/nvidia/cu13'
nvrtc = c.CDLL(str(root/'lib/libnvrtc.so.13'))
driver = c.CDLL('libcuda.so.1')


def check(result):
    assert result == 0, result


torch.cuda.init()
context_anchor = torch.empty(1, device='cuda')
program = c.c_void_p()
check(nvrtc.nvrtcCreateProgram(c.byref(program), source.encode(), b'nested.cu', 0, None, None))
arch = ''.join(map(str,torch.cuda.get_device_capability()))
options = (c.c_char_p*2)(f'--gpu-architecture=sm_{arch}'.encode(), b'--std=c++17')
status = nvrtc.nvrtcCompileProgram(program, 2, options)
log_size = c.c_size_t()
check(nvrtc.nvrtcGetProgramLogSize(program,c.byref(log_size)))
log = c.create_string_buffer(log_size.value)
check(nvrtc.nvrtcGetProgramLog(program,log))
assert status == 0, log.value.decode()
size = c.c_size_t()
check(nvrtc.nvrtcGetCUBINSize(program,c.byref(size)))
cubin = c.create_string_buffer(size.value)
check(nvrtc.nvrtcGetCUBIN(program,cubin))
module = c.c_void_p()
check(driver.cuModuleLoadData(c.byref(module),cubin))
function = c.c_void_p()
check(driver.cuModuleGetFunction(c.byref(function),module,b'nested'))
cases = []
for n in [31,32,33,255,256,257,4097]:
    raw = ((np.arange(n,dtype=np.int32)*37)%127)-63
    gpu = torch.from_numpy(raw).cuda()
    output = torch.empty_like(gpu)
    for seed in [0,7,-13,101]:
        values = [c.c_uint64(gpu.data_ptr()),c.c_uint64(output.data_ptr()),c.c_int(n),c.c_int(seed)]
        parameters = (c.c_void_p*4)(*[c.cast(c.byref(v),c.c_void_p) for v in values])
        check(driver.cuLaunchKernel(function, (n+255)//256,1,1,256,1,1,0,None,parameters,None))
        torch.cuda.synchronize()
        # Independent scalar CPU interpreter, with no CUDA calls or shared helper.
        expected = []
        for tid,x in enumerate(raw.tolist()):
            v = x+seed
            if tid%32 < 16:
                if x>0:
                    v = v*3+7
                    if x%2:
                        v -= 11
                        if x&4:
                            v = v*2+1
                        v += 13
                    v -= 17
                v += 19
            expected.append(v*2-x)
        actual = output.cpu().numpy()
        mismatch = int(np.count_nonzero(actual!=expected))
        cases.append({'n':n,'seed':seed,'mismatches':mismatch})
        assert mismatch==0, cases[-1]
check(driver.cuModuleUnload(module))
check(nvrtc.nvrtcDestroyProgram(c.byref(program)))
report={'purpose':'illustrative nested-divergence probe; not known 6111901 reproduction',
        'device':torch.cuda.get_device_name(),'architecture':arch,
        'source_sha256':hashlib.sha256(source.encode()).hexdigest(),
        'cubin_sha256':hashlib.sha256(cubin.raw).hexdigest(), 'cubin_bytes':size.value,
        'compiler_package':importlib.metadata.version('nvidia-cuda-nvrtc'),'cases':cases,'passed':True}
(Path(os.environ['TRAINER_QUALIFICATION_OUTPUT']) / 'divergence.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
