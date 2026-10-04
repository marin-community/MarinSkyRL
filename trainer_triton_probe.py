"""Retain the actual assembler invocation for a fresh Triton compilation."""

import hashlib
import json
import os
import subprocess
from pathlib import Path

import torch
import triton
import triton.language as tl
from triton.backends.nvidia.compiler import get_ptxas


@triton.jit
def fixed_arithmetic(x, y, count: tl.constexpr, BLOCK: tl.constexpr):
    index = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    value = tl.load(x + index, index < count, other=0)
    output = tl.where(index % 2 == 0, value * 2 + 0.25, value * 4 - 0.125)
    tl.store(y + index, output, index < count)


def main():
    cache = Path(os.environ['TRITON_CACHE_DIR'])
    assert not cache.exists() or not any(cache.iterdir()), 'Use a fresh owned Triton cache'
    capability = torch.cuda.get_device_capability()
    ptxas = Path(get_ptxas(capability[0] * 10 + capability[1]).path)
    assert ptxas.resolve() == Path(os.environ['CUDA_HOME'], 'bin/ptxas').resolve()
    with ptxas.open('rb') as stream:
        digest = hashlib.file_digest(stream, 'sha256').hexdigest()
    invocations = []
    original = subprocess.run

    def observed_run(command, *args, **kwargs):
        if isinstance(command, list) and Path(command[0]).resolve() == ptxas.resolve():
            invocations.append(list(map(str, command)))
        return original(command, *args, **kwargs)

    subprocess.run = observed_run
    try:
        x = (torch.arange(1025, dtype=torch.float32) % 17 - 8) / 8
        expected = torch.where(torch.arange(len(x)) % 2 == 0, x * 2 + 0.25, x * 4 - 0.125)
        gpu = x.cuda()
        result = torch.empty_like(gpu)
        compiled = fixed_arithmetic[(triton.cdiv(len(x), 256),)](gpu, result, len(x), BLOCK=256)
        torch.cuda.synchronize()
        torch.testing.assert_close(result.cpu(), expected, rtol=0, atol=0)
    finally:
        subprocess.run = original
    assert invocations, 'No fresh assembler invocation was recorded'
    output = Path(os.environ['TRAINER_QUALIFICATION_OUTPUT'])
    output.mkdir(parents=True, exist_ok=True)
    (output / 'fresh-triton.ptx').write_text(compiled.asm['ptx'])
    report = {
        'purpose': 'fresh compilation and exact arithmetic; not a known bug6111901 reproducer',
        'device': torch.cuda.get_device_name(), 'capability': capability,
        'ptxas_path': str(ptxas), 'ptxas_sha256': digest,
        'ptxas_version': subprocess.check_output([str(ptxas), '--version'], text=True),
        'actual_assembler_invocations': invocations,
        'ptx_sha256': hashlib.sha256(compiled.asm['ptx'].encode()).hexdigest(),
        'cubin_sha256': hashlib.sha256(compiled.asm['cubin']).hexdigest(),
        'passed': True,
    }
    (output / 'fresh-triton.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
