"""Exercise Hero's actual serving ShortConv Triton decode path against scalar CPU math."""

import hashlib
import json
import os
from pathlib import Path
import time

import torch
from triton.backends.nvidia.compiler import get_ptxas
from vllm.model_executor.layers.mamba.ops.causal_conv1d import causal_conv1d_update

torch.manual_seed(23)
rows=[]
for dtype in [torch.float32,torch.bfloat16]:
    # Binary fractions give an exact independent reference for four taps.
    states=(torch.randint(-8,9,(4,8,3)).float()/8).to(dtype)
    weights=(torch.randint(-8,9,(8,4)).float()/8).to(dtype)
    inputs=(torch.randint(-8,9,(3,8)).float()/8).to(dtype)
    indices=[2,1,3]
    expected=[]
    expected_state=states.clone()
    for b,index in enumerate(indices):
        row=[]
        for channel in range(8):
            taps=states[index,channel].float().tolist()+[float(inputs[b,channel])]
            row.append(sum(taps[j]*float(weights[channel,j]) for j in range(4)))
        expected.append(row)
        expected_state[index]=torch.cat((states[index,:,1:],inputs[b,:,None]),dim=1)
    expected=torch.tensor(expected).to(dtype)
    state=states.cuda()
    actual_input=inputs.cuda()
    start=time.monotonic()
    output=causal_conv1d_update(actual_input,state,weights.cuda(),conv_state_indices=torch.tensor(indices,device='cuda',dtype=torch.int32))
    torch.cuda.synchronize()
    torch.testing.assert_close(output.cpu(),expected,rtol=0,atol=0)
    torch.testing.assert_close(state.cpu(),expected_state,rtol=0,atol=0)
    rows.append({'dtype':str(dtype),'value_exact':True,'state_exact':True,'cold_seconds':time.monotonic()-start})
ptxas=Path(get_ptxas(torch.cuda.get_device_capability()[0]*10).path)
with ptxas.open('rb') as f:digest=hashlib.file_digest(f,'sha256').hexdigest()
result={'passed':True,'cases':rows,'ptxas_path':str(ptxas),'ptxas_sha256':digest}
(Path(os.environ['TRAINER_QUALIFICATION_OUTPUT']) / 'jit.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
