#!/bin/bash
# Native routed_experts validation on skyrl_megatron_vllm_r3_torch211.sif.
# NO overlay (native PR#39917 emits routed_experts directly). Run on a GPU node.
# Confirms routed_experts shape [gen_len, num_layers, top_k] matching the SkyRL
# consumers (generators/utils.py extract_routed_experts_from_rollout_details,
# models/router_replay.py: per-layer [num_tokens, top_k]).
set -x
SIF=/e/scratch/jureap59/feuer1/containers/skyrl_megatron_vllm_r3_torch211.sif
MODEL=$(ls -d /e/scratch/jureap59/feuer1/.cache/huggingface/hub/models--trl-internal-testing--tiny-Qwen3MoeForCausalLM/snapshots/*/ 2>/dev/null | head -1)
echo "MODEL=$MODEL"
PORT=8767
apptainer exec --nv --bind /e/scratch ${SIF} bash -lc "
set -x
export HF_HUB_OFFLINE=1 VLLM_LOGGING_LEVEL=WARNING
python -c 'import torch, vllm; print(\"TORCH\", torch.__version__, \"VLLM\", vllm.__version__)'
python -m vllm.entrypoints.openai.api_server \
  --model '${MODEL}' --served-model-name tinymoe \
  --enable-return-routed-experts --enforce-eager \
  --max-model-len 2048 --gpu-memory-utilization 0.30 --port ${PORT} \
  > /tmp/vllm_t211_server.log 2>&1 &
SRV=\$!
for i in \$(seq 1 90); do
  curl -s http://127.0.0.1:${PORT}/health >/dev/null 2>&1 && { echo READY_\$i; break; }
  kill -0 \$SRV 2>/dev/null || { echo SERVER_DIED; tail -60 /tmp/vllm_t211_server.log; exit 9; }
  sleep 3
done
curl -s http://127.0.0.1:${PORT}/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{\"model\":\"tinymoe\",\"messages\":[{\"role\":\"user\",\"content\":\"Hi\"}],\"max_tokens\":6,\"temperature\":0}' \
  > /tmp/vllm_t211_resp.json 2>&1
python - <<'PYEOF'
import json
d=json.load(open('/tmp/vllm_t211_resp.json')); ch=d['choices'][0]; msg=ch.get('message',{})
print('CHOICE_KEYS', list(ch.keys())); print('MSG_KEYS', list(msg.keys()))
re=ch.get('routed_experts') or msg.get('routed_experts')
print('routed_experts present:', re is not None)
if re is not None:
    def shp(x):
        s=[]
        while isinstance(x,list): s.append(len(x)); x=x[0] if x else None
        return s
    print('SHAPE [gen_len,L,K]:', shp(re))
    print('K (last dim) =', shp(re)[-1])
    print('first token row (L x K):', re[0] if re else None)
else:
    print(json.dumps(d, indent=2)[:1500])
PYEOF
kill \$SRV 2>/dev/null; sleep 2
"
