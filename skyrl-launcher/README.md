# MarinSkyRL launcher

`marinskyrl-launcher` is the launch-host package for MarinSkyRL. It contains the
Iris request, submission, monitoring, and task-bundle code without installing
the GPU training stack. `skyrl-train`, Ray, Torch, vLLM, and CUDA remain in the
immutable task image.

The machine interface accepts one JSON request and writes one JSON response:

```bash
marinskyrl iris launch --request request.json
```

Human-readable logs are written to stderr. A successful terminal response names
the exact durable checkpoint and Hugging Face policy export that downstream
evaluation may consume.
