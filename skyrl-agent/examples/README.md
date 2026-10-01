## Examples

This directory contains historical examples for the dormant skyrl-agent snapshot. GeneralReactTask examples
(MemAgent, Search-R1, BrowseComp, and OpenAI ReAct) have been removed; see the
[retirement guide](../../docs/coder1-retirement.md).

### 1) SWE Training

- Setup remote runtime/server:
  - Refer to the [SkyRL-OpenHands](https://github.com/NovaSky-AI/SkyRL-OpenHands) documentation to set up a remote sandbox server and cache train/eval images.
  - After setup, configure your the remote server URL and API key in the environment file (i.e., `.env`).

- Prepare dataset:
  - Run the dataset preparation script:
    ```bash
    python ./data/swe_data.py --output SWE_DATA_PATH
    ```

- Launch training (modify the corresponding path in the script first):
  - VERL-based:
    ```bash
    bash ./examples/run_verl/verl_oh.sh
    ```
  - SkyRL-Train-based:
    ```bash
    bash ./examples/run_skyrl/skyrl_swe.sh
    ```

- Run inference (demo):
Launch an OpenAI API-compatible serving (e.g., vLLM or similar), then configure the `api_url` in the corresponding YAML (typically under the backend config) to point to your serving endpoint. Then run:
    ```bash
    python ./examples/run_openai/test_vllm_oh_demo.py
    ```

### 2) Deep Research (web_research_hle.sh)

- Quick setup: `uv venv && uv sync`.

- Required `.env`: `WANDB_API_KEY`, `GOOGLE_SEARCH_KEY` (Serper key), `JINA_API_KEYS`, `WEB_SUMMARY_API_BASE`, `WEB_SUMMARY_MODEL` (e.g., `Qwen/Qwen3-32B`), `SKYAGENT_WEB_CACHE_DIR`, `STEM_LLM_JUDGE_URL`; optional blocklists.

- Dataset:
  ```bash
  python ./data/deep_research.py --output-dir DR_DATA_DIR
  ```

- Web summary (required):
  - Point `WEB_SUMMARY_API_BASE` to your remote OpenAI-compatible endpoint (e.g., `http://host:port/v1`).
  - Keep the model name in `WEB_SUMMARY_MODEL`.
- Optional router (for load-balancing/failover):
  ```bash
  SUMMARY_UPSTREAMS=http://host:port/v1 \
  SUMMARY_MODEL=Qwen/Qwen3-32B \
  PORT=8080 \
  bash services/run_router.sh
  ```
  then set `WEB_SUMMARY_API_BASE=http://<router-host>:8080/v1`.

- Optional: `TRAIN_OUTPUT_DIR`, `ROLLOUT_DIR`, `VAL_ROLLOUT_DIR` for storage paths.

- Run:
  ```bash
  bash ./examples/run_verl/web_research_hle.sh
  ```

### 3) OSWorld

Placeholder for now.
