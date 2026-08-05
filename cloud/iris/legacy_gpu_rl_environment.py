"""Legacy GPU-RL environment paths used by monitor and reshard entry points.

Delete this module when those entry points install the frozen root environment. The
remaining migration is tracked in marin-community/marin#7920.
"""

GPU_RL_ENV_DIR = "/opt/marin/envs/rl"
GPU_RL_PYTHON = f"{GPU_RL_ENV_DIR}/bin/python"
