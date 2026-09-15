"""Shared AIME 2024 protocol for Tinker and native evaluation."""

AIME24_DATASET = "HuggingFaceH4/aime_2024"
AIME24_REVISION = "2fe88a2f1091d5048c0f36abc874fb997b3dd99a"
AIME24_SPLIT = "train"
AIME24_SIZE = 30
MODEL_NAME = "Qwen/Qwen3.5-9B-Base"
RENDERER_NAME = "qwen3_5"
SYSTEM_PROMPT = "Put your final answer in \\boxed{}."
USER_INSTRUCTION = (
    "This is an AIME problem. The answer is an integer from 000 to 999. "
    "Show your work step by step, then put your final answer in \\boxed{}."
)
MAX_TOKENS = 64_000
CONTEXT_WINDOW = 65_536
TEMPERATURE = 1.0
TOP_P = 1.0
TOP_K = -1
NUM_SAMPLES = 1
