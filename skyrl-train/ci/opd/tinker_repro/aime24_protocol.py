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


def evaluation_hydra_arguments() -> tuple[str, ...]:
    """Use one AIME generation contract for inline and independent evaluation."""
    return (
        f"++generator.engine_init_kwargs.max_model_len={CONTEXT_WINDOW}",
        f"generator.eval_sampling_params.max_generate_length={MAX_TOKENS}",
        f"generator.eval_sampling_params.temperature={TEMPERATURE}",
        f"generator.eval_sampling_params.top_p={TOP_P}",
        f"generator.eval_sampling_params.top_k={TOP_K}",
        f"generator.eval_n_samples_per_prompt={NUM_SAMPLES}",
        f"environment.skyrl_gym.aime.evaluation_token_budget={MAX_TOKENS}",
        "environment.skyrl_gym.aime.strict_box_verify=true",
        f"environment.skyrl_gym.aime.max_gen_length={MAX_TOKENS}",
    )
