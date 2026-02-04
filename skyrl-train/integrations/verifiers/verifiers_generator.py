import logging
import warnings
from typing import List, Optional

import httpx
from omegaconf import DictConfig
from openai import AsyncOpenAI, APIError, APIConnectionError, RateLimitError, APITimeoutError
from verifiers import load_environment
from verifiers.types import GenerateOutputs, ProcessedOutputs, GenerateInputs

from skyrl_train.generators.base import GeneratorInterface, GeneratorInput, GeneratorOutput
from skyrl_train.generators.utils import get_rollout_metrics

logger = logging.getLogger(__name__)


class VerifiersGenerator(GeneratorInterface):
    # Default timeout in seconds for API requests
    DEFAULT_TIMEOUT = 600
    DEFAULT_CONNECT_TIMEOUT = 5.0
    DEFAULT_MAX_RETRIES = 10

    def __init__(
        self,
        generator_cfg: DictConfig,
        tokenizer,
        model_name: str,
    ):
        """
        Args:
            generator_cfg: DictConfig object containing the generator configuration
            tokenizer: tokenizer object for encoding and decoding text
        """
        self.generator_cfg = generator_cfg
        self.tokenizer = tokenizer
        self.model_name = model_name

        assert generator_cfg.enable_http_endpoint, "HTTP endpoint must be enabled for VerifiersGenerator"
        self.base_url = f"http://{generator_cfg.http_endpoint_host}:{generator_cfg.http_endpoint_port}/v1"
        self.client = self._setup_client(connection_limit=None)  # None means unlimited connections

    def _setup_client(self, connection_limit: Optional[int]) -> AsyncOpenAI:
        # Configurable timeout: generator_cfg.timeout_multiplier scales the default timeout
        timeout_multiplier = getattr(self.generator_cfg, "timeout_multiplier", 1.0)
        base_timeout = self.DEFAULT_TIMEOUT * timeout_multiplier
        connect_timeout = self.DEFAULT_CONNECT_TIMEOUT * timeout_multiplier

        # Configurable max retries
        max_retries = getattr(self.generator_cfg, "max_retries", self.DEFAULT_MAX_RETRIES)

        timeout = httpx.Timeout(timeout=base_timeout, connect=connect_timeout)
        limits = httpx.Limits(
            max_connections=connection_limit,  # OAI default: 1000
            max_keepalive_connections=connection_limit,  # OAI default: 100
        )
        http_client = httpx.AsyncClient(limits=limits, timeout=timeout)
        return AsyncOpenAI(
            base_url=self.base_url,
            api_key="dummy",  # Make OAI client happy.
            max_retries=max_retries,
            http_client=http_client,
        )

    async def generate(self, input_batch: GeneratorInput) -> GeneratorOutput:
        assert "env_extras" in input_batch, "Verifiers dataset fields are passed through env_extras"

        # Defaults are based on Verifiers' defaults.
        verifiers_dicts = [sample["verifiers"] for sample in input_batch["env_extras"]]
        generate_inputs = GenerateInputs(
            prompt=input_batch["prompts"],
            answer=[item.get("answer", "") for item in verifiers_dicts],
            info=[item.get("info", {}) for item in verifiers_dicts],
            task=[item.get("task", "default") for item in verifiers_dicts],
        )

        # Assumes all training samples correspond to the same Verifiers environment.
        # For now, if multiple environments are needed, use Verifiers' EnvGroup abstraction.
        environment_id = verifiers_dicts[0]["environment"]
        vf_env = load_environment(environment_id)

        # Verifiers requires logprobs from vLLM for post-processing.
        sampling_params = input_batch.get("sampling_params", {}).copy()
        sampling_params["logprobs"] = True
        sampling_params["top_logprobs"] = 1
        sampling_params["extra_body"] = {
            "return_tokens_as_token_ids": True,
        }

        # Clean the sampling params for Verifiers' a_generate.
        extra_body_keys = [
            "min_tokens",
            "skip_special_tokens",
            "include_stop_str_in_output",
            "top_k",
            "min_p",
            "repetition_penalty",
        ]
        for key in extra_body_keys:
            if key in sampling_params:
                sampling_params["extra_body"][key] = sampling_params[key]
                del sampling_params[key]

        # Generate the trajectories with error handling.
        try:
            generate_outputs: GenerateOutputs = await vf_env.a_generate(
                inputs=generate_inputs,
                client=self.client,
                model=self.model_name,
                sampling_args=sampling_params,
            )
        except (APIError, APIConnectionError, RateLimitError, APITimeoutError, httpx.TimeoutException) as e:
            # API failure after all retries exhausted - return zero rewards instead of crashing
            batch_size = len(input_batch["prompts"])
            warnings.warn(
                f"API request failed after retries for batch of {batch_size} samples: {type(e).__name__}: {e}. "
                f"Returning zero rewards for this batch.",
                RuntimeWarning,
            )
            logger.warning(
                f"API request failed after retries: {type(e).__name__}: {e}. "
                f"Batch size: {batch_size}. Returning zero rewards."
            )
            return self._create_zero_reward_output(input_batch)

        processed_outputs: ProcessedOutputs = vf_env.process_env_results_vllm(
            prompts=generate_outputs.prompt,
            completions=generate_outputs.completion,
            states=generate_outputs.state,
            rewards=generate_outputs.reward,
            processing_class=self.tokenizer,
            max_seq_len=self.generator_cfg.max_input_length + self.generator_cfg.sampling_params.max_generate_length,
            mask_env_responses=True,
        )

        # Convert output to SkyRL format.
        return GeneratorOutput(
            prompt_token_ids=processed_outputs.prompt_ids,
            response_ids=processed_outputs.completion_ids,
            rewards=processed_outputs.rewards,
            loss_masks=processed_outputs.completion_mask,
            rollout_logprobs=processed_outputs.completion_logprobs,
            rollout_metrics=get_rollout_metrics(processed_outputs.completion_ids, processed_outputs.rewards),
        )

    def _create_zero_reward_output(self, input_batch: GeneratorInput) -> GeneratorOutput:
        """Create a GeneratorOutput with zero rewards for failed API requests.

        This allows training to continue even when some batches fail due to API issues.
        """
        batch_size = len(input_batch["prompts"])

        # Encode prompts to get token IDs
        prompt_token_ids: List[List[int]] = []
        for prompt in input_batch["prompts"]:
            if isinstance(prompt, str):
                tokens = self.tokenizer.encode(prompt, add_special_tokens=False)
            else:
                tokens = list(prompt)
            prompt_token_ids.append(tokens)

        # Create empty responses (single EOS token per sample)
        eos_token_id = self.tokenizer.eos_token_id or 0
        response_ids = [[eos_token_id] for _ in range(batch_size)]

        # Zero rewards for all samples
        rewards = [[0.0] for _ in range(batch_size)]

        # Loss masks (all zeros since there's nothing to learn from failed samples)
        loss_masks = [[0] for _ in range(batch_size)]

        # Zero logprobs
        rollout_logprobs = [[0.0] for _ in range(batch_size)]

        return GeneratorOutput(
            prompt_token_ids=prompt_token_ids,
            response_ids=response_ids,
            rewards=rewards,
            loss_masks=loss_masks,
            rollout_logprobs=rollout_logprobs,
            rollout_metrics=get_rollout_metrics(response_ids, rewards),
        )
