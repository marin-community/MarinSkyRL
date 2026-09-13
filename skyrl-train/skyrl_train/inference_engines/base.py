from abc import ABC, abstractmethod
from typing import List, Dict, TypedDict, Any, Optional, Hashable, NotRequired

MessageType = Dict[str, str]
ConversationType = List[MessageType]


class InferenceEngineInput(TypedDict):
    # Either prompts or prompt_token_ids must be provided, but not both.
    prompts: Optional[List[ConversationType]]
    prompt_token_ids: Optional[List[List[int]]]
    sampling_params: Optional[Dict[str, Any]]
    session_ids: Optional[List[Hashable]]
    # Per-sample Responses-API options (tools, parallel_tool_calls, etc.) that
    # require the serving backend's resolved chat renderer.
    chat_completion_params: NotRequired[List[Dict[str, Any]]]


class InferenceEngineOutput(TypedDict):
    # We always return both tokens and text outputs. The tokens are the outputs
    # of inference engine, and the text is the decoded text output. Therefore,
    # it is guaranteed that tokenizer.decode(response_token_ids, skip_special_tokens=True) == responses,
    # but the reverse is not guaranteed, since there are multiple ways to
    # represent the same text with tokens. Therefore, for multi-turn generation,
    # please use token-in-token-out to ensure correctness.
    # `skip_special_tokens=True` is needed because string responses do not include EOS tokens like `<|im_end|>`
    responses: List[str]
    response_ids: List[List[int]]
    stop_reasons: List[str]
    response_logprobs: Optional[List[List[float]]]
    # prompt_logprobs: per-prompt-token top-K logprobs from vLLM (for teacher scoring).
    # Format: List[List[Optional[Dict[int, float]]]] — outer list is batch,
    # inner list is prompt positions, dict maps token_id → logprob.
    # Only populated when SamplingParams(prompt_logprobs=K) is used.
    prompt_logprobs: Optional[List[List[Optional[Dict[int, float]]]]]
    prompt_ids: NotRequired[List[List[int]]]
    assistant_messages: NotRequired[List[Dict[str, Any]]]


class NamedWeightsUpdateRequest(TypedDict):
    names: List[str]
    dtypes: List[str]
    shapes: List[List[int]]
    sizes: NotRequired[List[int]]
    extras: Optional[List[Dict[str, Any]]]
    packed: NotRequired[bool]


class InferenceEngineInterface(ABC):
    weight_sync_relative_rank_offset: int | None = None
    max_model_len: int | None = None

    def get_model_max_len(self) -> int | None:
        """Return the context limit resolved by the live serving backend."""
        return self.max_model_len

    @abstractmethod
    async def generate(self, input_batch: InferenceEngineInput) -> InferenceEngineOutput:
        raise NotImplementedError()

    @abstractmethod
    async def chat_completion(self, request_payload: Dict[str, Any]) -> Dict[str, Any]:
        """Handles OpenAI-compatible HTTP endpoint.

        Accepts a JSON payload: {"json": <request-body>, "headers": <headers-dict>}.
        The request body will be used to construct a ChatCompletionRequest.
        Returns a plain dict, either a ChatCompletionResponse or an ErrorResponse.
        The specific fields of the response/request depend on the engine's backend (e.g. for vllm
        these are defined in vllm.entrypoints.openai.protocol).
        """
        raise NotImplementedError()

    @abstractmethod
    async def completion(self, request_payload: Dict[str, Any]) -> Dict[str, Any]:
        """Handles OpenAI-compatible HTTP endpoint.

        Accepts a JSON payload: {"json": <request-body>, "headers": <headers-dict>}.
        The request body will be used to construct a CompletionRequest.
        Returns a plain dict, either a CompletionResponse or an ErrorResponse.
        The specific fields of the response/request depend on the engine's backend (e.g. for vllm
        these are defined in vllm.entrypoints.openai.protocol).
        """
        raise NotImplementedError()

    async def tokenize(self, request_payload: Dict[str, Any]) -> Dict[str, Any]:
        """Tokenize through the serving backend when supported.

        ``request_payload`` contains the JSON request body under ``json`` and
        serialized HTTP headers under ``headers``. Implementations return the
        backend's native tokenization or error response as a plain dictionary.
        """
        raise NotImplementedError()

    @abstractmethod
    async def wake_up(self, *args: Any, **kwargs: Any):
        raise NotImplementedError()

    @abstractmethod
    async def sleep(self, *args: Any, **kwargs: Any):
        raise NotImplementedError()

    @abstractmethod
    async def init_weight_update_communicator(
        self, master_addr, master_port, rank_offset, world_size, group_name, backend, override_existing: bool = False
    ):
        raise NotImplementedError()

    async def init_draft_transfer_communicator(
        self, master_addr, master_port, rank_offset, world_size, group_name, backend
    ) -> Any:
        """Join a persistent DraftTrainer-to-serving transfer group."""
        raise NotImplementedError()

    @abstractmethod
    async def update_named_weights(self, request: NamedWeightsUpdateRequest):
        raise NotImplementedError()

    @abstractmethod
    async def teardown(self):
        raise NotImplementedError()

    @abstractmethod
    async def reset_prefix_cache(self):
        raise NotImplementedError()

    @abstractmethod
    def tp_size(self) -> int:
        """Return the tensor parallel size of this inference engine."""
        raise NotImplementedError()

    @abstractmethod
    def pp_size(self) -> int:
        """Return the pipeline parallel size of this inference engine."""
        raise NotImplementedError()

    @abstractmethod
    def dp_size(self) -> int:
        """Return the data parallel size of this inference engine."""
        raise NotImplementedError()

    @abstractmethod
    async def pause_generation(self) -> None:
        """
        Pause the scheduler for a weight update after aborting all running and waiting
        requests. Running requests return their generated tokens with stop_reason "abort";
        waiting requests return zero completion tokens.
        """
        raise NotImplementedError()

    @abstractmethod
    async def resume_generation(self) -> None:
        """Resume the scheduler after a weight update."""
        raise NotImplementedError()

    async def begin_online_eagle_capture(self, config: Dict[str, Any]) -> Any:
        """Begin a bounded online-EAGLE capture interval when supported."""
        raise NotImplementedError()

    async def seal_online_eagle_capture(self, output_dir: str) -> Any:
        """Seal the active capture into an immutable local artifact."""
        raise NotImplementedError()

    async def export_online_eagle_capture(self, job: Dict[str, Any]) -> Any:
        """Return a metadata-only catalog for a sealed capture."""
        raise NotImplementedError()

    async def transfer_online_eagle_capture(self, transfer_plan: Dict[str, Any]) -> Any:
        """Send selected capture tensors directly to DraftTrainer."""
        raise NotImplementedError()

    async def stage_online_eagle_speculator(
        self,
        transfer_manifest: Dict[str, Any],
        incumbent_draft_revision: str,
    ) -> Any:
        """Validate a candidate on the serving node without activating it."""
        raise NotImplementedError()

    async def activate_online_eagle_speculator(
        self,
        transfer_manifest: Dict[str, Any],
    ) -> Any:
        """Activate the staged candidate on every rank."""
        raise NotImplementedError()

    async def commit_online_eagle_speculator(self, draft_revision: str) -> Any:
        """Commit a successful all-engine activation."""
        raise NotImplementedError()

    async def rollback_online_eagle_speculator(self, draft_revision: str) -> Any:
        """Restore the prior draft after a failed activation."""
        raise NotImplementedError()

    async def discard_online_eagle_capture(self) -> Any:
        """Discard the active capture interval after a failed rollout."""
        raise NotImplementedError()

    async def cleanup_online_eagle_scratch(self, scratch_root: str) -> Any:
        """Remove node-local online-EAGLE scratch when supported."""
        raise NotImplementedError()

    async def install_online_eagle_speculator(self, candidate_dir: str) -> Any:
        """Install a complete draft candidate across the serving engine."""
        raise NotImplementedError()

    async def publish_online_eagle_speculator(
        self,
        source_dir: str,
        destination: str,
        draft_revision: str,
        served_target_revision: str,
    ) -> Any:
        """Publish the exact served draft beside a policy checkpoint."""
        raise NotImplementedError()

    async def restore_online_eagle_speculator(self, source: str, destination: str) -> Any:
        """Restore the draft paired with a resumed policy checkpoint."""
        raise NotImplementedError()
