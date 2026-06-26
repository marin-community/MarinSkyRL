"""Qwen3.5/3.6 multimodal-shell → text-CausalLM unwrap for RL.

The Qwen3.5/3.6 MoE checkpoints (e.g. ``Qwen/Qwen3.6-35B-A3B``) ship as a
**multimodal wrapper** — top ``config.model_type == "qwen3_5_moe"`` with
``architectures == ["Qwen3_5MoeForConditionalGeneration"]`` and the actual
language model nested under ``model.language_model`` (a ``Qwen3_5MoeTextModel``),
alongside a vision tower (``model.visual``) and an MTP head. Because
``architectures`` names the *ForConditionalGeneration shell,
``AutoModelForCausalLM.from_pretrained`` instantiates that shell rather than the
text ``Qwen3_5MoeForCausalLM`` tower.

For RL we want the **text backbone only** (this mirrors hamishivi/tmax, which
loads ``AutoModelForCausalLM`` on the text tower and never instantiates the
*ForConditionalGeneration shell). Carrying the VLM shell breaks several
downstream assumptions that all read ``self.model`` / ``self.model.config`` as a
plain text CausalLM:

  * **FSDP wrap** — ``_no_split_modules`` on the shell lists ``Qwen3_5MoeVisionBlock``
    (absent from a text-only policy), and ``self.model.config`` is the VLM config
    (no top-level ``num_hidden_layers`` / MoE layer count), so
    ``count_moe_layers(self.model.config)`` and the wrap-policy auto-detect read
    the wrong config.
  * **MoE grouped-GEMM swap / router-replay** — ``count_moe_layers`` reads
    ``self.model.config`` which on the shell is the VLM config, not the text MoE
    config.
  * **vLLM weight-sync** — the shell prefixes every text weight with
    ``model.language_model.`` (vs the text CausalLM's ``model.``).

Empirically (gpu-rl image, transformers 5.12.1), the checkpoint's
``model.language_model.*`` (692) + ``lm_head.weight`` keys map **1:1, 0 missing /
0 extra** onto ``Qwen3_5MoeForCausalLM`` once the ``language_model.`` prefix is
stripped — i.e. the text tower is fully self-contained in the checkpoint. So the
unwrap is a pure reference re-point of the already-loaded submodules (no
re-download, no weight movement): take the loaded shell's
``model.language_model`` as the CausalLM ``.model`` and the shell's ``lm_head``
as the CausalLM ``.lm_head``, drop the vision tower + MTP head.

Gated on ``SKYRL_QWEN3_5_VLM_UNWRAP`` (default on) so it can be disabled.
"""

import os

from loguru import logger


def is_qwen3_5_vlm_shell(config) -> bool:
    """True iff ``config`` is a Qwen3.5/3.6 multimodal shell wrapping a text MoE
    tower that we should unwrap for RL.

    Detection (matches tmax's ``text_config`` / ``linear_conv_kernel_dim`` probe):
    a nested ``text_config`` carrying the GatedDeltaNet signature
    ``linear_conv_kernel_dim`` — i.e. the hybrid Qwen3.5/3.6 text tower — while
    the top-level config does NOT carry it (it is the shell). We additionally gate
    on the top ``model_type`` starting with ``qwen3_5`` so unrelated VLMs with a
    ``text_config`` are untouched.
    """
    if os.environ.get("SKYRL_QWEN3_5_VLM_UNWRAP", "1") not in ("1", "true", "True"):
        return False
    text_config = getattr(config, "text_config", None)
    if text_config is None:
        return False
    # GatedDeltaNet signature lives on the *text* config for the shell.
    text_is_qwen3_5_gdn = hasattr(text_config, "linear_conv_kernel_dim")
    top_is_qwen3_5 = str(getattr(config, "model_type", "")).startswith("qwen3_5")
    # Only unwrap when the top config itself is NOT already the text tower
    # (i.e. it is genuinely the shell, with the GDN signature one level down).
    top_is_shell = not hasattr(config, "linear_conv_kernel_dim")
    return bool(text_is_qwen3_5_gdn and top_is_qwen3_5 and top_is_shell)


def is_qwen3_5_text_tower(config) -> bool:
    """True iff ``config`` is the Qwen3.5/3.6 hybrid **text** tower config (the
    unwrapped ``Qwen3_5Moe(Text)Config`` the policy ends up with after
    :func:`unwrap_to_text_causal_lm`), NOT the multimodal shell.

    Where :func:`is_qwen3_5_vlm_shell` probes the *shell* (GDN signature one level
    down under ``config.text_config``), this probes the already-unwrapped text
    tower, whose ``model_type`` is e.g. ``qwen3_5_moe_text`` (or ``qwen3_5_text``)
    and which carries the GatedDeltaNet signature ``linear_conv_kernel_dim``
    DIRECTLY (no ``text_config`` nesting). Used by the weight-sync name mapper to
    decide whether the inference engine on the other side instantiated the VLM
    shell (``Qwen3_5MoeForConditionalGeneration``) and therefore expects the text
    weights under the ``model.language_model.`` HF namespace rather than ``model.``.

    Gated on the same ``SKYRL_QWEN3_5_VLM_UNWRAP`` flag as the unwrap, so toggling
    the unwrap off keeps the sender-side name namespace unchanged in lockstep.
    """
    if os.environ.get("SKYRL_QWEN3_5_VLM_UNWRAP", "1") not in ("1", "true", "True"):
        return False
    if config is None:
        return False
    top_is_qwen3_5 = str(getattr(config, "model_type", "")).startswith("qwen3_5")
    has_gdn = hasattr(config, "linear_conv_kernel_dim")
    # Exclude the shell itself (its top config does NOT carry the GDN signature;
    # it lives under text_config). Only the text tower carries it directly.
    return bool(top_is_qwen3_5 and has_gdn)


def map_text_name_to_vlm_engine(name: str) -> str:
    """HF→vLLM weight-sync name mapper for the Qwen3.5/3.6 VLM-shell engine.

    The RL policy is the unwrapped **text** ``Qwen3_5MoeForCausalLM`` (weights named
    ``model.*`` + ``lm_head.weight``), but the vLLM rollout engine instantiates the
    **multimodal shell** ``Qwen3_5MoeForConditionalGeneration`` whose
    ``load_weights`` runs the shell ``hf_to_vllm_mapper`` (vLLM
    ``qwen3_vl.py``)::

        orig_to_new_prefix = {
            "model.visual.":         "visual.",
            "lm_head.":              "language_model.lm_head.",
            "model.language_model.": "language_model.model.",
        }

    i.e. the shell expects the text decoder weights under the **HF** name
    ``model.language_model.*`` (which it then maps to its internal
    ``language_model.model.*``). A bare ``model.X`` from the text policy matches
    NONE of those prefixes, so ``AutoWeightsLoader`` looks up ``model.X`` directly
    on the shell and raises ``"There is no module or parameter named 'model' in
    Qwen3_5MoeForConditionalGeneration"`` — the exact Stage-2 weight-sync failure.

    This mapper bridges that gap on the SENDER side (the tmax analogue of
    ``_build_vlm_name_mapper`` / ``lambda name: f"language_model.{name}"``):
    rewrite the policy's ``model.X`` → ``model.language_model.X`` so the shell's
    own ``hf_to_vllm_mapper`` then carries it to ``language_model.model.X``.
    ``lm_head.weight`` is left UNCHANGED (the shell mapper already routes
    ``lm_head.`` → ``language_model.lm_head.``). Any name that is already prefixed
    (defensive) is passed through untouched.
    """
    if name == "lm_head.weight" or name.startswith("lm_head."):
        return name
    if name.startswith("model.language_model."):
        # Already in the shell-expected namespace; leave as-is (defensive).
        return name
    if name.startswith("model."):
        return "model.language_model." + name[len("model.") :]
    return name


def unwrap_to_text_causal_lm(vlm_model):
    """Convert a loaded Qwen3.5/3.6 ``*ForConditionalGeneration`` shell into its
    text ``Qwen3_5MoeForCausalLM`` tower, reusing the already-loaded submodules.

    Returns the new text CausalLM (with ``.model`` = the loaded
    ``Qwen3_5MoeTextModel``, ``.lm_head`` = the loaded lm_head, ``.config`` = the
    text config). The vision tower and MTP head are dropped (their parameters are
    released when the shell is garbage-collected).

    Raises if the shell does not have the expected ``model.language_model`` /
    ``lm_head`` structure (so a future arch change surfaces loudly rather than
    silently mis-training).
    """
    from transformers import AutoModelForCausalLM

    vlm_config = vlm_model.config
    text_config = getattr(vlm_config, "text_config", None)
    if text_config is None:
        raise ValueError("unwrap_to_text_causal_lm: config has no text_config")

    inner = getattr(vlm_model, "model", None)
    text_backbone = getattr(inner, "language_model", None) if inner is not None else None
    lm_head = getattr(vlm_model, "lm_head", None)
    if text_backbone is None or lm_head is None:
        raise ValueError(
            "unwrap_to_text_causal_lm: expected `model.language_model` + `lm_head` on "
            f"the shell, got model={type(inner).__name__ if inner is not None else None} "
            f"language_model={type(text_backbone).__name__ if text_backbone is not None else None} "
            f"lm_head={type(lm_head).__name__ if lm_head is not None else None}"
        )

    # Resolve the text CausalLM class from the *text* config's type.
    text_cls = AutoModelForCausalLM._model_mapping[type(text_config)]

    # Build the CausalLM shell structurally (meta — no weights), then re-point its
    # submodules at the already-loaded tensors. Building under `meta` avoids a
    # second materialization of the (large) decoder stack.
    import torch

    with torch.device("meta"):
        text_model = text_cls(text_config)
    text_model.model = text_backbone
    text_model.lm_head = lm_head
    # The CausalLM's `.config` must be the text config so every downstream reader
    # (count_moe_layers, wrap-policy auto-detect, generation) sees the real text
    # MoE topology rather than the VLM shell config.
    text_model.config = text_config

    # Drop the vision class from `_no_split_modules` so the FSDP wrap auto-detect
    # (get_fsdp_wrap_policy -> get_module_class_from_name) does not raise on a
    # class (`Qwen3_5MoeVisionBlock`) that no longer exists in this text-only
    # module. transformers stores `_no_split_modules` as a per-instance *set* on
    # the constructed model (the class attribute is a list); we overwrite the
    # instance attribute with a plain list of the surviving (text) classes.
    nsm = getattr(text_model, "_no_split_modules", None)
    if nsm:
        text_model._no_split_modules = [c for c in nsm if "Vision" not in c]

    logger.info(
        "[qwen3_5_vlm] unwrapped %s -> %s (text tower); dropped vision + MTP head. "
        "_no_split_modules=%s",
        type(vlm_model).__name__,
        type(text_model).__name__,
        getattr(text_model, "_no_split_modules", None),
    )
    return text_model
