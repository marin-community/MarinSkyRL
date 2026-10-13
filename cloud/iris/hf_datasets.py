"""Resolve prepared Hugging Face datasets to immutable source identities."""

from __future__ import annotations

from huggingface_hub import HfApi

from marinskyrl.hugging_face_retry import call_with_hugging_face_retry
from marinskyrl.resource_locator import HFDatasetSelector, parse_hf_dataset_selector


def resolve_hf_dataset_selector(value: str) -> HFDatasetSelector:
    """Resolve a prepared dataset selector to an immutable Hub commit."""
    selector = parse_hf_dataset_selector(value)
    if selector is None:
        raise ValueError(f"Invalid Hugging Face dataset selector: {value!r}")
    info = call_with_hugging_face_retry(
        lambda: HfApi().dataset_info(selector.repo_id, revision=selector.revision),
        operation=f"resolve Hugging Face dataset {selector.repo_id}@{selector.revision or 'main'}",
    )
    return HFDatasetSelector(selector.repo_id, info.sha, selector.subdir)
