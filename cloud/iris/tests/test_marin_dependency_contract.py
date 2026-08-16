from __future__ import annotations

import tomllib
from pathlib import Path

from iris.cluster.config import IrisClusterConfig
from packaging.requirements import Requirement

_REPOSITORY_ROOT = Path(__file__).resolve().parents[3]


def test_marin_runtime_dependencies_have_no_independent_constraints() -> None:
    project = tomllib.loads((_REPOSITORY_ROOT / "pyproject.toml").read_text())
    requirement_strings = list(project["project"]["dependencies"])
    requirement_strings.extend(
        requirement
        for requirements in project["project"]["optional-dependencies"].values()
        for requirement in requirements
    )
    requirement_strings.extend(
        requirement
        for requirements in project["dependency-groups"].values()
        for requirement in requirements
        if isinstance(requirement, str)
    )

    parsed_requirements = [Requirement(requirement) for requirement in requirement_strings]
    marin_requirements = [
        requirement
        for requirement in parsed_requirements
        if requirement.name.startswith("marin-") and requirement.name != "marin-style"
    ]
    constrained_requirements = [
        str(requirement) for requirement in marin_requirements if requirement.specifier or requirement.url is not None
    ]

    assert marin_requirements
    assert constrained_requirements == []


def test_resolved_iris_schema_accepts_current_cluster_budget_and_cache_fields() -> None:
    config = IrisClusterConfig.model_validate(
        {
            "kubernetes_provider": {"cache_max_age": {"seconds": 604800}},
            "user_budget_defaults": {"budget_limit": 400000, "max_band": "interactive"},
        }
    )

    assert config.kubernetes_provider is not None
    assert config.kubernetes_provider.cache_max_age is not None
    assert config.kubernetes_provider.cache_max_age.to_seconds() == 604800
    assert config.user_budget_defaults.budget_limit == 400000
    assert config.user_budget_defaults.max_band == 2
