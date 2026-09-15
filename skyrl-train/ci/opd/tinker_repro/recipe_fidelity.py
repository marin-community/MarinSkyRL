"""Apply local fidelity fixes around the pinned Tinker Cookbook recipes."""

from __future__ import annotations

import asyncio
import importlib
import sys
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from types import ModuleType
from typing import Protocol, cast, runtime_checkable

from training_plan import Recipe


DatasetLoader = Callable[..., object]
ConfigFactory = Callable[..., object]


class ChzEntrypoint(Protocol):
    def entrypoint(self, config_type: type[object]) -> object: ...


class CookbookRecipeModule(Protocol):
    chz: ChzEntrypoint
    CLIConfig: type[object]


@runtime_checkable
class OPDCLIConfig(Protocol):
    temperature: float


RECIPE_MODULES = {
    Recipe.SFT: "tinker_cookbook.recipes.distillation.off_policy_reasoning",
    Recipe.OPD: "tinker_cookbook.recipes.distillation.on_policy_distillation",
}


def pinned_dataset_loader(
    loader: DatasetLoader,
    *,
    repository: str,
    revision: str,
) -> DatasetLoader:
    """Bind a Cookbook dataset load to the reviewed repository revision."""

    def load_dataset(requested_repository: str, *args: object, **kwargs: object) -> object:
        if requested_repository != repository:
            raise RuntimeError(f"Recipe requested unexpected dataset {requested_repository}; expected {repository}")
        requested_revision = kwargs.pop("revision", revision)
        if requested_revision != revision:
            raise RuntimeError(f"Recipe requested unexpected revision {requested_revision}; expected {revision}")
        return loader(requested_repository, *args, revision=revision, **kwargs)

    return load_dataset


def opd_config_factory(factory: ConfigFactory, *, temperature: float) -> ConfigFactory:
    """Forward the reviewed sampling temperature into the Cookbook training config."""

    def build_config(**kwargs: object) -> object:
        if "temperature" in kwargs:
            raise RuntimeError("The pinned OPD adapter received temperature twice")
        return factory(temperature=temperature, **kwargs)

    return build_config


@contextmanager
def _replace_attribute(owner: object, name: str, replacement: object) -> Iterator[None]:
    original = getattr(owner, name)
    setattr(owner, name, replacement)
    try:
        yield
    finally:
        setattr(owner, name, original)


def _parse_cookbook_config(recipe_module: CookbookRecipeModule, arguments: list[str]) -> object:
    original_argv = sys.argv
    sys.argv = [original_argv[0], *arguments]
    try:
        return recipe_module.chz.entrypoint(recipe_module.CLIConfig)
    finally:
        sys.argv = original_argv


def validated_recipe_module(recipe: Recipe, module_name: str) -> ModuleType:
    expected = RECIPE_MODULES[recipe]
    if module_name != expected:
        raise RuntimeError(f"Plan selected unexpected {recipe.value} recipe module {module_name}; expected {expected}")
    return importlib.import_module(module_name)


def _run_sft(recipe_module: ModuleType, repository: str, revision: str, arguments: list[str]) -> None:
    config = _parse_cookbook_config(cast(CookbookRecipeModule, recipe_module), arguments)
    loader = pinned_dataset_loader(
        recipe_module.datasets.load_dataset,
        repository=repository,
        revision=revision,
    )
    with _replace_attribute(recipe_module.datasets, "load_dataset", loader):
        recipe_module.cli_main(config)


async def _run_opd(
    recipe_module: ModuleType,
    repository: str,
    revision: str,
    arguments: list[str],
) -> None:
    from tinker_cookbook.distillation import datasets as distillation_datasets  # noqa: PLC0415

    config = _parse_cookbook_config(cast(CookbookRecipeModule, recipe_module), arguments)
    if not isinstance(config, OPDCLIConfig):
        raise TypeError("Cookbook OPD CLI config does not expose temperature")
    loader = pinned_dataset_loader(
        distillation_datasets.load_dataset,
        repository=repository,
        revision=revision,
    )
    config_factory = opd_config_factory(
        recipe_module.train_on_policy.Config,
        temperature=config.temperature,
    )
    with (
        _replace_attribute(distillation_datasets, "load_dataset", loader),
        _replace_attribute(recipe_module.train_on_policy, "Config", config_factory),
    ):
        await recipe_module.cli_main(config)


def main(argv: list[str] | None = None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    if len(arguments) < 4:
        raise SystemExit("usage: recipe_fidelity.py {sft,opd} RECIPE_MODULE DATASET REVISION [RECIPE_ARGUMENT ...]")
    recipe = Recipe(arguments[0])
    recipe_module = validated_recipe_module(recipe, arguments[1])
    repository, revision = arguments[2:4]
    recipe_arguments = arguments[4:]
    if recipe is Recipe.SFT:
        _run_sft(recipe_module, repository, revision, recipe_arguments)
    else:
        asyncio.run(_run_opd(recipe_module, repository, revision, recipe_arguments))
    return 0


if __name__ == "__main__":
    main()
