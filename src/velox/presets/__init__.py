"""Bundled, versioned prop-firm rule presets (YAML -> RulesetSpec)."""

from __future__ import annotations

from functools import cache
from importlib import resources
from pathlib import Path

import yaml

from velox.domain.rules.spec import RulesetSpec


@cache
def _bundled() -> dict[str, RulesetSpec]:
    out: dict[str, RulesetSpec] = {}
    for f in resources.files(__package__).iterdir():
        if f.name.endswith(".yaml"):
            spec = RulesetSpec.model_validate(yaml.safe_load(f.read_text(encoding="utf-8")))
            out[spec.id] = spec
    return out


def list_presets() -> list[RulesetSpec]:
    return sorted(_bundled().values(), key=lambda s: s.id)


def load_preset(name_or_path: str) -> RulesetSpec:
    """Load a bundled preset by id (e.g. 'ftmo-2step') or a custom YAML file path."""
    if name_or_path in _bundled():
        return _bundled()[name_or_path]
    path = Path(name_or_path)
    if path.suffix in {".yaml", ".yml"} and path.exists():
        return RulesetSpec.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))
    known = ", ".join(sorted(_bundled()))
    raise KeyError(f"unknown preset '{name_or_path}' (bundled: {known})")
