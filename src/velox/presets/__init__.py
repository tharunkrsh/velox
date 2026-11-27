"""Bundled, versioned prop-firm rule presets (YAML -> RulesetSpec)."""

from __future__ import annotations

from functools import cache
from importlib import resources
from pathlib import Path

import yaml

from velox.domain.rules.spec import RulesetSpec


def parse_preset(text: str | bytes) -> RulesetSpec:
    """Validate a preset YAML document. Raises ``ValueError`` (pydantic's
    ``ValidationError`` included) when it is not a valid rule set."""
    try:
        data = yaml.safe_load(text)
    except yaml.YAMLError as e:
        raise ValueError(f"preset is not valid YAML: {e}") from e
    if not isinstance(data, dict):
        raise ValueError("preset YAML must be a mapping of rule-set fields")
    return RulesetSpec.model_validate(data)


@cache
def _bundled() -> dict[str, RulesetSpec]:
    out: dict[str, RulesetSpec] = {}
    for f in resources.files(__package__).iterdir():
        if f.name.endswith(".yaml"):
            spec = parse_preset(f.read_text(encoding="utf-8"))
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
        return parse_preset(path.read_text(encoding="utf-8"))
    known = ", ".join(sorted(_bundled()))
    raise KeyError(f"unknown preset '{name_or_path}' (bundled: {known})")
