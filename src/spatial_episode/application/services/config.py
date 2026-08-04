"""Strict YAML loading with paths resolved by the caller, never by cwd magic."""

from __future__ import annotations

from pathlib import Path
from typing import TypeVar

import yaml
from pydantic import BaseModel

ConfigT = TypeVar("ConfigT", bound=BaseModel)


def load_yaml_config(path: Path, model: type[ConfigT]) -> ConfigT:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"configuration root must be a mapping: {path}")
    return model.model_validate(payload)
