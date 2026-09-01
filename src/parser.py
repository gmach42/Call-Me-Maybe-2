"""Load and validate JSON input files (functions and prompts)."""

import json
import sys
from pathlib import Path
from typing import Any, TypeVar

from pydantic import BaseModel, ValidationError

from .pydantic_models import FunctionDefinition, PromptItem

ModelT = TypeVar("ModelT", bound=BaseModel)


def load_json_file(path: Path) -> Any:
    """Load and parse a JSON file, raising ValueError on missing or invalid."""
    try:
        with path.open("r", encoding="utf-8") as fh:
            return json.load(fh)
    except OSError as exc:
        raise ValueError(f"Cannot read {path}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in {path}: {exc}") from exc


def _validate_items(
    raw: list[Any], model: type[ModelT], path: Path
) -> list[ModelT]:
    """Validate each item, skipping invalid ones with a message."""
    items: list[ModelT] = []
    for i, entry in enumerate(raw, 1):
        try:
            items.append(model.model_validate(entry))
        except ValidationError as exc:
            print(f"Skipping invalid entry #{i} in {path}: {exc}",
                  file=sys.stderr)
    return items


def load_functions(path: Path) -> list[FunctionDefinition]:
    """Load and validate function definitions from a JSON file."""
    raw = load_json_file(path)
    if not isinstance(raw, list):
        raise ValueError(f"Expected a JSON array in {path}")
    return _validate_items(raw, FunctionDefinition, path)


def load_prompts(path: Path) -> list[PromptItem]:
    """Load and validate prompt items from a JSON file."""
    raw = load_json_file(path)
    if not isinstance(raw, list):
        raise ValueError(f"Expected a JSON array in {path}")
    return _validate_items(raw, PromptItem, path)
