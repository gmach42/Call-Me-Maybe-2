import json
from pathlib import Path
from typing import Any

from .pydantic_models import FunctionDefinition, PromptItem


def load_json_file(path: Path) -> Any:
    """Load and parse a JSON file, raising ValueError on missing or invalid."""  # noqa: E501
    try:
        with path.open("r", encoding="utf-8") as fh:
            return json.load(fh)
    except FileNotFoundError:
        raise ValueError(f"File not found: {path}")
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in {path}: {exc}") from exc


def load_functions(path: Path) -> list[FunctionDefinition]:
    """Load and validate function definitions from a JSON file."""
    raw = load_json_file(path)
    if not isinstance(raw, list):
        raise ValueError(f"Expected a JSON array in {path}")
    return [FunctionDefinition.model_validate(fn) for fn in raw]


def load_prompts(path: Path) -> list[PromptItem]:
    """Load and validate prompt items from a JSON file."""
    raw = load_json_file(path)
    if not isinstance(raw, list):
        raise ValueError(f"Expected a JSON array in {path}")
    return [PromptItem.model_validate(p) for p in raw]
