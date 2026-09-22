"""Load and validate JSON input files (functions and prompts)."""

import argparse
import json
import sys
from pathlib import Path
from typing import Any, TypeVar

from pydantic import BaseModel, ValidationError

from .pydantic_models import FunctionDefinition, PromptItem

# Type variable for Pydantic models (added for mypy type checking)
ModelT = TypeVar("ModelT", bound=BaseModel)

DEFAULT_FUNCTIONS = Path("data/input/functions_definition.json")
DEFAULT_INPUT = Path("data/input/function_calling_tests.json")
DEFAULT_OUTPUT = Path("data/output/function_calling_results.json")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    p = argparse.ArgumentParser(
        description=("Translate user prompts "
                     "into structured function calls."))
    p.add_argument(
        "--functions_definition",
        type=Path,
        default=DEFAULT_FUNCTIONS,
        metavar="FILE",
        help="Path to functions_definition.json",
    )
    p.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        metavar="FILE",
        help="Path to function_calling_tests.json",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        metavar="FILE",
        help="Path for the output JSON file",
    )
    return p.parse_args()


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
