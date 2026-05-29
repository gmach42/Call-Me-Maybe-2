"""Entry point: parse CLI arguments, load data, run the pipeline."""

import argparse
import sys
from pathlib import Path

from llm_sdk.llm_sdk import Small_LLM_Model

from .parser import load_functions, load_prompts
from .pipeline import run

DEFAULT_FUNCTIONS = Path("data/input/functions_definition.json")
DEFAULT_INPUT = Path("data/input/function_calling_tests.json")
DEFAULT_OUTPUT = Path("data/output/function_calling_results.json")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=("Translate natural-language prompts "
                     "into structured function calls."))
    p.add_argument(
        "--functions_definition",
        type=Path,
        default=DEFAULT_FUNCTIONS,
        metavar="FILE",
        help="Path to functions_definition.json",
    )  # noqa: E501
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


def main() -> None:
    """Main entry point."""
    args = _parse_args()

    try:
        functions = load_functions(args.functions_definition)
    except ValueError as exc:
        print(f"Error loading functions: {exc}", file=sys.stderr)
        sys.exit(1)

    try:
        prompts = load_prompts(args.input)
    except ValueError as exc:
        print(f"Error loading prompts: {exc}", file=sys.stderr)
        sys.exit(1)

    if not functions:
        print("No functions defined, nothing to do.", file=sys.stderr)
        sys.exit(1)

    if not prompts:
        print("No prompts found, nothing to do.", file=sys.stderr)
        sys.exit(1)

    print("Loading model…", file=sys.stderr)
    try:
        model = Small_LLM_Model()
    except Exception as exc:
        print(f"Error loading model: {exc}", file=sys.stderr)
        sys.exit(1)

    run(model, prompts, functions, args.output)


if __name__ == "__main__":
    main()
