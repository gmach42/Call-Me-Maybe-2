"""Entry point: parse CLI arguments, load data, run the pipeline."""

import sys
import time

from llm_sdk.llm_sdk import Small_LLM_Model

from .parser import load_functions, load_prompts, parse_args
from .pipeline import run


def main() -> None:
    """Run the CLI entry point."""
    start_time = time.time()
    args = parse_args()

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

    end_time = time.time()
    print(f"Execution time: {end_time - start_time:.1f} seconds\n",
          file=sys.stderr)


if __name__ == "__main__":
    main()
