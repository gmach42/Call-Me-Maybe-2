"""Processing pipeline: prompt → function call result."""

import json
import sys
from pathlib import Path
from typing import Any

from llm_sdk.llm_sdk import Small_LLM_Model

from .constrained_decoding import generate_function_name, generate_value
from .pydantic_models import FunctionCallResult, FunctionDefinition, PromptItem


def _base_prompt(prompt_text: str, functions: list[FunctionDefinition]) -> str:
    """Build the context prompt shown to the LLM for function selection."""
    lines = [
        "You are a function-calling assistant. "
        "Choose the single best function for the task.",
        "",
        "Available functions:",
    ]
    for fn in functions:
        params = ", ".join(f"{k}: {v.type}" for k, v in fn.parameters.items())
        lines.append(f"  {fn.name}({params}): {fn.description}")
    lines += ["", f"Task: {prompt_text}", "Function: "]
    return "\n".join(lines)


def _param_context(
    base: str,
    fn_name: str,
    param_key: str,
    param_type: str,
    collected: dict[str, Any],
) -> str:
    """Build the context string used to generate one parameter value.

    The returned string ends just before where the value should appear,
    so the LLM immediately generates the value tokens.
    For string parameters the opening quote is included in the suffix.
    """
    # base ends with "Function: ", so appending fn_name gives
    # "...Function: fn_add_numbers"
    ctx = base + fn_name + "\nParameters: {"

    items: list[str] = []
    for k, v in collected.items():
        if isinstance(v, bool):
            items.append(f'"{k}": {"true" if v else "false"}')
        elif isinstance(v, str):
            # Escape internal quotes to keep the JSON valid in context.
            escaped = v.replace('\\', '\\\\').replace('"', '\\"')
            items.append(f'"{k}": "{escaped}"')
        else:
            items.append(f'"{k}": {v}')

    if items:
        ctx += ", ".join(items) + ", "

    # Open the current key; for strings include the opening quote so the
    # model generates value tokens starting right after it.
    if param_type == "string":
        ctx += f'"{param_key}": "'
    else:
        ctx += f'"{param_key}": '

    return ctx


def process_prompt(
    model: Small_LLM_Model,
    item: PromptItem,
    functions: list[FunctionDefinition],
    cache: dict[int, str],
) -> FunctionCallResult:
    """Turn one PromptItem into a FunctionCallResult."""
    base = _base_prompt(item.prompt, functions)
    fn_input_ids: list[int] = model.encode(base)[0].tolist()

    fn_name = generate_function_name(model, fn_input_ids, functions)

    fn_def = next((fn for fn in functions if fn.name == fn_name), None)
    if fn_def is None:
        return FunctionCallResult(prompt=item.prompt,
                                  name=fn_name,
                                  parameters={})

    parameters: dict[str, Any] = {}
    for param_name, param_def in fn_def.parameters.items():
        ctx_str = _param_context(base, fn_name, param_name, param_def.type,
                                 parameters)
        param_ids: list[int] = model.encode(ctx_str)[0].tolist()
        parameters[param_name] = generate_value(model, param_ids,
                                                param_def.type, cache)

    return FunctionCallResult(
        prompt=item.prompt,
        name=fn_name,
        parameters=parameters,
    )


def run(
    model: Small_LLM_Model,
    prompts: list[PromptItem],
    functions: list[FunctionDefinition],
    output_path: Path,
) -> None:
    """Process every prompt and write results to output_path as JSON."""
    cache: dict[int, str] = {}
    results: list[dict[str, Any]] = []

    for idx, item in enumerate(prompts, 1):
        print(
            f"[{idx}/{len(prompts)}] {item.prompt[:60]}",
            file=sys.stderr,
        )
        try:
            result = process_prompt(model, item, functions, cache)
            results.append(result.model_dump())
        except Exception as exc:
            print(f"  Error: {exc}", file=sys.stderr)
            results.append({
                "prompt": item.prompt,
                "name": "",
                "parameters": {}
            })

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2, ensure_ascii=False)

    print(f"Output written to {output_path}", file=sys.stderr)
