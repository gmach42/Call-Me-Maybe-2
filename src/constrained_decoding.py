"""Constrained decoding utilities.

All generation is done token-by-token. Invalid tokens are excluded by
masking logits to -inf.
"""

import json
import math
import re
from typing import Any
import numpy as np
from llm_sdk.llm_sdk import Small_LLM_Model
from .pydantic_models import FunctionDefinition

# Matches a complete or partial valid JSON number (used as a prefix check).
_NUM_PREFIX_RE = re.compile(r'^-?[0-9]*\.?[0-9]*([eE][+-]?[0-9]*)?$')
# Matches a string made entirely of number characters.
_NUM_CHARS_RE = re.compile(r'^[-0-9.eE+]+$')

# Constants for constrained decoding.
NEGINF = float('-inf')
TOP_K = 100
MAX_NUM_STEPS = 32
MAX_STR_STEPS = 64

TYPE_MAP = {
    "int": "integer",
    "float": "number",
    "str": "string",
    "bool": "boolean",
}


def _token(model: Small_LLM_Model, tid: int, cache: dict[int, str]) -> str:
    """Decode a single token ID, with caching."""
    if tid not in cache:
        cache[tid] = model.decode([tid])
    return cache[tid]


def generate_function_name(
    model: Small_LLM_Model,
    input_ids: list[int],
    functions: list[FunctionDefinition],
) -> str:
    """Select a function name via constrained decoding.

    Requires a non-empty `functions` list. Given that, the loop below
    always returns via an exact match: `next_id` is always drawn from
    `valid_ids`, so `active` can never become empty before a full match
    is found. The two raises below guard states that should therefore
    be unreachable, rather than silently decoding a name that matches
    none of `functions`.
    """
    if not functions:
        raise ValueError(
            "generate_function_name requires a non-empty functions list"
        )

    fn_seqs: list[tuple[str, list[int]]] = [
        (fn.name, model.encode(fn.name)[0].tolist()) for fn in functions
    ]

    ctx: list[int] = list(input_ids)
    generated: list[int] = []
    active = list(range(len(fn_seqs)))

    while active:
        pos = len(generated)

        for i in active:
            if pos == len(fn_seqs[i][1]):
                return fn_seqs[i][0]

        valid_ids: set[int] = {
            fn_seqs[i][1][pos]
            for i in active if pos < len(fn_seqs[i][1])
        }

        if not valid_ids:
            raise RuntimeError(
                "generate_function_name: runtime error, no valid "
                "next token for any active candidate"
            )

        logits = model.get_logits_from_input_ids(ctx)
        next_id = max(valid_ids, key=lambda t: logits[t])

        ctx.append(next_id)
        generated.append(next_id)

        active = [
            i for i in active
            if pos < len(fn_seqs[i][1]) and fn_seqs[i][1][pos] == next_id
        ]

    raise RuntimeError(
        "generate_function_name: runtime error, no function name matched the "
        "generated token sequence within the maximum allowed steps"
    )


def generate_number_value(
    model: Small_LLM_Model,
    input_ids: list[int],
    cache: dict[int, str],
) -> float:
    """Generate a number value via constrained decoding.

    Raises:
        ValueError: If no valid number can be generated, or if the
            generated number is not finite (e.g. 1e9999999).
    """
    ctx: list[int] = list(input_ids)
    raw = ""

    for _ in range(MAX_NUM_STEPS):
        logits = model.get_logits_from_input_ids(ctx)
        arr = np.array(logits, dtype=np.float32)

        # argsort -> weakest first
        ascending_by_logit = np.argsort(arr)
        # flip -> strongest first
        descending_by_logit = np.flip(ascending_by_logit)
        # keep the TOP_K best tokens (index 0 of top_ids is the best)
        top_ids: list[int] = descending_by_logit[:TOP_K].tolist()

        greedy_str = _token(model, int(top_ids[0]), cache).strip()

        if raw and not _NUM_CHARS_RE.match(greedy_str):
            break

        best_num_id: int | None = None
        best_num_logit = NEGINF
        for tid in top_ids:
            s = _token(model, tid, cache).strip()
            if not s:
                continue
            if not _NUM_CHARS_RE.match(s):
                continue
            candidate = raw + s
            if not _NUM_PREFIX_RE.match(candidate):
                continue
            if logits[tid] > best_num_logit:
                best_num_logit = logits[tid]
                best_num_id = tid

        if best_num_id is None:
            break

        raw += _token(model, best_num_id, cache).strip()
        ctx.append(best_num_id)

    if not raw:
        raise ValueError("No valid number could be generated")

    try:
        value = float(raw)
    except ValueError as exc:
        raise ValueError(f"Generated number is not valid: {raw!r}") from exc

    if math.isinf(value) or math.isnan(value):
        raise ValueError(f"Generated number is not finite: {raw!r}")
    return value


def generate_integer_value(
    model: Small_LLM_Model,
    input_ids: list[int],
    cache: dict[int, str],
) -> int:
    """Generate an integer value via constrained decoding.

    We are just using generate_number_value and rounding to the nearest
    integer"""
    return round(generate_number_value(model, input_ids, cache))


def generate_string_value(
    model: Small_LLM_Model,
    input_ids: list[int],
    cache: dict[int, str],
) -> str:
    """Generate a string value via constrained decoding.

    Raises:
        ValueError: If the closing quote isn't reached within
            MAX_STR_STEPS, or the generated content isn't valid JSON
            once closed (e.g. a malformed escape sequence).
    """
    ctx: list[int] = list(input_ids)
    json_content = ""
    escaped = False
    closed = False

    for _ in range(MAX_STR_STEPS):
        logits = model.get_logits_from_input_ids(ctx)
        best_id = int(np.argmax(np.array(logits, dtype=np.float32)))
        best_str = _token(model, best_id, cache)

        stop_at: int | None = None
        for i, ch in enumerate(best_str):
            if escaped:
                escaped = False
                continue
            if ch == '\\':
                escaped = True
            elif ch == '"':
                stop_at = i
                break

        if stop_at is not None:
            json_content += best_str[:stop_at]
            closed = True
            break

        json_content += best_str
        ctx.append(best_id)

    if not closed:
        raise ValueError(
            f"Generated string was not closed within {MAX_STR_STEPS} "
            f"steps: {json_content!r}"
        )

    try:
        return str(json.loads(f'"{json_content}"'))
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"Generated string is not valid JSON: {json_content!r}"
        ) from exc


def generate_bool_value(
    model: Small_LLM_Model,
    input_ids: list[int],
    cache: dict[int, str],
) -> bool:
    """Generate a boolean value via constrained decoding."""
    logits = model.get_logits_from_input_ids(input_ids)
    arr = np.array(logits, dtype=np.float32)

    # argsort -> weakest first
    ascending_by_logit = np.argsort(arr)
    # flip -> strongest first
    descending_by_logit = np.flip(ascending_by_logit)
    # keep the TOP_K best tokens (index 0 of top_ids is the best)
    top_ids: list[int] = descending_by_logit[:TOP_K].tolist()

    best_true = NEGINF
    best_false = NEGINF

    for tid in top_ids:
        s = _token(model, tid, cache)
        if s == "true" and logits[tid] > best_true:
            best_true = logits[tid]
        elif s == "false" and logits[tid] > best_false:
            best_false = logits[tid]

    return best_true >= best_false


def generate_value(
    model: Small_LLM_Model,
    input_ids: list[int],
    param_type: str,
    cache: dict[int, str],
) -> Any:
    """Dispatch to the appropriate constrained generator by JSON type."""
    param_type = TYPE_MAP.get(param_type, param_type)
    if param_type == "number":
        return generate_number_value(model, input_ids, cache)
    if param_type == "integer":
        return generate_integer_value(model, input_ids, cache)
    if param_type == "string":
        return generate_string_value(model, input_ids, cache)
    if param_type == "boolean":
        return generate_bool_value(model, input_ids, cache)
    return None
