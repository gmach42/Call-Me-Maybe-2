"""Constrained decoding utilities.

All generation is done token-by-token. Invalid tokens are excluded by
masking logits to -inf (or by restricting candidate sets) before
picking the argmax, so the produced output is always schema-compliant.
"""

import json
import re
from typing import Any
import numpy as np
from llm_sdk.llm_sdk import Small_LLM_Model
from .pydantic_models import FunctionDefinition

# Matches a complete or partial valid JSON number (used as a prefix check).
_NUM_PREFIX_RE = re.compile(r'^-?[0-9]*\.?[0-9]*([eE][+-]?[0-9]*)?$')
# Matches a string made entirely of number characters.
_NUM_CHARS_RE = re.compile(r'^[-0-9.eE+]+$')

NEGINF = float('-inf')
TOP_K = 100  # number of top logits to inspect for value generation
MAX_NUM_STEPS = 32
MAX_STR_STEPS = 64

TYPE_MAP = {
    "int": "integer",
    "float": "number",
    "str": "string",
    "bool": "boolean",
    "list": "array",
    "dict": "object",
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

    Each function name is pre-tokenized once. At every generation step only
    the token IDs that are the correct next token in at least one still-active
    candidate are allowed. The winner is the first candidate whose token
    sequence is fully matched.
    """
    fn_seqs: list[tuple[str, list[int]]] = [
        (fn.name, model.encode(fn.name)[0].tolist()) for fn in functions
    ]

    ctx: list[int] = list(input_ids)
    generated: list[int] = []
    active = list(range(len(fn_seqs)))

    while active:
        pos = len(generated)

        # Return immediately if any active candidate is fully matched.
        for i in active:
            if pos == len(fn_seqs[i][1]):
                return fn_seqs[i][0]

        valid_ids: set[int] = {
            fn_seqs[i][1][pos]
            for i in active if pos < len(fn_seqs[i][1])
        }

        if not valid_ids:
            break

        logits = model.get_logits_from_input_ids(ctx)
        next_id = max(valid_ids, key=lambda t: logits[t])

        ctx.append(next_id)
        generated.append(next_id)

        active = [
            i for i in active
            if pos < len(fn_seqs[i][1]) and fn_seqs[i][1][pos] == next_id
        ]

    return model.decode(generated)


def generate_number_value(
    model: Small_LLM_Model,
    input_ids: list[int],
    cache: dict[int, str],
) -> float:
    """Generate a number value via constrained decoding.

    Inspects the TOP_K highest-logit tokens at each step and keeps only
    those whose decoded string consists entirely of valid number characters
    and whose concatenation with the accumulated string is still a valid
    number prefix.  Stops when the model's greedy choice is no longer a
    number character (and we already have at least one digit).
    """
    ctx: list[int] = list(input_ids)
    raw = ""

    for _ in range(MAX_NUM_STEPS):
        logits = model.get_logits_from_input_ids(ctx)
        arr = np.array(logits, dtype=np.float32)
        top_ids: list[int] = np.argsort(arr)[-TOP_K:][::-1].tolist()

        greedy_str = _token(model, int(top_ids[0]), cache).strip()

        # If we already have content and the greedy choice is not a number
        # character, the model wants to end the value — stop.
        if raw and not _NUM_CHARS_RE.match(greedy_str):
            break

        # Find the highest-logit valid number token in top-K.
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

    try:
        return float(raw) if raw else 0.0
    except ValueError:
        return 0.0


def generate_integer_value(
    model: Small_LLM_Model,
    input_ids: list[int],
    cache: dict[int, str],
) -> int:
    """Generate an integer value via constrained decoding.

    Reuses the number generator, then rounds to the nearest integer so a
    model output like ``3.0`` or ``3.4`` still yields a usable integer
    instead of being discarded.
    """
    num = generate_number_value(model, input_ids, cache)
    return round(num)


def generate_string_value(
    model: Small_LLM_Model,
    input_ids: list[int],
    cache: dict[int, str],
) -> str:
    """Generate a string value via constrained decoding.

    The context already contains the opening quote.  Tokens are generated
    greedily and appended to a raw JSON-escaped buffer, scanning character
    by character for the closing double-quote.  A quote preceded by an
    (unescaped) backslash is treated as an escaped quote belonging to the
    value rather than the terminator, so values that legitimately contain
    `"` or `\\` are preserved.  The buffer is finally run through the JSON
    decoder so escape sequences (`\\"`, `\\\\`, `\\n`, `\\t`, ...) collapse
    to their real characters.
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

    if closed:
        try:
            return str(json.loads(f'"{json_content}"'))
        except json.JSONDecodeError:
            return json_content
    return json_content


def generate_bool_value(
    model: Small_LLM_Model,
    input_ids: list[int],
    cache: dict[int, str],
) -> bool:
    """Generate a boolean value via constrained decoding.

    Scans the TOP_K tokens for tokens that decode to exactly 'true' or
    'false' and returns the boolean corresponding to the higher-logit one.
    """
    logits = model.get_logits_from_input_ids(input_ids)
    arr = np.array(logits, dtype=np.float32)
    top_ids: list[int] = np.argsort(arr)[-TOP_K:][::-1].tolist()

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
    # Unknown type: return None (will be caught by validation)
    return None
