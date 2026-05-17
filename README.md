*This project has been created as part of the 42 curriculum by gildas.*

# call me maybe — Introduction to function calling in LLMs

## Description

This project implements a **function calling** system for small language models.
Given a natural-language prompt and a set of function definitions (name, parameters,
return type, description), the program outputs a JSON file where each entry contains
the most appropriate function name and the extracted argument values.

The key challenge is reliability: small models (Qwen3-0.6B, ~600 M parameters) fail
to produce valid JSON on their own roughly 70 % of the time.  This implementation
achieves **100 % valid JSON** through **constrained decoding** — the model's logits are
filtered at every generation step so that only schema-compliant tokens can be selected.

## Instructions

### Installation

```bash
uv sync
```

### Run

```bash
uv run python -m src \
  --functions_definition data/input/functions_definition.json \
  --input               data/input/function_calling_tests.json \
  --output              data/output/function_calling_results.json
```

All three flags are optional and fall back to the paths above.

### Other Makefile targets

| Target | Effect |
|---|---|
| `make install` | `uv sync` |
| `make run` | run with default paths |
| `make debug` | run under `pdb` |
| `make lint` | flake8 + mypy |
| `make lint-strict` | flake8 + mypy --strict |
| `make test` | pytest |
| `make clean` | remove caches |

## Algorithm explanation

### Function name selection

Every function name is **pre-tokenized once** with `model.encode()`.  This yields,
for each function, an ordered list of token IDs.

At each generation step the decoder maintains a set of *active* candidates
(functions whose token sequence matches the tokens generated so far).  The
only token IDs considered at step *i* are those at position *i* in at least one
active candidate — all other logits are implicitly excluded.  The highest-logit
valid token is picked (argmax over the restricted set).  Candidates that no
longer match the chosen token are pruned.  Generation ends as soon as a candidate
is fully matched.

This guarantees the output is **always one of the defined function names**.

### Parameter value generation

After the function is chosen, the values are generated one parameter at a time.
For each parameter the pipeline builds a full context string:

```
…Function: fn_add_numbers
Parameters: {"a":
```

The LLM is asked to continue from that exact position.  Depending on the
declared JSON type, a different constrained generator is invoked:

* **number** — at each step the top-100 logits are inspected; only tokens
  whose decoded string consists entirely of number characters (`[-0-9.eE+]`)
  and whose concatenation with the accumulated value is still a valid number
  prefix are allowed.  Generation stops when the model's greedy choice is no
  longer a number character (and at least one digit has been produced).

* **string** — tokens are generated greedily (no masking needed) until the
  model's top choice contains a closing double-quote `"`, at which point only
  the content before the quote is kept.  The opening quote is part of the
  context, so the model generates purely the value.

* **boolean** — the top-100 tokens are scanned; the highest-logit token that
  decodes to exactly `"true"` or `"false"` determines the result.

All three generators use a **lazy decode cache** (`dict[int, str]`): each token
ID is decoded at most once per run, avoiding repeated calls to `model.decode`.

## Design decisions

* **No vocab file** — the `get_path_to_vocab_file()` helper is not used.
  Token strings are obtained on demand via `model.decode([token_id])` and
  memoised in a shared dictionary.  This keeps the implementation simple and
  independent of the file format.

* **Top-K inspection** (K = 100) for value types — iterating over all ~150 k
  vocabulary entries per step would be too slow.  In practice the correct value
  tokens always appear in the top 100 logits when the context is well-formed.

* **Prompt format** — a short, instruction-style plain-text prompt is used
  (no chat template).  It lists available functions with their signatures and
  descriptions, then states the task, and ends with `"Function: "` so the
  model immediately generates the function name.

* **Parameter context reuse** — previously generated parameter values are
  injected back into the context for each subsequent parameter, giving the
  model full visibility of what has already been filled in.

## Performance analysis

* **Accuracy** — on the provided 11-prompt test set the system correctly
  identifies both function name and all argument values.  The constrained
  decoding for function names is deterministic (100 % correct by construction);
  value accuracy depends on the model's language understanding.

* **JSON validity** — guaranteed 100 % by construction: the decoder can only
  produce tokens that are consistent with the schema at every step.

* **Speed** — each LLM call is one forward pass (~0.1–0.5 s on CPU, faster on
  GPU).  A typical prompt requires ~5–10 calls for the function name and
  ~10–30 calls per parameter.  The full 11-prompt test set completes in under
  5 minutes on standard CPU hardware.

## Challenges faced

* **BPE tokenization boundaries** — a function name like `fn_add_numbers` may
  tokenize differently depending on what precedes it.  Pre-tokenizing with
  `model.encode(fn.name)` (no special tokens, isolated string) proved reliable
  for function names composed of ASCII letters, digits, and underscores.

* **Number token ambiguity** — numbers such as `265` can be one token or
  several.  The top-K + prefix-regex approach handles both cases transparently.

* **Stopping condition for numbers** — deciding when the model has finished
  generating a number (vs. wanting to produce a longer value) required careful
  heuristics: stop when the greedy token is a non-number character AND at
  least one digit has already been accumulated.

## Testing strategy

Unit tests cover pydantic model validation (`tests/test_models.py`) and JSON
loading helpers (`tests/test_parser.py`).  End-to-end validation is done by
running the program on the provided input files and inspecting the output JSON
for correctness and schema compliance.

```bash
make test          # unit tests
make run           # full end-to-end run
```

## Example usage

```bash
# Default paths
uv run python -m src

# Custom paths
uv run python -m src \
  --functions_definition data/input/functions_definition.json \
  --input               data/input/function_calling_tests.json \
  --output              data/output/my_results.json
```

Example output entry:

```json
{
  "prompt": "What is the sum of 2 and 3?",
  "name": "fn_add_numbers",
  "parameters": {"a": 2.0, "b": 3.0}
}
```

## Resources

* Qwen3 model — https://huggingface.co/Qwen/Qwen3-0.6B
* Pydantic v2 — https://docs.pydantic.dev/latest/
* BPE tokenization — https://huggingface.co/learn/nlp-course/chapter6/5
* Constrained decoding overview — https://arxiv.org/abs/2407.09809
* JSON schema spec — https://json-schema.org/

**AI usage** — GitHub Copilot was used to help design and implement the
constrained decoding logic, the prompt format, and the pipeline structure.
All generated code was reviewed, understood, and tested before inclusion.
