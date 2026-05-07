import json
import numpy as np
from pathlib import Path
from llm_sdk.llm_sdk import Small_LLM_Model
from typing import Any
import re


# # ── 1. SETUP ──────────────────────────────────────────────

# functions = json.loads(Path("data/input/functions_definition.json").read_text())
# prompts = json.loads(Path("data/input/function_calling_tests.json").read_text())

# model = Small_LLM_Model()

# vocab_raw = json.loads(Path(model.get_path_to_vocab_file()).read_text())
# vocab = {k: v for k, v in vocab_raw.items()}   # id  → "fn_add"
# reverse_vocab = {v: k for k, v in vocab.items()}    # "fn_add" → id

# function_names = [f["name"] for f in functions]


# # ── 2. HELPERS ────────────────────────────────────────────

# def build_prompt(user_prompt: str) -> str:
#     fn_list = "\n".join(f"- {f['name']}: {f['description']}" for f in functions)
#     return (
#         f"Available functions:\n{fn_list}\n\n"
#         f"User request: {user_prompt}\n\n"
#         f"Response JSON:"
#     )


# def get_valid_ids(generated_so_far: list[int]) -> list[int]:
#     """TODO: retourner les token IDs valides selon l'état courant du JSON."""
#     # Hint : utilise ton trie pour la phase 1 (nom de fonction)
#     # et les types de paramètres pour la phase 2
#     raise NotImplementedError


# def is_complete(generated_so_far: list[int]) -> bool:
#     """TODO: retourner True quand le JSON est terminé (accolade fermante finale)."""
#     text = model.decode(generated_so_far)
#     return text.strip().endswith("}")


# # ── 3. BOUCLE DE GÉNÉRATION ───────────────────────────────

# def generate(prompt: str) -> dict:
#     input_ids: list[int] = model.encode(prompt)[0].tolist()
#     generated: list[int] = []

#     for _ in range(200):                                    # max 200 tokens
#         logits = model.get_logits_from_input_ids(input_ids)
#         logits_np = np.array(logits)

#         valid = get_valid_ids(generated)
#         if not valid:
#             break

#         mask = np.full(len(logits_np), float("-inf"))
#         mask[valid] = logits_np[valid]                      # garder seulement les valides

#         next_token = int(np.argmax(mask))
#         input_ids.append(next_token)
#         generated.append(next_token)

#         if is_complete(generated):
#             break

#     return json.loads(model.decode(generated))


# # ── 4. PIPELINE ───────────────────────────────────────────

# results = []
# for entry in prompts:
#     result = generate(build_prompt(entry["prompt"]))
#     results.append({
#         "prompt":     entry["prompt"],
#         "name":       result["name"],
#         "parameters": result["parameters"],
#     })

# Path("data/output").mkdir(parents=True, exist_ok=True)
# Path("data/output/function_calls.json").write_text(
#     json.dumps(results, indent=2)
# )
# print(f"Done — {len(results)} résultats écrits.")


functions = json.loads(Path("data/input/functions_definition.json").read_text())
prompts = json.loads(Path("data/input/function_calling_tests.json").read_text())
function_names = [f["name"] for f in functions]

model = Small_LLM_Model()


def is_token_valid(current_prompt: str, token_str: str, functions: list[str]) -> bool:
    attempt = current_prompt + token_str
    for function in functions:
        if function.startswith(attempt) or attempt.startswith(function):
            return True
    return False


def best_valid_token(logits_np: np.ndarray, current_text: str, candidates: list[str]) -> tuple[int, str]:
    logits = logits_np.copy()
    for _ in range(len(logits)):
        best_id = np.argmax(logits)
        best_str = model.decode(best_id)
        if is_token_valid(current_text, best_str, candidates):
            return best_id, best_str
        logits[best_id] = -np.inf
    raise RuntimeError


def generate_function_name(input_ids: list[int]) -> str:
    """Generate the function name token by token"""

    # looking for the name of the function with a maxlenght name of 50 characters
    generated_name: str = ""
    for _ in range(50):
        logits = model.get_logits_from_input_ids(input_ids)
        logits_np = np.array(logits)

        next_token_id, token_str = best_valid_token(logits_np, generated_name, function_names)
        generated_name += token_str
        input_ids.append(next_token_id)

        if generated_name in function_names:
            break

    return generated_name


def find_function_definition(function_name: str) -> dict[str, Any] | None:
    """Find the function definition in the list of functions"""
    for function in functions:
        if function["name"] == function_name:
            return function
    return None


def extract_quoted_texts(text: str) -> list[str]:
    matches = re.findall(r"'([^']+)'|\"([^\"]+)\"", text)
    return [m[0] or m[1] for m in matches]


def generate_parameters(prompt_text: str, function_definition: dict[str, Any]) -> dict[str, Any]:
    """Generate the parameters"""
    parameters: dict[str, Any] = {}
    schema = function_definition["parameters"]

    number_pattern = r'\d+'
    numbers = [int(n) for n in re.findall(number_pattern, prompt_text)]
    quoted_texts = extract_quoted_texts(prompt_text)
    words = prompt_text.split()

    number_index = 0
    quoted_index = 0

    for parameter_name, parameter_definition in schema.items():
        parameter_type = parameter_definition["type"]

        if parameter_type == "number":
            parameters[parameter_name] = numbers[number_index] if number_index < len(numbers) else 0
            number_index += 1

        elif parameter_type == "string":
            if quoted_index < len(quoted_texts):
                parameters[parameter_name] = quoted_texts[quoted_index]
                quoted_index += 1
            else:
                parameters[parameter_name] = words[-1] if words else ""

        else:
            parameters[parameter_name] = None

    return parameters


# main loop

results = []
for entry in prompts:

    prompt = entry['prompt']
    input_ids = model.encode(prompt)[0].tolist()

    function_name = generate_function_name(input_ids)
    function_definition = find_function_definition(function_name)
    if not function_definition:
        print(f"Function not found: {function_name}")
        continue
    parameters = generate_parameters(prompt, function_definition)

    results.append({
        "prompt": prompt,
        "name": function_name,
        "parameters": parameters
    })
print(results)
