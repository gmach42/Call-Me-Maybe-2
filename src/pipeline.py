from pathlib import Path

from .parser import load_json_file


def process_prompts(
    input_path: Path,
    functions_path: Path,
) -> list[dict[str, object]]:
    """Load inputs and build placeholder results."""
    prompts = load_json_file(input_path)
    load_json_file(functions_path)

    results: list[dict[str, object]] = []
    for item in prompts:
        prompt_text = item["prompt"]

        # Ici, tu appelleras le LLM pour choisir la fonction + arguments
        # puis tu valideras la sortie avec Pydantic.
        results.append(
            {
                "prompt": prompt_text,
                "chosen_function": "exemple_fonction",
                "arguments": {},
            }
        )

    return results
