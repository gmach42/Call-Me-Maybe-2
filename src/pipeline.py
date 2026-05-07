from pathlib import Path
from .parser import load_json_file
from .pydantic_models import PromptData


def run_pipeline(json_path: Path) -> None:
    try:
        raw_data = load_json_file(json_path)
        prompt_data = PromptData.model_validate(raw_data)
        print(f"Prompt: {prompt_data.prompt}")
    except ValueError as exc:
        print(f"Erreur lors du chargement du fichier JSON: {exc}")
    except Exception as exc:
        print(f"Erreur inattendue: {exc}")


