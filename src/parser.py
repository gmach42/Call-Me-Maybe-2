from pathlib import Path
import json
from typing import Any


def load_json_file(path: Path) -> Any:
    try:
        with path.open("r", encoding="utf-8") as file_handle:
            return json.load(file_handle)
    except FileNotFoundError:
        raise ValueError(f"Fichier introuvable: {path}")
    except json.JSONDecodeError as exc:
        raise ValueError(f"JSON invalide dans {path}: {exc}") from exc
