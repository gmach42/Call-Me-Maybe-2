from pathlib import Path

import pytest

from src.parser import load_json_file


def test_load_json_file_reads_valid_json(tmp_path: Path) -> None:
    file_path = tmp_path / "sample.json"
    file_path.write_text('{"prompt": "hello"}', encoding="utf-8")

    result = load_json_file(file_path)

    assert result == {"prompt": "hello"}


def test_load_json_file_raises_for_missing_file(tmp_path: Path) -> None:
    file_path = tmp_path / "missing.json"

    with pytest.raises(ValueError, match="File not found"):
        load_json_file(file_path)


def test_load_json_file_raises_for_invalid_json(tmp_path: Path) -> None:
    file_path = tmp_path / "broken.json"
    file_path.write_text("{broken", encoding="utf-8")

    with pytest.raises(ValueError, match="Invalid JSON"):
        load_json_file(file_path)
