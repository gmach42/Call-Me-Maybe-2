ifdef CACHE_DIR
export UV_CACHE_DIR=$(CACHE_DIR)/uv_cache
export UV_PROJECT_ENVIRONMENT=$(CACHE_DIR)/venv
export HF_HOME=$(CACHE_DIR)/hf_cache
endif

PYTHON := uv run python
FLAKE8 := uv run flake8
MYPY := uv run mypy
LINT_PATHS := src

install:
	uv sync

run:
	$(PYTHON) -m src

debug:
	$(PYTHON) -m pdb src/__main__.py

clean:
	find . -type d -name "__pycache__" -prune -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -prune -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".mypy_cache" -prune -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete

lint:
	$(FLAKE8) $(LINT_PATHS)
	$(MYPY) $(LINT_PATHS) --warn-return-any --warn-unused-ignores --ignore-missing-imports --disallow-untyped-defs --check-untyped-defs

lint-strict:
	$(FLAKE8) $(LINT_PATHS)
	$(MYPY) $(LINT_PATHS) --strict

.PHONY: install run debug clean lint lint-strict
