
export SRC_DIR=validator
export PYTHONPATH='$(shell pwd)'

dev:
	pip install -e ".[dev]"

lint:
	ruff check .

test:
	pytest ./tests

type:
	pyright validator

qa:
	make lint
	make type
	make test

dev-sync:
	uv sync --all-extras --no-cache

format:
	uv run ruff format

validate:
	uv run ruff format
	uv run ruff check --fix
	uv run mypy --ignore-missing-imports --install-types --non-interactive --package $(SRC_DIR)