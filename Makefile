.PHONY: lint format typecheck validate dev clean

lint:
	uv run ruff check app/

format:
	uv run ruff format app/

typecheck:
	uv run mypy app/ --ignore-missing-imports

validate: lint typecheck
	@echo "All checks passed"

dev:
	uv run python -m app.agent_worker

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true

install:
	uv sync

install-dev:
	uv sync --extra dev

pre-commit-install:
	uv run pre-commit install
