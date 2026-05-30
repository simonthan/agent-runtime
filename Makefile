.PHONY: dev lint format test build clean

dev:
	uv sync --all-groups

lint:
	uv run ruff format --check .
	uv run ruff check .
	uv run ty check src/ --exclude 'src/agent_runtime/safety/injection_detector.py'

format:
	uv run ruff format .
	uv run ruff check --fix .

test:
	uv run pytest

build:
	uv build

clean:
	rm -rf dist build *.egg-info .pytest_cache .ruff_cache
