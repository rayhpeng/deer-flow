install:
	uv sync

gateway:
	PYTHONPATH=. uv run uvicorn app.gateway.app:app --host 0.0.0.0 --port 8001

test:
	PYTHONPATH=. uv run pytest tests/ -v

lint:
	uvx ruff check .

format:
	uvx ruff check . --fix && uvx ruff format .
