.PHONY: setup test smoke lint
setup:
	python -m pip install -e '.[dev]'

test:
	pytest -q

smoke:
	python scripts/smoke_test.py

lint:
	ruff check src tests scripts
