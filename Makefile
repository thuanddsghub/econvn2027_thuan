.PHONY: setup test lint paper-audit
setup:
	python -m pip install -e '.[dev]'

test:
	pytest -q

lint:
	ruff check .

paper-audit:
	python scripts/audit_task20_final_experiment.py
