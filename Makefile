.PHONY: setup test lint paper-audit
setup:
	python -m pip install --no-deps --requirement requirements-lock.txt
	python -m pip install --no-deps --no-build-isolation --editable .

test:
	pytest -q

lint:
	ruff check .

paper-audit:
	python scripts/audit_task20_final_experiment.py
