VENV := .venv/bin

.PHONY: help install lint test simulate sensitivity ledger report all clean

help:
	@grep -E '^[a-z-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN{FS=":.*?## "}{printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}'

install: ## venv + Concerto editable
	uv venv --python 3.12
	uv pip install -e . pytest ruff mypy

lint: ## ruff + mypy + markdown link check
	$(VENV)/ruff check src tests scripts
	$(VENV)/ruff format --check src tests scripts
	$(VENV)/mypy src scripts
	$(VENV)/python scripts/check_links.py

test: ## pytest
	$(VENV)/python -m pytest tests -q

simulate: ## Every allocation arm at equilibrium, over N seeded trials
	$(VENV)/python -m concerto.cli simulate

sensitivity: ## Re-run every arm across the assumed-parameter grid
	$(VENV)/python -m concerto.cli sensitivity

ledger: ## Cardano enforcement model: what the validator holds, what leaks around it
	$(VENV)/python -m concerto.cli ledger

report: ## Render artifacts/results.html and docs/RESULTS.md
	$(VENV)/python scripts/build_report.py

all: lint test simulate sensitivity ledger report ## Everything, from a fixed seed

clean:
	rm -rf artifacts/*.json artifacts/results.html
