VENV := .venv/bin

.PHONY: help install lint test phase0 demo report all clean

help:
	@grep -E '^[a-z-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN{FS=":.*?## "}{printf "  \033[36m%-10s\033[0m %s\n", $$1, $$2}'

install: ## Create the venv and install Cadence + Timbre editable
	uv venv --python 3.12
	uv pip install -e ../cadence -e . pytest ruff mypy

lint: ## ruff + mypy + markdown link check
	$(VENV)/ruff check src tests scripts
	$(VENV)/ruff format --check src tests scripts
	$(VENV)/mypy src scripts
	$(VENV)/python scripts/check_links.py

test: ## pytest
	$(VENV)/python -m pytest tests -q

phase0: ## Run the falsification test and rule on Gate 0
	$(VENV)/python -m timbre.cli phase0

demo: ## Run the joint Cadence+Timbre demo across the query battery
	$(VENV)/python -m timbre.cli demo-report

report: ## Render artifacts/results.html from the JSON artifacts
	$(VENV)/python scripts/build_report.py

all: lint test phase0 demo report ## Everything, from a fixed seed

clean:
	rm -rf artifacts/*.json artifacts/*.pkl artifacts/results.html
