VENV := .venv/bin

.PHONY: help install lint test simulate report all clean

help:
	@grep -E '^[a-z-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN{FS=":.*?## "}{printf "  \033[36m%-9s\033[0m %s\n", $$1, $$2}'

install: ## venv + Cadence, Gamut and Ostinato editable
	uv venv --python 3.12
	uv pip install -e ../cadence -e ../gamut -e . pytest ruff mypy

lint: ## ruff + mypy + markdown link check
	$(VENV)/ruff check src tests scripts
	$(VENV)/ruff format --check src tests scripts
	$(VENV)/mypy src scripts
	$(VENV)/python scripts/check_links.py

test: ## pytest
	$(VENV)/python -m pytest tests -q

simulate: ## Run all three arms and record exposure drift per round
	$(VENV)/python -m ostinato.cli simulate

report: ## Render artifacts/results.html and docs/RESULTS.md
	$(VENV)/python scripts/build_report.py

all: lint test simulate report ## Everything, from a fixed seed

clean:
	rm -rf artifacts/*.json artifacts/results.html
