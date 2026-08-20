VENV := .venv/bin

.PHONY: help install lint test collect audit report all clean

help:
	@grep -E '^[a-z-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN{FS=":.*?## "}{printf "  \033[36m%-9s\033[0m %s\n", $$1, $$2}'

install: ## venv + Cadence and Gamut editable
	uv venv --python 3.12
	uv pip install -e ../cadence -e . pytest ruff mypy

lint: ## ruff + mypy + markdown link check
	$(VENV)/ruff check src tests scripts
	$(VENV)/ruff format --check src tests scripts
	$(VENV)/mypy src scripts
	$(VENV)/python scripts/check_links.py

test: ## pytest
	$(VENV)/python -m pytest tests -q

collect: ## Run Cadence over the held-out battery and cache what it surfaced
	$(VENV)/python -m gamut.cli collect

audit: ## Per-channel attribution + the accuracy/exposure frontier
	$(VENV)/python -m gamut.cli audit

report: ## Render artifacts/results.html and docs/RESULTS.md
	$(VENV)/python scripts/build_report.py

all: lint test collect audit report ## Everything, from a fixed seed

clean:
	rm -rf artifacts/*.json artifacts/*.npz artifacts/results.html
