VENV := .venv/bin

.PHONY: help install lint test build train evaluate demo report all clean

help:
	@grep -E '^[a-z-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN{FS=":.*?## "}{printf "  \033[36m%-10s\033[0m %s\n", $$1, $$2}'

install: ## venv + Cadence and Segue editable
	uv venv --python 3.12
	uv pip install -e ../cadence -e . pytest ruff mypy

lint: ## ruff + mypy
	$(VENV)/ruff check src tests scripts
	$(VENV)/ruff format --check src tests scripts
	$(VENV)/mypy src scripts

test: ## pytest
	$(VENV)/python -m pytest tests -q

build: ## Rebuild ordered playlist sequences from the raw MPD slices
	$(VENV)/python -m segue.cli build

train: ## Fit the transition operator
	$(VENV)/python -m segue.cli train

evaluate: ## Head-to-head on ordered challenges
	$(VENV)/python -m segue.cli evaluate

demo: ## Continue one held-out playlist
	$(VENV)/python -m segue.cli demo

report: ## Render artifacts/results.html and docs/RESULTS.md
	$(VENV)/python scripts/build_report.py

all: lint test build train evaluate report ## Everything, from a fixed seed

clean:
	rm -rf artifacts/*.json artifacts/*.pkl artifacts/*.npz artifacts/results.html
