.PHONY: help venv install data build splits train reranker eval eval-ab eval-constraints all play serve test lint typecheck clean

PY := .venv/bin/python
CADENCE := .venv/bin/cadence

help:
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}'

venv: ## create the virtualenv
	uv venv --python 3.12 .venv

install: ## install the package and dev extras
	uv pip install --python $(PY) -e ".[dev,llm]"

data: ## download MPD slices and audio features (~15 GB)
	$(PY) scripts/download_data.py

build: ## build catalog, interactions and folksonomy
	$(CADENCE) build --min-tag-playlists 5

splits: ## freeze the evaluation split
	$(CADENCE) splits --n-eval 2000

train: ## fit the retrieval spaces (excludes held-out playlists)
	$(CADENCE) train

reranker: ## fit the learned reranker
	$(CADENCE) train-reranker --n-queries 2000

eval: ## run the offline harness with ablations
	$(CADENCE) evaluate --limit 400 --ablations

eval-ab: ## price rrf_k against the shipped config with a paired band
	$(CADENCE) eval-ab --k 0 --limit 400 --arm rrf_k=30

eval-constraints: ## measure constraint satisfaction of the assembly stage
	$(CADENCE) eval-constraints

audit-lexicon: ## compare MOOD_LEXICON's audio targets with the folksonomy
	$(CADENCE) audit-lexicon

all: build splits train reranker eval eval-constraints ## full offline pipeline

play: ## example query
	$(CADENCE) play "chill acoustic songs for a rainy sunday, nothing explicit, about 40 minutes"

serve: ## start the HTTP API on :8000
	$(CADENCE) serve

test: ## run the test suite
	$(PY) -m pytest tests/ -q

lint: ## ruff
	$(PY) -m ruff check src tests scripts
	$(PY) -m ruff format --check src tests scripts

typecheck: ## mypy
	$(PY) -m mypy src/cadence

clean: ## remove build artifacts (keeps raw downloads)
	rm -rf artifacts data/processed .pytest_cache .ruff_cache .mypy_cache
	find . -name __pycache__ -type d -exec rm -rf {} +
