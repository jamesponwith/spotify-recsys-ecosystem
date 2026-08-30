APPS := cadence timbre segue gamut ostinato

.PHONY: help check-claims lint-all

help:
	@grep -E '^[a-z-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN{FS=":.*?## "}{printf "  \033[36m%-13s\033[0m %s\n", $$1, $$2}'

check-claims: ## Verify the root README's hand-typed numbers against each app's artifacts
	@ostinato/.venv/bin/python scripts/check_claims.py

lint-all: check-claims ## check-claims, then lint the shared tooling and each app
	@ostinato/.venv/bin/ruff check scripts
	@ostinato/.venv/bin/ruff format --check scripts
# An app whose venv is absent must fail the target, not be skipped: a skipped
# app reports green while checking nothing. The guard names the missing path
# rather than letting the recipe die with a bare "No such file or directory".
	@for a in $(APPS); do \
	  if [ ! -x $$a/.venv/bin/ruff ]; then \
	    echo "lint-all: $$a/.venv/bin/ruff is missing — run 'uv sync --extra dev --project $$a'"; \
	    exit 1; \
	  fi; \
	  echo "--- $$a"; $(MAKE) -C $$a lint || exit 1; \
	done
