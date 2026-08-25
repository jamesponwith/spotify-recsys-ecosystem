APPS := cadence timbre segue gamut ostinato

.PHONY: help check-claims lint-all

help:
	@grep -E '^[a-z-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN{FS=":.*?## "}{printf "  \033[36m%-13s\033[0m %s\n", $$1, $$2}'

check-claims: ## Verify the root README's hand-typed numbers against each app's artifacts
	@ostinato/.venv/bin/python scripts/check_claims.py

lint-all: check-claims ## check-claims, then lint the shared tooling and each app
	@ostinato/.venv/bin/ruff check scripts
	@ostinato/.venv/bin/ruff format --check scripts
	@for a in $(APPS); do \
	  if [ -x $$a/.venv/bin/ruff ]; then echo "--- $$a"; $(MAKE) -C $$a lint || exit 1; fi; \
	done
