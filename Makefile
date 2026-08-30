.PHONY: config lint format test sure

RUN := $(if $(shell command -v uv >/dev/null 2>&1 && echo yes),uv run,)

config: ## Create config.yaml from default_config (only if absent)
	@if [ ! -f config.yaml ]; then \
		cp default_config/default_config.yaml config.yaml; \
		echo "Created config.yaml from default_config/default_config.yaml"; \
	else \
		echo "config.yaml already exists, skipping"; \
	fi

lint: ## Run ruff linter
	$(RUN) ruff check .

format: ## Apply ruff formatting
	$(RUN) ruff format .

test: ## Run the test suite
	$(RUN) pytest -q

sure: lint format test ## lint, format, test
