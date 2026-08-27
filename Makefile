.PHONY: help lint format test sure

help: ## Show available targets
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}'

lint: ## Run ruff linter
	uv run ruff check .

format: ## Apply ruff formatting
	uv run ruff format .

test: ## Run the test suite
	uv run pytest -q

sure: lint format test ## lint, format, test
