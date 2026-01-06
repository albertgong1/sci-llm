.PHONY: pre-commit ruff ruff-format install-hooks

# Run all pre-commit hooks on all files
pre-commit:
	uv run pre-commit run --all-files

ruff:
	uv run pre-commit run ruff --all-files

ruff-format:
	uv run pre-commit run ruff-format --all-files

install-hooks:
	uv run pre-commit install
