# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Code best practices

Generally:
- Write clear and modular code with short functions and files for readability
- Functions and variables should have short yet descriptive names
- Use 1-2 line comments to describe complex algorithms or logic
- Always clarify if any prompts/instructions are ambiguous
- Use `# TODO:` comments when needed
- Avoid long and nested for loops, create separate functions and files for complex logic

For python files:
- Use python 3.13 version
- Use type hints like `x: int, list, dict, Any, Literal["value"], df: pd.DataFrame`. Also use nested types like `convo: list[dict[str, str]]`. Always use such type hints for function signatures, and often for variables that could be difficult to understand.
- Use `Path` for file paths
- Use a docstring at the top of a python file: 1-2 lines for the purpose, and a command usage
- Use pandas dataframes and CSVs generally for data analysis

For bash files:
- Start with shebang `!/usr/bin/env bash`
- Use a usage header if the bash script takes 1+ CLI arguments
- When 1+ CLI arguments, `shift` by the number of arguments, assign `cmd_args=$@` and pass `$cmd_args` to the script that bash files calls (if any)

## Common Commands

### Environment Setup
```bash
# Set up conda environment and install dependencies
./setup_conda_environment.sh

uv sync --all-groups
```

## Important File Locations

- Project code under `src/`
- Miscellaneous scripts under `scripts/`
- `pyproject.toml` for python dependencies managed by `uv` and `setup_conda_environment.sh` for environment setup and variables
