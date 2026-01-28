# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Strict code rules

- Never hard-code, store, or commit any API keys, always assume they are available through environment variables

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
- For CLI argument parsing, use the starter code below and add additional arguments using parser.add_argument(...):

```python
parser = argparse.ArgumentParser(
    description="Mass extract properties from PDF files using unsupervised LLM extraction"
)
parser = pbench.add_base_args(parser)
```
- Use a flat script structure for Python scripts within an examples/ subdirectory. Assume that the script will never be imported as a module, so do not use if __name__ == "__main__" to run the script.
- Always add argument type annotations and return type annotations.

For bash files:
- Start with shebang `!/usr/bin/env bash`
- Use a usage header if the bash script takes 1+ CLI arguments
- When 1+ CLI arguments, `shift` by the number of arguments, assign `cmd_args=$@` and pass `$cmd_args` to the script that bash files calls (if any)

## Matplotlib best practices
- For single-column figures, use figsize=(3.25, 2.5) as per the ICML guidelines
- For two-column figures, use figsize=(6.75, 2.5) as per the ICML guidelines

## Common Commands

### Environment Setup

```bash
uv sync --all-groups
```

## Important File Locations

- Project code under `src/`
- Miscellaneous scripts under `scripts/`
- `pyproject.toml` for python dependencies managed by `uv` and `setup_conda_environment.sh` for environment setup and variables
