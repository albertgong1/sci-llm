# SciLLM

## Setup Instructions

1. For first time setup, run the following script to create a conda environment `scillm` with python dependencies and packages.

```bash
./scripts/setup_conda_environment.sh
```

2. Update the packages with:

```bash
uv sync --all-groups
```

3. Set your API keys in a file named `.env` in the root directory and add

```bash
GOOGLE_API_KEY=xxxxx
OPENAI_API_KEY=xxxxx
```
