# SciLLM

## Setup Instructions

For first time setup, run the following script to create a conda environment `scillm` with python dependencies and packages.

```bash
./scripts/setup_conda_environment.sh
```

Update the packages with:

```bash
uv sync --all-groups
```

## Run Property Extraction Experiments

Set your API keys in a file named `.env` in the root directory and add

```bash
GOOGLE_API_KEY=xxxxx
OPENAI_API_KEY=xxxxx
```
