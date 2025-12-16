# SciLLM

## Setup Instructions

For first time setup, run the following script to create a conda environment `scillm` with python dependencies and packages.

```bash
./setup_conda_environment.sh
```

Update the packages with:

```bash
uv sync --all-groups
```

## Run experiments

First, set your API keys (`GOOGLE_API_KEY`, `OPENAI_API_KEY`), e.g. in your conda environment:

```bash
conda env config vars set GOOGLE_API_KEY="xxxxx"
conda env config vars set OPENAI_API_KEY="xxxxx"
```

### Property extraction

```bash
uv run python -m pbench_eval.extract_properties \
    --task supercon \
    --server gemini \
    --model_name gemini-2.5-flash \
    -od out/
```