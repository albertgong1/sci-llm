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

First, set your API keys (`GOOGLE_API_KEY`, `OPENAI_API_KEY`), e.g. in a `.env` file:

```bash
echo "GOOGLE_API_KEY=xxxx" >> .env
echo "OPENAI_API_KEY=xxxx" >> .env
```

### Property extraction

```bash
./src/pbench_eval/extract_properties.py \
    --task supercon \
    --server gemini \
    --model_name gemini-2.5-flash \
    -od out/
```