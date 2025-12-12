# SciLLM

## Setup Instructions

```bash
conda create -n sci-llm python=3.13
conda activate sci-llm
pip install uv
uv pip install -e ".[dev]"
# set up pre-commit
pre-commit install
```