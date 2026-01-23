# Flux Precedent Search Evaluation

This directory contains the workflow for evaluating the "Precedent Search" task, where an agent must determine if a material has been reported with flux growth.

## Detailed Workflow

### 1. Download Development Set (One-Time Setup)

Download [flux-material_dev-set.csv](https://drive.google.com/file/d/1lJrRh6D8Cj1Wq5Ta8w0_rUUSGh0wTtS3/view?usp=drive_link) and save to `examples/flux-precedent-search/`.

### 2. LLM API Workflow

We can prompt LLMs with web search grounding to perform flux precedent search.

```bash
uv run python run_precedent_search_with_llms.py --server gemini -m gemini-3-pro-preview -od out --use_web_search
```

This creates a CSV at `out/precedent_search__model=gemini-3-pro-preview__web_search.csv` with LLM predictions and DOI citations for material flux growth reports.

TODO: Add instructions for harbor tasks and evaluation
