# SciLLM

## Getting Started

To install our package, use the following pip and uv commands:

```bash
pip install uv
uv sync --all-groups
```

Configure API access to LLMs by creating a file named `.env` in the repository root and adding your API keys:

```bash
# If using Gemini for database construction or Harbor agent evaluation
GOOGLE_API_KEY=xxxxx
# If using OpenAI for database construction or Harbor agent evaluation
OPENAI_API_KEY=xxxxx
```

For example tasks, see our [SuperCon Extraction task](examples/supercon-extraction/README.md) and [Tc Precedent Search task](examples/tc-precedent-search/README.md).

## Reproducing the ICML figures

1. Please follow the steps at [Reproducing SuperCon Extraction Experiments](examples/supercon-extraction/README.md#reproducing-experiments) and [Reproducing Tc Precedent Search Experiments (steps 3--7)](examples/tc-precedent-search/README.md#3-run-trials).

2. Generate scatter plot of mean accuracy vs. total dollar cost for the SuperCon extraction and Tc precedent search tasks:

```bash
uv run python scripts/plot_accuracy_vs_tokens_supercon_tc.py --x-axis cost -od OUTPUT_DIR

# Example:
#   uv run python scripts/plot_accuracy_vs_tokens_supercon_tc.py --x-axis cost -od out-test-0319
```

3. Generate scatter plot of mean accuracy vs. total tokens for the SuperCon extraction and Tc precedent search tasks:

```bash
uv run python scripts/plot_accuracy_vs_tokens_supercon_tc.py --x-axis tokens -od OUTPUT_DIR

# Example:
#   uv run python scripts/plot_accuracy_vs_tokens_supercon_tc.py --x-axis tokens -od out-test-0319
```

4. Generate bar plots of failure rate for the SuperCon extraction task:

```bash
uv run python scripts/plot_failure_rate.py -od OUTPUT_DIR

# Example:
#   uv run python scripts/plot_failure_rate.py -od out-test-0319
```
