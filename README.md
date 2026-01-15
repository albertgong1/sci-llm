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

For example tasks, see our [SuperCon Extraction task](examples/supercon-extraction/README.md), [Biosurfactants Extraction task](examples/biosurfactants-extraction/README.md), and [Tc Precedent Search task](examples/tc-precedent-search/README.md).
