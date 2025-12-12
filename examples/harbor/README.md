# Harbor Eval

## Setup Instructions

1. Follow the instructions at [README.md](../../README.md#setup-instructions)

2. Install additional dependencies:

```bash
conda activate sci-llm
uv tool install harbor
```

3. Set up API keys:

```bash
conda env config vars set ANTHROPIC_API_KEY="your-api-key-here"
```

## Experiments

```bash
harbor run \
  -p ./ \
  -m anthropic/claude-haiku-4-5 \
  -a claude-code
```