# Harbor Eval

## Setup Instructions

1. Follow the instructions at [README.md](../../README.md#setup-instructions)

2. Install additional dependencies:

> \[!TIP\]
> Docker needs to be installed and running for harbor to work. Mac installation instructions: https://docs.docker.com/desktop/setup/install/mac-install/

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