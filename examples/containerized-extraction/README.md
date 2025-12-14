# SuperCon Property Extraction

> \[!NOTE\]
> Harbor tasks can be run with `gemini-cli` or `claude-code` agents. (You must provide
> the corresponding API key(s) in the repo root `.env`.)

## Setup (uv + Harbor)

1. Install project dependencies with uv (installs Harbor in the dev extra):

```bash
uv sync --extra dev
```

2. Provide API keys (either export them, or put them in the repo root `.env`):

```bash
export GOOGLE_API_KEY="your-api-key-here"
export ANTHROPIC_API_KEY="your-api-key-here"
```

Claude Code also supports OAuth-style auth via `CLAUDE_CODE_OAUTH_TOKEN=...`.

The PDF corpus should live under `examples/containerized-extraction/data/Paper_DB`
(15 PDFs shipped via PaperDB.tar in the upstream instructions).

## Build Harbor Tasks Locally

This repository generates Harbor-ready tasks (one per paper) into `out/harbor/...`.

### Easy mode (with transcription)

Includes a pre-extracted `paper.txt` inside the task environment.

```bash
uv run python examples/containerized-extraction/prepare_harbor_tasks.py \
  --task tc --paper-mode easy --write-job-config --force
```

Tasks are written to `out/harbor/supercon-mini/tc/easy/tasks`, and the job config is
saved at `out/harbor/supercon-mini/tc/easy/job.yaml`.

### Hard mode (PDF-only, no pre-transcription)

Omits `paper.txt` so the agent must use the PDF directly. The container includes
`pdftotext` to allow terminal agents to extract text on their own.

```bash
uv run python examples/containerized-extraction/prepare_harbor_tasks.py \
  --task tc --paper-mode hard --write-job-config --force
```

Tasks are written to `out/harbor/supercon-mini/tc/hard/tasks`, and the job config is
saved at `out/harbor/supercon-mini/tc/hard/job.yaml`.

### Build a single paper (optional)

Useful while iterating:

```bash
uv run python examples/containerized-extraction/prepare_harbor_tasks.py \
  --task tc --paper-mode easy --refno PR05001178 --write-job-config --force
```

## Run Harbor Locally

All Harbor runs should go through `run_harbor.py` so `.env` is loaded (and keys are
mapped to what each agent expects).

Note: generated `instruction.md` files start with `@paper.pdf` (hard) or `@paper.txt` (easy) so
the `gemini-cli` agent automatically attaches the paper content.

### Full jobs

- Easy + oracle:

```bash
uv run python examples/containerized-extraction/run_harbor.py jobs start \
  -c out/harbor/supercon-mini/tc/easy/job.yaml -a oracle
```

- Hard + oracle:

```bash
uv run python examples/containerized-extraction/run_harbor.py jobs start \
  -c out/harbor/supercon-mini/tc/hard/job.yaml -a oracle
```

- Easy + Gemini CLI:

```bash
uv run python examples/containerized-extraction/run_harbor.py jobs start \
  -c out/harbor/supercon-mini/tc/easy/job.yaml \
  -a gemini-cli -m gemini/gemini-2.5-flash
```

- Hard + Gemini CLI (PDF-only):

```bash
uv run python examples/containerized-extraction/run_harbor.py jobs start \
  -c out/harbor/supercon-mini/tc/hard/job.yaml \
  -a gemini-cli -m gemini/gemini-2.5-flash
```

- Easy + Claude Code:

```bash
uv run python examples/containerized-extraction/run_harbor.py jobs start \
  -c out/harbor/supercon-mini/tc/easy/job.yaml \
  -a claude-code
```

- Hard + Claude Code (PDF-only):

```bash
uv run python examples/containerized-extraction/run_harbor.py jobs start \
  -c out/harbor/supercon-mini/tc/hard/job.yaml \
  -a claude-code
```

### Single-task trials

Replace `<task-id>` with one from the tasks directory:

```bash
ls out/harbor/supercon-mini/tc/easy/tasks
```

- Easy + oracle:

```bash
uv run python examples/containerized-extraction/run_harbor.py trials start \
  -p out/harbor/supercon-mini/tc/easy/tasks/<task-id> -a oracle
```

- Easy + Gemini CLI:

```bash
uv run python examples/containerized-extraction/run_harbor.py trials start \
  -p out/harbor/supercon-mini/tc/easy/tasks/<task-id> \
  -a gemini-cli -m gemini/gemini-2.5-flash
```

- Hard + Gemini CLI (PDF-only):

```bash
uv run python examples/containerized-extraction/run_harbor.py trials start \
  -p out/harbor/supercon-mini/tc/hard/tasks/<task-id> \
  -a gemini-cli -m gemini/gemini-2.5-flash
```

- Easy + Claude Code:

```bash
uv run python examples/containerized-extraction/run_harbor.py trials start \
  -p out/harbor/supercon-mini/tc/easy/tasks/<task-id> \
  -a claude-code
```

- Hard + Claude Code (PDF-only):

```bash
uv run python examples/containerized-extraction/run_harbor.py trials start \
  -p out/harbor/supercon-mini/tc/hard/tasks/<task-id> \
  -a claude-code
```

## Smoke Test (Build + One Trial)

Builds both modes for one paper and runs a single trial per agent (skips agents whose
keys are missing):

```bash
uv run python examples/containerized-extraction/smoke_harbor.py --task tc --refno PR05001178
```

## Where Results Go

- Jobs: `jobs/<timestamp>/result.json`
- Trials: `trials/<trial-name>/verifier/reward.txt`
- Debugging mismatches: `trials/<trial-name>/verifier/details.json`
- Agent logs:
  - Gemini CLI: `trials/<trial-name>/agent/gemini-cli.txt`
  - Claude Code: `trials/<trial-name>/agent/claude-code.txt`