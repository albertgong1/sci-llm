# SuperCon Property Extraction

> \[!NOTE\]
> Harbor tasks can be run with `gemini-cli` or `claude-code` agents. (You must provide
> the corresponding API key(s) in the repo root `.env`.)

See `examples/containerized-extraction/knowledgebase.md` for a deeper explanation of the
task layout, “API” contracts, and debugging workflow.

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

### Choose a prompt template

Two template bundles live in `examples/containerized-extraction/`:
- `ground-template` (default): free-form extraction prompt (JSON examples allowed in template)
- `prompted-template`: question-based prompt (uses `{question_blocks}` etc.)

By default, `prepare_harbor_tasks.py` uses `ground-template`.

```bash
uv run python examples/containerized-extraction/prepare_harbor_tasks.py \
  --task tc --write-job-config --force
```

Tasks are written to `out/harbor/supercon-mini/tc/ground-template/tasks`, and the job config is
saved at `out/harbor/supercon-mini/tc/ground-template/job.yaml`.

To build the alternative template (`prompted-template`):

```bash
uv run python examples/containerized-extraction/prepare_harbor_tasks.py \
  --task tc --template prompted-template --write-job-config --force
```

### Template variables

`prepare_harbor_tasks.py` does *safe* placeholder substitution (so JSON braces inside prompts
won't break rendering). Missing placeholders do not error; they are left as-is.

Any of these placeholders can be used in `instruction.md.template`:

- `{refno}`, `{task}`, `{task_name}`, `{task_id}`
- `{pdf_path}`, `{meta_path}`, `{predictions_path}`
- `{paper_at_command}`, `{gemini_at_commands}`, `{claude_file_examples}`
- `{question_blocks}`, `{questions_json}`, `{task_meta_json}`

### Build a single paper (optional)

Useful while iterating:

```bash
uv run python examples/containerized-extraction/prepare_harbor_tasks.py \
  --task tc --refno PR05001178 --write-job-config --force
```

## Run Harbor Locally

All Harbor runs should go through `run_harbor.py` so `.env` is loaded (and keys are
mapped to what each agent expects).

Note: the provided templates begin with `@paper.pdf` via `{paper_at_command}`, so the
`gemini-cli` agent automatically attaches the PDF.

### Full jobs (ground-template default)

- Oracle:

```bash
uv run python examples/containerized-extraction/run_harbor.py jobs start \
  -c out/harbor/supercon-mini/tc/ground-template/job.yaml -a oracle
```

- Gemini CLI (PDF-only):

```bash
uv run python examples/containerized-extraction/run_harbor.py jobs start \
  -c out/harbor/supercon-mini/tc/ground-template/job.yaml \
  -a gemini-cli -m gemini/gemini-2.5-flash
```

- Claude Code (PDF-only):

```bash
uv run python examples/containerized-extraction/run_harbor.py jobs start \
  -c out/harbor/supercon-mini/tc/ground-template/job.yaml \
  -a claude-code
```

To run `prompted-template`, replace `ground-template` with `prompted-template` in the config path.

### Single-task trials

Replace `<task-id>` with one from the tasks directory:

```bash
ls out/harbor/supercon-mini/tc/ground-template/tasks
```

- Oracle:

```bash
uv run python examples/containerized-extraction/run_harbor.py trials start \
  -p out/harbor/supercon-mini/tc/ground-template/tasks/<task-id> -a oracle
```

- Gemini CLI (PDF-only):

```bash
uv run python examples/containerized-extraction/run_harbor.py trials start \
  -p out/harbor/supercon-mini/tc/ground-template/tasks/<task-id> \
  -a gemini-cli -m gemini/gemini-2.5-flash
```

- Claude Code (PDF-only):

```bash
uv run python examples/containerized-extraction/run_harbor.py trials start \
  -p out/harbor/supercon-mini/tc/ground-template/tasks/<task-id> \
  -a claude-code
```

## Smoke Test (Build + One Trial)

Builds one paper and runs a single trial per agent (skips agents whose keys are missing):

```bash
uv run python examples/containerized-extraction/smoke_harbor.py --task tc --refno PR05001178
```

To smoke-test the alternative prompt template:

```bash
uv run python examples/containerized-extraction/smoke_harbor.py --task tc --refno PR05001178 \
  --templates prompted-template
```

## Where Results Go

- Jobs: `jobs/<timestamp>/result.json`
- Trials: `trials/<trial-name>/verifier/reward.txt`
- Debugging mismatches: `trials/<trial-name>/verifier/details.json`
- Agent logs:
  - Gemini CLI: `trials/<trial-name>/agent/gemini-cli.txt`
  - Claude Code: `trials/<trial-name>/agent/claude-code.txt`
