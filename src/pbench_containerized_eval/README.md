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

## Publish Runs to Hugging Face

This directory includes two helper scripts to bundle a Harbor run directory and upload it
to a Hugging Face dataset repo without dropping any files.

### 1) Compile a run bundle (lossless)

Compiles the latest run under `jobs/` or `trials/` into `out/harbor-runs/<run-name>/`:

```bash
uv run python examples/containerized-extraction/compile_harbor_run.py
```

Or specify a run directory explicitly:

```bash
uv run python examples/containerized-extraction/compile_harbor_run.py \
  --run-dir jobs/<job-name>
```

The bundle contains:
- `harbor/<run-name>/...` (full copied Harbor run directory)
- `index/trials.jsonl` (one line per trial; easy to query)
- `index/files.jsonl` (sha256 manifest of every uploaded file)

### 2) Push to HF

Authenticate with either `hf auth login` or by exporting a token:

```bash
export HF_TOKEN="..."
```

Then upload (this will also compile if you pass `--run-dir`):

```bash
# Demo: run one paper with Gemini 2.5 Flash, then upload the full artifacts bundle.
uv sync --extra dev

uv run python examples/containerized-extraction/prepare_harbor_tasks.py \
  --task tc --template ground-template --refno PR05001178 \
  --write-job-config --force

uv run python examples/containerized-extraction/run_harbor.py trials start \
  -p out/harbor/supercon-mini/tc/ground-template/tasks/pr05001178--tc \
  -a gemini-cli -m gemini/gemini-2.5-flash

TRIAL_DIR=$(ls -td trials/* | head -1)
BUNDLE=$(uv run python examples/containerized-extraction/compile_harbor_run.py --run-dir "$TRIAL_DIR")

uv run python examples/containerized-extraction/push_harbor_run_to_hf.py \
  --repo-id kilian-group/foolmetwice-testing \
  --bundle-dir "$BUNDLE" \
  --write-root-readme
```

By default, each run is uploaded under `runs/<run-name>/` in the HF repo.

### 3) Query on the other end (example)

```python
from huggingface_hub import snapshot_download
from pathlib import Path
import json

repo_id = "kilian-group/foolmetwice-testing"
local = Path(snapshot_download(repo_id=repo_id, repo_type="dataset"))

run = local / "runs" / "<run-name>"
trials = [json.loads(line) for line in (run / "index" / "trials.jsonl").read_text().splitlines()]

failures = [t for t in trials if (t.get("reward") or 0.0) < 1.0 or t.get("exception_type")]
print("n_trials:", len(trials))
print("n_failures:", len(failures))

if failures:
    t = failures[0]
    print("trial:", t["trial_name"])
    print("agent logs:", t["paths"]["agent_logs"])
    print("verifier details:", t["paths"]["verifier_details"])
```

### Notes on preserving outputs

The template verifier entrypoint (`tests/test.sh`) copies `/app/output/` into the persisted
`/logs/verifier/app_output/` directory **even when the verifier fails**, so agent-written
artifacts (like `predictions.json`) survive Harbor's container cleanup. Rebuild tasks with
`prepare_harbor_tasks.py --force` to apply template changes to generated tasks.
