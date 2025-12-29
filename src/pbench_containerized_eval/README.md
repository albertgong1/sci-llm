# SuperCon Property Extraction (Harbor)

> Harbor tasks can be run with `gemini-cli` or `claude-code` agents. Make sure the repo
> root `.env` has your API keys.

See `src/pbench_containerized_eval/knowledgebase.md` for deeper context (task layout,
contracts, debugging playbook).

## Setup (uv + Harbor)

1) Install deps (Harbor is in the `dev` extra):
```bash
uv sync --extra dev
```

2) Add keys (export or put in `.env` at repo root):
```bash
export GOOGLE_API_KEY="..."
export ANTHROPIC_API_KEY="..."
```
Claude Code also supports `CLAUDE_CODE_OAUTH_TOKEN=...`.

PDF corpus lives at `src/pbench_containerized_eval/data/Paper_DB/<refno>.pdf` (15 PDFs).

## Build Harbor Tasks (supercon-mini-v2)

Templates live in `src/pbench_containerized_eval/`:
- `ground-template` (default free-form)
- `prompted-template` (question-style)

Build tasks for all properties with the default template (no task alias needed):
```bash
uv run python src/pbench_containerized_eval/prepare_harbor_tasks.py \
  --write-job-config --force
```
Outputs go to `out/harbor/supercon-mini-v2/ground-template/` with a `job.yaml`.

If you want to filter to a task alias (e.g., only Tc rows), pass `--task tc`. This
changes the output path to `out/harbor/supercon-mini-v2/tc/<template>/`.

Prompted template:
```bash
uv run python src/pbench_containerized_eval/prepare_harbor_tasks.py \
  --template prompted-template --write-job-config --force
```

Build a single paper while iterating:
```bash
uv run python src/pbench_containerized_eval/prepare_harbor_tasks.py \
  --refno PR05001178 --write-job-config --force
```

### Template placeholders
`prepare_harbor_tasks.py` does safe substitution; you can use:
`{refno}`, `{task}`, `{task_name}`, `{task_id}`, `{pdf_path}`, `{meta_path}`,
`{predictions_path}`, `{paper_at_command}`, `{gemini_at_commands}`,
`{claude_file_examples}`, `{question_blocks}`, `{questions_json}`,
`{task_meta_json}`.

## Run Harbor Locally

Always use `run_harbor.py` so `.env` is loaded and keys are mapped.
Templates start with `@paper.pdf`, so `gemini-cli` auto-attaches the PDF.
Parallelism for jobs is set with `--n-concurrent` (or edit `job.yaml`).
Results are written locally under `jobs/` or `trials/`, even when using Daytona.

### Run Harbor on Daytona (native)
Harbor supports Daytona as an environment backend. Ensure `DAYTONA_API_KEY` is in `.env`.
Use `--env daytona` (or the convenience wrapper `run_harbor_daytona.py`, which appends it).
`run_harbor.py` also enables a Daytona network allowlist patch; by default it sets
`PBENCH_DAYTONA_NETWORK_ALLOW_LIST=0.0.0.0/0` unless you override it.
```bash
uv run python src/pbench_containerized_eval/run_harbor.py jobs start \
  -c out/harbor/supercon-mini-v2/ground-template/job.yaml \
  -a gemini-cli -m gemini/gemini-2.5-flash \
  --env daytona --n-concurrent 4
```
or
```bash
uv run python src/pbench_containerized_eval/run_harbor_daytona.py \
  jobs start -c out/harbor/supercon-mini-v2/ground-template/job.yaml \
  -a gemini-cli -m gemini/gemini-2.5-flash \
  --n-concurrent 4
```

Single-task trial on Daytona:
```bash
uv run python src/pbench_containerized_eval/run_harbor.py trials start \
  -p out/harbor/supercon-mini-v2/ground-template/tasks/pr05001178 \
  -a gemini-cli -m gemini/gemini-2.5-flash \
  --env daytona
```
If Daytona hits disk or memory limits, lower resources with:
`--override-storage-mb 1024` and/or `--override-memory-mb 1024`, or rebuild tasks after
adjusting `[environment]` in `task.toml.template`.
If Gemini cannot reach the API (connection reset), try a custom allowlist:
```bash
PBENCH_DAYTONA_NETWORK_ALLOW_LIST="0.0.0.0/0" \
uv run python src/pbench_containerized_eval/run_harbor.py trials start \
  -p out/harbor/supercon-mini-v2/ground-template/tasks/pr05515300 \
  -a gemini-cli -m gemini/gemini-2.5-flash \
  --env daytona --override-memory-mb 1024 --override-storage-mb 1024
```
`run_harbor.py` defaults to a host-side Gemini fallback on Daytona. To force the
containerized Gemini CLI instead, set `PBENCH_GEMINI_HOST_AUTO=0`.

### Export traces to HF (built-in Harbor)
```bash
uv run python src/pbench_containerized_eval/run_harbor.py jobs start \
  -c out/harbor/supercon-mini-v2/ground-template/job.yaml \
  -a gemini-cli -m gemini/gemini-2.5-flash \
  --env daytona --n-concurrent 4 \
  --export-traces --export-push --export-repo your-org/your-traces-dataset
```

### Full jobs (ground-template)
- Oracle:
```bash
uv run python src/pbench_containerized_eval/run_harbor.py jobs start \
  -c out/harbor/supercon-mini-v2/ground-template/job.yaml -a oracle
```

- Gemini CLI:
```bash
uv run python src/pbench_containerized_eval/run_harbor.py jobs start \
  -c out/harbor/supercon-mini-v2/ground-template/job.yaml \
  -a gemini-cli -m gemini/gemini-2.5-flash
```

- Claude Code:
```bash
uv run python src/pbench_containerized_eval/run_harbor.py jobs start \
  -c out/harbor/supercon-mini-v2/ground-template/job.yaml \
  -a claude-code
```

To use `prompted-template`, swap the template name in the path.

### Single-task trials
Pick a task id from:
```bash
ls out/harbor/supercon-mini-v2/ground-template/tasks
```
- Oracle:
```bash
uv run python src/pbench_containerized_eval/run_harbor.py trials start \
  -p out/harbor/supercon-mini-v2/ground-template/tasks/<task-id> -a oracle
```
- Gemini CLI:
```bash
uv run python src/pbench_containerized_eval/run_harbor.py trials start \
  -p out/harbor/supercon-mini-v2/ground-template/tasks/<task-id> \
  -a gemini-cli -m gemini/gemini-2.5-flash
```
- Claude Code:
```bash
uv run python src/pbench_containerized_eval/run_harbor.py trials start \
  -p out/harbor/supercon-mini-v2/ground-template/tasks/<task-id> \
  -a claude-code
```

## Smoke Test (build + one trial)
Build one paper and run a single trial per agent (skips agents without keys):
```bash
uv run python src/pbench_containerized_eval/smoke_harbor.py --refno PR05001178
```
To smoke-test the prompted template:
```bash
uv run python src/pbench_containerized_eval/smoke_harbor.py --refno PR05001178 \
  --templates prompted-template
```

## Where Results Go
- Jobs: `jobs/<timestamp>/result.json`
- Trials: `trials/<trial-name>/verifier/reward.txt`
- Debug: `trials/<trial-name>/verifier/details.json`
- Agent logs: `trials/<trial-name>/agent/{gemini-cli|claude-code}.txt`
- Gemini CLI diagnostics: `trials/<trial-name>/agent/gemini-cli-network.txt` and `trials/<trial-name>/agent/gemini-cli.error.json`

## Publish Runs to Hugging Face

### 0) Export traces for a single trial (built-in Harbor)
```bash
uv run python -m harbor.cli.main traces export \
  -p trials/<trial-name> --push --repo your-org/your-traces-dataset
```

### 1) Compile a run bundle (lossless)
Latest run under `jobs/` or `trials/` → `out/harbor-runs/<run-name>/`:
```bash
uv run python src/pbench_containerized_eval/compile_harbor_run.py
```
Or specify a run:
```bash
uv run python src/pbench_containerized_eval/compile_harbor_run.py \
  --run-dir jobs/<job-name>
```

### 2) Push to HF
```bash
export HF_TOKEN="..."

# Demo: build one task, run Gemini, bundle, upload
uv sync --extra dev

uv run python src/pbench_containerized_eval/prepare_harbor_tasks.py \
  --template ground-template --refno PR05001178 \
  --write-job-config --force

uv run python src/pbench_containerized_eval/run_harbor.py trials start \
  -p out/harbor/supercon-mini-v2/ground-template/tasks/pr05001178 \
  -a gemini-cli -m gemini/gemini-2.5-flash

TRIAL_DIR=$(ls -td trials/* | head -1)
BUNDLE=$(uv run python src/pbench_containerized_eval/compile_harbor_run.py --run-dir "$TRIAL_DIR")

uv run python src/pbench_containerized_eval/push_harbor_run_to_hf.py \
  --repo-id kilian-group/foolmetwice-testing \
  --bundle-dir "$BUNDLE" \
  --write-root-readme
```
By default each run uploads to `runs/<run-name>/` in the HF dataset repo.

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
The verifier entrypoint (`tests/test.sh`) copies `/app/output/` into
`/logs/verifier/app_output/` **even when the verifier fails**, so agent-written artifacts
survive Harbor's container cleanup. Rebuild tasks with `prepare_harbor_tasks.py --force`
to apply template changes.
