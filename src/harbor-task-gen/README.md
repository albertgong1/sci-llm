# Harbor Task Generation (Harbor)

> Domain-agnostic task generation + Harbor runner. Use any dataset that follows the
> `refno` + `properties` list schema and any template under your workspace.
> Harbor tasks can be run with `gemini-cli`, `claude-code`, `openhands`, etc. Make
> sure the repo root `.env` has your API keys.

Workspace assets (templates, PDFs, outputs) live under
`examples/harbor-workspace/`. All commands below default to that workspace; pass
`--workspace /path/to/workspace` to override. The workspace README includes a
template authoring guide.
To clean generated outputs, use
`examples/harbor-workspace/clean_harbor_workspace.py`.

## Setup (uv + Harbor)

1) Install deps
```bash
uv sync
```

1) Add keys (export or put in `.env` at repo root):
```bash
export GOOGLE_API_KEY="..."
export ANTHROPIC_API_KEY="..."
export MODAL_TOKEN_ID="ak-..."
export MODAL_TOKEN_SECRET="as-..."
```
Claude Code also supports `CLAUDE_CODE_OAUTH_TOKEN=...`.

PDF corpus lives at `examples/harbor-workspace/data/Paper_DB/<refno>.pdf` (15 PDFs).

## Domains and Datasets
The domain is defined by the dataset + template. The dataset must contain rows with:
- `refno`: PDF id (matches `<refno>.pdf`)
- `properties`: list of dicts with `property_name`, `material_or_system`, `value_string`

For example:
- Superconductor extraction uses `kilian-group/supercon-mini-v2`
- Biosurfactants extraction uses a different template and dataset
- Precedent search uses a different template and scoring logic

## Build Harbor Tasks (example: supercon-mini-v2)

Templates live in `examples/harbor-workspace/`:
- `ground-template` (default free-form)
- `ground-template-easy` (the base template, but prompted with each property it needs to look for, better for testing if the LLM is following instructions)

Build tasks for all properties with the default template (no task alias needed):
```bash
uv run python src/harbor-task-gen/prepare_harbor_tasks.py \
  --gt-hf-repo kilian-group/supercon-mini-v2 --gt-hf-split full \
  --write-job-config --force
```
Outputs go to `examples/harbor-workspace/out/harbor/supercon-mini-v2/ground-template/`
with a `job.yaml`.

If you want to filter to a task alias (e.g., only Tc rows), pass `--task tc`. This
changes the output path to `examples/harbor-workspace/out/harbor/supercon-mini-v2/tc/<template>/`.

Easier template:
```bash
uv run python src/harbor-task-gen/prepare_harbor_tasks.py \
  --template ground-template-easy --write-job-config --force
```

Build a single paper while iterating:
```bash
uv run python src/harbor-task-gen/prepare_harbor_tasks.py \
  --refno PR05001178 --write-job-config --force
```

Build tasks without scoring (PDF-only, no dataset):
```bash
uv run python src/harbor-task-gen/prepare_harbor_tasks.py \
  --no-score --write-job-config --force
```
This writes tasks under `out/harbor/no-score/<template>/`. Run with verification
disabled (or use the generated job.yaml, which disables verification):
```bash
uv run python src/harbor-task-gen/run_harbor.py jobs start \
  -c out/harbor/no-score/ground-template/job.yaml \
  -a gemini-cli -m gemini/gemini-2.5-flash \
  --disable-verification
```

## Publish Tasks to HF (optional)

Upload a tasks directory to HF and write a Harbor `registry.json`:
```bash
uv run python src/harbor-task-gen/prepare_harbor_tasks.py \
  --template ground-template --write-job-config --force \
  --upload-hf --hf-repo-id YOUR_ORG/supercon-harbor-tasks
```
The script prints the dataset name and registry URL you will use to run from HF.
For public repos, HF auth is not required. For private repos, set `HF_TOKEN`
in `.env` (or `HUGGINGFACE_HUB_TOKEN` / `HF_API_TOKEN`).

### Template placeholders
`prepare_harbor_tasks.py` does safe substitution; you can use:
`{refno}`, `{task}`, `{task_name}`, `{task_id}`, `{pdf_path}`, `{meta_path}`,
`{predictions_path}`, `{paper_at_command}`, `{gemini_at_commands}`,
`{claude_file_examples}`, `{question_blocks}`, `{questions_json}`,
`{task_meta_json}`.

## Run Harbor

Always use `run_harbor.py` so `.env` is loaded and keys are mapped.
Templates start with `@paper.pdf`, so `gemini-cli` auto-attaches the PDF.
Parallelism for jobs is set with `--n-concurrent` (or edit `job.yaml`).
Results are written locally under `examples/harbor-workspace/jobs/` or
`examples/harbor-workspace/trials/`, even when using Modal. Paths shown below are
relative to the workspace root.
If you pass `-m gemini-2.5-flash`, the wrapper normalizes it to
`gemini/gemini-2.5-flash` automatically.

When you use Modal (`--env modal` or `--modal`), the agent runs inside a
remote Modal sandbox. Your machine just builds the task bundle, uploads it, and
streams logs/results back. Scaling is limited by Modal quotas and budget, plus network
latency.

### Run Harbor on Modal (native)
Harbor supports Modal as an environment backend. Ensure `MODAL_TOKEN_ID` and
`MODAL_TOKEN_SECRET` are in `.env` (or run `modal token set`).
Example: `modal token set --token-id "$MODAL_TOKEN_ID" --token-secret "$MODAL_TOKEN_SECRET"`.
Use `--env modal` or the wrapper flag `--modal` (which also enables cleanup defaults).
Modal environments are deleted after each run by default (`--delete`); pass
`--no-delete` only when you want to keep a sandbox for debugging.
Modal setup defaults:
- Agent setup timeout is increased to 900s for Modal runs. Override with
  `--agent-setup-timeout-sec 1200` or `HARBOR_AGENT_SETUP_TIMEOUT_SEC=1200`.
  Set the value to `0` to keep Harbor's default (360s).
- Log download is bounded to avoid hangs with
  `HARBOR_MODAL_LOG_DOWNLOAD_TIMEOUT_SEC` (default: 300). Set to `0` to disable.
```bash
uv run python src/harbor-task-gen/run_harbor.py jobs start \
  -c out/harbor/supercon-mini-v2/ground-template/job.yaml \
  -a gemini-cli -m gemini/gemini-2.5-flash \
  --modal --n-concurrent 4
```

Single-task trial on Modal:
```bash
uv run python src/harbor-task-gen/run_harbor.py trials start \
  -p out/harbor/supercon-mini-v2/ground-template/tasks/pr05001178 \
  -a gemini-cli -m gemini/gemini-2.5-flash \
  --modal
```
If Modal hits resource limits, lower resources with:
`--override-storage-mb 1024` and/or `--override-memory-mb 1024`, or rebuild tasks after
adjusting `[environment]` in `task.toml.template`.

Quick single-task test (Modal + overrides):
```bash
uv run python src/harbor-task-gen/run_harbor.py trials start \
  -p out/harbor/supercon-mini-v2/ground-template/tasks/pr05001178 \
  -a gemini-cli -m gemini/gemini-2.5-flash \
  --modal --override-storage-mb 1024 --override-memory-mb 1024
```
Concurrency for jobs is controlled with `--n-concurrent` (example above).

### Run Harbor from HF tasks
Jobs from an HF task registry:
```bash
uv run python src/harbor-task-gen/run_harbor.py jobs start \
  --hf-tasks-repo YOUR_ORG/supercon-harbor-tasks \
  -a gemini-cli -m gemini/gemini-2.5-flash \
  --modal --n-concurrent 4
```
Single-task trial from HF:
```bash
uv run python src/harbor-task-gen/run_harbor.py trials start \
  --hf-tasks-repo YOUR_ORG/supercon-harbor-tasks \
  --hf-task pr05001178 \
  -a gemini-cli -m gemini/gemini-2.5-flash \
  --modal
```
If your registry file or tasks live under a different path, add
`--hf-registry-path path/to/registry.json` or `--hf-tasks-path path/to/tasks`.
If you used a custom dataset name/version in the registry, pass
`--hf-tasks-dataset NAME` and `--hf-tasks-version VERSION`.
If needed, you can also set `--hf-registry-url` explicitly (e.g.,
`https://huggingface.co/datasets/<repo>/raw/main/registry.json`).

### Export traces to HF (built-in Harbor)
```bash
uv run python src/harbor-task-gen/run_harbor.py jobs start \
  -c out/harbor/supercon-mini-v2/ground-template/job.yaml \
  -a gemini-cli -m gemini/gemini-2.5-flash \
  --modal --n-concurrent 4 \
  --export-traces --export-push --export-repo your-org/your-traces-dataset
```

### Full jobs (ground-template)
- Oracle:
```bash
uv run python src/harbor-task-gen/run_harbor.py jobs start \
  -c out/harbor/supercon-mini-v2/ground-template/job.yaml -a oracle
```

- Gemini CLI:
```bash
uv run python src/harbor-task-gen/run_harbor.py jobs start \
  -c out/harbor/supercon-mini-v2/ground-template/job.yaml \
  -a gemini-cli -m gemini/gemini-2.5-flash
```

- Claude Code:
```bash
uv run python src/harbor-task-gen/run_harbor.py jobs start \
  -c out/harbor/supercon-mini-v2/ground-template/job.yaml \
  -a claude-code
```

To use `ground-template-easy`, swap the template name in the path.

### Single-task trials
Pick a task id from:
```bash
ls out/harbor/supercon-mini-v2/ground-template/tasks
```
- Oracle:
```bash
uv run python src/harbor-task-gen/run_harbor.py trials start \
  -p out/harbor/supercon-mini-v2/ground-template/tasks/<task-id> -a oracle
```
- Gemini CLI:
```bash
uv run python src/harbor-task-gen/run_harbor.py trials start \
  -p out/harbor/supercon-mini-v2/ground-template/tasks/<task-id> \
  -a gemini-cli -m gemini/gemini-2.5-flash
```
- Claude Code:
```bash
uv run python src/harbor-task-gen/run_harbor.py trials start \
  -p out/harbor/supercon-mini-v2/ground-template/tasks/<task-id> \
  -a claude-code
```

## Where Results Go
- Jobs: `examples/harbor-workspace/jobs/<timestamp>/result.json`
- Trials: `examples/harbor-workspace/trials/<trial-name>/verifier/reward.txt`
- Debug: `examples/harbor-workspace/trials/<trial-name>/verifier/details.json`
- Agent logs: `examples/harbor-workspace/trials/<trial-name>/agent/{gemini-cli|claude-code}.txt`

## Publish Runs to Hugging Face

### 0) Export traces for a single trial (built-in Harbor)
```bash
uv run python -m harbor.cli.main traces export \
  -p examples/harbor-workspace/trials/<trial-name> \
  --push --repo your-org/your-traces-dataset
```

### 1) Compile a run bundle (lossless)
Latest run under `examples/harbor-workspace/jobs/` or `examples/harbor-workspace/trials/`
→ `examples/harbor-workspace/out/harbor-runs/<run-name>/`:
```bash
uv run python src/harbor-task-gen/run_harbor.py compile-run
```
Or specify a run:
```bash
uv run python src/harbor-task-gen/run_harbor.py compile-run \
  --run-dir jobs/<job-name>
```
You can also append `--compile-run` to a Harbor invocation to bundle the latest run.

### 2) Push to HF
```bash
export HF_TOKEN="..."

# Demo: build one task, run Gemini, bundle, upload
uv sync --extra dev

uv run python src/harbor-task-gen/prepare_harbor_tasks.py \
  --template ground-template --refno PR05001178 \
  --write-job-config --force

uv run python src/harbor-task-gen/run_harbor.py trials start \
  -p out/harbor/supercon-mini-v2/ground-template/tasks/pr05001178 \
  -a gemini-cli -m gemini/gemini-2.5-flash

WORKSPACE=examples/harbor-workspace
TRIAL_DIR=$(ls -td "$WORKSPACE"/trials/* | head -1)
BUNDLE=$(uv run python src/harbor-task-gen/run_harbor.py compile-run --run-dir "$TRIAL_DIR")

uv run python src/harbor-task-gen/run_harbor.py push-run-to-hf \
  --repo-id kilian-group/foolmetwice-testing \
  --bundle-dir "$BUNDLE" \
  --write-root-readme
```
By default each run uploads to `runs/<run-name>/` in the HF dataset repo.
To push immediately after a Harbor run, add:
`--push-run-to-hf --hf-runs-repo <org/repo> --hf-runs-write-root-readme`.

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

## Flag Reference

### prepare_harbor_tasks.py
- `--gt-hf-repo`: HF dataset repo id (required unless `--no-score`).
- `--gt-hf-split`: HF dataset split (required unless `--no-score`).
- `--gt-hf-revision`: HF dataset revision (default: `main`).
- `--no-score`: Build tasks from PDFs only; skip verifier/solution; use `--disable-verification` at run time.
- `--workspace`: Workspace root (default: `examples/harbor-workspace`).
- `--template`: Template folder under workspace (default: `ground-template`).
- `--task`: Task alias to filter properties (e.g., `tc`); affects output path.
- `--pdf-dir`: Directory containing `<refno>.pdf` (default: `<workspace>/data/Paper_DB`).
- `--output-dir`: Output root (default: `<workspace>/out/harbor/<dataset>`).
- `--limit`: Cap number of tasks built.
- `--refno`: Only build specific refno(s) (repeatable).
- `--force`: Delete/rebuild existing output directory.
- `--write-job-config`: Emit `job.yaml` pointing at generated tasks.
- `--upload-hf`: Upload tasks to HF and write registry.json.
- `--hf-repo-id`: HF repo id (e.g., `ORG/supercon-harbor-tasks`).
- `--hf-repo-type`: HF repo type (`dataset`, `model`, `space`; default: `dataset`).
- `--hf-path-in-repo`: Path inside the repo for tasks (default: `tasks`).
- `--hf-registry-path`: Registry JSON path inside repo (default: `registry.json`).
- `--hf-dataset-name`: Dataset name stored in registry.json (default: repo id).
- `--hf-dataset-version`: Dataset version stored in registry.json (default: `head`).
- `--hf-description`: Dataset description stored in registry.json.
- `--hf-private`: Create repo as private if it does not exist.
- `--hf-public`: Create repo as public if it does not exist.
- `--hf-create/--no-hf-create`: Toggle repo creation (default: create).
- `--hf-no-input`: Disable interactive HF prompts.
- `--hf-tasks-root`: Override tasks root to upload (default: generated tasks dir).

### run_harbor.py wrapper flags (pass-through to Harbor otherwise)
- `--workspace`: Workspace root (default: `examples/harbor-workspace`).
- `--modal`: Shortcut for `--env modal` (jobs) / `--environment-type modal` (trials) and Modal defaults.
- `--agent-setup-timeout-sec`: Override Harbor agent setup timeout (seconds).
- `--hf-tasks-repo`: HF tasks repo id (enables HF task rewrite).
- `--hf-repo-type`: HF repo type for tasks (`dataset`, `model`, `space`).
- `--hf-tasks-dataset`: Dataset name in registry.json (default: repo id).
- `--hf-tasks-version`: Dataset version in registry.json (default: `head`).
- `--hf-registry-url`: Explicit registry.json URL (raw HF URL).
- `--hf-registry-path`: Path to registry.json inside repo (default: `registry.json`).
- `--hf-task`: Task id to run from HF (trials only).
- `--hf-task-path`: Full path to a task inside the repo (trials only).
- `--hf-tasks-path`: Root tasks folder inside repo (default: `tasks`).
- `--hf-task-commit`: Git commit/revision for HF tasks (optional).
- `--compile-run`: Compile latest run after Harbor finishes.
- `--compile-out-dir`: Output root for bundles (default: `<workspace>/out/harbor-runs`).
- `--compile-name`: Override bundle name (default: run dir name).
- `--compile-force`: Overwrite bundle directory if it exists.
- `--push-run-to-hf`: Upload compiled bundle to HF after Harbor finishes.
- `--hf-runs-repo`: HF repo id for run bundles (required with `--push-run-to-hf`).
- `--hf-runs-repo-type`: Repo type for bundles (`dataset`, `model`, `space`).
- `--hf-runs-path-in-repo`: Destination path inside the repo (default: `runs/<run-name>`).
- `--hf-runs-private`: Create repo as private if missing.
- `--hf-runs-public`: Create repo as public if missing.
- `--hf-runs-write-root-readme`: Create README.md at repo root if missing.
- `--hf-runs-force-root-readme`: Overwrite repo root README.md.

### run_harbor.py utilities
`compile-run` (standalone command):
- `--workspace`: Workspace root.
- `--run-dir`: Run directory (jobs/<name> or trials/<name>); default: latest.
- `--out-dir`: Bundle output root (default: `<workspace>/out/harbor-runs`).
- `--name`: Override bundle folder name.
- `--force`: Overwrite bundle directory if it exists.

`push-run-to-hf` (standalone command):
- `--workspace`: Workspace root.
- `--repo-id`: HF repo id (required).
- `--repo-type`: Repo type (`dataset`, `model`, `space`).
- `--run-dir`: Run directory (default: latest).
- `--bundle-dir`: Already compiled bundle; skips compile.
- `--out-dir`: Bundle output root (default: `<workspace>/out/harbor-runs`).
- `--path-in-repo`: Destination path inside repo (default: `runs/<run-name>`).
- `--private`: Create repo as private if missing.
- `--public`: Create repo as public if missing.
- `--write-root-readme`: Create README.md at repo root if missing.
- `--force-root-readme`: Overwrite repo root README.md.
- `--force`: Overwrite local bundle directory if it exists.
