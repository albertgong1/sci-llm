# SuperCon Property Extraction (Harbor)

> Harbor tasks can be run with `gemini-cli` or `claude-code` agents. Make sure the repo
> root `.env` has your API keys.

Workspace assets (templates, PDFs, outputs) live under `examples/harbor-workspace/`.
- All commands default to that workspace; override with `--workspace /path/to/workspace`
- Template authoring guide: `examples/harbor-workspace/README.md`
- Clean outputs: `examples/harbor-workspace/clean_harbor_workspace.py`

## Setup (uv + Harbor)

1) Install deps
```bash
uv sync
```

2) Add keys (export or put in `.env` at repo root):
```bash
export GOOGLE_API_KEY="..."
export ANTHROPIC_API_KEY="..."
export OPENROUTER_API_KEY="..."
export MODAL_TOKEN_ID="ak-..."
export MODAL_TOKEN_SECRET="as-..."
```
Claude Code also supports `CLAUDE_CODE_OAUTH_TOKEN=...`.

PDF corpus lives at `examples/harbor-workspace/data/Paper_DB/<refno>.pdf` (15 PDFs).

## Build Harbor Tasks (supercon-mini-v2)

Templates live in `examples/harbor-workspace/`:
- `ground-template` (default free-form)
- Add custom templates by copying `ground-template/` and passing `--template <name>`

Build tasks for all properties with the default template (no task alias needed):
```bash
uv run python src/harbor-task-gen/prepare_harbor_tasks.py \
  --write-job-config --force
```
Outputs go to `examples/harbor-workspace/out/harbor/supercon-mini-v2/ground-template/`
with a `job.yaml`.

If you want to filter to a task alias (e.g., only Tc rows), pass `--task tc`. This
changes the output path to `examples/harbor-workspace/out/harbor/supercon-mini-v2/tc/<template>/`.

Build a single paper while iterating:
```bash
uv run python src/harbor-task-gen/prepare_harbor_tasks.py \
  --refno PR05001178 --write-job-config --force
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
Results are written locally under `examples/harbor-workspace/jobs/` or
`examples/harbor-workspace/trials/`, even when using Modal. Paths shown below are
relative to the workspace root.
If you pass `-m gemini-2.5-flash`, the wrapper normalizes it to
`gemini/gemini-2.5-flash` automatically.

### Local runs (job + trial)
Full jobs (ground-template):
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

Single-task trials:
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

### Agent examples (native Harbor)
Use `--agent-kwarg key=value` (repeatable) to pass agent-specific options. The available
kwargs are defined on each agent's `__init__` in Harbor.

- Terminus 2 (multi-turn, capped):
```bash
uv run python src/harbor-task-gen/run_harbor.py jobs start \
  -c out/harbor/supercon-mini-v2/ground-template/job.yaml \
  -a terminus-2 -m openai/gpt-4o-mini \
  --agent-kwarg max_turns=1
```

- Terminus 2 via OpenRouter (Qwen Coder 3+):
```bash
uv run python src/harbor-task-gen/run_harbor.py jobs start \
  -c out/harbor/supercon-mini-v2/ground-template/job.yaml \
  -a terminus-2 -m openrouter/qwen/qwen3-coder-32b \
  --agent-kwarg max_turns=1
```
Requires `OPENROUTER_API_KEY` in `.env`. Swap the model slug to any OpenRouter
Qwen Coder 3+ listing you want to use.

- OpenHands (disable tool calls):
```bash
uv run python src/harbor-task-gen/run_harbor.py jobs start \
  -c out/harbor/supercon-mini-v2/ground-template/job.yaml \
  -a openhands -m openai/gpt-4o-mini \
  --agent-kwarg disable_tool_calls=true
```

- SWE-agent (installed agent wrapper):
```bash
uv run python src/harbor-task-gen/run_harbor.py jobs start \
  -c out/harbor/supercon-mini-v2/ground-template/job.yaml \
  -a swe-agent -m openai/gpt-4o-mini
```

Other built-ins include: `oracle`, `nop`, `claude-code`, `cline-cli`, `aider`,
`codex`, `cursor-cli`, `gemini-cli`, `goose`, `mini-swe-agent`, `opencode`, `qwen-coder`.

### Modal backend (native)
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
  -a gemini-cli -m gemini-2.5-flash \
  --modal
```
If Modal hits resource limits, lower resources with:
`--override-storage-mb 1024` and/or `--override-memory-mb 1024`, or rebuild tasks after
adjusting `[environment]` in `task.toml.template`.

Quick single-task test (Modal + overrides):
```bash
uv run python src/harbor-task-gen/run_harbor.py trials start \
  -p out/harbor/supercon-mini-v2/ground-template/tasks/pr05001178 \
  -a gemini-cli -m gemini-2.5-flash \
  --modal --override-storage-mb 1024 --override-memory-mb 1024
```
Concurrency for jobs is controlled with `--n-concurrent` (example above).

### Run Harbor from an HF task registry
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
