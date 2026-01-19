# Harbor Task Generator

This repo turns (PDF + labeled rows) into Harbor tasks and runs agents against them.
Use the helper scripts below; they load `.env`, normalize Gemini model names, and add
HF/Modal conveniences.

Workspace assets (templates, PDFs, outputs) live under `examples/harbor-workspace/` by
default. Use `--workspace /path/to/workspace` to point at a different task family.

## Base commands

Build tasks:
```bash
uv run python src/harbor-task-gen/prepare_harbor_tasks.py [flags]
```

Run Harbor (jobs/trials):
```bash
uv run python src/harbor-task-gen/run_harbor.py [--workspace PATH] <harbor args> [wrapper flags]
```

Compile or upload runs directly:
```bash
uv run python src/harbor-task-gen/run_harbor.py compile-run [flags]
uv run python src/harbor-task-gen/run_harbor.py push-run-to-hf [flags]
```

## Quick start (default SuperCon workspace)

```bash
uv run python src/harbor-task-gen/prepare_harbor_tasks.py \
  --write-job-config --force

uv run python src/harbor-task-gen/run_harbor.py jobs start \
  -c out/harbor/supercon-mini-v2/ground-template/job.yaml \
  -a gemini-cli -m gemini/gemini-2.5-flash
```

## Flag reference: prepare_harbor_tasks.py

Dataset selection:
- `--domain NAME` - dataset domain key (default: `supercon`). Known domains live in
  `src/pbench/datasets.yaml` (current: `supercon`, `precedent-search`, `biosurfactants`).
  The domain mapping supplies the HF dataset name, revision, and split.

Workspace + templates:
- `--workspace PATH` - workspace root (default: `examples/harbor-workspace`).
- `--template NAME` - template folder under the workspace (default: `ground-template`).
  Each template must include `instruction.md.template`, `task.toml.template`,
  `environment/Dockerfile`, and `tests/`.
- `--pdf-dir PATH` - directory containing `<refno>.pdf` files
  (default: `<workspace>/data/Paper_DB`).

Task filtering:
- `--task NAME` - task alias; filters properties using the internal map and inserts
  `<task>/` in the output path. Only `tc` is built-in today.
- `--refno REF` - only build specific refno(s) (repeatable).
- `--limit N` - cap number of tasks (papers) generated.

Output control:
- `--output-dir PATH` - output root (default: `<workspace>/out/harbor/<dataset>`).
- `--force` - overwrite the output directory if it already exists.
- `--write-job-config` - emit a `job.yaml` pointing at the generated tasks.

HF upload (optional):
- `--upload-hf` - upload tasks + registry after build.
- `--hf-repo-id ID` - HF repo id (e.g., `ORG/harbor-tasks`).
- `--hf-repo-type TYPE` - `dataset` (default), `model`, or `space`.
- `--hf-path-in-repo PATH` - tasks root inside the repo (default: `tasks`).
- `--hf-registry-path PATH` - registry file path (default: `registry.json`).
- `--hf-dataset-name NAME` - registry dataset name (default: repo id).
- `--hf-dataset-version VER` - registry dataset version (default: `head`).
- `--hf-description TEXT` - registry description.
- `--hf-private` / `--hf-public` - visibility when creating the repo.
- `--hf-create` / `--hf-no-create` - create repo if missing (default: create).
- `--hf-no-input` - disable interactive prompts during upload.
- `--hf-tasks-root PATH` - override the tasks root to upload.

Notes:
- The HF dataset must have rows shaped like `{refno, properties}` where each property
  includes `material_or_system`, `property_name`, and `value_string`.
- `rubric.csv` is read from `<workspace>/rubric.csv` to supply rubrics + definitions.

## Flag reference: run_harbor.py wrapper

Core wrapper flags:
- `--workspace PATH` - workspace root (default: `examples/harbor-workspace`).
- `--modal` - force `--env modal` and default `--delete`; enables Modal safety patches.
- `--agent-setup-timeout-sec N` - override agent setup timeout (Modal only).

Post-run hooks (append to a normal run):
- `--compile-run` - compile the latest run after execution.
- `--compile-out-dir PATH` - bundle output dir (default: `<workspace>/out/harbor-runs`).
- `--compile-name NAME` - override bundle folder name.
- `--compile-force` - overwrite existing bundle dir.
- `--push-run-to-hf` - compile + upload after run (requires `--hf-runs-repo`).
- `--hf-runs-repo ID` - HF repo id for run bundles.
- `--hf-runs-repo-type TYPE` - `dataset` (default), `model`, `space`.
- `--hf-runs-path-in-repo PATH` - upload path (default: `runs/<run-name>`).
- `--hf-runs-private` / `--hf-runs-public` - visibility when creating repo.
- `--hf-runs-write-root-readme` - add a root README if missing.
- `--hf-runs-force-root-readme` - overwrite root README.

HF task registry flags (jobs/trials):
- `--hf-tasks-repo ID` - HF repo id for tasks (required to use HF task flags).
- `--hf-repo-type TYPE` - `dataset` (default), `model`, `space`.
- `--hf-tasks-dataset NAME` - registry dataset name (default: repo id).
- `--hf-tasks-version VER` - registry dataset version (default: `head`).
- `--hf-registry-url URL` - direct URL to `registry.json`.
- `--hf-registry-path PATH` - path to registry inside repo (default: `registry.json`).
- `--hf-tasks-path PATH` - tasks root inside repo (default: `tasks`).
- `--hf-task ID` - task id for `trials start`.
- `--hf-task-path PATH` - explicit task path in repo for trials.
- `--hf-task-commit REV` - git commit/revision when cloning tasks.

Wrapper behavior:
- Loads repo root `.env` and maps keys:
  - `GOOGLE_API_KEY` <-> `GEMINI_API_KEY`
  - `CLAUDE_API_KEY` -> `ANTHROPIC_API_KEY`
  - `CLAUDE_CODE_TOKEN` / `CLAUDE_CODE_API_TOKEN` -> `CLAUDE_CODE_OAUTH_TOKEN`
- Normalizes Gemini model names: `gemini-2.5-flash` -> `gemini/gemini-2.5-flash`.
- Rewrites `--override-storage` to `--override-storage-mb`.
- Rewrites `--env` to `--environment-type` for trials.
- If `git-lfs` is missing, downloads an HF snapshot and runs from local files.

All other Harbor CLI flags pass through unchanged. Run `uv run harbor jobs start --help`
or `uv run harbor trials start --help` for native options.

## Flag reference: run_harbor.py subcommands

`compile-run`:
- `--workspace PATH` - workspace root (default: `examples/harbor-workspace`).
- `--run-dir PATH` - run dir (jobs/<name> or trials/<name>); default is latest.
- `--out-dir PATH` - bundle output dir (default: `<workspace>/out/harbor-runs`).
- `--name NAME` - override bundle folder name.
- `--force` - overwrite existing bundle dir.

`push-run-to-hf`:
- `--workspace PATH` - workspace root (default: `examples/harbor-workspace`).
- `--repo-id ID` - HF repo id (required).
- `--repo-type TYPE` - `dataset` (default), `model`, `space`.
- `--run-dir PATH` - run dir to compile (default: latest).
- `--bundle-dir PATH` - pre-compiled bundle dir (skips compilation).
- `--out-dir PATH` - bundle output dir (default: `<workspace>/out/harbor-runs`).
- `--path-in-repo PATH` - upload destination (default: `runs/<run-name>`; use '' for root).
- `--private` / `--public` - visibility when creating repo.
- `--write-root-readme` - create root README if missing.
- `--force-root-readme` - overwrite root README.
- `--force` - overwrite local bundle dir if it exists.

## Example gallery

Notes:
- Replace `<task-id>` or `<task_name>` with the actual folder name in `out/harbor/.../tasks/`.
- Add `--modal` for Modal runs (requires `MODAL_TOKEN_ID` + `MODAL_TOKEN_SECRET`).
- Agent/model names are Harbor-native. If your install uses provider prefixes, keep them
  (e.g., `gemini/...`, `openrouter/...`, `openai/...`, `anthropic/...`).

### SuperCon (default workspace)

Build all tasks:
```bash
uv run python src/harbor-task-gen/prepare_harbor_tasks.py \
  --write-job-config --force
```

Build Tc-only tasks:
```bash
uv run python src/harbor-task-gen/prepare_harbor_tasks.py \
  --task tc --write-job-config --force
```

Run job locally (Gemini):
```bash
uv run python src/harbor-task-gen/run_harbor.py jobs start \
  -c out/harbor/supercon-mini-v2/ground-template/job.yaml \
  -a gemini-cli -m gemini/gemini-2.5-flash
```

Run job on Modal with higher concurrency:
```bash
uv run python src/harbor-task-gen/run_harbor.py jobs start \
  -c out/harbor/supercon-mini-v2/ground-template/job.yaml \
  -a gemini-cli -m gemini/gemini-2.5-flash \
  --modal --n-concurrent 8 --override-memory-mb 2048 --override-storage-mb 2048
```

Run a single task (oracle sanity check):
```bash
uv run python src/harbor-task-gen/run_harbor.py trials start \
  -p out/harbor/supercon-mini-v2/ground-template/tasks/<task-id> \
  -a oracle
```

Run a single task (Gemini):
```bash
uv run python src/harbor-task-gen/run_harbor.py trials start \
  -p out/harbor/supercon-mini-v2/ground-template/tasks/<task-id> \
  -a gemini-cli -m gemini/gemini-2.5-flash
```

Compile + upload the latest run:
```bash
uv run python src/harbor-task-gen/run_harbor.py jobs start \
  -c out/harbor/supercon-mini-v2/ground-template/job.yaml \
  -a gemini-cli -m gemini/gemini-2.5-flash \
  --compile-run --push-run-to-hf \
  --hf-runs-repo YOUR_ORG/sci-llm-harbor-runs \
  --hf-runs-write-root-readme
```

Run from HF task registry (jobs):
```bash
uv run python src/harbor-task-gen/run_harbor.py jobs start \
  --hf-tasks-repo YOUR_ORG/supercon-harbor-tasks \
  -a gemini-cli -m gemini/gemini-2.5-flash --modal
```

Run one HF task (trials):
```bash
uv run python src/harbor-task-gen/run_harbor.py trials start \
  --hf-tasks-repo YOUR_ORG/supercon-harbor-tasks \
  --hf-task pr05001178 \
  -a oracle
```

### Tc precedent search (workspace: `examples/harbor-workspace`)

Build precedent-search tasks:
```bash
uv run python examples/tc-precedent-search/prepare_precedent_tasks.py \
  --force --write-job-config
```

Run the full job (Gemini on Modal):
```bash
uv run python src/harbor-task-gen/run_harbor.py --workspace examples/harbor-workspace \
  jobs start -c out/harbor/precedent-search/tc-precedent-search/job.yaml \
  -a gemini-cli -m gemini/gemini-2.5-flash --modal
```

Run one task (oracle):
```bash
uv run python src/harbor-task-gen/run_harbor.py --workspace examples/harbor-workspace \
  trials start -p out/harbor/precedent-search/tc-precedent-search/tasks/<task_name> \
  -a oracle
```

Run one task (Claude Code):
```bash
uv run python src/harbor-task-gen/run_harbor.py --workspace examples/harbor-workspace \
  trials start -p out/harbor/precedent-search/tc-precedent-search/tasks/<task_name> \
  -a claude-code -m anthropic/claude-3-5-sonnet
```

Run one task (OpenRouter via Terminus 2):
```bash
uv run python src/harbor-task-gen/run_harbor.py --workspace examples/harbor-workspace \
  trials start -p out/harbor/precedent-search/tc-precedent-search/tasks/<task_name> \
  -a terminus-2 -m openrouter/qwen/qwen3-coder-plus
```

Run one task (Codex):
```bash
uv run python src/harbor-task-gen/run_harbor.py --workspace examples/harbor-workspace \
  trials start -p out/harbor/precedent-search/tc-precedent-search/tasks/<task_name> \
  -a codex -m openai/gpt-4o-mini
```

### Biosurfactants (workspace: `examples/biosurfactants-extraction`)

Build tasks:
```bash
uv run python src/harbor-task-gen/prepare_harbor_tasks.py \
  --domain biosurfactants \
  --workspace examples/biosurfactants-extraction \
  --template biosurfactants-extraction-template \
  --write-job-config --force
```

Run job (Gemini on Modal):
```bash
uv run python src/harbor-task-gen/run_harbor.py \
  --workspace examples/biosurfactants-extraction jobs start \
  -c out/harbor/biosurfactants-mini/biosurfactants-extraction-template/job.yaml \
  -a gemini-cli -m gemini/gemini-2.5-flash --modal
```

Run one task (oracle):
```bash
uv run python src/harbor-task-gen/run_harbor.py \
  --workspace examples/biosurfactants-extraction trials start \
  -p out/harbor/biosurfactants-mini/biosurfactants-extraction-template/tasks/<task-id> \
  -a oracle
```

Run job (OpenRouter via Terminus 2):
```bash
uv run python src/harbor-task-gen/run_harbor.py \
  --workspace examples/biosurfactants-extraction jobs start \
  -c out/harbor/biosurfactants-mini/biosurfactants-extraction-template/job.yaml \
  -a terminus-2 -m openrouter/qwen/qwen3-coder-plus --modal
```

### Custom workspace (new task family)

```bash
uv run python src/harbor-task-gen/prepare_harbor_tasks.py \
  --domain supercon --workspace /tmp/harbor-ws \
  --write-job-config --force

uv run python src/harbor-task-gen/run_harbor.py --workspace /tmp/harbor-ws \
  jobs start -c out/harbor/supercon-mini-v2/ground-template/job.yaml \
  -a gemini-cli -m gemini/gemini-2.5-flash
```
