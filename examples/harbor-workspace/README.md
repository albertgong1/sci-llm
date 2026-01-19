# Harbor Workspace

This folder is the default workspace for task generation and Harbor runs. Templates,
PDFs, rubrics, and run outputs live here. The library code lives in
`src/harbor-task-gen/`.

## Layout
- `ground-template/` - default task template (add others as sibling folders).
- `rubric.csv` - scoring rubric used when tasks are built.
- `data/Paper_DB/` - PDF corpus (`<refno>.pdf`).
- `out/` - generated tasks and compiled run bundles.
- `jobs/` - Harbor job runs.
- `trials/` - Harbor trial runs.
- `collect_harbor_results.py` - helper to convert verifier outputs into prediction JSON.
- `clean_harbor_workspace.py` - clean up jobs/trials/out without touching templates.

## Using the library
All CLI helpers default to this workspace. From the repo root:

```bash
uv run python src/harbor-task-gen/prepare_harbor_tasks.py --write-job-config --force

uv run python src/harbor-task-gen/run_harbor.py jobs start \
  -c out/harbor/supercon-mini-v2/ground-template/job.yaml \
  -a gemini-cli -m gemini/gemini-2.5-flash --modal --n-concurrent 4
```

To use a different workspace, pass `--workspace /path/to/workspace` to the scripts.
To target a different dataset, use `--domain` (see `src/pbench/datasets.yaml`).

## Cleaning the workspace
Remove generated outputs (jobs/trials/out) while keeping templates and data:
```bash
uv run python examples/harbor-workspace/clean_harbor_workspace.py --dry-run
uv run python examples/harbor-workspace/clean_harbor_workspace.py
```

## Creating a new task template
1) Copy an existing template folder:
```bash
cp -R ground-template my-template
```
2) Edit the template files:
- `instruction.md.template`
- `tests/check_prediction.py`
- `task.toml.template`
- `environment/Dockerfile`

3) Generate tasks with the new template:
```bash
uv run python src/harbor-task-gen/prepare_harbor_tasks.py \
  --template my-template --write-job-config --force
```
4) Run Harbor on the generated job config:
```bash
uv run python src/harbor-task-gen/run_harbor.py jobs start \
  -c out/harbor/supercon-mini-v2/my-template/job.yaml \
  -a gemini-cli -m gemini/gemini-2.5-flash
```

For the full library API, CLI flags, and troubleshooting guides, see
`src/harbor-task-gen/README.md` and `src/harbor-task-gen/knowledgebase.md`.
