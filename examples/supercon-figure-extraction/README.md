# SuperCon Figure Extraction

This workspace benchmarks figure-only property extraction for SuperCon papers.
Tasks are built from MinerU figure crops and a figure inventory, and agents are
asked to recover superconducting properties using figure evidence only.

## Build Figure-Only Harbor Tasks

```bash
uv run python ../../src/harbor-task-gen/prepare_harbor_tasks.py \
  --workspace . \
  --template targeted-figure-template \
  --paper-source mineru_figures \
  --pdf-dir DATA_DIR/Paper_DB \
  --output-dir OUTPUT_DIR \
  --gt-hf-repo kilian-group/supercon-extraction \
  --gt-hf-split full \
  --gt-hf-revision main \
  --force
```

Useful notes:
- The task generator reuses the shared SuperCon verifier/output schema.
- The figure-only mode copies MinerU figure crops plus `figures.md` and `figures.json`.
- Later scoring changes can build on the same task format without regenerating the workspace layout.

## Run Harbor

```bash
uv run python ../../src/harbor-task-gen/run_harbor.py jobs start \
  --registry-path OUTPUT_DIR/targeted-figure-template/registry.json \
  --dataset supercon-extraction@main \
  -a gemini-cli -m gemini/gemini-3-pro-preview \
  --workspace . --jobs-dir JOBS_DIR --seed 1 --batch-size 10
```
