#!/usr/bin/env bash
# Run harbor extraction for multiple agent/model combinations
# Run from examples/biosurfactants-extraction/, not from scripts/
# Usage (local): ./scripts/run_harbor.sh JOBS_DIR [extra args...]
# Usage (Modal): ./scripts/run_harbor.sh JOBS_DIR --modal [extra args...]

# Exit entire script on Ctrl+C
trap "echo ' Interrupted, exiting...'; exit 130" INT

if [ $# -lt 1 ]; then
  echo "Usage (local): $0 JOBS_DIR [extra args...]"
  echo "Usage (Modal): $0 JOBS_DIR --modal [extra args...]"
  exit 1
fi

jobs_dir=$1
shift
cmd_args=$@

# Agent/model combinations
# Format: "agent:model" or "agent:model:kwarg1=value1,kwarg2=value2"
# TODO: add qwen-code with qwen3-max via OpenRouter
combinations=(
  # "gemini-cli:gemini/gemini-3-pro-preview"
  # "codex:openai/gpt-5.2-2025-12-11:reasoning_effort=medium"
  # "gemini-cli:gemini/gemini-3-flash-preview"
  "terminus-2:gemini/gemini-3-pro-preview"
  # "terminus-2:openai/gpt-5.2-2025-12-11:reasoning_effort=medium"
)
BATCH_SIZE=10
NUM_BATCHES=3

for combo in "${combinations[@]}"; do
  # Parse agent:model:kwargs format
  IFS=':' read -r agent model kwargs <<< "$combo"

  # Build --ak arguments from kwargs (comma-separated key=value pairs)
  ak_args=""
  if [ -n "$kwargs" ]; then
    IFS=',' read -ra kwarg_pairs <<< "$kwargs"
    for kv in "${kwarg_pairs[@]}"; do
      ak_args="$ak_args --ak $kv"
    done
  fi

  echo "========================================"
  echo "Running agent=${agent} model=${model} kwargs=${kwargs}"
  echo "========================================"

  for batch in $(seq 1 $NUM_BATCHES); do
    echo "Running batch ${batch}/${NUM_BATCHES}..."
    CMD="uv run python ../../src/harbor-task-gen/run_batch_harbor.py jobs start \
      --hf-tasks-repo kilian-group/biosurfactants-extraction-harbor-tasks --hf-tasks-version head \
      -a ${agent} -m ${model} ${ak_args} \
      --workspace . --jobs-dir ${jobs_dir} --seed 1 --batch-size ${BATCH_SIZE} --batch-number ${batch} $cmd_args"
    echo "Executing: $CMD"
    eval $CMD
  done

  echo "Completed ${agent}/${model}"
  echo ""
done

echo "All combinations completed!"
