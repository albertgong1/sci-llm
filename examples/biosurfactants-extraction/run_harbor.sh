#!/usr/bin/env bash
# Run harbor extraction for multiple agent/model combinations
# Usage: ./run_batches.sh JOBS_DIR

# Exit entire script on Ctrl+C
trap "echo ' Interrupted, exiting...'; exit 130" INT

if [ $# -lt 1 ]; then
  echo "Usage: $0 JOBS_DIR"
  exit 1
fi

jobs_dir=$1
shift
cmd_args=$@

# Agent/model combinations (50 samples each = 5 batches of 10)
combinations=(
  "gemini-cli:gemini/gemini-3-pro-preview"
  # "codex:openai/gpt-5.1-2025-11-13"
  "gemini-cli:gemini/gemini-3-flash-preview"
  # "codex:openai/gpt-5-mini-2025-08-07"
  "terminus-2:gemini/gemini-3-pro-preview"
  # "terminus-2:openai/gpt-5.1-2025-11-13"
)
BATCH_SIZE=50
NUM_BATCHES=1

for combo in "${combinations[@]}"; do
  agent="${combo%%:*}"
  model="${combo##*:}"

  echo "========================================"
  echo "Running agent=${agent} model=${model}"
  echo "========================================"

  for batch in $(seq 1 $NUM_BATCHES); do
    echo "Running batch ${batch}/${NUM_BATCHES}..."
    CMD="uv run python ../../src/harbor-task-gen/run_batch_harbor.py jobs start \
      --hf-tasks-repo kilian-group/biosurfactants-extraction-harbor-tasks --hf-tasks-version head \
      -a ${agent} -m ${model} \
      --workspace . --jobs-dir ${jobs_dir} --seed 1 --batch-size ${BATCH_SIZE} --batch-number ${batch} $cmd_args"
    echo "Executing: $CMD"
    eval $CMD
  done

  echo "Completed ${agent}/${model}"
  echo ""
done

echo "All combinations completed!"
