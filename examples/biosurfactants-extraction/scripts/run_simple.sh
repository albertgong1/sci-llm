#!/usr/bin/env bash
# Run pbench-eval for multiple model combinations
# Run from examples/biosurfactants-extraction/, not from scripts/
# Usage: ./scripts/run_simple.sh DATA_DIR OUTPUT_DIR [extra args...]

# Exit entire script on Ctrl+C
trap "echo ' Interrupted, exiting...'; exit 130" INT

if [ $# -lt 2 ]; then
  echo "Usage: $0 DATA_DIR OUTPUT_DIR [extra args...]"
  exit 1
fi

data_dir=$1
output_dir=$2
shift 2
cmd_args=$@

# Server/model combinations
# Format: "server:model" or "server:model:kwarg1=value1,kwarg2=value2"
combinations=(
  # "gemini:gemini-3-pro-preview"
  # "gemini:gemini-3-flash-preview"
  "openai:gpt-5.2-2025-12-11:openai_reasoning_effort=medium"
  # "openai:gpt-5.2-2025-12-11:openai_reasoning_effort=high"
  "openai:gpt-5-mini-2025-08-07:openai_reasoning_effort=medium"
  # "openai:gpt-5-mini-2025-08-07:openai_reasoning_effort=high"
)

for combo in "${combinations[@]}"; do
  # Parse server:model:kwargs format
  IFS=':' read -r server model kwargs <<< "$combo"

  # Build extra arguments from kwargs (comma-separated key=value pairs)
  extra_args=""
  if [ -n "$kwargs" ]; then
    IFS=',' read -ra kwarg_pairs <<< "$kwargs"
    for kv in "${kwarg_pairs[@]}"; do
      key="${kv%%=*}"
      value="${kv#*=}"
      extra_args="$extra_args --${key} ${value}"
    done
  fi

  echo "========================================"
  echo "Running server=${server} model=${model} kwargs=${kwargs}"
  echo "========================================"

  CMD="uv run pbench-eval -dd ${data_dir} --server ${server} -m ${model} \
    -pp prompts/benchmark_soft_prompt_01.md -od ${output_dir} \
    --hf_repo kilian-group/biosurfactants-extraction --hf_split full --hf_revision v0.0.0 --log_level INFO ${extra_args} $cmd_args"
  echo "Executing: $CMD"
  eval $CMD

  echo "Completed ${server}/${model}"
  echo ""
done

echo "All combinations completed!"
