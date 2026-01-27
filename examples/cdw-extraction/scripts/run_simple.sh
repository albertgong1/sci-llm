#!/usr/bin/env bash
# Run pbench-eval for multiple model combinations (SuperCon)
# Run from examples/supercon-extraction/, not from scripts/
# Usage: ./scripts/run_simple.sh DATA_DIR OUTPUT_DIR [extra args...]
#
# Note: This script uses registry_data.json to define the ordering.
# Ask ag2435@cornell.edu on Slack for a copy of this file.

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
  "gemini:gemini-3-pro-preview"
  "gemini:gemini-3-flash-preview"
  # "openai:gpt-5.2-2025-12-11:openai_reasoning_effort=high"
  "openai:gpt-5.2-2025-12-11:openai_reasoning_effort=medium"
  # "openai:gpt-5-mini-2025-08-07:openai_reasoning_effort=high"
  "openai:gpt-5-mini-2025-08-07:openai_reasoning_effort=medium"
)

# NOTE: this registry contains a shuffle of the tasks with Chao and Fatmagul validated ones at the front
REGISTRY_PATH=out-0126-harbor/targeted-stoichiometric-template/registry.json

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
    -pp prompts/targeted_extraction_prompt_03.md -od ${output_dir} \
    --harbor_task_ordering_registry_path ${REGISTRY_PATH} --max_num_papers 50 --log_level INFO ${extra_args} $cmd_args"
  echo "Executing: $CMD"
  eval $CMD

  echo "Completed ${server}/${model}"
  echo ""
done

echo "All combinations completed!"
