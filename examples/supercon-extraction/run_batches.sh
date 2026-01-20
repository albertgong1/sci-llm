#!/usr/bin/env bash
# Run harbor extraction for batches 1-10
# Usage: ./run_batches.sh JOBS_DIR [additional args]

if [ $# -lt 1 ]; then
  echo "Usage: $0 JOBS_DIR [additional args]"
  exit 1
fi

jobs_dir=$1
shift
cmd_args=$@

for batch in {1..1}; do
  echo "Running batch ${batch}..."
  uv run python ../../src/harbor-task-gen/run_batch_harbor.py jobs start \
    --hf-tasks-repo kilian-group/supercon-extraction-harbor-tasks --hf-tasks-version v0.1.0 \
    -a gemini-cli -m gemini/gemini-3-pro-preview \
    --workspace . --jobs-dir "${jobs_dir}" --seed 1 --batch-size 10 --batch-number "${batch}" \
    $cmd_args
done

echo "All batches completed!"
