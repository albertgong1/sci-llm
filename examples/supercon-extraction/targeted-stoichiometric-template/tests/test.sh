#!/bin/bash
set -euo pipefail

set +e
python /tests/check_prediction.py 2>&1 | tee /logs/verifier/log.txt
status=${PIPESTATUS[0]}
set -e

# Preserve the agent-written predictions file even on verifier failure (Harbor deletes containers).
if [[ -f /app/predictions.json ]]; then
  mkdir -p /logs/verifier/app_output
  cp /app/predictions.json /logs/verifier/app_output/ 2>/dev/null || true
fi

if [[ -f /app/task_meta.json ]]; then
  cp /app/task_meta.json /logs/verifier/task_meta.json 2>/dev/null || true
fi

exit $status
