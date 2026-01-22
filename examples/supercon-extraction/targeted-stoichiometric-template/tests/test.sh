#!/bin/bash
set -euo pipefail

set +e
python /tests/check_prediction.py 2>&1 | tee /logs/verifier/log.txt
status=${PIPESTATUS[0]}
set -e

# Preserve *all* agent-written outputs even on verifier failure (Harbor deletes containers).
if [[ -d /app/output ]]; then
  mkdir -p /logs/verifier/app_output
  cp -R /app/output/. /logs/verifier/app_output/ 2>/dev/null || true
fi

if [[ -f /app/task_meta.json ]]; then
  cp /app/task_meta.json /logs/verifier/task_meta.json 2>/dev/null || true
fi

exit $status
