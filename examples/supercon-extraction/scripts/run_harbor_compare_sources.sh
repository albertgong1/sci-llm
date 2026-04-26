#!/usr/bin/env bash
# Run the same Harbor batch grid against both PDF and MinerU task registries.
# Run from examples/supercon-extraction/, not from scripts/
# Usage: ./scripts/run_harbor_compare_sources.sh JOBS_ROOT [extra args...]

set -euo pipefail

trap "echo ' Interrupted, exiting...'; exit 130" INT

if [ $# -lt 1 ]; then
  echo "Usage: $0 JOBS_ROOT [extra args...]"
  exit 1
fi

jobs_root=$1
shift
declare -a cmd_args=()
if [ $# -gt 0 ]; then
  cmd_args=("$@")
fi

compare_sources=${HARBOR_COMPARE_SOURCES:-$'pdf\nmineru'}
compare_dataset=${HARBOR_COMPARE_DATASET:-supercon-extraction@v0.2.1}
pdf_registry=${HARBOR_COMPARE_PDF_REGISTRY:-out-pdf-harbor-200/targeted-stoichiometric-template/registry.json}
mineru_registry=${HARBOR_COMPARE_MINERU_REGISTRY:-out-mineru-harbor-200/targeted-stoichiometric-template/registry.json}
batch_size=${HARBOR_BATCH_SIZE:-50}
batch_start=${HARBOR_BATCH_START:-1}
num_batches=${HARBOR_NUM_BATCHES:-4}
score_after=${HARBOR_COMPARE_SCORE_AFTER:-0}
score_root=${HARBOR_COMPARE_SCORE_ROOT:-${jobs_root}-scores}
progress_score_after_batch=${HARBOR_COMPARE_PROGRESS_SCORE_AFTER_BATCH:-1}
progress_score_root=${HARBOR_COMPARE_PROGRESS_SCORE_ROOT:-${score_root}}
compare_continue_on_error=${HARBOR_COMPARE_CONTINUE_ON_ERROR:-1}
harbor_resume_existing=${HARBOR_RESUME_EXISTING:-1}
harbor_continue_on_error=${HARBOR_CONTINUE_ON_ERROR:-${compare_continue_on_error}}
compare_manifest_path=${HARBOR_COMPARE_MANIFEST_PATH:-${jobs_root}/compare_manifest.json}
modal_requested=0
n_concurrent_set=0
compare_failures=0
if [ ${#cmd_args[@]} -gt 0 ]; then
  for arg in "${cmd_args[@]}"; do
    case "${arg}" in
      --modal)
        modal_requested=1
        ;;
      --n-concurrent|--n-concurrent=*)
        n_concurrent_set=1
        ;;
    esac
  done
fi

if [ "${modal_requested}" = "1" ] && [ "${n_concurrent_set}" = "0" ]; then
  cmd_args+=(--n-concurrent "${HARBOR_MODAL_N_CONCURRENT:-10}")
fi

mkdir -p "${jobs_root}"
HARBOR_COMPARE_JOBS_ROOT="${jobs_root}" \
HARBOR_COMPARE_SCORE_ROOT_VALUE="${score_root}" \
HARBOR_COMPARE_DATASET_VALUE="${compare_dataset}" \
HARBOR_COMPARE_PDF_REGISTRY_VALUE="${pdf_registry}" \
HARBOR_COMPARE_MINERU_REGISTRY_VALUE="${mineru_registry}" \
HARBOR_COMPARE_BATCH_SIZE_VALUE="${batch_size}" \
HARBOR_COMPARE_BATCH_START_VALUE="${batch_start}" \
HARBOR_COMPARE_NUM_BATCHES_VALUE="${num_batches}" \
HARBOR_COMPARE_SCORE_AFTER_VALUE="${score_after}" \
HARBOR_COMPARE_PROGRESS_SCORE_AFTER_BATCH_VALUE="${progress_score_after_batch}" \
HARBOR_COMPARE_CONTINUE_ON_ERROR_VALUE="${compare_continue_on_error}" \
HARBOR_COMPARE_RESUME_EXISTING_VALUE="${harbor_resume_existing}" \
HARBOR_COMPARE_MANIFEST_SOURCES="${compare_sources}" \
HARBOR_COMPARE_MANIFEST_CMD_ARGS="${cmd_args[*]-}" \
python3 - "${compare_manifest_path}" <<'PY'
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

manifest_path = Path(sys.argv[1])
payload = {
    "generated_at": datetime.now(timezone.utc).isoformat(),
    "jobs_root": os.environ.get("HARBOR_COMPARE_JOBS_ROOT"),
    "score_root": os.environ.get("HARBOR_COMPARE_SCORE_ROOT_VALUE"),
    "compare_dataset": os.environ.get("HARBOR_COMPARE_DATASET_VALUE"),
    "pdf_registry": os.environ.get("HARBOR_COMPARE_PDF_REGISTRY_VALUE"),
    "mineru_registry": os.environ.get("HARBOR_COMPARE_MINERU_REGISTRY_VALUE"),
    "batch_size": int(os.environ.get("HARBOR_COMPARE_BATCH_SIZE_VALUE", "0") or 0),
    "batch_start": int(os.environ.get("HARBOR_COMPARE_BATCH_START_VALUE", "0") or 0),
    "num_batches": int(os.environ.get("HARBOR_COMPARE_NUM_BATCHES_VALUE", "0") or 0),
    "score_after": os.environ.get("HARBOR_COMPARE_SCORE_AFTER_VALUE") == "1",
    "progress_score_after_batch": os.environ.get(
        "HARBOR_COMPARE_PROGRESS_SCORE_AFTER_BATCH_VALUE"
    )
    == "1",
    "continue_on_error": os.environ.get("HARBOR_COMPARE_CONTINUE_ON_ERROR_VALUE")
    == "1",
    "resume_existing": os.environ.get("HARBOR_COMPARE_RESUME_EXISTING_VALUE") == "1",
    "sources": [
        line
        for line in os.environ.get("HARBOR_COMPARE_MANIFEST_SOURCES", "").splitlines()
        if line.strip()
    ],
    "extra_args_shell": os.environ.get("HARBOR_COMPARE_MANIFEST_CMD_ARGS", ""),
}
manifest_path.write_text(json.dumps(payload, indent=2) + "\n")
PY

while IFS= read -r source; do
  if [ -z "${source}" ]; then
    continue
  fi

  case "${source}" in
    pdf)
      registry_path=${pdf_registry}
      ;;
    mineru)
      registry_path=${mineru_registry}
      ;;
    *)
      echo "Unknown source '${source}'. Expected one of: pdf, mineru"
      exit 1
      ;;
  esac

  if [ ! -f "${registry_path}" ]; then
    echo "Registry file not found for source '${source}': ${registry_path}"
    exit 1
  fi

  echo "========================================"
  echo "Running source=${source} registry=${registry_path}"
  echo "========================================"

  run_status=0
  if [ ${#cmd_args[@]} -gt 0 ]; then
    if HARBOR_TASKS_REGISTRY_PATH="${registry_path}" \
      HARBOR_TASKS_DATASET="${compare_dataset}" \
      HARBOR_BATCH_SIZE="${batch_size}" \
      HARBOR_BATCH_START="${batch_start}" \
      HARBOR_NUM_BATCHES="${num_batches}" \
      HARBOR_RESUME_EXISTING="${harbor_resume_existing}" \
      HARBOR_CONTINUE_ON_ERROR="${harbor_continue_on_error}" \
      HARBOR_PROGRESS_SCORE_AFTER_BATCH="${progress_score_after_batch}" \
      HARBOR_PROGRESS_SCORE_OUTPUT_DIR="${progress_score_root}/${source}" \
      HARBOR_PROGRESS_SCORE_SOURCE="${source}" \
      HARBOR_PROGRESS_SCORE_SNAPSHOT_DIR="${progress_score_root}/${source}/completed_jobs_snapshot" \
      ./scripts/run_harbor.sh "${jobs_root}/${source}" "${cmd_args[@]}"
    then
      run_status=0
    else
      run_status=$?
    fi
  else
    if HARBOR_TASKS_REGISTRY_PATH="${registry_path}" \
      HARBOR_TASKS_DATASET="${compare_dataset}" \
      HARBOR_BATCH_SIZE="${batch_size}" \
      HARBOR_BATCH_START="${batch_start}" \
      HARBOR_NUM_BATCHES="${num_batches}" \
      HARBOR_RESUME_EXISTING="${harbor_resume_existing}" \
      HARBOR_CONTINUE_ON_ERROR="${harbor_continue_on_error}" \
      HARBOR_PROGRESS_SCORE_AFTER_BATCH="${progress_score_after_batch}" \
      HARBOR_PROGRESS_SCORE_OUTPUT_DIR="${progress_score_root}/${source}" \
      HARBOR_PROGRESS_SCORE_SOURCE="${source}" \
      HARBOR_PROGRESS_SCORE_SNAPSHOT_DIR="${progress_score_root}/${source}/completed_jobs_snapshot" \
      ./scripts/run_harbor.sh "${jobs_root}/${source}"
    then
      run_status=0
    else
      run_status=$?
    fi
  fi

  if [ "${score_after}" = "1" ] && [ -d "${jobs_root}/${source}" ]; then
    if HARBOR_COMPARE_SOURCES="${source}" ./scripts/score_harbor_compare_sources.sh "${jobs_root}" "${score_root}"; then
      :
    else
      score_status=$?
      compare_failures=$((compare_failures + 1))
      echo "Scoring failed for source=${source} with exit code ${score_status}"
      if [ "${compare_continue_on_error}" != "1" ]; then
        exit "${score_status}"
      fi
    fi
  fi

  if [ "${run_status}" -ne 0 ]; then
    compare_failures=$((compare_failures + 1))
    echo "Run failed for source=${source} with exit code ${run_status}"
    if [ "${compare_continue_on_error}" != "1" ]; then
      exit "${run_status}"
    fi
  fi
done <<< "${compare_sources}"

if [ "${compare_failures}" -gt 0 ]; then
  echo "Completed with ${compare_failures} source-level failure(s)."
  exit 1
fi
