#!/usr/bin/env bash
# Run harbor extraction for multiple agent/model combinations
# Run from examples/supercon-extraction/, not from scripts/
# Usage (local): ./scripts/run_harbor.sh JOBS_DIR [extra args...]
# Usage (Modal): ./scripts/run_harbor.sh JOBS_DIR --modal [extra args...]

set -euo pipefail

# Exit entire script on Ctrl+C
trap "echo ' Interrupted, exiting...'; exit 130" INT

if [ $# -lt 1 ]; then
  echo "Usage (local): $0 JOBS_DIR [extra args...]"
  echo "Usage (Modal): $0 JOBS_DIR --modal [extra args...]"
  exit 1
fi

jobs_dir=$1
shift
declare -a cmd_args=()
if [ $# -gt 0 ]; then
  cmd_args=("$@")
fi
resume_existing=${HARBOR_RESUME_EXISTING:-1}
continue_on_error=${HARBOR_CONTINUE_ON_ERROR:-0}
fail_fast_on_quota=${HARBOR_FAIL_FAST_ON_QUOTA:-1}
block_provider_on_quota=${HARBOR_BLOCK_PROVIDER_ON_QUOTA:-1}
quota_error_threshold=${HARBOR_QUOTA_ERROR_THRESHOLD:-3}
quota_error_poll_sec=${HARBOR_QUOTA_ERROR_POLL_SEC:-5}
run_manifest_path=${HARBOR_RUN_MANIFEST_PATH:-${jobs_dir}/run_manifest.json}
user_force_all=0
blocked_providers=""
modal_requested=0

if [ ${#cmd_args[@]} -gt 0 ]; then
  for arg in "${cmd_args[@]}"; do
    case "${arg}" in
      --force|--force=*)
        user_force_all=1
        ;;
      --modal)
        modal_requested=1
        ;;
    esac
  done
fi

if [ "${modal_requested}" = "1" ] && [ -z "${HARBOR_MODAL_LOG_DOWNLOAD_TIMEOUT_SEC+x}" ]; then
  export HARBOR_MODAL_LOG_DOWNLOAD_TIMEOUT_SEC=30
fi

# Optional environment overrides:
#   HARBOR_USE_HF_TASKS=1
#   HARBOR_HF_TASKS_REPO=ORG/REPO
#   HARBOR_HF_TASKS_VERSION=v0.0.0
#   HARBOR_TASKS_REGISTRY_PATH=out-mineru-harbor/targeted-stoichiometric-template/registry.json
#   HARBOR_TASKS_DATASET=supercon-extraction@v0.2.1
#   HARBOR_INCLUDE_FLASH=1
#   HARBOR_COMBINATIONS=$'codex:openai/gpt-5.2-2025-12-11:reasoning_effort=medium'
#   HARBOR_BATCH_SIZE=50
#   HARBOR_BATCH_START=1
#   HARBOR_NUM_BATCHES=4
#   HARBOR_OPENAI_N_CONCURRENT=1
#   HARBOR_OPENAI_BATCH_COOLDOWN_SEC=30
#   HARBOR_PROGRESS_SCORE_AFTER_BATCH=1
#   HARBOR_PROGRESS_SCORE_OUTPUT_DIR=out-progress/mineru
#   HARBOR_PROGRESS_SCORE_SOURCE=mineru
#   HARBOR_PROGRESS_SCORE_CONTINUE_ON_ERROR=1
use_hf_tasks=${HARBOR_USE_HF_TASKS:-0}
default_registry_path=out-0121-harbor/targeted-stoichiometric-template/registry.json
default_tasks_dataset=supercon-extraction@main
if [ -f "out-mineru-harbor/targeted-stoichiometric-template/registry.json" ]; then
  default_registry_path=out-mineru-harbor/targeted-stoichiometric-template/registry.json
  default_tasks_dataset=supercon-extraction@v0.2.1
fi
tasks_registry_path=${HARBOR_TASKS_REGISTRY_PATH:-${default_registry_path}}
tasks_dataset=${HARBOR_TASKS_DATASET:-${default_tasks_dataset}}
hf_tasks_repo=${HARBOR_HF_TASKS_REPO:-kilian-group/supercon-extraction-harbor-tasks}
hf_tasks_version=${HARBOR_HF_TASKS_VERSION:-v0.0.0}

if [ "$use_hf_tasks" != "1" ] && [ ! -f "${tasks_registry_path}" ]; then
  echo "Registry file not found: ${tasks_registry_path}"
  exit 1
fi

# Agent/model combinations
# Format: "agent:model" or "agent:model:kwarg1=value1,kwarg2=value2"
if [ -n "${HARBOR_COMBINATIONS:-}" ]; then
  combinations_spec=${HARBOR_COMBINATIONS}
else
  combinations_spec=$'gemini-cli:gemini/gemini-3-pro-preview\ncodex:openai/gpt-5.2-2025-12-11:reasoning_effort=medium\nterminus-2:gemini/gemini-3-pro-preview\nterminus-2:openai/gpt-5.2-2025-12-11:reasoning_effort=medium'
  if [ "${HARBOR_INCLUDE_FLASH:-0}" = "1" ]; then
    combinations_spec+=$'\ngemini-cli:gemini/gemini-3-flash-preview\ncodex:openai/gpt-5-mini-2025-08-07:reasoning_effort=medium'
  fi
fi
BATCH_SIZE=${HARBOR_BATCH_SIZE:-50}
BATCH_START=${HARBOR_BATCH_START:-1}
NUM_BATCHES=${HARBOR_NUM_BATCHES:-4}
openai_batch_cooldown_sec=${HARBOR_OPENAI_BATCH_COOLDOWN_SEC:-0}
progress_score_after_batch=${HARBOR_PROGRESS_SCORE_AFTER_BATCH:-0}
progress_score_output_dir=${HARBOR_PROGRESS_SCORE_OUTPUT_DIR:-}
progress_score_source=${HARBOR_PROGRESS_SCORE_SOURCE:-}
progress_score_snapshot_dir=${HARBOR_PROGRESS_SCORE_SNAPSHOT_DIR:-}
progress_score_continue_on_error=${HARBOR_PROGRESS_SCORE_CONTINUE_ON_ERROR:-1}
overall_failures=0

write_run_manifest() {
  local manifest_path=$1
  mkdir -p "$(dirname "${manifest_path}")"
  HARBOR_RUN_MANIFEST_COMBINATIONS="${combinations_spec}" \
  HARBOR_RUN_MANIFEST_EXTRA_ARGS="${cmd_args[*]-}" \
  python3 - "${manifest_path}" <<'PY'
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

manifest_path = Path(sys.argv[1])
payload = {
    "generated_at": datetime.now(timezone.utc).isoformat(),
    "jobs_dir": os.environ.get("HARBOR_RUN_JOBS_DIR"),
    "tasks_registry_path": os.environ.get("HARBOR_RUN_TASKS_REGISTRY_PATH"),
    "tasks_dataset": os.environ.get("HARBOR_RUN_TASKS_DATASET"),
    "use_hf_tasks": os.environ.get("HARBOR_RUN_USE_HF_TASKS") == "1",
    "hf_tasks_repo": os.environ.get("HARBOR_RUN_HF_TASKS_REPO"),
    "hf_tasks_version": os.environ.get("HARBOR_RUN_HF_TASKS_VERSION"),
    "batch_size": int(os.environ.get("HARBOR_RUN_BATCH_SIZE", "0") or 0),
    "batch_start": int(os.environ.get("HARBOR_RUN_BATCH_START", "0") or 0),
    "num_batches": int(os.environ.get("HARBOR_RUN_NUM_BATCHES", "0") or 0),
    "resume_existing": os.environ.get("HARBOR_RUN_RESUME_EXISTING") == "1",
    "continue_on_error": os.environ.get("HARBOR_RUN_CONTINUE_ON_ERROR") == "1",
    "fail_fast_on_quota": os.environ.get("HARBOR_RUN_FAIL_FAST_ON_QUOTA") == "1",
    "block_provider_on_quota": os.environ.get("HARBOR_RUN_BLOCK_PROVIDER_ON_QUOTA")
    == "1",
    "modal_requested": os.environ.get("HARBOR_RUN_MODAL_REQUESTED") == "1",
    "progress_score_after_batch": os.environ.get(
        "HARBOR_RUN_PROGRESS_SCORE_AFTER_BATCH"
    )
    == "1",
    "progress_score_output_dir": os.environ.get("HARBOR_RUN_PROGRESS_SCORE_OUTPUT_DIR"),
    "progress_score_source": os.environ.get("HARBOR_RUN_PROGRESS_SCORE_SOURCE"),
    "combinations": [
        line
        for line in os.environ.get("HARBOR_RUN_MANIFEST_COMBINATIONS", "").splitlines()
        if line.strip()
    ],
    "extra_args_shell": os.environ.get("HARBOR_RUN_MANIFEST_EXTRA_ARGS", ""),
}
manifest_path.write_text(json.dumps(payload, indent=2) + "\n")
PY
}

get_job_model_slug() {
  local model_name=$1
  local model_slug=${model_name##*/}
  model_slug=${model_slug//:/-}
  printf '%s\n' "${model_slug}"
}

get_model_provider() {
  local model_name=$1
  if [[ "${model_name}" == */* ]]; then
    printf '%s\n' "${model_name%%/*}"
  else
    printf 'unknown\n'
  fi
}

provider_is_blocked() {
  local provider_name=$1
  case " ${blocked_providers} " in
    *" ${provider_name} "*)
      return 0
      ;;
    *)
      return 1
      ;;
  esac
}

with_provider_cmd_args() {
  local provider_name=$1
  shift

  if [ "${provider_name}" != "openai" ] || [ -z "${HARBOR_OPENAI_N_CONCURRENT:-}" ]; then
    printf '%s\0' "$@"
    return
  fi

  local filtered=()
  local skip_next=0
  local arg

  for arg in "$@"; do
    if [ "${skip_next}" = "1" ]; then
      skip_next=0
      continue
    fi

    case "${arg}" in
      --n-concurrent)
        skip_next=1
        ;;
      --n-concurrent=*)
        ;;
      *)
        filtered+=("${arg}")
        ;;
    esac
  done

  filtered+=(--n-concurrent "${HARBOR_OPENAI_N_CONCURRENT}")
  printf '%s\0' "${filtered[@]}"
}

job_log_has_quota_errors() {
  local job_log=$1
  local min_hits=${2:-1}
  local hit_count

  if [ ! -f "${job_log}" ]; then
    return 1
  fi

  hit_count=$(grep -E -c 'insufficient_quota|exceeded your current quota|billing details' "${job_log}" || true)
  [ "${hit_count}" -ge "${min_hits}" ]
}

monitor_batch_for_quota_errors() {
  local command_pid=$1
  local job_log=$2

  while kill -0 "${command_pid}" 2>/dev/null; do
    if job_log_has_quota_errors "${job_log}" "${quota_error_threshold}"; then
      echo "Detected repeated quota errors in ${job_log}; terminating batch early."
      kill "${command_pid}" 2>/dev/null || true
      return 0
    fi
    sleep "${quota_error_poll_sec}"
  done

  return 1
}

get_batch_job_dir() {
  local batch_number=$1
  local agent_name=$2
  local model_name=$3
  local model_slug
  model_slug=$(get_job_model_slug "${model_name}")
  printf '%s\n' "${jobs_dir}/bn${batch_number}-bs${BATCH_SIZE}-${agent_name}-${model_slug}-s1"
}

get_batch_job_status() {
  local job_dir=$1
  local result_path=${job_dir}/result.json

  if [ ! -d "${job_dir}" ]; then
    printf 'missing\n'
    return
  fi

  if [ ! -f "${result_path}" ]; then
    printf 'incomplete\n'
    return
  fi

  if uv run python - "${result_path}" "${job_dir}" <<'PY' >/dev/null 2>&1
import json
import sys
from pathlib import Path

result_path = Path(sys.argv[1])
job_dir = Path(sys.argv[2])
data = json.loads(result_path.read_text())
stats = data.get("stats") or {}
finished = bool(data.get("finished_at"))
n_total_trials = data.get("n_total_trials")
n_trials = stats.get("n_trials")
n_errors = stats.get("n_errors")

if not (
    finished
    and n_total_trials is not None
    and n_trials == n_total_trials
    and int(n_errors or 0) == 0
):
    raise SystemExit(1)

for trial_dir in job_dir.iterdir():
    if not trial_dir.is_dir() or "__" not in trial_dir.name:
        continue

    codex_log = trial_dir / "agent" / "codex.txt"
    if codex_log.exists():
        codex_text = codex_log.read_text(errors="ignore")
        if "Quota exceeded. Check your plan and billing details." in codex_text:
            raise SystemExit(1)

    verifier_details = trial_dir / "verifier" / "details.json"
    if verifier_details.exists():
        details = json.loads(verifier_details.read_text())
        error = str(details.get("error") or "")
        if "Missing predictions file" in error:
            raise SystemExit(1)

raise SystemExit(0)
PY
  then
    printf 'complete\n'
  else
    printf 'incomplete\n'
  fi
}

run_progress_scoring() {
  local trigger_label=$1
  local progress_cmd
  local status

  if [ "${progress_score_after_batch}" != "1" ] || [ -z "${progress_score_output_dir}" ]; then
    return 0
  fi

  progress_cmd=(
    uv run python ../../src/pbench_eval/score_completed_harbor_progress.py
    -jd "${jobs_dir}"
    -od "${progress_score_output_dir}"
  )
  if [ -n "${progress_score_source}" ]; then
    progress_cmd+=(--source "${progress_score_source}")
  fi
  if [ -n "${progress_score_snapshot_dir}" ]; then
    progress_cmd+=(--snapshot_dir "${progress_score_snapshot_dir}")
  fi

  echo "Updating rolling F1 after ${trigger_label}..."
  printf 'Executing:'
  printf ' %q' "${progress_cmd[@]}"
  printf '\n'

  if "${progress_cmd[@]}"; then
    return 0
  else
    status=$?
  fi

  echo "Rolling F1 scoring failed after ${trigger_label} with exit code ${status}"
  if [ "${progress_score_continue_on_error}" != "1" ]; then
    exit "${status}"
  fi
  return 0
}

mkdir -p "${jobs_dir}"
HARBOR_RUN_JOBS_DIR="${jobs_dir}" \
HARBOR_RUN_TASKS_REGISTRY_PATH="${tasks_registry_path}" \
HARBOR_RUN_TASKS_DATASET="${tasks_dataset}" \
HARBOR_RUN_USE_HF_TASKS="${use_hf_tasks}" \
HARBOR_RUN_HF_TASKS_REPO="${hf_tasks_repo}" \
HARBOR_RUN_HF_TASKS_VERSION="${hf_tasks_version}" \
HARBOR_RUN_BATCH_SIZE="${BATCH_SIZE}" \
HARBOR_RUN_BATCH_START="${BATCH_START}" \
HARBOR_RUN_NUM_BATCHES="${NUM_BATCHES}" \
HARBOR_RUN_RESUME_EXISTING="${resume_existing}" \
HARBOR_RUN_CONTINUE_ON_ERROR="${continue_on_error}" \
HARBOR_RUN_FAIL_FAST_ON_QUOTA="${fail_fast_on_quota}" \
HARBOR_RUN_BLOCK_PROVIDER_ON_QUOTA="${block_provider_on_quota}" \
HARBOR_RUN_MODAL_REQUESTED="${modal_requested}" \
HARBOR_RUN_PROGRESS_SCORE_AFTER_BATCH="${progress_score_after_batch}" \
HARBOR_RUN_PROGRESS_SCORE_OUTPUT_DIR="${progress_score_output_dir}" \
HARBOR_RUN_PROGRESS_SCORE_SOURCE="${progress_score_source}" \
write_run_manifest "${run_manifest_path}"

while IFS= read -r combo; do
  if [ -z "${combo}" ]; then
    continue
  fi
  # Parse agent:model:kwargs format
  IFS=':' read -r agent model kwargs <<< "$combo"
  provider=$(get_model_provider "${model}")

  if [ "${block_provider_on_quota}" = "1" ] && provider_is_blocked "${provider}"; then
    echo "Skipping agent=${agent} model=${model} because provider ${provider} is blocked after a quota failure."
    echo ""
    continue
  fi

  declare -a combo_cmd_args=()
  if [ ${#cmd_args[@]} -gt 0 ]; then
    while IFS= read -r -d '' item; do
      combo_cmd_args+=("${item}")
    done < <(with_provider_cmd_args "${provider}" "${cmd_args[@]}")
  fi

  # Build --ak arguments from kwargs (comma-separated key=value pairs)
  ak_args=()
  if [ -n "$kwargs" ]; then
    IFS=',' read -ra kwarg_pairs <<< "$kwargs"
    for kv in "${kwarg_pairs[@]}"; do
      ak_args+=(--ak "$kv")
    done
  fi

  echo "========================================"
  echo "Running agent=${agent} model=${model} kwargs=${kwargs}"
  echo "========================================"

  for batch in $(seq $BATCH_START $NUM_BATCHES); do
    echo "Running batch ${batch}/${NUM_BATCHES}..."
    batch_force=0
    job_dir=$(get_batch_job_dir "${batch}" "${agent}" "${model}")

    if [ "${provider}" = "openai" ] && [ "${openai_batch_cooldown_sec}" -gt 0 ] && [ "${batch}" -gt "${BATCH_START}" ]; then
      echo "Cooling down ${openai_batch_cooldown_sec}s before OpenAI batch ${batch}/${NUM_BATCHES}..."
      sleep "${openai_batch_cooldown_sec}"
    fi

    if [ "${user_force_all}" != "1" ] && [ "${resume_existing}" = "1" ]; then
      job_status=$(get_batch_job_status "${job_dir}")
      case "${job_status}" in
        complete)
          echo "Skipping completed batch ${batch}/${NUM_BATCHES}: ${job_dir}"
          run_progress_scoring "completed batch ${batch}/${NUM_BATCHES}"
          continue
          ;;
        incomplete)
          echo "Reprocessing incomplete batch ${batch}/${NUM_BATCHES}: ${job_dir}"
          batch_force=1
          ;;
      esac
    fi

    if [ "$use_hf_tasks" = "1" ]; then
      cmd=(
        uv run python ../../src/harbor-task-gen/run_batch_harbor.py jobs start
        --hf-tasks-repo "${hf_tasks_repo}"
        --hf-tasks-version "${hf_tasks_version}"
        -a "${agent}"
        -m "${model}"
        --workspace .
        --jobs-dir "${jobs_dir}"
        --seed 1
        --batch-size "${BATCH_SIZE}"
        --batch-number "${batch}"
      )
    else
      cmd=(
        uv run python ../../src/harbor-task-gen/run_batch_harbor.py jobs start
        --registry-path "${tasks_registry_path}"
        --dataset "${tasks_dataset}"
        -a "${agent}"
        -m "${model}"
        --workspace .
        --jobs-dir "${jobs_dir}"
        --seed 1
        --batch-size "${BATCH_SIZE}"
        --batch-number "${batch}"
      )
    fi
    if [ "${batch_force}" = "1" ]; then
      cmd+=(--force)
    fi
    if [ ${#ak_args[@]} -gt 0 ]; then
      cmd+=("${ak_args[@]}")
    fi
    if [ ${#combo_cmd_args[@]} -gt 0 ]; then
      cmd+=("${combo_cmd_args[@]}")
    fi
    printf 'Executing:'
    printf ' %q' "${cmd[@]}"
    printf '\n'
    status=0
    quota_monitor_pid=""
    if [ "${fail_fast_on_quota}" = "1" ]; then
      job_log="${job_dir}/job.log"
      "${cmd[@]}" &
      command_pid=$!
      monitor_batch_for_quota_errors "${command_pid}" "${job_log}" &
      quota_monitor_pid=$!
      if wait "${command_pid}"; then
        status=0
      else
        status=$?
      fi
      if [ -n "${quota_monitor_pid}" ]; then
        kill "${quota_monitor_pid}" 2>/dev/null || true
        wait "${quota_monitor_pid}" 2>/dev/null || true
      fi
    else
      if "${cmd[@]}"; then
        status=0
      else
        status=$?
      fi
    fi

    if [ "${status}" -eq 0 ]; then
      run_progress_scoring "batch ${batch}/${NUM_BATCHES}"
    else
      provider_blocked_this_batch=0
      overall_failures=$((overall_failures + 1))
      echo "Batch ${batch}/${NUM_BATCHES} failed for ${agent}/${model} with exit code ${status}"
      if [ "${block_provider_on_quota}" = "1" ] && job_log_has_quota_errors "${job_dir}/job.log"; then
        echo "Blocking provider ${provider} for the rest of this run due to quota exhaustion."
        blocked_providers="${blocked_providers} ${provider}"
        provider_blocked_this_batch=1
      fi
      if [ "${continue_on_error}" != "1" ]; then
        exit "${status}"
      fi
      if [ "${provider_blocked_this_batch}" = "1" ]; then
        break
      fi
    fi
  done

  echo "Completed ${agent}/${model}"
  echo ""
done <<< "${combinations_spec}"

if [ "${overall_failures}" -gt 0 ]; then
  echo "Completed with ${overall_failures} failed batch(es)."
  exit 1
fi

echo "All combinations completed!"
