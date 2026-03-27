#!/usr/bin/env bash
# Score Harbor comparison runs and write aggregate outputs per paper source.
# Run from examples/supercon-extraction/, not from scripts/
# Usage: ./scripts/score_harbor_compare_sources.sh JOBS_ROOT [OUTPUT_ROOT]

set -euo pipefail

trap "echo ' Interrupted, exiting...'; exit 130" INT

if [ $# -lt 1 ]; then
  echo "Usage: $0 JOBS_ROOT [OUTPUT_ROOT]"
  exit 1
fi

jobs_root=$1
output_root=${2:-${jobs_root}-scores}

compare_sources=${HARBOR_COMPARE_SOURCES:-$'pdf\nmineru'}
match_model=${SUPERCON_SCORE_MATCH_MODEL:-gemini-2.5-flash}
hf_repo=${SUPERCON_SCORE_HF_REPO:-kilian-group/supercon-extraction}
hf_split=${SUPERCON_SCORE_HF_SPLIT:-full}
hf_revision=${SUPERCON_SCORE_HF_REVISION:-v0.2.1}
prompt_path=${SUPERCON_SCORE_PROMPT_PATH:-prompts/property_matching_prompt.md}
rubric_path=${SUPERCON_SCORE_RUBRIC_PATH:-scoring/rubric_4.csv}
conversion_factors_path=${SUPERCON_SCORE_CONVERSION_FACTORS_PATH:-scoring/si_conversion_factors.csv}
matching_mode=${SUPERCON_SCORE_MATCHING_MODE:-material}
log_level=${SUPERCON_SCORE_LOG_LEVEL:-ERROR}
force_score=${SUPERCON_SCORE_FORCE:-1}
skip_missing=${HARBOR_SCORE_SKIP_MISSING:-1}
continue_on_error=${HARBOR_SCORE_CONTINUE_ON_ERROR:-1}
score_failures=0
scored_sources=0

if [ ! -d "${jobs_root}" ]; then
  echo "Jobs root not found: ${jobs_root}"
  exit 1
fi

if [ ! -f "${prompt_path}" ]; then
  echo "Prompt file not found: ${prompt_path}"
  exit 1
fi

if [ ! -f "${rubric_path}" ]; then
  echo "Rubric file not found: ${rubric_path}"
  exit 1
fi

if [ ! -f "${conversion_factors_path}" ]; then
  echo "Conversion factors file not found: ${conversion_factors_path}"
  exit 1
fi

force_args=()
if [ "${force_score}" = "1" ]; then
  force_args=(--force)
fi

while IFS= read -r source; do
  if [ -z "${source}" ]; then
    continue
  fi

  jobs_dir="${jobs_root}/${source}"
  output_dir="${output_root}/${source}"

  if [ ! -d "${jobs_dir}" ]; then
    if [ "${skip_missing}" = "1" ]; then
      echo "Skipping source '${source}' because jobs directory is missing: ${jobs_dir}"
      continue
    fi
    echo "Jobs directory not found for source '${source}': ${jobs_dir}"
    exit 1
  fi

  first_job_dir=$(find "${jobs_dir}" -mindepth 1 -maxdepth 1 -type d -print -quit)
  if [ -z "${first_job_dir}" ]; then
    if [ "${skip_missing}" = "1" ]; then
      echo "Skipping source '${source}' because no Harbor job directories were found in: ${jobs_dir}"
      continue
    fi
    echo "No Harbor job directories found for source '${source}': ${jobs_dir}"
    exit 1
  fi

  mkdir -p "${output_dir}"
  scored_sources=$((scored_sources + 1))

  echo "========================================"
  echo "Scoring source=${source}"
  echo "========================================"

  cmd=(
    uv run pbench-pred-embeddings
    -jd "${jobs_dir}"
    -od "${output_dir}"
  )
  if [ ${#force_args[@]} -gt 0 ]; then
    cmd+=("${force_args[@]}")
  fi
  printf 'Executing:'
  printf ' %q' "${cmd[@]}"
  printf '\n'
  if "${cmd[@]}"; then
    :
  else
    status=$?
    score_failures=$((score_failures + 1))
    echo "Failed scoring step pbench-pred-embeddings for source=${source} with exit code ${status}"
    if [ "${continue_on_error}" != "1" ]; then
      exit "${status}"
    fi
    continue
  fi

  cmd=(
    uv run pbench-generate-matches
    -jd "${jobs_dir}"
    -od "${output_dir}"
    -m "${match_model}"
    --hf_repo "${hf_repo}"
    --hf_split "${hf_split}"
    --hf_revision "${hf_revision}"
    --prompt_path "${prompt_path}"
  )
  if [ ${#force_args[@]} -gt 0 ]; then
    cmd+=("${force_args[@]}")
  fi
  printf 'Executing:'
  printf ' %q' "${cmd[@]}"
  printf '\n'
  if "${cmd[@]}"; then
    :
  else
    status=$?
    score_failures=$((score_failures + 1))
    echo "Failed scoring step pbench-generate-matches for source=${source} with exit code ${status}"
    if [ "${continue_on_error}" != "1" ]; then
      exit "${status}"
    fi
    continue
  fi

  cmd=(
    uv run pbench-score-precision
    -jd "${jobs_dir}"
    -od "${output_dir}"
    -m "${match_model}"
    --rubric_path "${rubric_path}"
    --conversion_factors_path "${conversion_factors_path}"
    --matching_mode "${matching_mode}"
    --log_level "${log_level}"
  )
  if [ ${#force_args[@]} -gt 0 ]; then
    cmd+=("${force_args[@]}")
  fi
  printf 'Executing:'
  printf ' %q' "${cmd[@]}"
  printf '\n'
  if "${cmd[@]}"; then
    :
  else
    status=$?
    score_failures=$((score_failures + 1))
    echo "Failed scoring step pbench-score-precision for source=${source} with exit code ${status}"
    if [ "${continue_on_error}" != "1" ]; then
      exit "${status}"
    fi
    continue
  fi

  cmd=(
    uv run pbench-score-recall
    -jd "${jobs_dir}"
    -od "${output_dir}"
    -m "${match_model}"
    --rubric_path "${rubric_path}"
    --conversion_factors_path "${conversion_factors_path}"
    --matching_mode "${matching_mode}"
    --log_level "${log_level}"
  )
  if [ ${#force_args[@]} -gt 0 ]; then
    cmd+=("${force_args[@]}")
  fi
  printf 'Executing:'
  printf ' %q' "${cmd[@]}"
  printf '\n'
  if "${cmd[@]}"; then
    :
  else
    status=$?
    score_failures=$((score_failures + 1))
    echo "Failed scoring step pbench-score-recall for source=${source} with exit code ${status}"
    if [ "${continue_on_error}" != "1" ]; then
      exit "${status}"
    fi
    continue
  fi

  cmd=(
    uv run pbench-score-f1
    -jd "${jobs_dir}"
    -od "${output_dir}"
    -m "${match_model}"
    --rubric_path "${rubric_path}"
    --conversion_factors_path "${conversion_factors_path}"
    --matching_mode "${matching_mode}"
    --log_level "${log_level}"
  )
  if [ ${#force_args[@]} -gt 0 ]; then
    cmd+=("${force_args[@]}")
  fi
  printf 'Executing:'
  printf ' %q' "${cmd[@]}"
  printf '\n'
  if "${cmd[@]}"; then
    :
  else
    status=$?
    score_failures=$((score_failures + 1))
    echo "Failed scoring step pbench-score-f1 for source=${source} with exit code ${status}"
    if [ "${continue_on_error}" != "1" ]; then
      exit "${status}"
    fi
    continue
  fi

  if uv run python format_accuracy.py -jd "${jobs_dir}" | tee "${output_dir}/accuracy_summary.txt"; then
    :
  else
    status=$?
    score_failures=$((score_failures + 1))
    echo "Failed formatting accuracy summary for source=${source} with exit code ${status}"
    if [ "${continue_on_error}" != "1" ]; then
      exit "${status}"
    fi
    continue
  fi

  if uv run python format_tokens.py -jd "${jobs_dir}" | tee "${output_dir}/token_summary.txt"; then
    :
  else
    status=$?
    score_failures=$((score_failures + 1))
    echo "Failed formatting token summary for source=${source} with exit code ${status}"
    if [ "${continue_on_error}" != "1" ]; then
      exit "${status}"
    fi
    continue
  fi
done <<< "${compare_sources}"

if [ "${scored_sources}" -eq 0 ]; then
  echo "No sources were scored."
  exit 1
fi

if [ "${score_failures}" -gt 0 ]; then
  echo "Completed scoring with ${score_failures} failed step(s)."
  exit 1
fi
