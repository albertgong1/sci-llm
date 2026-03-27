#!/usr/bin/env bash
# Build local Harbor task registries for the PDF vs MinerU comparison.
# Run from examples/supercon-extraction/, not from scripts/
# Usage: ./scripts/build_harbor_compare_sources.sh [extra args...]

set -euo pipefail

trap "echo ' Interrupted, exiting...'; exit 130" INT

cmd_args=("$@")

compare_sources=${HARBOR_COMPARE_SOURCES:-$'pdf\nmineru'}
workspace_root=${SUPERCON_WORKSPACE:-.}
template_name=${SUPERCON_TEMPLATE:-targeted-stoichiometric-template}
pdf_dir=${SUPERCON_PDF_DIR:-data-official/Paper_DB}
registry_path=${SUPERCON_REGISTRY_PATH:-registry_data.json}
max_num_papers=${SUPERCON_MAX_NUM_PAPERS:-200}
gt_hf_repo=${SUPERCON_GT_HF_REPO:-kilian-group/supercon-extraction}
gt_hf_split=${SUPERCON_GT_HF_SPLIT:-full}
gt_hf_revision=${SUPERCON_GT_HF_REVISION:-v0.2.1}
pdf_output_dir=${SUPERCON_PDF_OUTPUT_DIR:-out-pdf-harbor-200}
mineru_output_dir=${SUPERCON_MINERU_OUTPUT_DIR:-out-mineru-harbor-200}
mineru_cache_dir=${SUPERCON_MINERU_CACHE_DIR:-${mineru_output_dir}/mineru-cache}
mineru_binary=${SUPERCON_MINERU_BINARY:-./scripts/run_mineru_cli.sh}
mineru_backend=${SUPERCON_MINERU_BACKEND:-pipeline}
mineru_method=${SUPERCON_MINERU_METHOD:-}

if [ ! -d "${pdf_dir}" ]; then
  echo "Paper directory not found: ${pdf_dir}"
  exit 1
fi

if [ ! -f "${registry_path}" ]; then
  echo "Registry file not found: ${registry_path}"
  exit 1
fi

while IFS= read -r source; do
  if [ -z "${source}" ]; then
    continue
  fi

  cmd=(
    uv run python ../../src/harbor-task-gen/prepare_harbor_tasks.py
    --workspace "${workspace_root}"
    --template "${template_name}"
    --pdf-dir "${pdf_dir}"
    --gt-hf-repo "${gt_hf_repo}"
    --gt-hf-split "${gt_hf_split}"
    --gt-hf-revision "${gt_hf_revision}"
    --harbor-task-ordering-registry-path "${registry_path}"
    --max-num-papers "${max_num_papers}"
  )

  case "${source}" in
    pdf)
      cmd+=(--output-dir "${pdf_output_dir}" --paper-source pdf)
      ;;
    mineru)
      if [ ! -x "${mineru_binary}" ] && ! command -v "${mineru_binary}" >/dev/null 2>&1; then
        echo "MinerU binary not found or not executable: ${mineru_binary}"
        exit 1
      fi
      cmd+=(
        --output-dir "${mineru_output_dir}"
        --paper-source mineru
        --mineru-binary "${mineru_binary}"
        --mineru-cache-dir "${mineru_cache_dir}"
        --mineru-backend "${mineru_backend}"
      )
      if [ -n "${mineru_method}" ]; then
        cmd+=(--mineru-method "${mineru_method}")
      fi
      ;;
    *)
      echo "Unknown source '${source}'. Expected one of: pdf, mineru"
      exit 1
      ;;
  esac

  if [ ${#cmd_args[@]} -gt 0 ]; then
    cmd+=("${cmd_args[@]}")
  fi

  echo "========================================"
  echo "Building source=${source}"
  echo "========================================"
  printf 'Executing:'
  printf ' %q' "${cmd[@]}"
  printf '\n'
  "${cmd[@]}"
done <<< "${compare_sources}"
