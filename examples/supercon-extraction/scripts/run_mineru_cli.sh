#!/usr/bin/env bash
# Run MinerU from a dedicated tool virtualenv with compatible dependencies.
# Run from examples/supercon-extraction/, not from scripts/
# Usage: ./scripts/run_mineru_cli.sh [mineru args...]

set -euo pipefail

trap "echo ' Interrupted, exiting...'; exit 130" INT

tool_root=${MINERU_TOOL_VENV_DIR:-${XDG_CACHE_HOME:-$HOME/.cache}/sci-llm/mineru-tool}
tool_python=${MINERU_TOOL_PYTHON:-3.13}
tool_package_spec=${MINERU_TOOL_PACKAGE_SPEC:-mineru[core]==2.7.6}
tool_extra_spec=${MINERU_TOOL_EXTRA_SPEC:-huggingface-hub<1}
tool_stamp="${tool_root}/.bootstrap-complete"
mineru_bin="${tool_root}/bin/mineru"

if ! command -v uv >/dev/null 2>&1; then
  echo "uv is required to bootstrap the MinerU tool environment."
  exit 1
fi

if [ "${MINERU_TOOL_REINSTALL:-0}" = "1" ]; then
  rm -rf "${tool_root}"
fi

if [ ! -x "${mineru_bin}" ] || [ ! -f "${tool_stamp}" ]; then
  mkdir -p "$(dirname "${tool_root}")"
  rm -rf "${tool_root}"
  uv venv "${tool_root}" --python "${tool_python}"
  uv pip install --python "${tool_root}/bin/python" "${tool_package_spec}" "${tool_extra_spec}"
  date -u +"%Y-%m-%dT%H:%M:%SZ" > "${tool_stamp}"
fi

exec "${mineru_bin}" "$@"
