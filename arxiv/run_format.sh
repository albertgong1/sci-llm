#!/bin/bash
set -euo pipefail

# cd to the location of this file on disk
cd "$(dirname -- "$0")"

# run ruff
ruff format src/ tests/