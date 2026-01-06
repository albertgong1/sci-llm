#!/bin/bash
set -euo pipefail

# cd to the location of this file on disk
cd "$(dirname -- "$0")"

cleanup() {
    unset GOOGLE_RECAPTCHA_KEY
}

# handle getting google recaptcha if it exists
read -s -p "Enter Google Recaptcha Key (optional): " input
echo
if [[ -n "$input" ]]; then
    export GOOGLE_RECAPTCHA_KEY="$input"
fi

# run pytest
PYTHONPATH=. pytest