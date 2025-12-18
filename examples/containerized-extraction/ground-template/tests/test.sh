#!/bin/bash
set -euo pipefail

python /tests/check_prediction.py 2>&1 | tee /logs/verifier/log.txt
