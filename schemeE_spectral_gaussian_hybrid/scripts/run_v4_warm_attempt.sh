#!/usr/bin/env bash
set -Eeuo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."
BASE_CONFIG="${BASE_CONFIG:-configs/v3_5090.json}"
CONFIG="configs/v4_attempt3_warm_structured.json"
mkdir -p logs reports/generated artifacts/v4

run() {
  echo "[$(date '+%F %T')] $*"
  "$@"
}

run python scripts/prepare_v4_attempts.py --base "$BASE_CONFIG"
run python scripts/inspect_architecture.py --config "$CONFIG"
run python -m unittest discover -s tests -v
run python scripts/train_hybrid.py --config "$CONFIG" --stage fold0 --resume
run python scripts/verify_completion.py --config "$CONFIG" --stage fold0 \
  --output reports/generated/v4_attempt3_completion.json
run python scripts/scan_v2_policy.py --config "$CONFIG" \
  --output reports/generated/v4_attempt3_policy.json
run python scripts/scan_output_projection.py --config "$CONFIG" \
  --policy reports/generated/v4_attempt3_policy.json \
  --output reports/generated/v4_attempt3_output_projection.json
run python scripts/select_v4_attempt.py
echo "[$(date '+%F %T')] Scheme E-v4 warm-start attempt complete"
