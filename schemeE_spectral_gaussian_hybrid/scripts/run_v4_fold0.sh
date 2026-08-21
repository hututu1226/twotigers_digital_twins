#!/usr/bin/env bash
set -Eeuo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."
BASE_CONFIG="${BASE_CONFIG:-configs/v3_5090.json}"
mkdir -p logs reports/generated artifacts/v4

run() {
  echo "[$(date '+%F %T')] $*"
  "$@"
}

run python scripts/prepare_v4_attempts.py --base "$BASE_CONFIG"
run python scripts/inspect_architecture.py --config configs/v4_attempt1_structured.json
run python -m unittest discover -s tests -v

attempt_configs=(
  configs/v4_attempt1_structured.json
  configs/v4_attempt2_decoder.json
  configs/v4_attempt3_warm_structured.json
)
for index in 1 2 3; do
  config="${attempt_configs[$((index-1))]}"
  run python scripts/train_hybrid.py --config "$config" --stage fold0 --resume
  run python scripts/verify_completion.py --config "$config" --stage fold0 \
    --output "reports/generated/v4_attempt${index}_completion.json"
  run python scripts/scan_v2_policy.py --config "$config" \
    --output "reports/generated/v4_attempt${index}_policy.json"
  run python scripts/scan_output_projection.py --config "$config" \
    --policy "reports/generated/v4_attempt${index}_policy.json" \
    --output "reports/generated/v4_attempt${index}_output_projection.json"
done

run python scripts/select_v4_attempt.py
echo "[$(date '+%F %T')] Scheme E-v4 Fold0 complete"
