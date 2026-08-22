#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"

CONFIG="configs/l2_001_adaptive_hybrid.json"
OUTPUT_DIR="artifacts/scheme_e_065/l2_001_adaptive_hybrid/hybrid"
REPORT="reports/generated/l2_001_adaptive_hybrid_breakdown.json"
STATUS="logs/l2_001_adaptive_hybrid_status.txt"
BACKUP_ROOT="${BACKUP_ROOT:-/root/autodl-fs/scheme_e_065_l2_001}"
mkdir -p logs reports/generated "$BACKUP_ROOT"

failed() {
  code=$?
  printf 'status=FAILED\nexit_code=%s\ntime=%s\n' "$code" "$(date -Iseconds)" > "$STATUS"
  exit "$code"
}
trap failed ERR

test -s artifacts/v6/fold0/adaptive_local_bank_priors.npz
test "$(stat -c '%s' artifacts/v4/fold0_attempt1/hybrid/best.pt)" -gt 1000000
test -s reports/generated/v4_attempt1_policy.json

python scripts/prepare_l2_adaptive_hybrid_config.py --output "$CONFIG"
python scripts/inspect_architecture.py --config "$CONFIG"
python -m unittest tests.test_core
python scripts/train_hybrid.py --config "$CONFIG" --stage fold0
python scripts/verify_completion.py --config "$CONFIG" --stage fold0 \
  --output reports/generated/l2_001_adaptive_hybrid_completion.json
python scripts/evaluate_breakdown.py --config "$CONFIG" \
  --policy reports/generated/v4_attempt1_policy.json \
  --output "$REPORT"

mkdir -p "$BACKUP_ROOT/artifacts" "$BACKUP_ROOT/reports" "$BACKUP_ROOT/logs"
cp -a "$OUTPUT_DIR" "$BACKUP_ROOT/artifacts/"
cp -a "$CONFIG" "$REPORT" \
  reports/generated/l2_001_adaptive_hybrid_completion.json "$BACKUP_ROOT/reports/"
cp -a logs/scheme_e_065_l2_001_adaptive_hybrid.log "$BACKUP_ROOT/logs/" 2>/dev/null || true
printf 'status=SUCCESS\ntime=%s\nreport=%s\nbackup=%s\n' \
  "$(date -Iseconds)" "$REPORT" "$BACKUP_ROOT" > "$STATUS"
