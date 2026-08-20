#!/usr/bin/env bash
set -Eeuo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."
mkdir -p logs reports/generated
LIMIT="${LEGACY_SCAN_LIMIT:-192}"
ROOT_ARGS=()
if [[ -n "${LEGACY_RUN_ROOT:-}" ]]; then
  ROOT_ARGS=(--legacy-root "$LEGACY_RUN_ROOT")
fi

set -o pipefail
python scripts/scan_scheme_d.py "${ROOT_ARGS[@]}" --limit "$LIMIT" \
  --output reports/generated/scheme_d_scan.json \
  2>&1 | tee logs/scheme_d_scan.log
python scripts/scan_scheme_e.py "${ROOT_ARGS[@]}" --limit "$LIMIT" \
  --output reports/generated/scheme_e_scan.json \
  2>&1 | tee logs/scheme_e_scan.log

python scripts/select_scheme_f_revision.py \
  --scheme-d reports/generated/scheme_d_scan.json \
  --scheme-e reports/generated/scheme_e_scan.json \
  --base configs/fold0_5090.json \
  --output configs/fold0_selected.json \
  --report reports/generated/legacy_scan_decision.json
