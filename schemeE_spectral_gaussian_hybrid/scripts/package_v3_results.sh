#!/usr/bin/env bash
set -Eeuo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."
stamp="$(date '+%Y%m%d_%H%M%S')"
archive="schemeE_v3_results_${stamp}.tar.gz"
tar -czf "$archive" \
  configs/v3_5090.json \
  configs/v3_attempt1_conservative.json \
  configs/v3_attempt2_flexible.json \
  configs/v3_attempt3_decoder.json \
  configs/v3_fold_best.json \
  configs/v3_final_selected.json \
  artifacts/v3/fold0/carrier_fit.json \
  artifacts/v3/fold0_attempt1/hybrid/summary.json \
  artifacts/v3/fold0_attempt1/hybrid/best.pt \
  artifacts/v3/fold0_attempt1/hybrid/history.jsonl \
  artifacts/v3/fold0_attempt2/hybrid/summary.json \
  artifacts/v3/fold0_attempt2/hybrid/best.pt \
  artifacts/v3/fold0_attempt2/hybrid/history.jsonl \
  artifacts/v3/fold0_attempt3/hybrid/summary.json \
  artifacts/v3/fold0_attempt3/hybrid/best.pt \
  artifacts/v3/fold0_attempt3/hybrid/history.jsonl \
  artifacts/v3/final/carrier_fit.json \
  artifacts/v3/final/hybrid/best.pt \
  artifacts/v3/final/hybrid/summary.json \
  outputs/v3/Round2_Test_Channel.npy \
  reports/generated/v3_attempt1_policy.json \
  reports/generated/v3_attempt2_policy.json \
  reports/generated/v3_attempt3_policy.json \
  reports/generated/v3_attempt_selection.json \
  reports/generated/v3_final_inference.json \
  reports/generated/v3_final_output_check.json \
  reports/generated/schemeE_v3_final_experiment_report.json \
  reports/generated/schemeE_v3_final_EXPERIMENT_REPORT.md
sha256sum "$archive" > "$archive.sha256"
echo "$archive"
