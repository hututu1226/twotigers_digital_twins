#!/usr/bin/env bash
set -Eeuo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."
stamp="$(date '+%Y%m%d_%H%M%S')"
archive="schemeE_v2_results_${stamp}.tar.gz"
tar -czf "$archive" \
  configs/v2_5090.json \
  configs/v2_attempt1_safe.json \
  configs/v2_attempt2_reference.json \
  configs/v2_attempt3_decoder.json \
  configs/v2_fold_best.json \
  configs/v2_final_selected.json \
  artifacts/v2/fold0/spectral_teacher/strict_report.json \
  artifacts/v2/fold0_attempt1/hybrid/summary.json \
  artifacts/v2/fold0_attempt1/hybrid/best.pt \
  artifacts/v2/fold0_attempt1/hybrid/history.jsonl \
  artifacts/v2/fold0_attempt2/hybrid/summary.json \
  artifacts/v2/fold0_attempt2/hybrid/best.pt \
  artifacts/v2/fold0_attempt2/hybrid/history.jsonl \
  artifacts/v2/fold0_attempt3/hybrid/summary.json \
  artifacts/v2/fold0_attempt3/hybrid/best.pt \
  artifacts/v2/fold0_attempt3/hybrid/history.jsonl \
  artifacts/v2/final/spectral_teacher/oof_report.json \
  artifacts/v2/final/spectral_teacher/final_report.json \
  artifacts/v2/final/spectral_teacher/model.pkl \
  artifacts/v2/final/hybrid/best.pt \
  artifacts/v2/final/hybrid/summary.json \
  outputs/v2/Round2_Test_Channel.npy \
  reports/generated/v2_attempt1_policy.json \
  reports/generated/v2_attempt2_policy.json \
  reports/generated/v2_attempt3_policy.json \
  reports/generated/v2_attempt_selection.json \
  reports/generated/v2_final_inference.json \
  reports/generated/v2_final_output_check.json \
  reports/generated/schemeE_v2_final_experiment_report.json \
  reports/generated/schemeE_v2_final_EXPERIMENT_REPORT.md
sha256sum "$archive" > "$archive.sha256"
echo "$archive"
