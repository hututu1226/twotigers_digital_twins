#!/usr/bin/env bash
set -Eeuo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."
timestamp="$(date '+%Y%m%d_%H%M%S')"
archive="schemeE_results_${timestamp}.tar.gz"

required=(
  artifacts/fold0/spectral_teacher/oof_report.json
  artifacts/fold0/hybrid/best.pt
  artifacts/fold0/hybrid/summary.json
  artifacts/final/spectral_teacher/final_report.json
  artifacts/final/spectral_teacher/model.pkl
  artifacts/final/hybrid/best.pt
  artifacts/final/hybrid/summary.json
  outputs/final/Round2_Test_Channel.npy
  reports/generated/schemeE_fold0_experiment_report.json
  reports/generated/schemeE_final_experiment_report.json
  configs/final_selected.json
)
for path in "${required[@]}"; do
  [[ -s "$path" ]] || { echo "Missing package input: $path" >&2; exit 1; }
done

tar -czf "$archive" \
  configs/fold0_5090.json configs/final_selected.json \
  docs README.md pyproject.toml requirements.txt \
  artifacts/fold0/spectral_teacher/oof_report.json \
  artifacts/fold0/hybrid/best.pt artifacts/fold0/hybrid/summary.json \
  artifacts/fold0/hybrid/history.jsonl \
  artifacts/final/spectral_teacher/final_report.json \
  artifacts/final/spectral_teacher/model.pkl \
  artifacts/final/hybrid/best.pt artifacts/final/hybrid/summary.json \
  artifacts/final/hybrid/history.jsonl \
  outputs/final/Round2_Test_Channel.npy reports/generated
sha256sum "$archive" > "${archive}.sha256"
tar -tzf "$archive" >/dev/null
sha256sum -c "${archive}.sha256"
echo "$archive"
