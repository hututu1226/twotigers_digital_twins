#!/usr/bin/env bash
set -Eeuo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."
timestamp="$(date '+%Y%m%d_%H%M%S')"
archive="schemeG_results_${timestamp}.tar.gz"
required=(
  artifacts/fold0/context/best.pt
  artifacts/fold0/context/evaluation.json
  artifacts/final/context/final.pt
  outputs/final/Round2_Test_Channel.npy
  reports/generated/schemeG_fold0_experiment_report.json
  reports/generated/schemeG_final_experiment_report.json
  configs/final_selected.json
)
for path in "${required[@]}"; do
  [[ -s "$path" ]] || { echo "Missing package input: $path" >&2; exit 1; }
done
tar -czf "$archive" \
  configs/fold0_5090.json configs/final_selected.json docs README.md pyproject.toml requirements.txt \
  artifacts/fold0/context/best.pt artifacts/fold0/context/evaluation.json \
  artifacts/fold0/context/outage_scan.json artifacts/fold0/context/summary.json \
  artifacts/final/context/final.pt artifacts/final/context/summary.json \
  outputs/final/Round2_Test_Channel.npy outputs/final/Round2_Test_Channel.json \
  reports/generated
sha256sum "$archive" > "${archive}.sha256"
tar -tzf "$archive" >/dev/null
sha256sum -c "${archive}.sha256"
echo "$archive"
