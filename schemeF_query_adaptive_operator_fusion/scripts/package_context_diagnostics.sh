#!/usr/bin/env bash
set -euo pipefail

OUTPUT_DIR="${OUTPUT_DIR:-artifacts/fold0/context_diagnostics}"
STAMP="$(date +%Y%m%d_%H%M%S)"
ARCHIVE="schemeF_context_diagnostics_${STAMP}.tar.gz"

required=(
  "$OUTPUT_DIR/report.json"
  "$OUTPUT_DIR/SUMMARY.md"
)
optional=(
  artifacts/fold0/autoencoder/evaluation.json
  artifacts/fold0/autoencoder/ablation.json
  artifacts/fold0/context/evaluation.json
  artifacts/fold0/context/outage_scan.json
  artifacts/fold0/stage_gap.json
  configs/fold0_5090.json
)

for path in "${required[@]}"; do
  if [[ ! -f "$path" ]]; then
    printf 'Missing diagnostic artifact: %s\n' "$path" >&2
    exit 1
  fi
done

files=("${required[@]}")
for path in "${optional[@]}"; do
  if [[ -f "$path" ]]; then
    files+=("$path")
  fi
done
if [[ -f logs/context_diagnostics.log ]]; then
  files+=(logs/context_diagnostics.log)
fi
tar -czf "$ARCHIVE" "${files[@]}"
sha256sum "$ARCHIVE" > "${ARCHIVE}.sha256"
printf 'Created %s\n' "$ARCHIVE"
cat "${ARCHIVE}.sha256"
