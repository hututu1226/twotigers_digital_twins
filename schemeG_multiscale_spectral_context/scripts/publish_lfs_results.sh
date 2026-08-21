#!/usr/bin/env bash
set -Eeuo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/../.."
branch="${BRANCH:-0821_schemeG}"
git switch "$branch"
git pull --ff-only origin "$branch"
git lfs install
git lfs track '*.pt' '*.npy' '*.npz' '*.pkl' '*.tar.gz'
paths=(
  .gitattributes
  schemeG_multiscale_spectral_context/configs/final_selected.json
  schemeG_multiscale_spectral_context/artifacts/fold0/context/best.pt
  schemeG_multiscale_spectral_context/artifacts/fold0/context/evaluation.json
  schemeG_multiscale_spectral_context/artifacts/fold0/context/outage_scan.json
  schemeG_multiscale_spectral_context/artifacts/fold0/context/summary.json
  schemeG_multiscale_spectral_context/artifacts/final/context/final.pt
  schemeG_multiscale_spectral_context/artifacts/final/context/summary.json
  schemeG_multiscale_spectral_context/outputs/final/Round2_Test_Channel.npy
  schemeG_multiscale_spectral_context/reports/generated
)
git add -f -- "${paths[@]}"
git status --short -- "${paths[@]}"
if git diff --cached --quiet; then
  echo "Scheme G artifacts are already committed; nothing new to commit."
else
  git commit -m "Add Scheme G formal training artifacts"
fi
git push origin "$branch"
