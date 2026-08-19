#!/usr/bin/env bash
set -Eeuo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/../.."
branch="${BRANCH:-0819}"
git switch "$branch"
git pull --ff-only origin "$branch"
git lfs install
git lfs track '*.pt' '*.npy' '*.npz' '*.pkl' '*.tar.gz'
paths=(
  .gitattributes
  schemeD_transport_residual_context/configs/final_selected.json
  schemeD_transport_residual_context/artifacts/fold0/context/best.pt
  schemeD_transport_residual_context/artifacts/fold0/context/evaluation.json
  schemeD_transport_residual_context/artifacts/fold0/context/outage_scan.json
  schemeD_transport_residual_context/artifacts/fold0/context/summary.json
  schemeD_transport_residual_context/artifacts/final/context/final.pt
  schemeD_transport_residual_context/artifacts/final/context/summary.json
  schemeD_transport_residual_context/outputs/final/Round2_Test_Channel.npy
  schemeD_transport_residual_context/reports/generated
)
git add -f -- "${paths[@]}"
git status --short -- "${paths[@]}"
if git diff --cached --quiet; then
  echo "Scheme D artifacts are already committed; nothing new to commit."
else
  git commit -m "Add Scheme D formal training artifacts"
fi
git push origin "$branch"
