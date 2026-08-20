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
  schemeF_query_adaptive_operator_fusion/configs/final_selected.json
  schemeF_query_adaptive_operator_fusion/artifacts/fold0/context/best.pt
  schemeF_query_adaptive_operator_fusion/artifacts/fold0/context/evaluation.json
  schemeF_query_adaptive_operator_fusion/artifacts/fold0/context/outage_scan.json
  schemeF_query_adaptive_operator_fusion/artifacts/fold0/context/summary.json
  schemeF_query_adaptive_operator_fusion/artifacts/final/context/final.pt
  schemeF_query_adaptive_operator_fusion/artifacts/final/context/summary.json
  schemeF_query_adaptive_operator_fusion/outputs/final/Round2_Test_Channel.npy
  schemeF_query_adaptive_operator_fusion/reports/generated
)
git add -f -- "${paths[@]}"
git status --short -- "${paths[@]}"
if git diff --cached --quiet; then
  echo "Scheme F artifacts are already committed; nothing new to commit."
else
  git commit -m "Add Scheme F formal training artifacts"
fi
git push origin "$branch"
