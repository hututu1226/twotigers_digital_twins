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
  schemeE_spectral_gaussian_hybrid/configs/final_selected.json
  schemeE_spectral_gaussian_hybrid/artifacts/fold0/spectral_teacher/oof_report.json
  schemeE_spectral_gaussian_hybrid/artifacts/fold0/hybrid/best.pt
  schemeE_spectral_gaussian_hybrid/artifacts/fold0/hybrid/summary.json
  schemeE_spectral_gaussian_hybrid/artifacts/final/spectral_teacher/final_report.json
  schemeE_spectral_gaussian_hybrid/artifacts/final/spectral_teacher/model.pkl
  schemeE_spectral_gaussian_hybrid/artifacts/final/hybrid/best.pt
  schemeE_spectral_gaussian_hybrid/artifacts/final/hybrid/summary.json
  schemeE_spectral_gaussian_hybrid/outputs/final/Round2_Test_Channel.npy
  schemeE_spectral_gaussian_hybrid/reports/generated
)
git add -f -- "${paths[@]}"
git status --short -- "${paths[@]}"
if git diff --cached --quiet; then
  echo "Scheme E artifacts are already committed; nothing new to commit."
else
  git commit -m "Add Scheme E formal training artifacts"
fi
git push origin "$branch"
