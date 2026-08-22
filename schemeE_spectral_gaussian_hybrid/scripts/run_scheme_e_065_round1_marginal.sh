#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"

mkdir -p logs ../research/scheme_e_065

python -u scripts/diagnose_round1_marginal_expert.py \
  --config configs/v4_fold_best.json \
  --policy reports/generated/v4_attempt1_policy.json \
  --projection-report reports/generated/v4_attempt1_output_projection.json \
  --prior-wave-number -140.33 \
  --minimum-quality 0.5 \
  --neighbors 24 \
  --distance-power 2.0 \
  --iterations 8 \
  --expected-baseline 0.631581059599534 \
  --report ../research/scheme_e_065/L0_015_ROUND1_MARGINAL_EXPERT.json \
  --device cuda
