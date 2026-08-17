#!/usr/bin/env bash
set -euo pipefail

CONFIG="${CONFIG:-configs/fold0_5090.json}"
mkdir -p artifacts/capacity logs

python scripts/overfit_autoencoder.py \
  --config "$CONFIG" \
  --samples 1 \
  --steps 200 \
  --batch-size 1 \
  --learning-rate 0.001 \
  --minimum-score 0.90 \
  --report-interval 25 \
  --output artifacts/capacity/one_sample.json

python scripts/overfit_autoencoder.py \
  --config "$CONFIG" \
  --samples 32 \
  --steps 1200 \
  --batch-size 8 \
  --learning-rate 0.001 \
  --minimum-score 0.85 \
  --report-interval 100 \
  --output artifacts/capacity/thirty_two_samples.json
