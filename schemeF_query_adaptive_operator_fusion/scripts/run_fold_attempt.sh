#!/usr/bin/env bash
set -Eeuo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."
CONFIG="${CONFIG:-configs/fold0_selected.json}"
CONTEXT_DIR="$(python -c 'import sys; from scheme_f.config import load_config; print(load_config(sys.argv[1])["context"]["output_dir"])' "$CONFIG")"
CHECKPOINT="$CONTEXT_DIR/best.pt"
mkdir -p "$CONTEXT_DIR" logs reports/generated

if [[ ! -s artifacts/preprocessed_scheme_f/manifest.json ]] || ! \
  python -c 'import numpy as np; s=np.load("artifacts/preprocessed_scheme_f/metadata.npz"); assert "train_geometry_features" in s.files and s["train_geometry_features"].shape[1] == 71'; then
  python scripts/preprocess.py --config "$CONFIG" --force
fi
[[ -s artifacts/fold0/encoded.npz ]] || \
  python scripts/encode_latents.py --config "$CONFIG"

if [[ -s "$CHECKPOINT" && -s "$CONTEXT_DIR/summary.json" ]]; then
  echo "Existing Scheme F attempt found at $CONTEXT_DIR; reusing it."
elif [[ "${RESUME:-1}" == "1" && -s "$CONTEXT_DIR/last.pt" ]]; then
  python scripts/train_context.py --config "$CONFIG" --resume
else
  python scripts/train_context.py --config "$CONFIG"
fi

python scripts/scan_outage.py \
  --config "$CONFIG" \
  --checkpoint "$CHECKPOINT" \
  --output "$CONTEXT_DIR/outage_scan.json"
threshold="$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["best_threshold"])' "$CONTEXT_DIR/outage_scan.json")"
soft_strength="$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["best_soft_strength"])' "$CONTEXT_DIR/outage_scan.json")"
prior_alpha="$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["best_spectral_prior_alpha"])' "$CONTEXT_DIR/outage_scan.json")"
python scripts/evaluate.py \
  --config "$CONFIG" --stage context \
  --checkpoint "$CHECKPOINT" \
  --outage-threshold "$threshold" \
  --soft-outage-strength "$soft_strength" \
  --spectral-prior-alpha "$prior_alpha" \
  --output "$CONTEXT_DIR/evaluation.json"
