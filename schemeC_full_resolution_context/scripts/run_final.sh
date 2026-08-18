#!/usr/bin/env bash
set -euo pipefail

if [[ -f configs/final_selected.json ]]; then
  DEFAULT_CONFIG="configs/final_selected.json"
else
  DEFAULT_CONFIG="configs/final_5090.json"
fi
CONFIG="${CONFIG:-$DEFAULT_CONFIG}"

if [[ ! -f artifacts/preprocessed_scheme_c/manifest.json ]]; then
  python scripts/preprocess.py --config "$CONFIG"
fi

python scripts/ensure_run_compatibility.py --config "$CONFIG" --run final

if [[ "${RESUME:-0}" == "1" && -f artifacts/final/autoencoder/last.pt ]]; then
  python scripts/train_autoencoder.py --config "$CONFIG" --resume
else
  python scripts/train_autoencoder.py --config "$CONFIG"
fi

python scripts/analyze_context_masks.py \
  --config "$CONFIG" \
  --output artifacts/final/context_mask_report.json
python scripts/encode_latents.py --config "$CONFIG"
python scripts/ensure_context_compatibility.py --config "$CONFIG"

if [[ "${RESUME:-0}" == "1" && -f artifacts/final/context/last.pt ]]; then
  python scripts/train_context.py --config "$CONFIG" --resume
else
  python scripts/train_context.py --config "$CONFIG"
fi

python scripts/infer.py --config "$CONFIG"
python scripts/inspect_output.py outputs/final/Round2_Test_Channel.npy
