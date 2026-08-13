#!/usr/bin/env bash
set -euo pipefail

CONFIG="${CONFIG:-configs/final_4090.json}"
AE_ARGS=()
SPATIAL_ARGS=()
SKIP_AE=0
SKIP_SPATIAL=0
if [[ "${RESUME:-0}" == "1" && -f artifacts/final/autoencoder/final.pt ]]; then
  SKIP_AE=1
elif [[ "${RESUME:-0}" == "1" && -f artifacts/final/autoencoder/last.pt ]]; then
  AE_ARGS+=(--resume)
fi
if [[ "${RESUME:-0}" == "1" && -f artifacts/final/spatial/final.pt ]]; then
  SKIP_SPATIAL=1
elif [[ "${RESUME:-0}" == "1" && -f artifacts/final/spatial/last.pt ]]; then
  SPATIAL_ARGS+=(--resume)
fi

if [[ ! -f artifacts/preprocessed_1m/manifest.json ]]; then
  python scripts/preprocess.py --config "$CONFIG"
fi
if [[ "$SKIP_AE" == "0" ]]; then
  python scripts/train_autoencoder.py --config "$CONFIG" "${AE_ARGS[@]}"
else
  printf 'AE final checkpoint exists; skipping completed stage.\n'
fi
python scripts/encode_latents.py --config "$CONFIG"
if [[ "$SKIP_SPATIAL" == "0" ]]; then
  python scripts/train_spatial.py --config "$CONFIG" "${SPATIAL_ARGS[@]}"
else
  printf 'Spatial final checkpoint exists; skipping completed stage.\n'
fi
python scripts/infer.py --config "$CONFIG"
python scripts/inspect_output.py outputs/final/Round2_Test_Channel.npy
