#!/usr/bin/env bash
set -euo pipefail

CONFIG="configs/fold0_4090.json"
AE_ARGS=()
SPATIAL_ARGS=()
SKIP_AE=0
SKIP_SPATIAL=0
if [[ "${RESUME:-0}" == "1" && -f artifacts/fold0/autoencoder/final.pt ]]; then
  SKIP_AE=1
elif [[ "${RESUME:-0}" == "1" && -f artifacts/fold0/autoencoder/last.pt ]]; then
  AE_ARGS+=(--resume)
fi
if [[ "${RESUME:-0}" == "1" && -f artifacts/fold0/spatial/final.pt ]]; then
  SKIP_SPATIAL=1
elif [[ "${RESUME:-0}" == "1" && -f artifacts/fold0/spatial/last.pt ]]; then
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
python scripts/analyze_latents.py --config "$CONFIG" --output artifacts/fold0/latent_diagnostics.json
if [[ "$SKIP_SPATIAL" == "0" ]]; then
  python scripts/train_spatial.py --config "$CONFIG" "${SPATIAL_ARGS[@]}"
else
  printf 'Spatial final checkpoint exists; skipping completed stage.\n'
fi
python scripts/evaluate.py --config "$CONFIG" --stage spatial --checkpoint artifacts/fold0/spatial/best.pt
python scripts/scan_outage.py --config "$CONFIG" --checkpoint artifacts/fold0/spatial/best.pt --output artifacts/fold0/spatial/outage_scan.json
THRESHOLD="$(python -c "import json; print(json.load(open('artifacts/fold0/spatial/outage_scan.json'))['best_threshold'])")"
python scripts/infer.py --config "$CONFIG" --outage-threshold "$THRESHOLD"
python scripts/inspect_output.py outputs/fold0/Round2_Test_Channel.npy
