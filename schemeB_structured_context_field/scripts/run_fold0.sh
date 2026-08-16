#!/usr/bin/env bash
set -euo pipefail

CONFIG="${CONFIG:-configs/fold0_4090.json}"

if [[ ! -f artifacts/preprocessed_dual/manifest.json ]]; then
  python scripts/preprocess.py --config "$CONFIG"
fi

AE_ARGS=()
if [[ "${RESUME:-0}" == "1" && -f artifacts/fold0/autoencoder/final.pt ]]; then
  printf 'AE final checkpoint exists; skipping completed stage.\n'
elif [[ "${RESUME:-0}" == "1" && -f artifacts/fold0/autoencoder/last.pt ]]; then
  AE_ARGS+=(--resume)
  python scripts/train_autoencoder.py --config "$CONFIG" "${AE_ARGS[@]}"
else
  python scripts/train_autoencoder.py --config "$CONFIG"
fi

python scripts/encode_latents.py --config "$CONFIG"

CONTEXT_ARGS=()
if [[ "${RESUME:-0}" == "1" && -f artifacts/fold0/context/final.pt ]]; then
  printf 'Context final checkpoint exists; skipping completed stage.\n'
elif [[ "${RESUME:-0}" == "1" && -f artifacts/fold0/context/last.pt ]]; then
  CONTEXT_ARGS+=(--resume)
  python scripts/train_context.py --config "$CONFIG" "${CONTEXT_ARGS[@]}"
else
  python scripts/train_context.py --config "$CONFIG"
fi

JOINT_ARGS=()
if [[ "${RESUME:-0}" == "1" && -f artifacts/fold0/joint/final.pt ]]; then
  printf 'Joint final checkpoint exists; skipping completed stage.\n'
elif [[ "${RESUME:-0}" == "1" && -f artifacts/fold0/joint/last.pt ]]; then
  JOINT_ARGS+=(--resume)
  python scripts/finetune_joint.py --config "$CONFIG" "${JOINT_ARGS[@]}"
else
  python scripts/finetune_joint.py --config "$CONFIG"
fi

python scripts/evaluate.py --config "$CONFIG" --stage autoencoder --checkpoint artifacts/fold0/autoencoder/best.pt --output artifacts/fold0/autoencoder/evaluation.json
python scripts/evaluate.py --config "$CONFIG" --stage context --checkpoint artifacts/fold0/context/best.pt --output artifacts/fold0/context/evaluation.json
python scripts/evaluate.py --config "$CONFIG" --stage joint --checkpoint artifacts/fold0/joint/best.pt --output artifacts/fold0/joint/evaluation.json
python scripts/report_stage_gap.py
python scripts/scan_outage.py \
  --config "$CONFIG" \
  --checkpoint artifacts/fold0/joint/best.pt \
  --output artifacts/fold0/joint/outage_scan.json

THRESHOLD="$(python -c "import json; print(json.load(open('artifacts/fold0/joint/outage_scan.json'))['best_threshold'])")"
python scripts/infer.py --config "$CONFIG" --outage-threshold "$THRESHOLD"
python scripts/inspect_output.py outputs/fold0/Round2_Test_Channel.npy
