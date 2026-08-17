#!/usr/bin/env bash
set -euo pipefail

CONFIG="${CONFIG:-configs/fold0_5090.json}"

if [[ ! -f artifacts/preprocessed_scheme_c/manifest.json ]]; then
  python scripts/preprocess.py --config "$CONFIG"
fi

python scripts/ensure_run_compatibility.py --config "$CONFIG" --run fold0
python scripts/verify_completion.py --stage capacity

if [[ "${RESUME:-0}" == "1" && -f artifacts/fold0/autoencoder/last.pt ]]; then
  python scripts/train_autoencoder.py --config "$CONFIG" --resume
else
  python scripts/train_autoencoder.py --config "$CONFIG"
fi

python scripts/evaluate.py \
  --config "$CONFIG" \
  --stage autoencoder \
  --checkpoint artifacts/fold0/autoencoder/best.pt \
  --output artifacts/fold0/autoencoder/evaluation.json
python scripts/evaluate_ae_ablation.py \
  --config "$CONFIG" \
  --checkpoint artifacts/fold0/autoencoder/best.pt \
  --output artifacts/fold0/autoencoder/ablation.json
python scripts/check_ae_gate.py \
  --config "$CONFIG" \
  --evaluation artifacts/fold0/autoencoder/evaluation.json \
  --ablation artifacts/fold0/autoencoder/ablation.json \
  --output artifacts/fold0/autoencoder/quality_gate.json

python scripts/encode_latents.py --config "$CONFIG"

if [[ "${RESUME:-0}" == "1" && -f artifacts/fold0/context/last.pt ]]; then
  python scripts/train_context.py --config "$CONFIG" --resume
else
  python scripts/train_context.py --config "$CONFIG"
fi

if [[ "${RESUME:-0}" == "1" && -f artifacts/fold0/joint/last.pt ]]; then
  python scripts/finetune_joint.py --config "$CONFIG" --resume
else
  python scripts/finetune_joint.py --config "$CONFIG"
fi

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
