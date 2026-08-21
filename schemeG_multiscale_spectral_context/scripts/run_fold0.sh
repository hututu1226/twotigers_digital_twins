#!/usr/bin/env bash
set -Eeuo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."
CONFIG="${CONFIG:-configs/fold0_5090.json}"

[[ -s artifacts/preprocessed_scheme_g/manifest.json ]] || \
  python scripts/preprocess.py --config "$CONFIG"
[[ -s artifacts/fold0/encoded.npz ]] || \
  python scripts/encode_latents.py --config "$CONFIG"

if [[ -s artifacts/fold0/context/best.pt && -s artifacts/fold0/context/summary.json ]]; then
  echo "Existing Scheme G Fold0 training summary found; reusing it."
elif [[ "${RESUME:-1}" == "1" && -s artifacts/fold0/context/last.pt ]]; then
  python scripts/train_context.py --config "$CONFIG" --resume
else
  python scripts/train_context.py --config "$CONFIG"
fi

python scripts/scan_outage.py \
  --config "$CONFIG" \
  --checkpoint artifacts/fold0/context/best.pt \
  --output artifacts/fold0/context/outage_scan.json
threshold="$(python -c "import json; print(json.load(open('artifacts/fold0/context/outage_scan.json'))['best_threshold'])")"
python scripts/evaluate.py \
  --config "$CONFIG" --stage context \
  --checkpoint artifacts/fold0/context/best.pt \
  --outage-threshold "$threshold" \
  --output artifacts/fold0/context/evaluation.json
python scripts/verify_completion.py --config "$CONFIG" --stage fold0 \
  --output reports/generated/fold0_completion.json
python scripts/report_experiment.py --stage fold0
