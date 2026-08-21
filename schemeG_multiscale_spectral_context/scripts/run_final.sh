#!/usr/bin/env bash
set -Eeuo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."
CONFIG="${CONFIG:-configs/final_selected.json}"

[[ -s artifacts/preprocessed_scheme_g/manifest.json ]] || \
  python scripts/preprocess.py --config "$CONFIG"
[[ -s artifacts/final/encoded.npz ]] || \
  python scripts/encode_latents.py --config "$CONFIG"

if [[ -s artifacts/final/context/final.pt && -s artifacts/final/context/summary.json ]]; then
  echo "Existing Scheme G final training summary found; reusing it."
elif [[ "${RESUME:-1}" == "1" && -s artifacts/final/context/last.pt ]]; then
  python scripts/train_context.py --config "$CONFIG" --resume
else
  python scripts/train_context.py --config "$CONFIG"
fi

python scripts/infer.py --config "$CONFIG"
python scripts/inspect_output.py outputs/final/Round2_Test_Channel.npy --expected-count 500
python scripts/verify_completion.py --config "$CONFIG" --stage final \
  --output reports/generated/final_completion.json
python scripts/report_experiment.py --stage final
