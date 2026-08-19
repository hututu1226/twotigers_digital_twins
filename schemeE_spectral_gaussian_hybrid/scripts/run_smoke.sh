#!/usr/bin/env bash
set -Eeuo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."
CONFIG="configs/smoke.json"

python -m unittest discover -s tests -v
python scripts/make_smoke_data.py --output artifacts/smoke_data --force
python scripts/preprocess.py --config "$CONFIG" --force
python scripts/extract_spectral_targets.py --config "$CONFIG" --force
python scripts/train_spectral_teacher.py --config "$CONFIG" --mode oof
python scripts/train_hybrid.py --config "$CONFIG" --stage fold0
python scripts/train_spectral_teacher.py --config "$CONFIG" --mode final
python scripts/train_hybrid.py --config "$CONFIG" --stage final
python scripts/infer.py --config "$CONFIG"
python scripts/inspect_output.py outputs/smoke/Round2_Test_Channel.npy \
  --samples 2 --report reports/generated/smoke_output_check.json
python scripts/verify_completion.py --config "$CONFIG" --stage smoke \
  --output reports/generated/smoke_completion.json

echo "Scheme E smoke test PASS"
