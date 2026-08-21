#!/usr/bin/env bash
set -Eeuo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."
python scripts/make_smoke_data.py --output artifacts/smoke_data --force
python scripts/preprocess.py --config configs/smoke.json --force
python scripts/extract_spectral_targets.py --config configs/smoke.json --force
python scripts/prepare_v2_smoke_config.py
CONFIG="configs/v2_smoke_generated.json"
rm -rf artifacts/smoke_v2/hybrid artifacts/smoke_v2/final_hybrid \
  artifacts/smoke_v2/spectral_teacher outputs/smoke_v2
python scripts/train_spectral_teacher.py --config "$CONFIG" --mode oof
python scripts/train_hybrid.py --config "$CONFIG" --stage fold0
python scripts/train_spectral_teacher.py --config "$CONFIG" --mode final
python scripts/train_hybrid.py --config "$CONFIG" --stage final
python scripts/infer.py --config "$CONFIG"
python scripts/inspect_output.py outputs/smoke_v2/Round2_Test_Channel.npy \
  --samples 2 --report reports/generated/smoke_v2_output_check.json
echo "Scheme E-v2 reference-aware smoke PASS"
