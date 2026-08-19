#!/usr/bin/env bash
set -Eeuo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."
CONFIG="${CONFIG:-configs/fold0_5090.json}"
FINAL_CONFIG="${FINAL_CONFIG:-configs/final_selected.json}"
mkdir -p logs reports/generated outputs/final artifacts/final

run() {
  echo "[$(date '+%F %T')] $*"
  "$@"
}

run python scripts/check_environment.py --config "$CONFIG" --require-cuda --strict-boosters

if [[ ! -s artifacts/preprocessed_scheme_e/manifest.json ]]; then
  run python scripts/preprocess.py --config "$CONFIG"
fi
run python scripts/inspect_architecture.py --config "$CONFIG"
if [[ ! -s artifacts/spectral/channel_targets.npz ]]; then
  run python scripts/extract_spectral_targets.py --config "$CONFIG"
fi
if [[ ! -s artifacts/fold0/spectral_teacher/oof_priors.npz || ! -s artifacts/fold0/spectral_teacher/oof_report.json ]]; then
  run python scripts/train_spectral_teacher.py --config "$CONFIG" --mode oof
fi
if ! python scripts/verify_completion.py --config "$CONFIG" --stage fold0 >/dev/null 2>&1; then
  run python scripts/train_hybrid.py --config "$CONFIG" --stage fold0 --resume
fi
run python scripts/verify_completion.py --config "$CONFIG" --stage fold0 \
  --output reports/generated/fold0_completion.json
run python scripts/evaluate_breakdown.py --config "$CONFIG" \
  --output reports/generated/fold0_breakdown.json
run python scripts/report_experiment.py --stage fold0

run python scripts/prepare_final_config.py --base "$CONFIG" --output "$FINAL_CONFIG"

if [[ ! -s artifacts/final/spectral_teacher/test_priors.npz || ! -s artifacts/final/spectral_teacher/model.pkl || ! -s artifacts/final/spectral_teacher/final_report.json ]]; then
  run python scripts/train_spectral_teacher.py --config "$FINAL_CONFIG" --mode final
fi
if [[ ! -s artifacts/final/hybrid/best.pt || ! -s artifacts/final/hybrid/summary.json ]]; then
  run python scripts/train_hybrid.py --config "$FINAL_CONFIG" --stage final --resume
fi
if [[ ! -s outputs/final/Round2_Test_Channel.npy || artifacts/final/hybrid/best.pt -nt outputs/final/Round2_Test_Channel.npy ]]; then
  run python scripts/infer.py --config "$FINAL_CONFIG"
fi
run python scripts/inspect_output.py outputs/final/Round2_Test_Channel.npy \
  --samples 500 --report reports/generated/final_output_check.json
run python scripts/verify_completion.py --config "$FINAL_CONFIG" --stage final \
  --output reports/generated/final_completion.json
run python scripts/report_experiment.py --stage final
run bash scripts/package_results.sh

echo "[$(date '+%F %T')] Scheme E complete"
