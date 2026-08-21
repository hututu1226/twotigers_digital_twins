#!/usr/bin/env bash
set -Eeuo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."
BASE_CONFIG="${BASE_CONFIG:-configs/v2_5090.json}"
mkdir -p logs reports/generated outputs/v2 artifacts/v2/final

run() {
  echo "[$(date '+%F %T')] $*"
  "$@"
}

run python scripts/check_environment.py --config "$BASE_CONFIG" --require-cuda --strict-boosters
if [[ ! -s artifacts/preprocessed_scheme_e/manifest.json ]]; then
  run python scripts/preprocess.py --config "$BASE_CONFIG"
fi
if [[ ! -s artifacts/spectral/channel_targets.npz ]]; then
  run python scripts/extract_spectral_targets.py --config "$BASE_CONFIG"
fi
run python scripts/inspect_architecture.py --config "$BASE_CONFIG"
run python -m unittest discover -s tests -v
run bash scripts/run_v2_smoke.sh

if [[ ! -s artifacts/v2/fold0/spectral_teacher/strict_priors.npz || ! -s artifacts/v2/fold0/spectral_teacher/strict_report.json ]]; then
  run python scripts/build_strict_fold_prior.py --config "$BASE_CONFIG"
fi
run python scripts/prepare_v2_attempts.py --base "$BASE_CONFIG"

attempt_configs=(
  configs/v2_attempt1_safe.json
  configs/v2_attempt2_reference.json
  configs/v2_attempt3_decoder.json
)
for index in 1 2 3; do
  config="${attempt_configs[$((index-1))]}"
  if ! python scripts/verify_completion.py --config "$config" --stage fold0 >/dev/null 2>&1; then
    run python scripts/train_hybrid.py --config "$config" --stage fold0 --resume
  fi
  run python scripts/verify_completion.py --config "$config" --stage fold0 \
    --output "reports/generated/v2_attempt${index}_completion.json"
  run python scripts/scan_v2_policy.py --config "$config" \
    --output "reports/generated/v2_attempt${index}_policy.json"
done

run python scripts/select_v2_attempt.py
run python scripts/prepare_v2_final_config.py
FINAL_CONFIG="configs/v2_final_selected.json"

if [[ ! -s artifacts/v2/final/spectral_teacher/oof_priors.npz || ! -s artifacts/v2/final/spectral_teacher/oof_report.json ]]; then
  run python scripts/train_spectral_teacher.py --config "$FINAL_CONFIG" --mode oof
fi
if [[ ! -s artifacts/v2/final/spectral_teacher/test_priors.npz || ! -s artifacts/v2/final/spectral_teacher/model.pkl ]]; then
  run python scripts/train_spectral_teacher.py --config "$FINAL_CONFIG" --mode final
fi
if [[ ! -s artifacts/v2/final/hybrid/best.pt || ! -s artifacts/v2/final/hybrid/summary.json ]]; then
  run python scripts/train_hybrid.py --config "$FINAL_CONFIG" --stage final --resume
fi
run python scripts/infer.py --config "$FINAL_CONFIG"
run python scripts/inspect_output.py outputs/v2/Round2_Test_Channel.npy \
  --samples 500 --report reports/generated/v2_final_output_check.json
run python scripts/verify_completion.py --config "$FINAL_CONFIG" --stage final \
  --output reports/generated/v2_final_completion.json
run python scripts/report_v2.py
run bash scripts/package_v2_results.sh

echo "[$(date '+%F %T')] Scheme E-v2 complete"
