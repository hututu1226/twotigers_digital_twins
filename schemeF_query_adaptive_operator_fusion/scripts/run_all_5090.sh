#!/usr/bin/env bash
set -Eeuo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."
mkdir -p logs reports/generated outputs/final artifacts/final
BASE_CONFIG="${CONFIG:-configs/fold0_5090.json}"

bash scripts/prepare_shared_assets.sh
python scripts/check_environment.py --config "$BASE_CONFIG" --require-cuda
bash scripts/run_legacy_scans.sh
SELECTED_CONFIG="configs/fold0_selected.json"
python scripts/inspect_architecture.py \
  --config "$SELECTED_CONFIG" --output reports/generated/architecture.json

bash scripts/run_smoke.sh --device cuda

RESUME=1 CONFIG="$SELECTED_CONFIG" bash scripts/run_fold_attempt.sh
python scripts/prepare_second_attempt.py \
  --base "$SELECTED_CONFIG" \
  --evaluation artifacts/fold0_attempt1/context/evaluation.json \
  --output configs/fold0_attempt2.json \
  --decision reports/generated/attempt2_decision.json
if [[ "$(python -c 'import json; print(int(json.load(open("reports/generated/attempt2_decision.json"))["run_second_attempt"]))')" == "1" ]]; then
  RESUME=1 CONFIG=configs/fold0_attempt2.json bash scripts/run_fold_attempt.sh
fi

python scripts/select_best_attempt.py \
  --configs "$SELECTED_CONFIG" configs/fold0_attempt2.json \
  --output-config configs/fold0_best.json
python scripts/evaluate_breakdown.py --config configs/fold0_best.json
python scripts/verify_completion.py --config configs/fold0_best.json --stage fold0 \
  --output reports/generated/fold0_completion.json
python scripts/report_experiment.py --stage fold0

python scripts/prepare_final_config.py \
  --template configs/final_5090.json \
  --selected-fold-config configs/fold0_best.json \
  --context-checkpoint artifacts/fold0/context/best.pt \
  --outage-report artifacts/fold0/context/outage_scan.json \
  --output configs/final_selected.json
RESUME=1 CONFIG=configs/final_selected.json bash scripts/run_final.sh
bash scripts/package_results.sh
