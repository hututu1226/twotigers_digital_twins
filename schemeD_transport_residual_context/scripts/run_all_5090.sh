#!/usr/bin/env bash
set -Eeuo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."
mkdir -p logs reports/generated
CONFIG="${CONFIG:-configs/fold0_5090.json}"

python scripts/check_environment.py --config "$CONFIG" --require-cuda
python scripts/inspect_architecture.py --config "$CONFIG" --output reports/generated/architecture.json
RESUME=1 CONFIG="$CONFIG" bash scripts/run_fold0.sh
python scripts/prepare_final_config.py \
  --template configs/final_5090.json \
  --context-checkpoint artifacts/fold0/context/best.pt \
  --outage-report artifacts/fold0/context/outage_scan.json \
  --output configs/final_selected.json
RESUME=1 CONFIG=configs/final_selected.json bash scripts/run_final.sh
bash scripts/package_results.sh
