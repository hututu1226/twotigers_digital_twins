#!/usr/bin/env bash
set -euo pipefail

CONFIG="${CONFIG:-configs/fold0_5090.json}"
CHECKPOINT="${CHECKPOINT:-artifacts/fold0/context/best.pt}"
OUTPUT_DIR="${OUTPUT_DIR:-artifacts/fold0/context_diagnostics}"
LIMIT="${LIMIT:-0}"

mkdir -p logs "$OUTPUT_DIR"
set -o pipefail
python scripts/diagnose_context.py \
  --config "$CONFIG" \
  --checkpoint "$CHECKPOINT" \
  --output-dir "$OUTPUT_DIR" \
  --limit "$LIMIT" \
  2>&1 | tee logs/context_diagnostics.log

test "$(python -c "import json; print(json.load(open('$OUTPUT_DIR/report.json'))['status'])")" = "PASS"
OUTPUT_DIR="$OUTPUT_DIR" bash scripts/package_context_diagnostics.sh

if [[ "${CONFIRM_AUTODL_SHUTDOWN:-NO}" == "YES" ]]; then
  if [[ ! -x /usr/bin/shutdown ]]; then
    printf '/usr/bin/shutdown is unavailable; diagnostics succeeded but shutdown was skipped.\n' >&2
    exit 1
  fi
  sync
  /usr/bin/shutdown
fi
