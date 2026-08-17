#!/usr/bin/env bash
set -euo pipefail

python -m unittest discover -s tests -v
python scripts/smoke_test.py --config configs/smoke.json "$@"
python scripts/inspect_architecture.py --config configs/fold0_5090.json
