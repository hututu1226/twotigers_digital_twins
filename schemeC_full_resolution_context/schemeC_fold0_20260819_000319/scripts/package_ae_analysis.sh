#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$PROJECT_DIR"

STAMP="$(date +%Y%m%d_%H%M%S)"
GATE_STATUS="$(python - <<'PY'
import json
from pathlib import Path

report = json.loads(
    Path("artifacts/fold0/autoencoder/quality_gate.json").read_text(encoding="utf-8")
)
print(str(report.get("status", "UNKNOWN")).upper())
PY
)"
case "$GATE_STATUS" in
  PASS|FAIL|SKIPPED) ;;
  *)
    printf 'Unexpected AE quality gate status: %s\n' "$GATE_STATUS" >&2
    exit 1
    ;;
esac

ARCHIVE="schemeC_ae_analysis_${GATE_STATUS}_${STAMP}.tar.gz"
mkdir -p artifacts/fold0/autoencoder

{
  printf 'packaged_at=%s\n' "$(date -Is)"
  printf 'quality_gate=%s\n' "$GATE_STATUS"
  printf 'git_commit=' && (cd .. && git rev-parse HEAD) || true
  printf 'git_branch=' && (cd .. && git branch --show-current) || true
  python --version
  python -c "import numpy, torch; print('numpy=' + numpy.__version__); print('torch=' + torch.__version__); print('torch_cuda=' + str(torch.version.cuda)); print('cuda_available=' + str(torch.cuda.is_available())); print('gpu=' + (torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu'))"
  if command -v nvidia-smi >/dev/null 2>&1; then nvidia-smi; fi
} > artifacts/fold0/autoencoder/repro_environment.txt 2>&1
python -m pip freeze > artifacts/fold0/autoencoder/pip_freeze.txt

required=(
  artifacts/preprocessed_scheme_c/manifest.json
  artifacts/preprocessed_scheme_c/metadata.npz
  artifacts/capacity/one_sample.json
  artifacts/capacity/thirty_two_samples.json
  artifacts/fold0/autoencoder/best.pt
  artifacts/fold0/autoencoder/last.pt
  artifacts/fold0/autoencoder/final.pt
  artifacts/fold0/autoencoder/history.jsonl
  artifacts/fold0/autoencoder/summary.json
  artifacts/fold0/autoencoder/resolved_config.json
  artifacts/fold0/autoencoder/evaluation.json
  artifacts/fold0/autoencoder/ablation.json
  artifacts/fold0/autoencoder/quality_gate.json
  artifacts/fold0/autoencoder/repro_environment.txt
  artifacts/fold0/autoencoder/pip_freeze.txt
  configs/fold0_5090.json
  scheme_c
  scripts
  tests
  docs
  README.md
  pyproject.toml
  requirements.txt
)

for path in "${required[@]}"; do
  if [[ ! -e "$path" ]]; then
    printf 'Missing required AE analysis artifact: %s\n' "$path" >&2
    exit 1
  fi
done

optional=(
  artifacts/capacity/completion_report.json
  artifacts/fold0/autoencoder/best_spectrum.pt
  artifacts/fold0/autoencoder/completion_report.json
  logs/ae_capacity.log
  logs/ae_fold0_v4.log
  logs/smoke_v4.log
  logs/overnight_pipeline.log
  logs/overnight_launcher.log
  logs/overnight_status.txt
)

files=("${required[@]}")
for path in "${optional[@]}"; do
  if [[ -e "$path" ]]; then
    files+=("$path")
  fi
done

tar --exclude='__pycache__' --exclude='*.pyc' -czf "$ARCHIVE" "${files[@]}"
sha256sum "$ARCHIVE" > "${ARCHIVE}.sha256"
tar -tzf "$ARCHIVE" >/dev/null
sha256sum -c "${ARCHIVE}.sha256"
printf 'Created %s\n' "$ARCHIVE"

