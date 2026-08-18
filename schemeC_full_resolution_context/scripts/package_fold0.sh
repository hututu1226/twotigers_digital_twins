#!/usr/bin/env bash
set -euo pipefail

STAMP="$(date +%Y%m%d_%H%M%S)"
ARCHIVE="schemeC_fold0_${STAMP}.tar.gz"
mkdir -p artifacts/fold0

{
  printf 'packaged_at=%s\n' "$(date -Is)"
  printf 'git_commit=' && (cd .. && git rev-parse HEAD) || true
  printf 'git_branch=' && (cd .. && git branch --show-current) || true
  python --version
  python -c "import numpy, torch; print('numpy=' + numpy.__version__); print('torch=' + torch.__version__); print('torch_cuda=' + str(torch.version.cuda)); print('gpu=' + (torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu'))"
  if command -v nvidia-smi >/dev/null 2>&1; then nvidia-smi; fi
} > artifacts/fold0/repro_environment.txt 2>&1
python -m pip freeze > artifacts/fold0/pip_freeze.txt

required=(
  artifacts/preprocessed_scheme_c/manifest.json
  artifacts/preprocessed_scheme_c/metadata.npz
  artifacts/capacity/one_sample.json
  artifacts/capacity/thirty_two_samples.json
  artifacts/fold0/autoencoder/best.pt
  artifacts/fold0/autoencoder/summary.json
  artifacts/fold0/autoencoder/evaluation.json
  artifacts/fold0/autoencoder/ablation.json
  artifacts/fold0/autoencoder/quality_gate.json
  artifacts/fold0/encoded.npz
  artifacts/fold0/context_mask_report.json
  artifacts/fold0/context/best.pt
  artifacts/fold0/context/final.pt
  artifacts/fold0/context/history.jsonl
  artifacts/fold0/context/evaluation.json
  artifacts/fold0/context/outage_scan.json
  artifacts/fold0/stage_gap.json
  artifacts/fold0/repro_environment.txt
  artifacts/fold0/pip_freeze.txt
  outputs/fold0/Round2_Test_Channel.npy
  outputs/fold0/Round2_Test_Channel.json
  configs/fold0_5090.json
  configs/final_5090.json
  configs/smoke.json
  scheme_c scripts tests docs README.md pyproject.toml requirements.txt
)

for path in "${required[@]}"; do
  if [[ ! -e "$path" ]]; then
    printf 'Missing required Fold0 result: %s\n' "$path" >&2
    exit 1
  fi
done

files=("${required[@]}")
if [[ -f logs/fold0.log ]]; then files+=(logs/fold0.log); fi
tar --exclude='__pycache__' --exclude='*.pyc' -czf "$ARCHIVE" "${files[@]}"
sha256sum "$ARCHIVE" > "${ARCHIVE}.sha256"
printf 'Created %s\n' "$ARCHIVE"
cat "${ARCHIVE}.sha256"
