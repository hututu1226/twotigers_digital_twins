#!/usr/bin/env bash
set -euo pipefail

STAMP="$(date +%Y%m%d_%H%M%S)"
ARCHIVE="schemeB_results_${STAMP}.tar.gz"
mkdir -p artifacts/final

{
  printf 'packaged_at=%s\n' "$(date -Is)"
  printf 'git_commit='
  (cd .. && git rev-parse HEAD) || true
  printf 'git_branch='
  (cd .. && git branch --show-current) || true
  python --version
  python -c "import numpy, torch; print('numpy=' + numpy.__version__); print('torch=' + torch.__version__); print('torch_cuda=' + str(torch.version.cuda)); print('cuda_available=' + str(torch.cuda.is_available())); print('device=' + (torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu'))"
  if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi
  fi
} > artifacts/final/repro_environment.txt 2>&1
python -m pip freeze > artifacts/final/pip_freeze.txt

required=(
  artifacts/preprocessed_dual/manifest.json
  artifacts/preprocessed_dual/metadata.npz
  artifacts/preprocessed_dual/context_static_cell_0.npz
  artifacts/preprocessed_dual/context_static_cell_1.npz
  artifacts/preprocessed_dual/environment_static_cell_0.npz
  artifacts/preprocessed_dual/environment_static_cell_1.npz
  artifacts/final/autoencoder/final.pt
  artifacts/final/autoencoder/history.jsonl
  artifacts/final/encoded.npz
  artifacts/final/context/final.pt
  artifacts/final/context/history.jsonl
  artifacts/final/joint/final.pt
  artifacts/final/joint/history.jsonl
  outputs/final/Round2_Test_Channel.npy
  outputs/final/Round2_Test_Channel.json
  configs/final_4090.json
  configs/fold0_4090.json
  configs/smoke.json
  structured_context_field
  scripts
  tests
  docs
  README.md
  pyproject.toml
  requirements.txt
  artifacts/final/repro_environment.txt
  artifacts/final/pip_freeze.txt
)

for path in "${required[@]}"; do
  if [[ ! -e "$path" ]]; then
    printf 'Missing required result: %s\n' "$path" >&2
    exit 1
  fi
done

optional=(
  configs/final_selected.json
  artifacts/fold0/autoencoder/best.pt
  artifacts/fold0/autoencoder/history.jsonl
  artifacts/fold0/autoencoder/summary.json
  artifacts/fold0/autoencoder/evaluation.json
  artifacts/fold0/encoded.npz
  artifacts/fold0/encoded.json
  artifacts/fold0/context/best.pt
  artifacts/fold0/context/history.jsonl
  artifacts/fold0/context/summary.json
  artifacts/fold0/context/evaluation.json
  artifacts/fold0/joint/best.pt
  artifacts/fold0/joint/history.jsonl
  artifacts/fold0/joint/summary.json
  artifacts/fold0/joint/evaluation.json
  artifacts/fold0/joint/outage_scan.json
  artifacts/fold0/stage_gap.json
  artifacts/final/autoencoder/summary.json
  artifacts/final/context/summary.json
  artifacts/final/joint/summary.json
  logs/fold0.log
  logs/final.log
)

files=("${required[@]}")
for path in "${optional[@]}"; do
  if [[ -e "$path" ]]; then
    files+=("$path")
  fi
done

tar --exclude='__pycache__' --exclude='*.pyc' -czf "$ARCHIVE" "${files[@]}"
sha256sum "$ARCHIVE" > "${ARCHIVE}.sha256"
printf 'Created %s\n' "$ARCHIVE"
printf 'Checksum: '
cat "${ARCHIVE}.sha256"
