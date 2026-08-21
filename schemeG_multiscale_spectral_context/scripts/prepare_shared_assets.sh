#!/usr/bin/env bash
set -Eeuo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/../.."
REPO_ROOT="$PWD"
LEGACY_ROOT="${LEGACY_RUN_ROOT:-}"
if [[ -z "$LEGACY_ROOT" ]]; then
  for candidate in \
    /root/autodl-fs/schemeF_0820_20260820 \
    /root/autodl-fs/twotigers_0819_run \
    /root/autodl-fs/twotigers_digital_twins; do
    if [[ -d "$candidate" && "$candidate" != "$REPO_ROOT" ]]; then
      LEGACY_ROOT="$candidate"
      break
    fi
  done
fi

link_file() {
  local source="$1" destination="$2"
  [[ -f "$source" ]] || return 0
  [[ -e "$destination" || -L "$destination" ]] && return 0
  mkdir -p "$(dirname "$destination")"
  ln -s "$source" "$destination"
}

link_dir() {
  local source="$1" destination="$2"
  [[ -d "$source" ]] || return 0
  [[ -e "$destination" || -L "$destination" ]] && return 0
  mkdir -p "$(dirname "$destination")"
  ln -s "$source" "$destination"
}

if [[ -n "$LEGACY_ROOT" ]]; then
  echo "Reusing persistent assets from $LEGACY_ROOT where available."
  link_file \
    "$LEGACY_ROOT/schemeC_full_resolution_context/artifacts/fold0/autoencoder/best.pt" \
    "$REPO_ROOT/schemeC_full_resolution_context/artifacts/fold0/autoencoder/best.pt"
  link_dir \
    "$LEGACY_ROOT/schemeE_spectral_gaussian_hybrid/artifacts/preprocessed_scheme_e" \
    "$REPO_ROOT/schemeE_spectral_gaussian_hybrid/artifacts/preprocessed_scheme_e"
  link_dir \
    "$LEGACY_ROOT/schemeE_spectral_gaussian_hybrid/artifacts/spectral" \
    "$REPO_ROOT/schemeE_spectral_gaussian_hybrid/artifacts/spectral"
  link_dir \
    "$LEGACY_ROOT/schemeE_spectral_gaussian_hybrid/artifacts/fold0/spectral_teacher" \
    "$REPO_ROOT/schemeE_spectral_gaussian_hybrid/artifacts/fold0/spectral_teacher"
  link_dir \
    "$LEGACY_ROOT/schemeE_spectral_gaussian_hybrid/artifacts/final/spectral_teacher" \
    "$REPO_ROOT/schemeE_spectral_gaussian_hybrid/artifacts/final/spectral_teacher"
fi

AE="$REPO_ROOT/schemeC_full_resolution_context/artifacts/fold0/autoencoder/best.pt"
if [[ ! -s "$AE" || "$(stat -c '%s' "$AE")" -lt 1000000 ]]; then
  echo "Scheme C AE checkpoint is unavailable or is still an LFS pointer: $AE" >&2
  exit 1
fi

cd "$REPO_ROOT/schemeE_spectral_gaussian_hybrid"
CONFIG="configs/fold0_5090.json"
if [[ ! -s artifacts/preprocessed_scheme_e/metadata.npz ]]; then
  python scripts/preprocess.py --config "$CONFIG"
fi
if [[ ! -s artifacts/spectral/channel_targets.npz ]]; then
  python scripts/extract_spectral_targets.py --config "$CONFIG"
fi
if [[ ! -s artifacts/fold0/spectral_teacher/oof_priors.npz ]]; then
  python scripts/train_spectral_teacher.py --config "$CONFIG" --mode oof
fi
if [[ ! -s artifacts/final/spectral_teacher/test_priors.npz ]]; then
  python scripts/train_spectral_teacher.py --config "$CONFIG" --mode final
fi

python - <<'PY'
from pathlib import Path
import numpy as np

paths = [
    Path("artifacts/fold0/spectral_teacher/oof_priors.npz"),
    Path("artifacts/final/spectral_teacher/test_priors.npz"),
]
expected = [4000, 500]
for path, count in zip(paths, expected, strict=True):
    with np.load(path) as source:
        rows = len(source["pas_log"])
        required = {"pas_log", "pdp_log", "log_power", "uncertainty", "outage_probability"}
        if not required.issubset(source.files) or rows != count:
            raise SystemExit(f"Invalid Scheme E prior cache: {path}, rows={rows}")
print("Scheme E OOF/test priors verified.")
PY

cd "$REPO_ROOT/schemeG_multiscale_spectral_context"
if [[ ! -s artifacts/fold0/spectral_teacher/strict_priors.npz ]]; then
  python scripts/build_strict_fold_prior.py --fold 0
fi
python - <<'PY'
from pathlib import Path
import numpy as np

path = Path("artifacts/fold0/spectral_teacher/strict_priors.npz")
with np.load(path) as source:
    if len(source["pas_log"]) != 4000 or not bool(source["available"].all()):
        raise SystemExit(f"Invalid strict Fold0 prior: {path}")
print("Strict Fold0 prior verified.")
PY
