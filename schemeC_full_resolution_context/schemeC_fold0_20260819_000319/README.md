# Scheme C: Geometry-Warped Context V2

This directory contains the current Huawei Round 2 pipeline. AE v4 remains the
validated channel representation, while Context V2 replaces the original
same-bin attention field.

## Current architecture

- AE v4: `6,144` Spectrum values plus `24,576` complex Detail values.
- Context V2: `18,614,162` parameters and no `30,720 -> hundreds -> 30,720`
  global bottleneck.
- A learned router examines every same-BS observation and selects 64 candidates.
- Geometry-conditioned 3D warping aligns moving angle-delay paths before
  attention.
- Axial latent attention mixes neighboring and distant angle-delay bins.
- Ordered corridor attention preserves where an obstacle occurs between the BS
  and user.
- The two BSs use separate latent statistics and station-specific FiLM adapters.
- Context and the AE decoder train in one end-to-end run. There is no separate
  Joint stage in Context V2.
- No fixed KNN weighted interpolation, manual amplitude calibration, or ray
  tracing is used.

The Fold0 target is `0.70`, but this remains an experimental target rather than
a guaranteed score. The existing AE ceiling is `0.9491`; only a formal Context
V2 Fold0 run can measure how much of that ceiling spatial prediction retains.

## Documents

- [Context V2 algorithm design](docs/context_v2_design.md)
- [Context V2 AutoDL 5090 runbook](docs/context_v2_autodl.md)
- [AE v4 failure-driven redesign](docs/ae_v4_failure_driven_redesign.md)
- [Unattended execution and automatic shutdown](docs/overnight_autorun.md)

The older [combined algorithm document](docs/algorithm_design.md) and
[legacy AutoDL guide](docs/autodl_5090_guide.md) retain AE history, but their
Context V1 and separate Joint-stage descriptions are superseded by the two
Context V2 documents above.

## Local verification

From `schemeC_full_resolution_context`:

```bash
python -m unittest discover -s tests -v
python scripts/smoke_test.py --config configs/smoke.json --device cpu
python scripts/analyze_context_masks.py --config configs/fold0_5090.json
python scripts/inspect_architecture.py --config configs/fold0_5090.json
```

The smoke test must finish with `"status": "PASS"` and produce a finite
`complex64` array with shape `[2,256,4,192]`.

## Fold0 on AutoDL

The verified AE `best.pt` is stored in Git LFS together with its evaluation,
ablation, summary, and quality-gate JSON files. After cloning, download the real
checkpoint before running Scheme C:

```bash
git lfs install
git lfs pull
```

`run_fold0.sh` then reuses this AE, regenerates per-BS float32 latent data,
trains Context V2, scans the outage threshold, evaluates Fold0, generates the
500 test channels, and checks format.

```bash
mkdir -p logs
set -o pipefail
RESUME=1 bash scripts/run_fold0.sh 2>&1 | tee logs/context_v2_fold0.log
```

For unattended execution with backup and shutdown, use the commands in
`docs/context_v2_autodl.md`.
