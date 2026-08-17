# Scheme C: Full-Resolution Context Field

Scheme C is the new Huawei Round 2 pipeline. It keeps the improved 30,720-element
AE representation and replaces Scheme B's flat MLP/large output heads with a
full-resolution latent-token context model.

The design target is a Fold0 validation score near `0.7`, but that number is a
research target, not a guaranteed result. The code records the AE ceiling,
Context score, Joint score, and their gap so the next change can be based on
evidence.

## Key properties

- AE v4 keeps a `6,144`-element power branch and an isolated `24,576`-element
  complex-detail branch; neither branch is flattened into a small global vector.
- Formal training uses coarse/detail/joint stages and an AE quality gate. Context
  is not started below `0.75` AE Score or when detail ablations show collapse.
- Context never performs `30,720 -> hundreds -> 30,720` global compression.
- Every angle-delay latent bin attends to all available users from its serving BS.
- Both BSs share one backbone and use a learned station embedding.
- Environment features are learned from BEV maps and BS-to-user corridor samples.
- There is no KNN weighted interpolation, manual amplitude calibration, or ray tracing.
- Training supports validation early stopping, runtime limits, checkpoint resume,
  GPU latent caching, AMP, and gradient checkpointing.

## Documents

- [Detailed algorithm design](docs/algorithm_design.md)
- [AE v4 failure analysis and redesign evidence](docs/ae_v4_failure_driven_redesign.md)
- [AutoDL 5090 operating guide](docs/autodl_5090_guide.md)
- [Overnight training and automatic shutdown](docs/overnight_autorun.md)

## Quick verification

From this directory:

```bash
python scripts/check_environment.py --config configs/fold0_5090.json
python -m unittest discover -s tests -v
python scripts/smoke_test.py --config configs/smoke.json --device cuda
python scripts/inspect_architecture.py --config configs/fold0_5090.json
```

The smoke test must end with `"status": "PASS"` and produce a complex64 NPY
file with shape `[2, 256, 4, 192]`.

Before spending hours on Fold0, run the deliberate 1-sample and 32-sample AE
capacity checks on the 5090:

```bash
set -o pipefail
bash scripts/run_ae_capacity_gates.sh 2>&1 | tee logs/ae_capacity.log
```

These checks must pass before the formal run. They prove training-set capacity,
not validation generalization.

## Formal Fold0 run

```bash
mkdir -p logs
set -o pipefail
bash scripts/run_fold0.sh 2>&1 | tee logs/fold0.log
```

Resume after interruption:

```bash
RESUME=1 bash scripts/run_fold0.sh 2>&1 | tee -a logs/fold0.log
```

Do not run the all-data final stage until Fold0 results have been reviewed. The
exact sequence and artifact locations are in the AutoDL guide.
