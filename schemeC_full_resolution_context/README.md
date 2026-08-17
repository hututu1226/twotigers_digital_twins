# Scheme C: Full-Resolution Context Field

Scheme C is the new Huawei Round 2 pipeline. It keeps the improved 30,720-element
AE representation and replaces Scheme B's flat MLP/large output heads with a
full-resolution latent-token context model.

The design target is a Fold0 validation score near `0.7`, but that number is a
research target, not a guaranteed result. The code records the AE ceiling,
Context score, Joint score, and their gap so the next change can be based on
evidence.

## Key properties

- AE v3 uses a `6,144`-element spectrum branch and `24,576`-element detail branch.
- Context never performs `30,720 -> hundreds -> 30,720` global compression.
- Every angle-delay latent bin attends to all available users from its serving BS.
- Both BSs share one backbone and use a learned station embedding.
- Environment features are learned from BEV maps and BS-to-user corridor samples.
- There is no KNN weighted interpolation, manual amplitude calibration, or ray tracing.
- Training supports validation early stopping, runtime limits, checkpoint resume,
  GPU latent caching, AMP, and gradient checkpointing.

## Documents

- [Detailed algorithm design](docs/algorithm_design.md)
- [AutoDL 5090 operating guide](docs/autodl_5090_guide.md)

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
