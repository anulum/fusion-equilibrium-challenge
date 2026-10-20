# Training DIII-D from raw data for #986068

Install `requirements-training-cpu.txt` in a fresh `.venv` environment. For NVIDIA
training, use the separate `requirements-training-cuda.txt` environment instead.
Keep temporary files and package caches on a disk with enough space.

Run the complete small integration test:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 \
.venv/bin/python train.py --raw-dir /data/diii_d_train --out /work/smoke --limit 4
```

It uses four fitting shots, two calibration shots and two excluded validation
shots, at most 48 frames per shot, one epoch and batch 64. Network width 512,
depth 4, output orders and ensemble membership remain unchanged. All 69 map
networks receive their nine geometry readouts; all four D9 heads, eight joint
q95/betaN networks and eight static X-point networks are fitted. Trees use five
boosting rounds with patience two. These reductions exercise the whole chain;
the resulting models are unsuitable for competition submission.

For the full production recipe, omit `--limit`:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 JAX_PLATFORMS=cuda,cpu \
JAX_ENABLE_X64=1 XLA_PYTHON_CLIENT_PREALLOCATE=false \
.venv/bin/python train.py --raw-dir /data/diii_d_train --out /work/full --platform cuda
```

ROCm uses its separately installed JAX environment and `--platform rocm`.
The default tree stage uses CPU histogram training with the original D9
parameters; `--tree-device cuda` selects NVIDIA tree training. The CPU port
must be declared when reporting reproduction accuracy.

The full basis uses only the original native training split. Production neural
fits use every permitted non-calibration training-directory shot; recipes marked
TC also fit on calibration rows. Calibration selects the best map checkpoint
using the original trimmed flux error. Scalar families retain epoch 60. Static
X models initialize the active trunk on CPU, append 32 zero features and use
one common scale in metres. Their unused temporal sidecars are unnecessary for
static inference. Trees fit the original training and validation splits and
use calibration for early stopping. Previously exposed validation and holdout
shots reused for production are not a fresh validation set.

Deployment #986068 mixes joint q95 at share 0.46 and joint betaN at share
0.87. This does not change the individual network fitting recipes.

The output contains newly fitted DIII-D models, `deployment.json`, model fit
receipts and a copy of the portable inference runtime. No existing DIII-D
weights are loaded. Frozen MAST inference assets are copied with the runtime;
this entrypoint does not retrain MAST. Run `infer.py` from the output directory
to exercise the newly fitted manifest. Preserve and review accuracy before
submitting a full retrained candidate: backend and data-preparation rounding
can change trained weights.

Output directories must be new. The entrypoint fails on a nonfinite loss,
insufficient finite targets or a failed full static-seed quality check. It does
not replace seeds or retry numerical failures. Use a memory ceiling and monitor
available host RAM for long runs. Full production fitting and A100 wall time
remain separate qualifications from the small integration test.
