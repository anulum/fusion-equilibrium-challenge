# Compute declaration for submission #986068

The submitted model contains 69 map networks, four tree heads, eight joint
q95/betaN networks and eight static X-point networks. D61 changes scalar
mixing only; it does not add fitted models.

| Selected model component | Fits | Recorded fit wall seconds | Hardware evidence |
|---|---:|---:|---|
| Map networks with 14 scalar labels, GPU subset | 8 | 2,092 | Kaggle Tesla T4 campaign |
| Map networks with 78 scalar labels, GPU subset | 24 | 6,334 | Kaggle Tesla T4 campaign |
| Map networks with 14 scalar labels, CPU subset | 13 | 66,310 | Kaggle/ML350 CPU; individual CPU models unrecorded |
| Map networks with 78 scalar labels, CPU subset | 24 | 113,440 | Kaggle/ML350 CPU; individual CPU models unrecorded |
| Joint q95/betaN networks | 8 | 1,554.926 | Local AMD RX 6600 XT / ROCm campaign |
| Static X-point networks | 8 | 1,281.317 | Local AMD RX 6600 XT / ROCm campaign |
| D9 tree heads | 4 | 1,915 | Original notebook log identifies Tesla T4 |

These are sums of selected-fit wall times, not elapsed time of a parallel
campaign or GPU-kernel-only timings. Historical CUDA device-index strings
were bound to Tesla T4 using the original campaign ledger and notebook log;
they do not describe the workstation's NVIDIA card.

Current CUDA qualification used one NVIDIA GTX 1060 with 6 GiB VRAM, a
separate pinned JAX CUDA 12 environment and one fit per network family:

| Family | Seed | Epochs | Batch | Fitting frames | Fit wall seconds |
|---|---:|---:|---:|---:|---:|
| Map, 14 labels | 93 | 60 | 4,096 | 1,480,465 | 279.410 |
| Map, 78 labels | 200 | 60 | 4,096 | 1,480,465 | 325.591 |
| Joint scalars | 200 | 60 | 4,096 | 1,480,465 | 289.519 |
| Static X points | 500 | 60 | 2,048 | 1,480,465 | 283.103 |

The sum is 1,177.623 seconds (0.3271 hours). It includes normalization,
transfer, compilation and fitting. A fixed set of 358 development shots /
78,875 frames was excluded from fitting and checkpoint selection. This
already exposed development set measures backend differences, not fresh
scientific generalization. GPU allocator fraction 0.95 was needed after the
default allocation limit failed; no data, batch or epoch reductions were
made. The matched ROCm accuracy comparison remains pending.

CPU hardware observed for current runs was an Intel Core i5-11600K,
6 physical / 12 logical cores, on the workstation, and two Intel Xeon
E5-2640 processors, 12 physical / 24 logical cores total, on ML350.
The complete small raw-to-manifest chain took 228.9997 seconds on the
workstation with CUDA and 329.8650 seconds on ML350 CPU in an independent
fresh environment. Both runs fitted every model component with reduced
data and epochs; they do not measure full production preparation. The
workstation's matched-backend table preparation took 87.9676 seconds.

A compute-only A100 estimate uses approximately 5.719e16 dense training
FLOPs for the 85 full-production neural fits, calculated as six FLOPs per
active matrix weight per fitting frame per epoch (forward plus two
backward matrix products). At the A100's published 19.5 TFLOPS FP32 peak,
the ideal dense-operation time is 0.8147 hours. Assuming 25–50% of that peak
gives an illustrative **1.63–3.26 hours of dense computation**. The utilization
assumption is not benchmarked. This excludes compilation, transfer,
activations, optimizer updates, calibration evaluation, geometry readouts,
trees, basis/data preparation, MAST fitting and experimental search.
It is neither measured A100 wall time nor a budget-compliance certificate.
The hardware peak comes from the [NVIDIA A100 datasheet](https://www.nvidia.com/content/dam/en-zz/Solutions/Data-Center/a100/pdf/nvidia-a100-datasheet-nvidia-us-2188504-web.pdf).

A complete search-budget ledger, per-member historical CPU model bindings
and isolated full-production CPU-stage timings are unavailable in the
recovered selected-fit reports. No compute-light badge or complete
end-to-end production timing is claimed. MAST training costs remain a
separate accounting scope.

