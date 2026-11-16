# Focused pipeline profiling

In the scaling notebook, load dependencies/models and select one recorded clip.
Set `RUN_MODE='profile'` in section 4, then run sections 4 and 5. The regular
matrix remains available with `RUN_MODE='benchmark'`. Model weights, thresholds,
input sizes and ordinary inference code are unchanged.

The profile processes at most the first 250 source frames, inferring every fifth
frame by default. It performs two separate passes:

1. A normal no-export combined baseline, including head pose when selected.
   `baseline/summary.json` and `frames.csv` measure decoding and inference.
   Sampled `nvidia-smi` readings distinguish GPU busy percent from memory use.
2. An instrumented pass with a PyTorch CPU/CUDA trace, labeled head CPU transforms
   and network execution, and an ONNX Runtime action-provider trace.

Download the generated `SEND-THIS-*.zip`. Read `profile-summary.json` first.
`torch-cpu.txt` and `torch-device.txt` summarize framework operations;
`torch-trace.json` and `ort-trace*.json` provide timelines for deeper inspection.
No video, source images or weights are exported by profiling mode. Traces can
contain machine paths and operation metadata; keep the results private.

## Interpretation

- Profile instrumentation adds overhead. Use the separate baseline for normal
  throughput; do not treat the instrumented pass as a speed comparison.
- GPU utilization samples are device-wide, not process-exclusive. They cover
  baseline initialization, warmup and processing, and can miss short bursts.
- `memory_used_mib` is memory allocation, not compute utilization.
- ORT provider event duration sums include runtime overhead and warmup. They are
  not pure GPU kernel durations, wall time or utilization percentages. Concurrent
  events can overlap.
- PyTorch traces alone do not prove where ONNX Runtime action operators execute.
  Inspect the separate ORT provider summary/trace for CPU fallback.
- The baseline includes decode and skipped source frames, but no browser transfer,
  display or movie export. Use the existing output cases to measure export costs.
- If the selected window has no usable faces, it cannot profile head-network cost.
- This is a bottleneck diagnostic, not accuracy evaluation or live camera latency.

## Local verification

The focused tests cover provider-event accounting, bounded execution, separation
of baseline and traces, and media-free packaging. A real CPU smoke run with the
registered YuNet, refined SCB and head models completed three updates and exported
ORT events plus labeled head CPU/network regions. CUDA profiling still needs a
real Colab GPU trial; no GPU speed improvement is claimed.
