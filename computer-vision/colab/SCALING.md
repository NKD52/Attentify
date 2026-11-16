# Recorded-video scaling benchmark

[Open the separate benchmark notebook](https://colab.research.google.com/github/NKD52/Attentify/blob/feature/scaling-benchmark/computer-vision/colab/Attentify_Scaling_Benchmark.ipynb). The ordinary tester stays on its existing branch and keeps its 640 defaults.

## First run

1. Select a GPU runtime and run setup.
2. Connect Drive and set the private model-folder path in your own notebook copy. The public notebook deliberately contains a placeholder.
3. Upload a short clip, or give its Colab/Drive path. Optionally label its visible student count; leave unknown counts as `None`.
4. Run the matrix and download its results ZIP. The default eight cases and two repeats produce 16 trials per clip. CPU runs can be slow; use a smaller frame window for a local smoke test.

The matrix contains faces-only, actions-only and combined baselines at 640; independently larger action input and face width; both larger together; and two output widths with fixed 640 inference. It deliberately changes one model-size variable at a time before the joint case. All cases read the same initial source-frame window, infer at the same stride, and use the same weights and thresholds. Even repeats reverse the order. No automatic parameter tuning or promotion occurs.

To explore larger classes, provide **real clips with different actual student counts**. Do not tile one student, duplicate frames or treat detector counts as ground truth. This module records user-supplied counts and mean detected faces as separate quantities. One clip cannot establish general classroom scaling.

## What each timing means

- Initialization: model checksum/load/session setup, before steady-state timing.
- Warm-up: repeated inference on the first source frame; tracker resets before the timed window.
- Pipeline: decode all source frames, infer on scheduled frames, render/encode output when enabled, and finalize the encoder. CSV/JSON exports, review image generation, ZIP creation and transfers are outside this interval.
- Model stages: YuNet plus tracking; action preprocessing; ONNX Runtime call (including its transfers/CPU operators); action postprocessing; association. These are wall times, not isolated GPU-kernel durations.
- Throughput: decoded source frames per wall second and actual inference updates per wall second are separate. The real-time factor is source-window duration divided by pipeline time; above one means this offline window processed faster than its source duration, not proof of real-time capture/display.
- Snapshot gap: source-video time between the current frame and last inferred snapshot. It is **not measured camera capture-to-display latency**, which stays null.
- Resources: sampled whole-process RSS and GPU memory during initialization, warm-up and timed processing. NVSMI polling runs about once per second. Values include retained runtime contexts and other allocations in the same Python process; they are not exact peaks or model-exclusive requirements. Unavailable measurements remain null. GPU polling can perturb timing slightly; record whether sampling was enabled.

Preview movies encode every decoded source frame at the source FPS, with current raw source left and a matching dated inference snapshot right. Snapshot overlays expire after two source seconds. Output width changes only rendering/encoding, not inference resolution. The two-panel canvas height is derived from the source aspect ratio and header; a 3840-wide output is not necessarily a 3840x2160 movie. Actual output dimensions and FPS are recorded in `case.json`.

## Results and review

`comparison.csv` provides flat timing columns; `comparison.json` preserves detailed summaries. Each case folder has settings/model info, `frames.csv`, `observations.jsonl`, `summary.json`, optional `preview.mp4`, and optional `review/`.

Review images are generated after the timed run when `review_every>0`. They include frames with no predictions. Open `review/index.html`, assess visible actions and missing/wrong boxes, and export your answers. Keep the review JSON with the original benchmark ZIP. Sampling and proposal review do not produce complete ground-truth labels or an accuracy score. A faster resolution cannot be recommended without checking quality, especially small/distant faces, hands and phones.

Failed trials remain explicitly failed, retain errors/partial observations and do not silently fall back to CPU or a smaller size. Check `suite.json` for failed counts and provenance. Never average failed cases into a performance result.

## Local execution

Use the existing CPU dependencies plus optional `psutil` for RSS sampling. Prepare an unpacked verified model cache with `workspace_storage.ensure_assets`, then from this directory:

```text
python benchmark.py --clips clips.json --assets PATH_TO_VERIFIED_MODELS --backend cpu --frames 250 --repeats 2
```

`clips.json` is a JSON list, for example `[{"path":"PATH_TO_CLIP.mp4","label":"classroom-1","visible_students":null}]`. The `--assets` directory contains `yunet.onnx` and `scb-refined.onnx`, not transport segments. The notebook handles model-cache preparation automatically. Programmatic `run_suite` supports a custom case list, inference stride, repeats and output/review options.

## Validation limits

Automated checks cover resizing/box remapping, case matrix, preserved preview FPS/frame counts, raw-versus-snapshot separation, failure continuation and metadata. Actual-model CPU checks cover baseline prediction equivalence and configurable input sizes. Hosted CUDA benchmarking and larger real-classroom populations require user-side runs; local/synthetic checks are not minimum-hardware evidence. Head-pose scheduling, model precision changes, smaller alternatives and live-stream transport remain future experiments.
