# Attentify pose model comparison

Separate Colab experiment for original pretrained **YOLO11s-Pose** and
**YOLOX-m HumanArt + RTMPose-m**. No SCB/face/head model or existing notebook is
replaced. These models return COCO17 person joints; they do **not** classify the
six requested sitting postures or student attention.

## Run on hosted hardware

Open `Attentify_Posture_Comparison.ipynb` in Colab, select a GPU runtime and run
the cells in order. The notebook downloads public weights automatically. Upload
the same image/video inputs once; both pipelines use the same source-frame
window and inference stride. A laptop GPU is unnecessary for hosted inference.
Choose `BACKEND='cpu'` only for a deliberately separate CPU comparison.

Your personal notebook can cache weights and save results on authorized Drive
storage. No repeated private model ZIP or manually entered model path is needed.
Other teammates can use temporary Colab storage without your Drive account.

Defaults: two repeats, the first 750 source frames, stride 5, five warmups and
six evenly spaced review samples per case. At 25 FPS this covers 30 seconds;
the predictions update five times per source second. Short clips stop at EOF.
Change the visible settings for longer clips. No images are exported when
`REVIEW_SAMPLES=0`; raw inputs/weights are never bundled automatically.

## Honest performance accounting

- Records actual CPU count, GPU name/VRAM, CUDA/cuDNN, Python/package versions,
  code revision/hashes, source hashes, model hashes, original image dimensions,
  input sizes, native thresholds and provider registration.
- CUDA runs fail when the required provider/device is unavailable. Registered
  CUDA still permits some ONNX operators on CPU; it is not a per-operator proof.
- `median_inference_ms`/p95 are synchronized host elapsed times including input
  preparation, execution, result transfers and conversion. They are not pure
  GPU kernel time. Framework-internal phase timers are descriptive and should
  not be compared as identical quantities across implementations.
- Total offline processing includes decoding all source frames, including those
  skipped for inference. Initialization/downloads, warmups, review generation,
  JSON/ZIP creation, browser transfer and display are excluded.
- RTMPose timings include the person detector and per-person pose work. YOLO
  includes detection and pose in one network. This compares runnable pipelines,
  not backbones with matched detections. Their native detector thresholds differ.
- Repeats reverse model order. Sampled resources describe the whole process and
  can include retained contexts, are not exact peaks and add overhead. Disable
  `SAMPLE_RESOURCES` for a separate timing-only experiment. Do not compare one
  model with sampling enabled to another with it disabled.
- Offline throughput is not live webcam FPS or camera-to-display latency.
  Images have no real-time factor. Colab GPU results do not specify the minimum
  hardware required by a different laptop or a larger classroom.
- Failures stay failed, without silent smaller-model or CPU substitution.

## Inspect results

Download the unique `SEND-THIS-*.zip`; no earlier results are overwritten.

- `suite.json`: source/code/environment/settings provenance.
- `comparison.csv`/`.json`: per-source/model/repeat summaries.
- Case folders: `settings.json`, `frames.csv`, `observations.jsonl`, summary and
  any error. Predictions use original image pixels and native keypoint scores.
- Optional `review/`: matching raw and skeleton-overlay images, including
  inference frames without detections. Confidence 0.3 is drawing-only.

Review the correct person and joint ownership, missed/merged people, visible
shoulders, hips and wrists, and behavior under desk occlusion. Native confidence
scores and person counts are not calibrated accuracy. `posture_accuracy` and
`keypoint_accuracy` remain null without reference annotations. Subject/source
separation will matter once a classifier or reference evaluation is introduced.

## Current posture goal

The supervisor's requested categories are upright, forward lean, backward lean,
bending over a desk, head resting on hand/desk and slouching. This comparison is
the first skeleton-quality/throughput check before selecting an existing posture
classifier or another evaluated decision layer. Forward lean, slouch and hand
support can overlap; occluded cases must not be forced into confident labels.

## Reproducibility and sources

Adapters are pinned to `ultralytics==8.3.200`, `rtmlib==0.0.14` and
`onnxruntime(-gpu)==1.23.2`. Colab's installed Torch/torchvision are recorded rather
than replaced. Original public checkpoint URLs and actual downloaded hashes are
recorded; no upstream cryptographic checksum verification is claimed.

- https://docs.ultralytics.com/models/yolo11/
- https://github.com/Tau-J/rtmlib
- https://github.com/open-mmlab/mmpose/tree/main/projects/rtmpose

Respect the model/code and underlying dataset terms. This repository contains
adapters, not redistributed model weights, student footage or personal results.
