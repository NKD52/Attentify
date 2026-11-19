# Validation status

Checked 19 November 2026.

- Six focused tests passed: identical source-frame windows and reversed repeats,
  explicit failed-case retention, no full-image pose when no person is detected,
  invalid-setting rejection, refusal of a GPU run without CUDA, and incomplete
  PyTorch checkpoint rejection.
- The 36 existing portable-workspace tests also passed before publication.
- All generated notebook Python cells parse successfully.
- Real CPU smoke test passed using the original downloaded YOLO11s-Pose,
  YOLOX-m HumanArt and RTMPose-m models. Both pipelines produced 17-joint person
  proposals and exported matching raw/annotated review images. This establishes
  execution and schema compatibility, not pose accuracy or GPU performance.
- An interrupted YOLO checkpoint was detected during verification. The downloader
  now preserves an invalid cached checkpoint under an `.invalid-*` name, refuses
  partial HTTP responses/length mismatches, checks PyTorch archive integrity and
  installs complete downloads atomically. The fresh checkpoint loaded correctly.
- The supplied YOLOX ONNX has embedded NMS and outputs five values per box. Its
  wrapper's score/NMS settings are not necessarily embedded graph thresholds;
  the native final score filter is documented in exported engine configuration.

## Still requiring hosted evaluation

The notebook must be run on an actual Colab GPU to verify CUDA library/provider
compatibility, measure that runtime's speed and record its hardware. No GPU
speed result or minimum-hardware claim was inferred from these CPU checks.

The two pipelines' detection thresholds and native confidence scales differ.
Human review/reference annotations are needed to judge missed people, joint
ownership and occlusion quality. Neither pipeline includes a trained six-class
sitting-posture classifier or an attention predictor.
