# Head-pose validation - 16 November 2026

The official checkpoint requested by the friend's sixdrepnet wrapper was downloaded during controlled setup and verified: SHA-256 `9bdcd0e072df0d21e92288b40ea1d1b3cf7173c44728ab9c45c4d673f361e018`, 157315191 bytes. Its registered private transport parts stay outside GitHub. No friend footage or virtual environment was uploaded.

Actual-model CPU checks:

- Safe weights-only loading and identical preprocessing matched the upstream `SixDRepNet(gpu_id=-1, dict_path=...)` reference on a local face crop within 0.002 degrees.
- Batched and single prediction results matched within 0.002 degrees; track ordering was retained.
- The combined YuNet + refined ONNX actions + 6DRepNet pipeline produced six head estimates for six shared face detections on a local multi-face sample.
- Full image inference exported JSON observations, annotated output and review artifacts with head model hashes and source frame IDs. All media remained local.

Automated tests cover crop clipping/minimum size, Python-native serializable coordinates, Euler order/signs on known constructed rotations, missing checkpoints, separate ZIP/folder head loading, disabled feature isolation, head requiring shared faces, track/frame propagation, and both head benchmark cases. Existing workspace/scaling tests remain in the suite.

Independent review found duplicate head initialization keywords in benchmark cases and a ZIP/folder setup incompatibility. Regression tests reproduced both and fixes were applied. An actual export smoke test additionally caught NumPy crop integers not serializing to JSON; this was reproduced and corrected.

These checks establish execution and reference parity. They do not establish classroom angular accuracy, calibrated confidence, mirrored left/right categories, or hosted CUDA head-model throughput. A known-motion trial and the head-on/head-off scaling cases remain the next user evaluations. No head angle alone is an attention label.
