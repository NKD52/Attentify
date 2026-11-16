# Head-pose integration

Implementation target: optional versioned 6DRepNet using the same source-frame YuNet face detections as the face/action pipeline. No second face detector and no attentive/inattentive label is added.

The pretrained checkpoint is the same 300W-LP/AFLW2000 model requested by the friend's `sixdrepnet==0.1.6` wrapper. Its provenance, checksum and private storage parts are registered. Inference never silently downloads weights. Setup loads verified local/private-storage assets.

Output: one timestamped head observation per detected face, including temporary track, original/crop box, pitch/yaw/roll degrees, finite-estimate status or unknown reason, model version and separate face detector score. Head confidence stays null because this regressor does not provide a calibrated confidence. Angles follow the upstream Euler convention; human left/right and camera mirroring require a known-motion check before categorical directions are used.

Preprocessing mirrors the upstream wrapper: BGR to RGB, PIL image, resize shorter edge to 224, center crop 224, tensor and ImageNet normalization. Valid face crops are batched with a configurable chunk limit. CPU and CUDA are explicit; no silent GPU-to-CPU fallback. Head inference is disabled by default in the Python API, preserving existing callers. The notebook exposes the toggle and backend.

Tests cover invalid/partial/small crops, Euler order and signs on constructed rotations, track/source association, missing weights and disabled-feature isolation. Actual-model tests compare the single-face path to the upstream wrapper and batched results to single predictions; a combined media smoke run verifies export and timings. Such execution checks do not establish classroom head-angle accuracy.

Updates use a new model ID/checksum and compatible adapter settings; previous code/model versions remain reproducible. The new head phase is timed separately and included in scaling cases only when explicitly enabled.
