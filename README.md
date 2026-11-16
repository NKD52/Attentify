# Attentify

Final Year Project (FYP) repository for student attention detection research.

## Structure

- [`computer-vision/colab/`](computer-vision/colab/): portable face/action testing workspace. Open its notebook from GitHub, load private models from Drive or a local path, select faces/actions/combined, and export predictions and offline reviews.
- [`computer-vision/team-pilot/`](computer-vision/team-pilot/): shared preprocessing, decoding, tracking and association helpers reused by the portable workspace. The private Windows camera handoff remains a separate bundle.

- [`computer-vision/face-tracking-prototype/`](computer-vision/face-tracking-prototype/): Prototype 1, local multi-face detection and temporary tracking.
- [`computer-vision/detector-comparison/`](computer-vision/detector-comparison/): Windows launcher for Attentify MediaPipe and OpenCV Zoo YuNet on cameras, clips, and images.
- `experiments/random-forest/`: reserved for Random Forest experiments.

## Status

The portable workspace runs the existing YuNet and experimental refined SCB
YOLOv7 model independently or together. Its model register preserves the
original action baseline and a registered optional 6DRepNet head-pose adapter. See
[`docs/EXPERIMENT-REGISTER.md`](docs/EXPERIMENT-REGISTER.md) for selected historical
work and migration limitations. Basic face/action Colab video and camera execution
has been tested. Head pose is checked locally for execution and upstream parity;
its hosted GPU performance and classroom angular accuracy still require evaluation.

Enable `ENABLE_HEAD_POSE` in the notebook to show pitch/yaw/roll per shared face
track, or disable it to compare the face/action baseline. The notebooks load
`main` by default. See [head-pose integration](docs/HEADPOSE-INTEGRATION.md).

Prototype 1 implements local video input, MediaPipe Face Landmarker, temporary
geometric tracking IDs, face counts, processing FPS, and annotated MP4 output.
See its README for installation, usage, classroom testing, and limitations.
It does not implement recognition, permanent student identity, attention
classification, cross-class histories, or a dashboard.

## Data privacy

Do not commit private student videos, face images, personal information, or private datasets. Keep research data outside this repository.
