# Attentify

Final Year Project (FYP) repository for student attention detection research.

## Structure

- [`computer-vision/face-tracking-prototype/`](computer-vision/face-tracking-prototype/): Prototype 1, local multi-face detection and temporary tracking.
- [`computer-vision/detector-comparison/`](computer-vision/detector-comparison/): Windows launcher for Attentify MediaPipe and OpenCV Zoo YuNet on cameras, clips, and images.
- `experiments/random-forest/`: reserved for Random Forest experiments.

## Status

Prototype 1 implements local video input, MediaPipe Face Landmarker, temporary
geometric tracking IDs, face counts, processing FPS, and annotated MP4 output.
See its README for installation, usage, classroom testing, and limitations.
It does not implement recognition, permanent student identity, attention
classification, cross-class histories, or a dashboard.

## Data privacy

Do not commit private student videos, face images, personal information, or private datasets. Keep research data outside this repository.
