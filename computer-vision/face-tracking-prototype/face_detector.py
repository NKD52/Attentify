"""MediaPipe Tasks Face Landmarker wrapper and explicit model download."""

from pathlib import Path
from urllib.request import urlopen
import math
import shutil

import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from tracker import Box

MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
    "face_landmarker/float16/1/face_landmarker.task"
)
DEFAULT_MODEL = Path(__file__).resolve().parent / "models" / "face_landmarker.task"


def download_model(destination: Path) -> None:
    """Download Google's pretrained model once, without transmitting video."""
    if destination.exists():
        raise FileExistsError(f"Model already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(".download")
    try:
        with urlopen(MODEL_URL, timeout=60) as response, temporary.open("wb") as output:
            shutil.copyfileobj(response, output)
        if temporary.stat().st_size < 1_000_000:
            raise RuntimeError("Downloaded model is unexpectedly small")
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)


class FaceDetector:
    def __init__(self, model_path: Path, max_faces: int = 5, confidence: float = 0.5):
        if not model_path.is_file():
            raise FileNotFoundError(
                f"Model not found: {model_path}. Run: python main.py --download-model"
            )
        options = vision.FaceLandmarkerOptions(
            base_options=python.BaseOptions(
                model_asset_path=str(model_path), delegate=python.BaseOptions.Delegate.CPU
            ),
            running_mode=vision.RunningMode.VIDEO,
            num_faces=max_faces,
            min_face_detection_confidence=confidence,
            min_face_presence_confidence=confidence,
            min_tracking_confidence=confidence,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False,
        )
        self.landmarker = vision.FaceLandmarker.create_from_options(options)
        self.last_timestamp_ms = -1

    def detect(self, frame, timestamp_ms: int) -> list[Box]:
        if timestamp_ms <= self.last_timestamp_ms:
            raise ValueError("MediaPipe timestamps must be strictly increasing")
        self.last_timestamp_ms = timestamp_ms
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = self.landmarker.detect_for_video(
            mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), timestamp_ms
        )
        height, width = frame.shape[:2]
        boxes = []
        for landmarks in result.face_landmarks:
            xs, ys = [p.x for p in landmarks], [p.y for p in landmarks]
            if not xs or not all(math.isfinite(v) for v in xs + ys):
                continue
            # Small margin around the landmark extent; clamp to the image.
            pad_x, pad_y = (max(xs) - min(xs)) * 0.08, (max(ys) - min(ys)) * 0.08
            left = max(0, min(width - 1, math.floor((min(xs) - pad_x) * width)))
            top = max(0, min(height - 1, math.floor((min(ys) - pad_y) * height)))
            right = max(0, min(width - 1, math.ceil((max(xs) + pad_x) * width)))
            bottom = max(0, min(height - 1, math.ceil((max(ys) + pad_y) * height)))
            if right > left and bottom > top:
                boxes.append((left, top, right, bottom))
        return boxes

    def close(self) -> None:
        self.landmarker.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
