"""MediaPipe Tasks Face Landmarker wrapper and explicit model download."""

from pathlib import Path
from dataclasses import replace
from urllib.request import urlopen
import math
import shutil

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from tracker import Box, iou

MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
    "face_landmarker/float16/1/face_landmarker.task"
)
DEFAULT_MODEL = Path(__file__).resolve().parent / "models" / "face_landmarker.task"


def tile_regions(width: int, height: int, grid: int):
    """Cover the frame with overlapping crops (25% larger than grid cells)."""
    if grid < 2:
        return []
    tile_w, tile_h = min(width, math.ceil(width * 1.25 / grid)), min(height, math.ceil(height * 1.25 / grid))
    return [(round(col * (width - tile_w) / (grid - 1)),
             round(row * (height - tile_h) / (grid - 1)), tile_w, tile_h)
            for row in range(grid) for col in range(grid)]


def merge_boxes(boxes: list[Box], limit: int) -> list[Box]:
    """Suppress duplicate detections from overlapping tiles; prefer larger boxes."""
    kept = []
    for box in sorted(boxes, key=lambda b: (b[2] - b[0]) * (b[3] - b[1]), reverse=True):
        if not any(iou(box, other) > 0.3 for other in kept):
            kept.append(box)
        if len(kept) == limit:
            break
    return kept


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
    def __init__(self, model_path: Path, max_faces: int = 5, confidence: float = 0.5,
                 tile_grid: int = 1, running_mode: str = "VIDEO"):
        if running_mode not in ("IMAGE", "VIDEO") or tile_grid not in (1, 2, 3, 4):
            raise ValueError("Mode must be IMAGE/VIDEO and tile grid must be 1..4")
        if not model_path.is_file():
            raise FileNotFoundError(
                f"Model not found: {model_path}. Run: python main.py --download-model"
            )
        options = vision.FaceLandmarkerOptions(
            base_options=python.BaseOptions(
                model_asset_path=str(model_path), delegate=python.BaseOptions.Delegate.CPU
            ),
            running_mode=getattr(vision.RunningMode, running_mode),
            num_faces=max_faces,
            min_face_detection_confidence=confidence,
            min_face_presence_confidence=confidence,
            min_tracking_confidence=confidence,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False,
        )
        self.landmarker = vision.FaceLandmarker.create_from_options(options)
        self.tile_landmarker = None
        try:
            if tile_grid > 1 and running_mode == "VIDEO":
                # Crops are unrelated views: do not share VIDEO tracking state
                # or invent timestamps between tiles of the same source frame.
                self.tile_landmarker = vision.FaceLandmarker.create_from_options(
                    replace(options, running_mode=vision.RunningMode.IMAGE)
                )
        except Exception:
            self.landmarker.close()
            raise
        self.running_mode = running_mode
        self.tile_grid = tile_grid
        self.max_faces = max_faces
        self.last_counts = {"full_frame": 0, "tile_raw": 0, "merged": 0}
        self.last_timestamp_ms = -1

    def detect(self, frame, timestamp_ms: int) -> list[Box]:
        if self.running_mode == "VIDEO" and timestamp_ms <= self.last_timestamp_ms:
            raise ValueError("MediaPipe timestamps must be strictly increasing")
        self.last_timestamp_ms = timestamp_ms
        if frame.dtype != np.uint8 or frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError("Expected an OpenCV uint8 HxWx3 BGR frame")
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb))
        result = (self.landmarker.detect_for_video(image, timestamp_ms)
                  if self.running_mode == "VIDEO" else self.landmarker.detect(image))
        height, width = frame.shape[:2]
        boxes = self._boxes(result, width, height)
        full_count = len(boxes)
        tile_count = 0
        for x, y, tile_w, tile_h in tile_regions(width, height, self.tile_grid):
            crop = mp.Image(image_format=mp.ImageFormat.SRGB,
                            data=np.ascontiguousarray(rgb[y:y + tile_h, x:x + tile_w]))
            detector = self.tile_landmarker if self.tile_landmarker is not None else self.landmarker
            tile_boxes = self._boxes(detector.detect(crop), tile_w, tile_h)
            tile_count += len(tile_boxes)
            boxes.extend((left + x, top + y, right + x, bottom + y)
                         for left, top, right, bottom in tile_boxes)
        if self.tile_grid > 1:
            boxes = merge_boxes(boxes, self.max_faces)
        self.last_counts = {"full_frame": full_count, "tile_raw": tile_count, "merged": len(boxes)}
        return boxes

    @staticmethod
    def _boxes(result, width: int, height: int) -> list[Box]:
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
        try:
            self.landmarker.close()
        finally:
            if self.tile_landmarker is not None:
                self.tile_landmarker.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
