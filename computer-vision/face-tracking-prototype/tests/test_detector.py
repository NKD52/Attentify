from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
from mediapipe.tasks.python import vision

from face_detector import FaceDetector, merge_boxes, tile_regions
from main import prepare_frame


class FakeLandmarker:
    def __init__(self):
        self.images = []
        self.timestamps = []

    def detect(self, image):
        self.images.append(image.numpy_view().copy())
        return SimpleNamespace(face_landmarks=[])

    def detect_for_video(self, image, timestamp):
        self.timestamps.append(timestamp)
        return self.detect(image)

    def close(self):
        pass


class DetectorTests(unittest.TestCase):
    @patch.object(Path, "is_file", return_value=True)
    def test_rgb_video_timestamps_and_separate_stateless_tiles(self, _):
        full, tiles = FakeLandmarker(), FakeLandmarker()
        with patch.object(vision.FaceLandmarker, "create_from_options", side_effect=[full, tiles]) as create:
            with FaceDetector(Path("unused.task"), tile_grid=2) as detector:
                frame = np.full((32, 48, 3), (10, 20, 30), dtype=np.uint8)
                detector.detect(frame, 0)
                detector.detect(frame, 40)
                self.assertEqual(full.timestamps, [0, 40])
                self.assertEqual(tiles.timestamps, [])
                self.assertEqual(len(tiles.images), 8)
                self.assertEqual(full.images[0][0, 0].tolist(), [30, 20, 10])
                with self.assertRaises(ValueError):
                    detector.detect(frame, 40)
            modes = [call.args[0].running_mode for call in create.call_args_list]
            self.assertEqual(modes, [vision.RunningMode.VIDEO, vision.RunningMode.IMAGE])

    @patch.object(Path, "is_file", return_value=True)
    def test_invalid_pixel_format_is_reported(self, _):
        with patch.object(vision.FaceLandmarker, "create_from_options", return_value=FakeLandmarker()):
            with FaceDetector(Path("unused.task")) as detector:
                with self.assertRaisesRegex(ValueError, "uint8 HxWx3 BGR"):
                    detector.detect(np.zeros((10, 10), dtype=np.uint8), 0)

    def test_tile_coverage_and_duplicate_suppression(self):
        regions = tile_regions(1280, 720, 4)
        self.assertEqual(len(regions), 16)
        self.assertEqual(regions[0][:2], (0, 0))
        x, y, w, h = regions[-1]
        self.assertEqual((x + w, y + h), (1280, 720))
        for x, y, w, h in regions:
            self.assertTrue(0 <= x < x + w <= 1280 and 0 <= y < y + h <= 720)
        self.assertEqual(merge_boxes([(10, 10, 50, 50), (12, 12, 48, 48),
                                     (100, 10, 140, 50)], 5),
                         [(10, 10, 50, 50), (100, 10, 140, 50)])
        self.assertEqual(len(merge_boxes([(0, 0, 10, 10), (20, 20, 30, 30)], 1)), 1)

    def test_native_and_aspect_preserving_resize(self):
        frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
        self.assertIs(prepare_frame(frame, 0), frame)
        self.assertEqual(prepare_frame(frame, 1280).shape, (720, 1280, 3))
        self.assertIs(prepare_frame(frame, 3840), frame)

    @patch.object(Path, "is_file", return_value=True)
    def test_adaptive_rechecks_faces_and_does_not_display_lost_boxes(self, _):
        full, tiles, roi = FakeLandmarker(), FakeLandmarker(), FakeLandmarker()
        frame = np.zeros((100, 200, 3), dtype=np.uint8)
        with patch.object(vision.FaceLandmarker, "create_from_options", side_effect=[full, tiles, roi]) as create:
            with FaceDetector(Path("unused.task"), tile_grid=2, adaptive=True, scan_every=3) as detector:
                # Controlled discovery. Later frames must use fresh crop results.
                with patch.object(detector, "_boxes", return_value=[(20, 20, 40, 40)]), \
                        patch.object(detector, "_crop_boxes", return_value=[]):
                    detector.detect(frame, 0)
                self.assertTrue(detector.last_counts["scan"])
                self.assertEqual(detector.last_counts["passes"], 1 + 4 + 9)
                for timestamp in (40, 80):
                    self.assertEqual(detector.detect(frame, timestamp), [])
                    self.assertFalse(detector.last_counts["scan"])
                    self.assertEqual(detector.last_counts["passes"], 1)
                self.assertEqual(full.timestamps, [0])
                detector.detect(frame, 120)
                self.assertTrue(detector.last_counts["scan"])
                self.assertEqual(full.timestamps, [0, 120])
                self.assertEqual(roi.timestamps, [])
                self.assertEqual(create.call_args_list[2].args[0].num_faces, 1)

    @patch.object(Path, "is_file", return_value=True)
    def test_adaptive_resolution_change_forces_new_scan(self, _):
        with patch.object(vision.FaceLandmarker, "create_from_options", side_effect=[FakeLandmarker() for _ in range(3)]):
            with FaceDetector(Path("unused.task"), tile_grid=2, adaptive=True) as detector:
                detector.detect(np.zeros((100, 200, 3), dtype=np.uint8), 0)
                detector.detect(np.zeros((120, 200, 3), dtype=np.uint8), 40)
                self.assertTrue(detector.last_counts["scan"])

    @patch.object(Path, "is_file", return_value=True)
    def test_crop_boxes_are_remapped_into_the_full_frame(self, _):
        fake = FakeLandmarker()
        with patch.object(vision.FaceLandmarker, "create_from_options", return_value=fake):
            with FaceDetector(Path("unused.task"), running_mode="IMAGE") as detector:
                with patch.object(detector, "_boxes", return_value=[(1, 2, 11, 12)]):
                    result = detector._crop_boxes(np.zeros((100, 200, 3), dtype=np.uint8), (30, 40, 50, 50))
                self.assertEqual(result, [(31, 42, 41, 52)])

    def test_adaptive_requires_a_tile_grid(self):
        with self.assertRaisesRegex(ValueError, "tile_grid"):
            FaceDetector(Path("unused.task"), adaptive=True)


if __name__ == "__main__":
    unittest.main()
