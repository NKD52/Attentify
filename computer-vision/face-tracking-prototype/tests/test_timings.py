from contextlib import redirect_stdout
from io import StringIO
from itertools import count
import unittest
from unittest.mock import MagicMock, patch

import numpy as np

from main import TimingStats, parse_args, print_timings, run


class TimingTests(unittest.TestCase):
    def test_online_aggregation_and_phase_share(self):
        search, regular = TimingStats(), TimingStats()
        search.add(0.080)
        regular.add(0.010)
        regular.add(0.030)
        self.assertEqual(regular.count, 2)
        self.assertAlmostEqual(regular.average_ms, 20)
        self.assertAlmostEqual(regular.minimum, 0.010)
        self.assertAlmostEqual(regular.maximum, 0.030)
        with redirect_stdout(StringIO()) as log:
            print_timings({"search": search, "regular": regular}, {}, 0.5, 0.12)
        self.assertIn("startup_ms=500.00", log.getvalue())
        self.assertIn("phase=search frames=1", log.getvalue())
        self.assertIn("detector_share_pct=66.7", log.getvalue())
        self.assertIn("phase=regular frames=2", log.getvalue())

    def test_summary_only_groups_search_and_regular_without_frame_logging(self):
        args = parse_args(["--camera", "0", "--tile-grid", "2", "--adaptive", "--no-output", "--timings"])
        capture = MagicMock()
        capture.isOpened.return_value = True
        capture.get.return_value = 25
        frame = np.zeros((120, 160, 3), dtype=np.uint8)
        capture.read.side_effect = [(True, frame.copy()), (True, frame.copy()), (False, None)]
        detector = MagicMock()
        scans = iter([True, False])

        def detect(*_):
            detector.last_counts = {"scan": next(scans)}
            return []

        detector.detect.side_effect = detect
        clock = count(0, 0.001)
        with patch("main.cv2.VideoCapture", return_value=capture), patch("main.FaceDetector") as factory, \
                patch("main.perf_counter", side_effect=lambda: next(clock)), redirect_stdout(StringIO()) as log:
            factory.return_value.__enter__.return_value = detector
            run(args)
        output = log.getvalue()
        self.assertIn("phase=search frames=1", output)
        self.assertIn("phase=regular frames=1", output)
        self.assertIn("stage=read calls=2", output)
        self.assertIn("stage=resize calls=2", output)
        self.assertNotIn("[frame]", output)
        self.assertNotIn("stage=write", output)
        self.assertNotIn("stage=display", output)

    def test_ctrl_c_still_prints_completed_frame_timings(self):
        args = parse_args(["--camera", "0", "--no-output", "--timings"])
        capture = MagicMock()
        capture.isOpened.return_value = True
        capture.get.return_value = 25
        capture.read.side_effect = [(True, np.zeros((120, 160, 3), dtype=np.uint8)), KeyboardInterrupt]
        with patch("main.cv2.VideoCapture", return_value=capture), patch("main.FaceDetector") as factory, \
                redirect_stdout(StringIO()) as log:
            factory.return_value.__enter__.return_value.detect.return_value = []
            run(args)
        self.assertIn("Stopped by user", log.getvalue())
        self.assertIn("phase=fixed frames=1", log.getvalue())
        capture.release.assert_called_once()

    def test_preview_and_logging_are_timed_outside_detection(self):
        args = parse_args(["--camera", "0", "--no-output", "--display", "--diagnostics"])
        capture = MagicMock()
        capture.isOpened.return_value = True
        capture.get.return_value = 25
        capture.read.return_value = (True, np.zeros((120, 160, 3), dtype=np.uint8))
        with patch("main.cv2.VideoCapture", return_value=capture), patch("main.FaceDetector") as factory, \
                patch("main.cv2.imshow"), patch("main.cv2.waitKey", return_value=ord("q")), \
                patch("main.cv2.destroyAllWindows"), redirect_stdout(StringIO()) as log:
            factory.return_value.__enter__.return_value.detect.return_value = []
            run(args)
        output = log.getvalue()
        self.assertIn("phase=fixed detector_ms=", output)
        self.assertIn("display_ms=", output)
        self.assertIn("stage=display calls=1", output)
        self.assertIn("stage=logging calls=1", output)
        self.assertIn("phase=fixed frames=1", output)


if __name__ == "__main__":
    unittest.main()
