from contextlib import redirect_stdout, redirect_stderr
from io import StringIO
from pathlib import Path
import tempfile
import unittest
from unittest.mock import MagicMock, patch

import numpy as np

from main import parse_args, run


class OutputTests(unittest.TestCase):
    def test_no_output_never_opens_writer(self):
        args = parse_args(["--camera", "0", "--no-output"])
        capture = MagicMock()
        capture.isOpened.return_value = True
        capture.get.return_value = 25
        capture.read.side_effect = [(True, np.zeros((120, 160, 3), dtype=np.uint8)), (False, None)]
        with patch("main.cv2.VideoCapture", return_value=capture), patch("main.cv2.VideoWriter") as writer, \
                patch("main.FaceDetector") as detector, redirect_stdout(StringIO()) as log:
            detector.return_value.__enter__.return_value.detect.return_value = []
            run(args)
            writer.assert_not_called()
            capture.release.assert_called_once()
        self.assertIn("no output video saved", log.getvalue())
        self.assertNotIn("Saved 1 annotated", log.getvalue())

    def test_existing_output_requires_explicit_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "result.mp4"
            output.write_bytes(b"previous output")
            args = parse_args(["--camera", "0", "--output", str(output)])
            with self.assertRaisesRegex(FileExistsError, "--overwrite"):
                run(args)
            self.assertEqual(output.read_bytes(), b"previous output")
            args.overwrite = True
            with patch("main.cv2.VideoCapture") as capture:
                capture.return_value.isOpened.return_value = False
                with self.assertRaisesRegex(RuntimeError, "Cannot open input"):
                    run(args)
            # Failed input initialization does not truncate the old output.
            self.assertEqual(output.read_bytes(), b"previous output")

    def test_input_cannot_be_overwritten(self):
        with tempfile.TemporaryDirectory() as directory:
            video = Path(directory) / "input.mp4"
            video.write_bytes(b"input")
            args = parse_args(["--video", str(video), "--output", str(video), "--overwrite"])
            with self.assertRaisesRegex(ValueError, "different from input"):
                run(args)
            self.assertEqual(video.read_bytes(), b"input")

    def test_conflicting_output_options_are_rejected(self):
        with redirect_stderr(StringIO()):
            for flags in (["--no-output", "--overwrite"], ["--no-output", "--output", "out.mp4"]):
                with self.assertRaises(SystemExit):
                    parse_args(["--camera", "0", *flags])


if __name__ == "__main__":
    unittest.main()
