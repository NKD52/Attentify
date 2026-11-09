"""Run Attentify MediaPipe and OpenCV Zoo YuNet on a shared source."""

import argparse
from contextlib import ExitStack
from dataclasses import dataclass
from math import isfinite
from pathlib import Path
import sys
from time import perf_counter
from urllib.request import urlopen


HERE = Path(__file__).resolve().parent
CACHE = HERE / "assets"
ATTENTIFY = HERE.parent / "face-tracking-prototype"
ZOO_REV = "48b10fa505aeb51fbd127fc60a5c660040e823b7"
ZOO_MODEL = "face_detection_yunet_2023mar.onnx"
ZOO_PATH = f"opencv/opencv_zoo/{ZOO_REV}/models/face_detection_yunet"
# OpenCV Zoo's BGR colors, in YuNet's output order.
YUNET_LANDMARK_COLORS = (
    (255, 0, 0),    # Right eye
    (0, 0, 255),    # Left eye
    (0, 255, 0),    # Nose tip
    (255, 0, 255),  # Right mouth corner
    (0, 255, 255),  # Left mouth corner
)
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff")
MODES = {
    "both": ("MediaPipe", "YuNet"),
    "mediapipe": ("MediaPipe",),
    "yunet": ("YuNet",),
}


@dataclass(frozen=True)
class DetectorSettings:
    mp_width: int = 640
    mp_confidence: float = 0.5
    mp_max_faces: int = 50
    mp_tile_grid: int = 4
    mp_adaptive: bool = True
    mp_scan_every: int = 20
    yunet_width: int = 640
    yunet_confidence: float = 0.9
    yunet_nms: float = 0.3
    yunet_top_k: int = 5000


DEFAULT_SETTINGS = DetectorSettings()


def download(url, target, minimum_size=1):
    if target.is_file() and target.stat().st_size >= minimum_size:
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".part")
    print(f"Downloading {target.name} ...", flush=True)
    try:
        with urlopen(url, timeout=45) as response:
            temporary.write_bytes(response.read())
        if temporary.stat().st_size < minimum_size:
            raise RuntimeError(f"Incomplete download for {target.name}")
        temporary.replace(target)
    finally:
        temporary.unlink(missing_ok=True)
    return target


def build_processors(mode, stack, settings=DEFAULT_SETTINGS):
    import cv2

    processors = {}
    if "MediaPipe" in MODES[mode]:
        if str(ATTENTIFY) not in sys.path:
            sys.path.insert(0, str(ATTENTIFY))
        from face_detector import DEFAULT_MODEL, FaceDetector, download_model
        from tracker import FaceTracker

        if not DEFAULT_MODEL.is_file():
            print(f"Downloading {DEFAULT_MODEL.name} ...", flush=True)
            download_model(DEFAULT_MODEL)
        detector = stack.enter_context(
            FaceDetector(
                DEFAULT_MODEL,
                max_faces=settings.mp_max_faces,
                confidence=settings.mp_confidence,
                tile_grid=settings.mp_tile_grid,
                adaptive=settings.mp_adaptive,
                scan_every=settings.mp_scan_every,
            )
        )
        tracker = FaceTracker(1.0, 1.0)

        def mediapipe(frame, seconds, timestamp_ms):
            tracks = tracker.update(detector.detect(frame, timestamp_ms), seconds)
            for track in tracks:
                x1, y1, x2, y2 = track.box
                cv2.rectangle(frame, (x1, y1), (x2, y2), (80, 230, 100), 2)
                cv2.putText(
                    frame, f"ID {track.track_id}", (x1, max(15, y1 - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (80, 230, 100), 1,
                )
            return len(tracks)

        processors["MediaPipe"] = mediapipe

    if "YuNet" in MODES[mode]:
        source = CACHE / "zoo"
        download(f"https://raw.githubusercontent.com/{ZOO_PATH}/yunet.py", source / "yunet.py")
        model = download(
            f"https://media.githubusercontent.com/media/{ZOO_PATH}/{ZOO_MODEL}",
            CACHE / ZOO_MODEL, 100_000,
        )
        if str(source) not in sys.path:
            sys.path.insert(0, str(source))
        from yunet import YuNet

        detector_yunet = YuNet(
            str(model),
            confThreshold=settings.yunet_confidence,
            nmsThreshold=settings.yunet_nms,
            topK=settings.yunet_top_k,
        )

        def yunet(frame, seconds, timestamp_ms):
            height, width = frame.shape[:2]
            detector_yunet.setInputSize((width, height))
            faces = detector_yunet.infer(frame)
            for face in faces:
                x, y, w, h = map(int, face[:4])
                cv2.rectangle(frame, (x, y), (x + w, y + h), (70, 200, 255), 2)
                cv2.putText(
                    frame, f"{face[-1]:.2f}", (x, max(15, y - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (70, 200, 255), 1,
                )
                landmarks = face[4:14].astype(int).reshape(5, 2)
                for point, color in zip(landmarks, YUNET_LANDMARK_COLORS):
                    cv2.circle(frame, tuple(point), 2, color, 2)
            return len(faces)

        processors["YuNet"] = yunet
    return processors


def resize_frame(frame, width, height=None):
    import cv2

    current_height, current_width = frame.shape[:2]
    if height is None:
        height = max(1, round(current_height * width / current_width))
    if (width, height) == (current_width, current_height):
        return frame
    interpolation = cv2.INTER_AREA if width < current_width else cv2.INTER_LINEAR
    return cv2.resize(frame, (width, height), interpolation=interpolation)


def run(mode, source, *, image=False, headless=False, limit=None, settings=DEFAULT_SETTINGS):
    import cv2

    title = "Attentify detector comparison"
    counts = dict.fromkeys(MODES[mode], 0)
    durations = dict.fromkeys(MODES[mode], 0.0)
    frames = 0
    with ExitStack() as stack:
        if image:
            import numpy as np

            data = Path(source).read_bytes()
            still = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR) if data else None
            if still is None:
                raise RuntimeError(f"Could not read image {source!r}.")
            limit = 1
            fps = 1.0
        else:
            capture = cv2.VideoCapture(source)
            stack.callback(capture.release)
            if not capture.isOpened():
                raise RuntimeError(f"Could not open {source!r}. Check the file or camera index.")
            fps = capture.get(cv2.CAP_PROP_FPS)
            if not isfinite(fps) or fps <= 0:
                fps = 30.0
        processors = build_processors(mode, stack, settings)
        width_caps = {"MediaPipe": settings.mp_width, "YuNet": settings.yunet_width}
        if not headless:
            cv2.namedWindow(title, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(title, 640 * len(processors), 480)
            stack.callback(cv2.destroyAllWindows)

        camera = isinstance(source, int)
        start = perf_counter()
        last_timestamp = -1
        while limit is None or frames < limit:
            ok, frame = (True, still) if image else capture.read()
            if not ok:
                if camera or not frames:
                    raise RuntimeError(f"Could not read a frame from {source!r}.")
                break
            seconds = perf_counter() - start if camera else frames / fps
            timestamp = max(last_timestamp + 1, round(seconds * 1000))
            last_timestamp = timestamp
            source_height, source_width = frame.shape[:2]
            widths = {
                name: min(width_caps[name] or source_width, source_width) for name in processors
            }
            resized_frames = {width: resize_frame(frame, width) for width in set(widths.values())}
            frames += 1
            if frames == 1:
                print("Input widths: " + ", ".join(f"{name} {width} px" for name, width in widths.items()))
            panels = []
            for name, process in processors.items():
                width = widths[name]
                panel = resized_frames[width].copy()
                begin = perf_counter()
                count = process(panel, seconds, timestamp)
                durations[name] += perf_counter() - begin
                counts[name] += count
                if not headless:
                    label = f"{name} | {width} px | {count} faces | F{frames}"
                    display_width = min(640, source_width)
                    display_height = max(1, round(source_height * display_width / source_width))
                    panel = resize_frame(panel, display_width, display_height)
                    panel = cv2.copyMakeBorder(
                        panel, 28, 0, 0, 0, cv2.BORDER_CONSTANT, value=(25, 25, 25)
                    )
                    cv2.putText(panel, label, (8, 20), cv2.FONT_HERSHEY_SIMPLEX,
                                0.55, (255, 255, 255), 1)
                    panels.append(panel)
            if not headless:
                cv2.imshow(title, cv2.hconcat(panels))
                key = cv2.waitKey(1) & 0xFF
                if image or key == ord(" "):
                    while cv2.getWindowProperty(title, cv2.WND_PROP_VISIBLE) >= 1:
                        if key in (ord("q"), 27):
                            break
                        key = cv2.waitKey(50) & 0xFF
                        if not image and key == ord(" "):
                            break
                if key in (ord("q"), 27) or cv2.getWindowProperty(
                    title, cv2.WND_PROP_VISIBLE
                ) < 1:
                    break

    print(f"Processed {frames} source frames.")
    for name in processors:
        print(f"{name}: {counts[name]} total boxes, {durations[name] * 1000 / frames:.1f} ms/frame "
              "for detection, tracking if enabled, and box drawing")


def open_launcher(settings=DEFAULT_SETTINGS):
    import tkinter as tk
    from tkinter import filedialog, messagebox, ttk

    root = tk.Tk()
    root.title("Attentify Detector Compare")
    root.resizable(False, False)
    form = ttk.Frame(root, padding=18)
    form.grid(sticky="nsew")
    mode = tk.StringVar(value="Side by side")
    camera = tk.StringVar(value="0")
    clip = tk.StringVar()
    choices = {"Side by side": "both", "Attentify MediaPipe": "mediapipe", "OpenCV Zoo YuNet": "yunet"}

    def browse():
        path = filedialog.askopenfilename(
            filetypes=[
                ("Videos and images", tuple(f"*{ext}" for ext in IMAGE_EXTENSIONS)
                 + ("*.mp4", "*.avi", "*.mov", "*.mkv", "*.webm", "*.m4v")),
                ("All files", "*.*"),
            ]
        )
        if path:
            clip.set(path)

    def preview(use_camera):
        try:
            if use_camera:
                value = camera.get().strip()
                if not value.isascii() or not value.isdecimal():
                    raise ValueError("Enter camera index 0, 1, 2, and so on.")
                source = int(value)
            else:
                source = clip.get().strip()
            if not use_camera and (not source or not Path(source).is_file()):
                raise ValueError("Browse to a video clip or image first.")
            image = not use_camera and Path(source).suffix.lower() in IMAGE_EXTENSIONS
            root.withdraw()
            root.update()
            run(choices[mode.get()], source, image=image, settings=settings)
        except Exception as error:
            root.deiconify()
            messagebox.showerror("Could not run preview", str(error), parent=root)
        finally:
            root.deiconify()

    ttk.Label(form, text="Detector mode").grid(row=0, column=0, sticky="w", pady=5)
    ttk.Combobox(form, textvariable=mode, values=list(choices), state="readonly",
                 width=25).grid(row=0, column=1, sticky="w")
    ttk.Label(form, text="Camera index").grid(row=1, column=0, sticky="w", pady=5)
    ttk.Spinbox(form, from_=0, to=20, textvariable=camera, width=6).grid(row=1, column=1, sticky="w")
    ttk.Button(form, text="Start camera", command=lambda: preview(True)).grid(row=1, column=2, sticky="ew")
    ttk.Label(form, text="Clip or image").grid(row=2, column=0, sticky="w", pady=5)
    ttk.Entry(form, textvariable=clip, width=42).grid(row=2, column=1, padx=8)
    ttk.Button(form, text="Browse...", command=browse).grid(row=2, column=2, sticky="ew")
    ttk.Button(form, text="Open file", command=lambda: preview(False)).grid(row=3, column=2, sticky="ew")
    ttk.Label(form, text="Preview controls   Space pauses or resumes   Q or Esc closes").grid(
        row=4, column=0, columnspan=3, sticky="w", pady=(14, 0),
    )
    root.mainloop()


def probability(value):
    number = float(value)
    if not 0 <= number <= 1:
        raise argparse.ArgumentTypeError("expected a value from 0 to 1")
    return number


def positive_int(value):
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("expected a positive integer")
    return number


def input_width(value):
    width = int(value)
    if width != 0 and width < 320:
        raise argparse.ArgumentTypeError("expected 0 (native width) or at least 320")
    return width


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        fromfile_prefix_chars="@",
    )
    parser.add_argument("--mode", choices=MODES, default="both")
    sources = parser.add_mutually_exclusive_group()
    sources.add_argument("--camera", type=int)
    sources.add_argument("--clip", type=Path)
    sources.add_argument("--image", type=Path)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--limit", type=int)
    mp = parser.add_argument_group("Attentify MediaPipe settings")
    mp.add_argument("--mp-width", type=input_width, help="maximum input width, or 0 for native")
    mp.add_argument("--mp-confidence", type=probability, help="detection, presence and tracking threshold")
    mp.add_argument("--mp-max-faces", type=positive_int, help="maximum number of faces")
    mp.add_argument("--mp-tile-grid", type=int, choices=(1, 2, 3, 4), help="grid size per axis; 1 disables tiling")
    mp.add_argument("--mp-adaptive", action=argparse.BooleanOptionalAction, help="enable adaptive detection")
    mp.add_argument("--mp-scan-every", type=positive_int, help="frames between adaptive scans")
    yu = parser.add_argument_group("OpenCV Zoo YuNet settings")
    yu.add_argument("--yunet-width", type=input_width, help="maximum input width, or 0 for native")
    yu.add_argument("--yunet-confidence", type=probability, help="detection confidence threshold")
    yu.add_argument("--yunet-nms", type=probability, help="overlap threshold for suppressing duplicate boxes")
    yu.add_argument("--yunet-top-k", type=positive_int, help="candidate limit before duplicate suppression")
    parser.set_defaults(**vars(DEFAULT_SETTINGS))
    args = parser.parse_args()
    settings = DetectorSettings(**{name: getattr(args, name) for name in vars(DEFAULT_SETTINGS)})
    if settings.mp_adaptive and settings.mp_tile_grid < 2:
        parser.error("--mp-tile-grid 1 requires --no-mp-adaptive")
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")
    if args.camera is not None and args.camera < 0:
        parser.error("--camera must be nonnegative")
    if args.camera is None and args.clip is None and args.image is None:
        if args.headless or args.limit is not None:
            parser.error("--headless and --limit require --camera, --clip, or --image")
        open_launcher(settings)
    else:
        path = args.image or args.clip
        source = args.camera if path is None else str(path)
        try:
            run(args.mode, source, image=args.image is not None,
                headless=args.headless, limit=args.limit, settings=settings)
        except (OSError, RuntimeError, ValueError) as error:
            parser.exit(1, f"{error}\n")


if __name__ == "__main__":
    main()
