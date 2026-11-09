"""Run Attentify MediaPipe and OpenCV Zoo YuNet on a shared source."""

import argparse
from contextlib import ExitStack
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
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff")
MODES = {
    "both": ("MediaPipe", "YuNet"),
    "mediapipe": ("MediaPipe",),
    "yunet": ("YuNet",),
}


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


def build_processors(mode, stack):
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
            FaceDetector(DEFAULT_MODEL, max_faces=50, tile_grid=4, adaptive=True)
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

        detector_yunet = YuNet(str(model), confThreshold=0.9)

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
            return len(faces)

        processors["YuNet"] = yunet
    return processors


def run(mode, source, *, image=False, headless=False, limit=None):
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
        processors = build_processors(mode, stack)
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
            height, width = frame.shape[:2]
            if width > 640:
                frame = cv2.resize(
                    frame, (640, max(1, round(height * 640 / width))),
                    interpolation=cv2.INTER_AREA,
                )
            frames += 1
            panels = []
            for name, process in processors.items():
                panel = frame.copy()
                begin = perf_counter()
                count = process(panel, seconds, timestamp)
                durations[name] += perf_counter() - begin
                counts[name] += count
                if not headless:
                    label = f"{name} | faces {count} | frame {frames}"
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


def open_launcher():
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
            run(choices[mode.get()], source, image=image)
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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=MODES, default="both")
    sources = parser.add_mutually_exclusive_group()
    sources.add_argument("--camera", type=int)
    sources.add_argument("--clip", type=Path)
    sources.add_argument("--image", type=Path)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")
    if args.camera is not None and args.camera < 0:
        parser.error("--camera must be nonnegative")
    if args.camera is None and args.clip is None and args.image is None:
        if args.headless or args.limit is not None:
            parser.error("--headless and --limit require --camera, --clip, or --image")
        open_launcher()
    else:
        path = args.image or args.clip
        source = args.camera if path is None else str(path)
        try:
            run(args.mode, source, image=args.image is not None,
                headless=args.headless, limit=args.limit)
        except (OSError, RuntimeError, ValueError) as error:
            parser.exit(1, f"{error}\n")


if __name__ == "__main__":
    main()
