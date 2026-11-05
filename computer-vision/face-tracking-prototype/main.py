"""Local multi-face detection, temporary tracking, and annotated video output."""

import argparse
from collections import deque
import math
from pathlib import Path
import sys
from time import perf_counter

import cv2
import mediapipe as mp

from face_detector import DEFAULT_MODEL, FaceDetector, download_model
from tracker import FaceTracker


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--video", type=Path, help="Local input video path")
    source.add_argument("--camera", type=int, help="Webcam index, e.g. 0")
    parser.add_argument("--output", type=Path, default=Path("outputs/annotated.mp4"))
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--download-model", action="store_true", help="Download model and exit")
    parser.add_argument("--max-faces", type=int, default=5)
    parser.add_argument("--sample-every", type=int, default=1, help="Process every Nth frame")
    parser.add_argument("--max-width", type=int, default=1280, help="Resize down to this width; 0 keeps native size")
    parser.add_argument("--confidence", type=float, default=0.5)
    parser.add_argument("--tile-grid", type=int, choices=(1, 2, 3, 4), default=1,
                        help="Optional overlapping NxN IMAGE crops; 1 disables tiling")
    parser.add_argument("--diagnostics", action="store_true", help="Print config and counts for every sampled frame")
    parser.add_argument("--inspect-frame", type=int, metavar="INDEX",
                        help="Compare one zero-based video frame in fresh IMAGE/VIDEO modes, then exit")
    parser.add_argument("--max-lost-seconds", type=float, default=1.0)
    parser.add_argument("--match-distance", type=float, default=1.0,
                        help="Maximum match distance in face-box diagonals")
    parser.add_argument("--fps", type=float, help="Override input FPS; required if metadata is invalid")
    parser.add_argument("--display", action="store_true", help="Show preview; Q or Esc stops")
    args = parser.parse_args()
    if not args.download_model and args.video is None and args.camera is None:
        parser.error("provide --video PATH or --camera INDEX")
    if min(args.max_faces, args.sample_every) < 1 or (args.max_width != 0 and args.max_width < 320):
        parser.error("max-faces/sample-every must be positive; max-width must be 0 or >= 320")
    if args.inspect_frame is not None and (args.video is None or args.inspect_frame < 0):
        parser.error("inspect-frame requires --video and a nonnegative frame index")
    if not math.isfinite(args.confidence) or not 0 <= args.confidence <= 1:
        parser.error("confidence must be between 0 and 1")
    if not math.isfinite(args.max_lost_seconds) or args.max_lost_seconds < 0:
        parser.error("max-lost-seconds must be finite and >= 0")
    if not math.isfinite(args.match_distance) or args.match_distance <= 0:
        parser.error("match-distance must be finite and > 0")
    if args.fps is not None and (not math.isfinite(args.fps) or args.fps <= 0):
        parser.error("fps must be finite and > 0")
    if args.camera is not None and args.camera < 0:
        parser.error("camera index must be >= 0")
    return args


def prepare_frame(frame, max_width):
    height, width = frame.shape[:2]
    if max_width and width > max_width:
        return cv2.resize(frame, (max_width, max(1, round(height * max_width / width))),
                          interpolation=cv2.INTER_AREA)
    return frame


def inspect_frame(args):
    """Compare decoded pixels directly; do not save student images or video."""
    capture = cv2.VideoCapture(str(args.video))
    try:
        if not capture.isOpened():
            raise RuntimeError("Cannot open video for frame inspection")
        # Sequential decoding avoids approximate keyframe seeks selecting a different frame.
        for _ in range(args.inspect_frame + 1):
            ok, frame = capture.read()
            if not ok:
                raise ValueError("inspect-frame is beyond readable video frames")
        print(f"[inspect] frame={args.inspect_frame} orientation_meta={capture.get(cv2.CAP_PROP_ORIENTATION_META)} "
              f"orientation_auto={capture.get(cv2.CAP_PROP_ORIENTATION_AUTO)} "
              f"opencv={cv2.__version__} mediapipe={mp.__version__}")
        for label, pixels in (("native", frame), ("processed", prepare_frame(frame, args.max_width))):
            for mode in ("IMAGE", "VIDEO"):
                with FaceDetector(args.model, args.max_faces, args.confidence, args.tile_grid, mode) as detector:
                    boxes = detector.detect(pixels, 0)
                    print(f"[inspect] size={label}:{pixels.shape[1]}x{pixels.shape[0]} "
                          f"dtype={pixels.dtype} input=BGR inference=RGB mode={mode} timestamp_ms=0 "
                          f"confidence={args.confidence} max_faces={args.max_faces} tile_grid={args.tile_grid} "
                          f"counts={detector.last_counts} boxes={boxes}", flush=True)
    finally:
        capture.release()


def annotate(frame, tracks, fps):
    for track in tracks:
        x1, y1, x2, y2 = track.box
        color = (60 + track.track_id * 61 % 196,
                 60 + track.track_id * 97 % 196, 60 + track.track_id * 37 % 196)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, f"ID {track.track_id}", (x1, max(18, y1 - 7)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    cv2.rectangle(frame, (0, 0), (min(frame.shape[1] - 1, 620), 66), (25, 25, 25), -1)
    cv2.putText(frame, f"Detected faces: {len(tracks)} | Processing FPS: {fps:.1f}",
                (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    status = "Temporary IDs - no recognition" if tracks else "No faces detected"
    cv2.putText(frame, status, (10, 53), cv2.FONT_HERSHEY_SIMPLEX,
                0.55, (200, 220, 255), 1)


def run(args):
    if args.download_model:
        download_model(args.model)
        print(f"Downloaded model: {args.model}")
        return
    if args.video is not None and not args.video.is_file():
        raise FileNotFoundError(f"Input video not found: {args.video}")
    if args.inspect_frame is not None:
        inspect_frame(args)
        return
    if args.output.suffix.lower() != ".mp4":
        raise ValueError("Use an .mp4 output path")
    if args.video is not None and args.output.resolve() == args.video.resolve():
        raise ValueError("Output must be different from input")
    if args.output.exists():
        raise FileExistsError(f"Output already exists; choose another path: {args.output}")
    source = str(args.video) if args.video is not None else args.camera
    capture = cv2.VideoCapture(source)
    writer = None
    processed = 0
    index = 0
    last_ms = -1
    durations = deque(maxlen=30)
    start = perf_counter()
    try:
        if not capture.isOpened():
            raise RuntimeError("Cannot open input. Check path, video codec, or camera permissions.")
        source_fps = args.fps or capture.get(cv2.CAP_PROP_FPS)
        if not math.isfinite(source_fps) or source_fps <= 0:
            if args.camera is None:
                raise ValueError("Video has no valid FPS metadata. Supply --fps, e.g. --fps 25")
            source_fps = 30.0
            print("Camera FPS unavailable; output uses nominal 30 FPS.")
        tracker = FaceTracker(args.max_lost_seconds, args.match_distance)
        if args.diagnostics:
            print(f"[config] opencv={cv2.__version__} mediapipe={mp.__version__} backend={capture.getBackendName()} "
                  f"mode=VIDEO tile_mode=IMAGE model={args.model.name} max_faces={args.max_faces} "
                  f"detection/presence/tracking_confidence={args.confidence} max_width={args.max_width} "
                  f"sample_every={args.sample_every} source_fps={source_fps} "
                  f"orientation_meta={capture.get(cv2.CAP_PROP_ORIENTATION_META)} "
                  f"orientation_auto={capture.get(cv2.CAP_PROP_ORIENTATION_AUTO)} tile_grid={args.tile_grid}", flush=True)
        zero_frames = 0
        with FaceDetector(args.model, args.max_faces, args.confidence, args.tile_grid) as detector:
            while True:
                ok, frame = capture.read()
                if not ok:
                    if processed == 0:
                        raise RuntimeError("Input opened but no frames could be decoded")
                    if args.camera is not None:
                        print("Camera stopped delivering frames; closing output.")
                    break
                frame_index = index
                index += 1
                if frame_index % args.sample_every:
                    continue
                tick = perf_counter()
                source_height, source_width = frame.shape[:2]
                frame = prepare_frame(frame, args.max_width)
                height, width = frame.shape[:2]
                seconds = frame_index / source_fps if args.video is not None else tick - start
                timestamp_ms = max(last_ms + 1, round(seconds * 1000))
                last_ms = timestamp_ms
                boxes = detector.detect(frame, timestamp_ms)
                zero_frames += not boxes
                tracks = tracker.update(boxes, seconds)
                if args.diagnostics:
                    print(f"[frame] index={frame_index} timestamp_ms={timestamp_ms} "
                          f"decoded={source_width}x{source_height} inference={width}x{height} "
                          f"dtype={frame.dtype} input=BGR inference_color=RGB "
                          f"counts={detector.last_counts} visible_tracks={len(tracks)}", flush=True)
                durations.append(perf_counter() - tick)
                processing_fps = len(durations) / max(sum(durations), 1e-9)
                annotate(frame, tracks, processing_fps)
                if writer is None:
                    args.output.parent.mkdir(parents=True, exist_ok=True)
                    writer = cv2.VideoWriter(str(args.output), cv2.VideoWriter_fourcc(*"mp4v"),
                                             source_fps / args.sample_every, (width, height))
                    if not writer.isOpened():
                        raise RuntimeError("Cannot create MP4 output. Check codec and destination permissions.")
                    output_size = (width, height)
                if (width, height) != output_size:
                    raise RuntimeError("Input resolution changed during capture")
                writer.write(frame)
                processed += 1
                if args.display:
                    cv2.imshow("Attentify - Prototype 1", frame)
                    if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                        break
    finally:
        capture.release()
        if writer is not None:
            writer.release()
        if args.display:
            cv2.destroyAllWindows()
        if processed:
            print(f"Saved {processed} annotated frames to {args.output.resolve()}")
    if processed:
        print(f"Total elapsed: {perf_counter() - start:.2f}s; IDs are temporary within this run.")
        if args.diagnostics:
            print(f"[summary] processed={processed} zero_face_frames={zero_frames}")


if __name__ == "__main__":
    try:
        run(parse_args())
    except KeyboardInterrupt:
        print("Stopped by user.")
    except Exception as error:
        print(f"Error: {error}", file=sys.stderr)
        sys.exit(1)
