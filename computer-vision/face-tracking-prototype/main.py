"""Local multi-face detection, temporary tracking, and annotated video output."""

import argparse
from collections import deque
from dataclasses import dataclass
import math
from pathlib import Path
import sys
from time import perf_counter

import cv2
import mediapipe as mp

from face_detector import DEFAULT_MODEL, FaceDetector, download_model
from tracker import FaceTracker


@dataclass
class TimingStats:
    """Constant-memory wall-clock totals; no frame/image data is retained."""

    count: int = 0
    total: float = 0.0
    minimum: float = math.inf
    maximum: float = 0.0

    def add(self, seconds):
        self.count += 1
        self.total += seconds
        self.minimum = min(self.minimum, seconds)
        self.maximum = max(self.maximum, seconds)

    @property
    def average_ms(self):
        return self.total * 1000 / self.count if self.count else 0.0


def print_timings(phases, stages, startup_seconds, detection_seconds):
    print(f"[timing] startup_ms={startup_seconds * 1000:.2f}")
    for phase, stats in phases.items():
        share = 100 * stats.total / detection_seconds if detection_seconds else 0.0
        print(f"[timing] phase={phase} frames={stats.count} "
              f"detector_total_ms={stats.total * 1000:.2f} detector_avg_ms={stats.average_ms:.2f} "
              f"detector_min_ms={stats.minimum * 1000:.2f} detector_max_ms={stats.maximum * 1000:.2f} "
              f"detector_share_pct={share:.1f}")
    for stage, stats in stages.items():
        print(f"[timing] stage={stage} calls={stats.count} total_ms={stats.total * 1000:.2f} "
              f"avg_ms={stats.average_ms:.2f} max_ms={stats.maximum * 1000:.2f}")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--video", type=Path, help="Local input video path")
    source.add_argument("--camera", type=int, help="Webcam index, e.g. 0")
    output = parser.add_mutually_exclusive_group()
    output.add_argument("--output", type=Path, default=Path("outputs/annotated.mp4"))
    output.add_argument("--no-output", action="store_true", help="Process without saving a video; use --display for preview")
    parser.add_argument("--overwrite", action="store_true", help="Replace an existing output video")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--download-model", action="store_true", help="Download model and exit")
    parser.add_argument("--max-faces", type=int, default=5)
    parser.add_argument("--sample-every", type=int, default=1, help="Process every Nth frame")
    parser.add_argument("--max-width", type=int, default=1280, help="Resize down to this width; 0 keeps native size")
    parser.add_argument("--confidence", type=float, default=0.5)
    parser.add_argument("--tile-grid", type=int, choices=(1, 2, 3, 4), default=1,
                        help="Optional overlapping NxN IMAGE crops; 1 disables tiling")
    parser.add_argument("--adaptive", action="store_true",
                        help="Periodically search two grids and re-detect known face regions between scans")
    parser.add_argument("--scan-every", type=int, default=20,
                        help="Adaptive discovery interval in processed frames (default 20)")
    parser.add_argument("--diagnostics", action="store_true", help="Print config and counts for every sampled frame")
    parser.add_argument("--timings", action="store_true", help="Print timing summary without per-frame diagnostic logging")
    parser.add_argument("--inspect-frame", type=int, metavar="INDEX",
                        help="Compare one zero-based video frame in fresh IMAGE/VIDEO modes, then exit")
    parser.add_argument("--max-lost-seconds", type=float, default=1.0)
    parser.add_argument("--match-distance", type=float, default=1.0,
                        help="Maximum match distance in face-box diagonals")
    parser.add_argument("--fps", type=float, help="Override input FPS; required if metadata is invalid")
    parser.add_argument("--display", action="store_true", help="Show preview; Q or Esc stops")
    args = parser.parse_args(argv)
    if args.no_output:
        if args.overwrite:
            parser.error("--overwrite cannot be combined with --no-output")
        args.output = None
    if not args.download_model and args.video is None and args.camera is None:
        parser.error("provide --video PATH or --camera INDEX")
    if min(args.max_faces, args.sample_every) < 1 or (args.max_width != 0 and args.max_width < 320):
        parser.error("max-faces/sample-every must be positive; max-width must be 0 or >= 320")
    if args.inspect_frame is not None and (args.video is None or args.inspect_frame < 0):
        parser.error("inspect-frame requires --video and a nonnegative frame index")
    if args.scan_every < 1 or (args.adaptive and args.tile_grid < 2):
        parser.error("scan-every must be >= 1; --adaptive requires --tile-grid 2, 3 or 4")
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
                with FaceDetector(args.model, args.max_faces, args.confidence, args.tile_grid, mode,
                                  args.adaptive, args.scan_every) as detector:
                    boxes = detector.detect(pixels, 0)
                    print(f"[inspect] size={label}:{pixels.shape[1]}x{pixels.shape[0]} "
                          f"dtype={pixels.dtype} input=BGR inference=RGB mode={mode} timestamp_ms=0 "
                          f"confidence={args.confidence} max_faces={args.max_faces} tile_grid={args.tile_grid} adaptive={args.adaptive} "
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
    if args.output is not None:
        if args.output.suffix.lower() != ".mp4":
            raise ValueError("Use an .mp4 output path")
        if args.video is not None and (
            args.output.resolve() == args.video.resolve()
            or (args.output.exists() and args.output.samefile(args.video))
        ):
            raise ValueError("Output must be different from input, even with --overwrite")
        if args.output.exists() and not args.overwrite:
            raise FileExistsError(f"Output already exists; use --overwrite to replace it: {args.output}")
    source = str(args.video) if args.video is not None else args.camera
    capture = cv2.VideoCapture(source)
    writer = None
    processed = 0
    index = 0
    last_ms = -1
    durations = deque(maxlen=30)
    detection_seconds = 0.0
    total_faces = 0
    phase_timings = {}
    stage_timings = {}
    startup_seconds = 0.0
    timing_enabled = args.timings or args.diagnostics

    def record_stage(name, seconds):
        if timing_enabled:
            stage_timings.setdefault(name, TimingStats()).add(seconds)

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
                  f"orientation_auto={capture.get(cv2.CAP_PROP_ORIENTATION_AUTO)} tile_grid={args.tile_grid} "
                  f"adaptive={args.adaptive} scan_every={args.scan_every}", flush=True)
        zero_frames = 0
        with FaceDetector(args.model, args.max_faces, args.confidence, args.tile_grid,
                          adaptive=args.adaptive, scan_every=args.scan_every) as detector:
            startup_seconds = perf_counter() - start
            while True:
                read_start = perf_counter()
                ok, frame = capture.read()
                read_seconds = perf_counter() - read_start
                if not ok:
                    if processed == 0:
                        raise RuntimeError("Input opened but no frames could be decoded")
                    if args.camera is not None:
                        print("Camera stopped delivering frames; closing output.")
                    break
                record_stage("read", read_seconds)
                frame_index = index
                index += 1
                if frame_index % args.sample_every:
                    continue
                tick = perf_counter()
                source_height, source_width = frame.shape[:2]
                frame = prepare_frame(frame, args.max_width)
                resize_seconds = perf_counter() - tick
                record_stage("resize", resize_seconds)
                height, width = frame.shape[:2]
                seconds = frame_index / source_fps if args.video is not None else tick - start
                timestamp_ms = max(last_ms + 1, round(seconds * 1000))
                last_ms = timestamp_ms
                detection_start = perf_counter()
                boxes = detector.detect(frame, timestamp_ms)
                detector_seconds = perf_counter() - detection_start
                detection_seconds += detector_seconds
                phase = ("search" if detector.last_counts.get("scan", False) else "regular") if args.adaptive else "fixed"
                if timing_enabled:
                    phase_timings.setdefault(phase, TimingStats()).add(detector_seconds)
                total_faces += len(boxes)
                zero_frames += not boxes
                tracking_start = perf_counter()
                tracks = tracker.update(boxes, seconds)
                tracking_seconds = perf_counter() - tracking_start
                record_stage("tracking", tracking_seconds)
                durations.append(perf_counter() - tick)
                processing_fps = len(durations) / max(sum(durations), 1e-9)
                draw_start = perf_counter()
                annotate(frame, tracks, processing_fps)
                draw_seconds = perf_counter() - draw_start
                record_stage("draw", draw_seconds)
                write_seconds = 0.0
                write_start = perf_counter()
                if args.output is not None and writer is None:
                    args.output.parent.mkdir(parents=True, exist_ok=True)
                    writer = cv2.VideoWriter(str(args.output), cv2.VideoWriter_fourcc(*"mp4v"),
                                             source_fps / args.sample_every, (width, height))
                    if not writer.isOpened():
                        raise RuntimeError("Cannot create MP4 output. Check codec and destination permissions.")
                    output_size = (width, height)
                if writer is not None:
                    if (width, height) != output_size:
                        raise RuntimeError("Input resolution changed during capture")
                    writer.write(frame)
                    write_seconds = perf_counter() - write_start
                    record_stage("write", write_seconds)
                processed += 1
                display_seconds = 0.0
                stop = False
                if args.display:
                    display_start = perf_counter()
                    cv2.imshow("Attentify - Prototype 1", frame)
                    stop = cv2.waitKey(1) & 0xFF in (ord("q"), 27)
                    display_seconds = perf_counter() - display_start
                    record_stage("display", display_seconds)
                if args.diagnostics:
                    log_start = perf_counter()
                    print(f"[frame] index={frame_index} timestamp_ms={timestamp_ms} "
                          f"decoded={source_width}x{source_height} inference={width}x{height} "
                          f"dtype={frame.dtype} input=BGR inference_color=RGB "
                          f"counts={detector.last_counts} visible_tracks={len(tracks)} "
                          f"phase={phase} detector_ms={detector_seconds * 1000:.2f} "
                          f"read_ms={read_seconds * 1000:.2f} resize_ms={resize_seconds * 1000:.2f} "
                          f"tracking_ms={tracking_seconds * 1000:.2f} draw_ms={draw_seconds * 1000:.2f} "
                          f"write_ms={write_seconds * 1000:.2f} display_ms={display_seconds * 1000:.2f}", flush=True)
                    record_stage("logging", perf_counter() - log_start)
                if stop:
                    break
    except KeyboardInterrupt:
        # Let a normal Ctrl+C stop reach the same timing summary as Q or EOF.
        print("Stopped by user.")
    finally:
        capture.release()
        if writer is not None:
            writer.release()
        if args.display:
            cv2.destroyAllWindows()
        if processed and writer is not None:
            print(f"Saved {processed} annotated frames to {args.output.resolve()}")
        elif processed:
            print(f"Processed {processed} frames; no output video saved.")
    if processed:
        elapsed = perf_counter() - start
        print(f"Total elapsed: {elapsed:.2f}s; overall throughput: {processed / elapsed:.1f} FPS; "
              f"detector throughput: {processed / max(detection_seconds, 1e-9):.1f} FPS; "
              "IDs are temporary within this run.")
        if args.diagnostics:
            print(f"[summary] processed={processed} zero_face_frames={zero_frames} mean_faces={total_faces / processed:.2f}")
        if timing_enabled:
            print_timings(phase_timings, stage_timings, startup_seconds, detection_seconds)


if __name__ == "__main__":
    try:
        run(parse_args())
    except KeyboardInterrupt:
        print("Stopped by user.")
    except Exception as error:
        print(f"Error: {error}", file=sys.stderr)
        sys.exit(1)
