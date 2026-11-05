# Prototype 1: Multi-Student Face Detection and Tracking

Local Python prototype for **Attentify: AI-Powered Student Attention Monitoring
and Engagement System**. This stage detects visible faces, draws bounding boxes,
and associates them with temporary track IDs. It does not identify students,
classify attention, infer engagement, or maintain histories across classes.

## Pipeline

1. OpenCV reads a local video (or an explicitly selected webcam).
2. Every Nth frame is optionally resized and passed to the MediaPipe Tasks
   **Face Landmarker** in VIDEO mode, using increasing timestamps.
3. Each face's landmark extent becomes a padded bounding box. Blendshape and
   transformation outputs are disabled.
4. A small geometric tracker scores old/new box pairs by centre distance and
   intersection over union. Greedy one-to-one matching retains temporary IDs.
5. Missing tracks remain eligible for matching for one second of source time
   by default, then expire. Only faces detected now are drawn and counted.
6. OpenCV writes an annotated MP4 and optionally shows a live preview.

MediaPipe performs its own internal detection/tracking to produce landmarks;
the separate tracker here supplies our temporary IDs. No face embeddings,
recognition database, Holistic, custom training, or external tracking package
is used. The pretrained model is downloaded separately from Google.

## Files

```text
face-tracking-prototype/
  main.py             # Command-line options, video input, overlay, output
  face_detector.py    # MediaPipe wrapper and model download
  tracker.py          # Temporary box association and expiry
  requirements.txt
  README.md
  tests/test_tracker.py
  models/             # Created on download; model ignored by Git
  outputs/            # Created on run; videos ignored by Git
```

## Windows installation

Use a 64-bit Python installation. The pinned dependencies were selected for
Python 3.13 on Windows; see the validation notes below for actual checks.
Run these commands in PowerShell after installing Python and Git:

```powershell
git clone https://github.com/NKD52/Attentify.git
cd Attentify\computer-vision\face-tracking-prototype
py -3.13 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe main.py --download-model
```

If you already cloned the repository, update it with `git pull` and enter the
prototype folder. Activation is optional; the commands above use the virtual
environment's Python directly, avoiding PowerShell activation-policy issues.
On macOS/Linux, use `python3` and `.venv/bin/python` instead.

The download command retrieves Google's version-1 `face_landmarker.task` model
into `models/`. It refuses to overwrite an existing model. Use `--model PATH`
for a different compatible local model. Ordinary video processing does not
download the model or upload video. MediaPipe's upstream package documents
on-device inference and may collect usage/performance metrics; consult its
[privacy notice](https://pypi.org/project/mediapipe/) for deployment decisions.

## Run a local video

Keep your input outside the repository. Use a short video you have permission
to process, with clearly visible faces, for the first test:

```powershell
.\.venv\Scripts\python.exe main.py --video "D:\ClassroomVideos\test.mp4" --output "outputs\test-annotated.mp4" --display
```

The default maximum is five faces. To reduce processing work:

```powershell
.\.venv\Scripts\python.exe main.py --video "D:\ClassroomVideos\test.mp4" --output "outputs\sampled.mp4" --max-faces 5 --sample-every 2 --max-width 1280
```

Omit `--display` for file-only processing. With the preview focused, press Q
or Esc to stop; Ctrl+C also stops and closes the video writer. An existing
output is protected by default. Use `--overwrite` to reuse the same filename.

For preview only, without writing any output video:

```powershell
.\.venv\Scripts\python.exe main.py --video "D:\Videos\classroom.mp4" --tile-grid 4 --no-output --display
```

To save repeatedly to the same output path:

```powershell
.\.venv\Scripts\python.exe main.py --video "D:\Videos\classroom.mp4" --tile-grid 4 --output "outputs\test.mp4" --overwrite --display
```

`--overwrite` replaces the previous output when writing begins; the input video
cannot be used as the output, even with this flag. `--no-output` creates no
video writer or output directory. Add `--diagnostics` if you want console counts
without a preview. It cannot be combined with `--output` or `--overwrite`.

### Optional webcam

```powershell
.\.venv\Scripts\python.exe main.py --camera 0 --output "outputs\webcam-test.mp4" --display
```

This command **records an annotated webcam video** until stopped. No webcam
is opened unless `--camera` is supplied. Webcam output uses the nominal camera
FPS (30 if unavailable); if inference cannot keep up, playback can be faster
than real time. It is a simple demo, not a real-time recording synchronizer.

## Main options

### Faster classroom mode with small-face searches

```powershell
.\.venv\Scripts\python.exe main.py --video "D:\Videos\classroom.mp4" --tile-grid 4 --adaptive --no-output --display
```

`--adaptive` keeps MediaPipe Face Landmarker and the existing 0.5 confidence
thresholds. Instead of rescanning sixteen tiles on every frame, it:

1. Searches the full scene using the selected grid and the next finer grid
   (4x4 and 5x5 with the command above). The alternate scale and crop boundaries
   can expose small faces missed by one grid alone.
2. Between searches, runs MediaPipe on compact regions around previously seen
   faces. These regions are twice the face-box size and use a single-face
   IMAGE task to avoid repeatedly predicting meshes for neighbours.
3. Repeats the full search every 20 processed frames by default. Each returned
   box still comes from a fresh inference; old positions are never counted as
   current detections. Failed regions are retried twice, then discarded.

The mode is opt-in: omit `--adaptive` to retain the existing fixed-grid path.
It adds one model instance for single-face region checks, using more memory
but fewer typical per-frame inference calls. It adds no dependencies or model
downloads. Temporary IDs still come from the same geometric tracker.

Use `--scan-every 10` for quicker discovery of new or returning faces at higher
CPU cost; use 30 for less discovery overhead but a longer delay. The interval
counts **processed frames**, so `--sample-every` also increases discovery latency
in source time. At 25 FPS with every frame processed, 20 frames is about 0.8s.
Fast motion, camera cuts, or a face leaving its search region can cause a miss
until the next search. A frame-size change immediately resets regions and
forces a fresh search. IMAGE-mode single-frame inspection always performs a
full adaptive search, since it has no preceding video frames to reuse.

The overlay still reports rolling processing FPS and can dip during search
frames. The final console summary now also reports **overall throughput**
(includes input decoding, startup, drawing, output, and preview waits) and
**detector throughput** (time inside detection only). Compare matching settings
and the same clip; the larger detector number is not overall application FPS.
With `--diagnostics`, `scan` and `passes` show the work done on each frame.

### Separate search and regular-frame timings

For a quieter performance comparison, use `--timings` instead of per-frame logs:

```powershell
.\.venv\Scripts\python.exe main.py --video "D:\Videos\classroom.mp4" --tile-grid 4 --adaptive --no-output --display --timings
```

Stop with Q, Esc, or Ctrl+C, or let the clip end. The final `[timing]` lines show:

- `phase=search`: detector time for the complete discovery call, including any
  known-face region checks performed on that search frame.
- `phase=regular`: detector time for face-region checks between searches.
- `frames`, `detector_total_ms`, `detector_avg_ms`, `detector_min_ms`, and
  `detector_max_ms` for each observed phase. `detector_share_pct` shows how much
  of total detection time that group consumed, not its share of all runtime.
- `phase=fixed` when adaptive mode is off; these frames are never mislabeled as
  adaptive regular frames. Groups with no observations are omitted.
- Stage totals/averages/maxima for `read`, `resize`, `tracking`, and `draw`.
  `write` and `display` appear only when enabled. `display` includes both
  `imshow` and the keyboard/event wait. `write` includes first-writer setup.
- `startup_ms` covers setup and model initialization after the capture is opened;
  it is excluded from the search/regular detector groups. It does not include
  Python import time or initial VideoCapture opening.

`--diagnostics` includes these same summaries and adds `phase`, `detector_ms`,
`read_ms`, `resize_ms`, `tracking_ms`, `draw_ms`, `write_ms`, and `display_ms` to
each processed frame line. Its console print/flush cost is separately aggregated
as `stage=logging`. `--timings` does not enable per-frame logging, so it helps
check whether terminal output itself affects performance.

All values measure elapsed wall time on the current machine, including thread
scheduling delays. Stage sums need not equal the overall elapsed time: cleanup,
other Python overhead, and final EOF reads are not separately categorized.
Read statistics count all successfully decoded frames (including skipped frames);
other per-frame stages count processed frames. No output writing or display time
is attributed to MediaPipe's detector time. Detection settings, crop scheduling,
thresholds, and frame sampling are unchanged by these diagnostics.

For more than five visible faces, increase `--max-faces`, for example 10. This
raises the global detection limit and the number of regions to check, so it can
reduce FPS. Camera placement, occlusion, blur, and actual face size still limit
back-row detection; adaptive processing cannot reconstruct hidden faces.

Local comparison on the same 256-frame 4K classroom clip, downscaled to 1280
width with every frame processed, file-only input, no output recording, no
preview, and diagnostics enabled:

- Fixed 4x4: 46.42s total, 5.5 overall FPS, 6.8 detector FPS, 4.81 mean faces.
- Adaptive 4x4 plus 5x5 discovery every 20 frames: 28.25s total, 9.1 overall FPS,
  12.8 detector FPS, 4.98 mean faces.
- Both processed all 256 frames without zero-face frames. The adaptive initial
  search also found the rear face missed by fixed 4x4 on the first frame.

These are one-machine timing and detection-count comparisons, not labelled
precision/recall or a guaranteed FPS for other scenes. A higher face count alone
is not proof of correct detection. Seventeen tests passed, including scan cadence,
crop coordinate mapping, disappearance without stale boxes, and resolution resets.
No classroom frames, videos, or per-student data are stored in this repository.

- `--video PATH` or `--camera INDEX`: select exactly one input.
- `--output PATH`: MP4 destination, default `outputs/annotated.mp4`.
- `--max-faces 5`: maximum number MediaPipe should return, not a guaranteed count.
- `--sample-every 1`: process every frame; 2 processes every second frame.
- `--max-width 1280`: downscale wide frames; never upscale small faces.
- `--confidence 0.5`: detection, presence, and internal tracking thresholds.
- `--max-lost-seconds 1.0`: how long unmatched tracks can be reassociated.
- `--match-distance 1.0`: maximum centre motion in face-box diagonal lengths.
- `--fps 25`: override source FPS when metadata is wrong or missing.
- `--display`: show preview; default is file-only output.
- `--model PATH` / `--download-model`: local model selection / one-time setup.
- `--adaptive`: alternate periodic scene discovery with per-face region checks.
- `--scan-every 20`: discovery interval in processed frames for adaptive mode.

For constant-frame-rate files, sampled output uses source FPS divided by N,
so playback duration is preserved to within a sampling interval. Only sampled
frames appear in the output. Variable-frame-rate input is interpreted using
nominal FPS; convert to constant frame rate if timing matters. Audio is not
copied. The on-screen processing FPS is a rolling measure of resize, inference,
and tracking time; it excludes decoding, drawing, writing, and preview waits.
It is not the video's playback FPS or complete application throughput.

## Classroom test procedure

1. Start with a 10–30 second clip showing three to five volunteers with visible
   faces. Avoid backlit, distant, or heavily occluded faces for the first run.
2. Run at `--sample-every 1`. Compare the face count against the visible faces.
3. Check that each box follows its face and retains its ID during slow motion.
4. Have someone briefly cover their face. Their box should disappear immediately;
   their ID may recover if the face returns near its old position before expiry.
5. Test an absence longer than the timeout. A returning face should receive a
   new ID after the old track expires. IDs restart from 1 on every new run.
6. Test an empty scene: count should be zero without a crash. Test people
   crossing and document any ID swaps instead of interpreting IDs as identity.
7. Open the saved MP4 and check that it decodes, contains overlays, and has the
   expected duration. Try sampling only after establishing baseline behavior.

Use `--max-width 1920` to retain more detail from a high-resolution source;
this may be slower. Increasing resolution cannot overcome the model's
short-range detection limits. Do not upload classroom input or annotated
output to GitHub. Annotated videos still contain students' faces.

## Limitations

- The bundled detector is short-range. Wide classroom scenes with small faces
  are a difficult case; support for 3–5 faces depends on visibility, scale,
  lighting, and camera placement. There is no validated classroom accuracy claim.
- Position/overlap matching cannot know who a person is. IDs can swap when faces
  cross, jump after motion, or change after occlusion/re-entry. A new face near
  a lost face's location may inherit its ID. Do not attach student records to IDs.
- Greedy matching is deliberately simple and can be suboptimal in crowded scenes.
  There is no motion prediction or appearance-based reassociation.
- Sampling too sparsely increases motion between frames and ID instability.
  Lost tracks are retained but not extrapolated or counted as visible faces.
- Boxes follow landmark extents; hair or the whole head may lie outside a box.
  MediaPipe smoothing is limited when configured for multiple faces.
- File decoding stops at EOF or a failed read; a truncated/corrupt video may
  produce a shorter output. It is not possible to reliably distinguish all
  decoder failures from EOF through this simple OpenCV loop.
- No attention labels, emotion inference, recognition, persistent IDs, student
  histories, frontend, backend, or database are included.

## Troubleshooting

### Classroom video returns zero faces

The full-frame model can miss faces that occupy only a small part of a wide
scene, even when they look clear to a person. Increasing the source resolution
alone does not increase face size relative to the detector's internal input.
The default path remains unchanged: VIDEO mode, RGB input, confidence 0.5,
and no tiled inference.

For a wide classroom view, enable the same model on overlapping crops:

```powershell
.\.venv\Scripts\python.exe main.py --video "D:\Videos\classroom.mp4" --output "outputs\classroom-tiled.mp4" --tile-grid 4 --diagnostics
```

`--tile-grid 4` adds sixteen overlapping IMAGE-mode crops for each sampled
frame. Boxes are mapped back into frame coordinates, duplicates are suppressed
by overlap, and results are limited to `--max-faces`. Crops use stateless IMAGE
mode rather than reusing VIDEO state between unrelated regions. Full-frame
inference still uses VIDEO mode with strictly increasing source timestamps.
This is slower; `--sample-every 5` can help with an initial comparison, but
reduces tracking continuity. A 3x3 grid is cheaper and may miss more small faces.
Tile seams, occlusion, and the simple duplicate suppression can still cause
misses, duplicate boxes, or unstable IDs. Tiling is optional, not a guaranteed
classroom detection solution.

Inspect one decoded frame without writing images or an output video:

```powershell
.\.venv\Scripts\python.exe main.py --video "D:\Videos\classroom.mp4" --inspect-frame 0
.\.venv\Scripts\python.exe main.py --video "D:\Videos\classroom.mp4" --inspect-frame 0 --tile-grid 4
```

The index is zero-based. Inspection decodes sequentially and compares native
and configured resized pixels in fresh IMAGE and VIDEO instances. This isolates
resizing and temporal state without reusing detections from previous frames.
Use an index where faces are visible; it need not be frame 0.

`--diagnostics` prints package versions, capture backend, effective FPS,
orientation metadata/auto-rotation setting, model filename, all three confidence
thresholds, maximum faces, resize limit, sampling interval, and tile grid. Each
sampled frame reports its index, increasing timestamp, decoded/inference size,
pixel type, BGR-to-RGB conversion, full-frame/tile/merged counts, and visible
tracks. A final summary reports zero-detection frames. Logs contain no images.
Run the same frame with `--confidence 0.3` to test thresholds; lower values may
increase false positives and are not the default fix. `--max-width 0` disables
resizing for a controlled comparison.

An upright decoded frame and `orientation_meta=0` rule out a rotation issue for
that clip. If another clip's decoded image is sideways, inspect its rotation
metadata and codec/backend behavior rather than swapping RGB/BGR channels.
Inference now explicitly rejects anything other than uint8 three-channel BGR
input, and makes cropped RGB arrays contiguous before passing them to MediaPipe.

Local investigation of one failing 3840x2160, 25 FPS classroom clip found:

- Frames 0, 50, and 125 returned zero full-frame faces in fresh IMAGE mode
  at both native and 1280 width, at confidence 0.5 and 0.3.
- The decoded orientation was upright with zero rotation metadata. Pixel format
  was uint8 BGR; the existing RGB conversion and VIDEO timestamps were correct.
- A close crop of frame 0 detected two faces at 0.5. The 4x4 tiled pass recovered
  four boxes on that frame at native and 1280 width, in IMAGE and VIDEO comparisons.

This supports relative face scale as the cause for that clip; it does not
establish the cause for every video. No classroom images or video are committed.

Across the whole clip, sampling every fifth frame produced 52 output frames.
The baseline had 52 zero-face frames; 4x4 tiling had none. Both outputs decoded
at the expected 5 FPS. Runtime was approximately 12 seconds for the baseline
and 24 seconds with tiling on the test machine. These are detection-count
checks, not a measured precision/recall evaluation. All nine unit tests passed,
including RGB conversion, timestamp handling, separate IMAGE-mode tile calls,
pixel validation, resize behavior, and duplicate suppression. The earlier
controlled five-face integration checks also passed again with tiling disabled.

- **Model not found:** run the download command, or pass `--model` with its path.
- **Cannot open input:** check the full filename, permission, and video codec.
- **No faces:** try closer/larger faces, better lighting, and less profile rotation.
  Lowering confidence can increase false positives and does not solve range limits.
- **Camera fails:** close other camera apps and check OS camera access.
- **Cannot create MP4:** use a writable directory and an available OpenCV codec.
- **Preview unavailable:** run without `--display`; install the non-headless
  dependencies in a fresh environment. Do not mix multiple OpenCV packages.
- **ID swaps:** reduce sampling, improve visibility, and test smaller matching
  distances. A longer timeout helps short gaps but increases incorrect reassociation.

## Checks

Run the geometric tracking tests without a camera or MediaPipe model:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

These tests cover multi-face ID continuity with reordered detections, empty
frames, expiry, and distance gating. They do not measure detection accuracy.

Validation performed on Windows with Python 3.13 and the pinned dependencies:

- All five tracker tests passed, and Python compilation and dependency checks passed.
- The real CPU Face Landmarker returned five boxes for a controlled scene made
  by tiling the public portrait from the official MediaPipe example five times.
- Two-second MP4 integration inputs containing that scene followed by blank
  frames were processed successfully. Full sampling produced 40 readable frames
  at 20 FPS; every-second-frame sampling produced 20 frames at 10 FPS.
- Missing-input error handling was checked. The controlled test media is kept
  outside the repository and is not classroom evidence.

Real classroom video accuracy, moving people crossing/occluding one another,
webcam capture, and the on-screen preview still require manual testing. This
controlled repeated-portrait test verifies plumbing and multi-face configuration,
not performance on five different students.

git clone https://github.com/NKD52/Attentify.git

cd Attentify\computer-vision\face-tracking-prototype

py -3.13 -m venv .venv

.\.venv\Scripts\python.exe -m pip install -r requirements.txt

.\.venv\Scripts\python.exe main.py --download-model

.\.venv\Scripts\python.exe main.py --video "D:\Videos\classroom.mp4" --output 
"outputs\test.mp4" --display

## References

- [MediaPipe Face Landmarker Python guide](https://developers.google.com/edge/mediapipe/solutions/vision/face_landmarker/python)
- [Model and multi-face options](https://developers.google.com/edge/mediapipe/solutions/vision/face_landmarker)
- [Official pretrained model](https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task)
