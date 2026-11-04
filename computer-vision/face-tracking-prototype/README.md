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
output is never overwritten. Choose another filename for each run.

### Optional webcam

```powershell
.\.venv\Scripts\python.exe main.py --camera 0 --output "outputs\webcam-test.mp4" --display
```

This command **records an annotated webcam video** until stopped. No webcam
is opened unless `--camera` is supplied. Webcam output uses the nominal camera
FPS (30 if unavailable); if inference cannot keep up, playback can be faster
than real time. It is a simple demo, not a real-time recording synchronizer.

## Main options

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
