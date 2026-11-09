# Attentify Detector Compare

1. Clone or download the Attentify repo on your Windows PC and install 64-bit Python 3.13 if needed. Keep the default Tcl/Tk and pip components.
2. Open `computer-vision/detector-comparison` and double-click **Start-Compare.cmd**. The first setup installs the prototype's dependencies. The first preview downloads any missing model assets and the official Zoo wrapper.
3. Choose **Side by side**, **Attentify MediaPipe**, or **OpenCV Zoo YuNet**.
4. Select a camera index and click **Start camera**, or **Browse** to a clip or image and click **Open file**. Start with camera 0. Try 1 for another connected camera.
5. In the preview, press **Space** to pause or resume a camera or clip, and **Q** or **Esc** to return to the launcher. A photo stays open until you close it.

Side by side uses one camera, clip, or photo at a time. Both detectors receive a copy of the same pixels. The preview processes frames as quickly as inference allows. Only the MediaPipe panel shows Attentify's temporary track IDs. Camera frames, clips, and photos are processed locally.

Both panels default to a maximum input width of 640 px. MediaPipe uses Attentify's adaptive 4-by-4 classroom configuration with a 50-face limit and its existing tracker. YuNet uses the official Zoo detector wrapper, the 2023mar FP32 weights, and the Zoo demo's 0.90 score threshold.

The launcher prints average processing time per detector after each run. Timing includes inference, tracking if enabled, and box drawing. Use individual runs on the same clip to compare times without the other pipeline active.

## Optional commands

```powershell
.\Start-Compare.cmd --mode both --camera 0
.\Start-Compare.cmd --mode yunet --camera 1
.\Start-Compare.cmd --mode mediapipe --clip "D:\clips\classroom.mp4"
.\Start-Compare.cmd --mode both --image "D:\clips\classroom.jpg"
.\Start-Compare.cmd --mode both --clip "D:\clips\classroom.mp4" --headless --limit 100
```

## Tune each detector

The same flags work in individual and side-by-side runs. Each detector can use a different input width while receiving the same source frame. The side-by-side view scales the two previews to the same display size and labels each with its actual input width.

| MediaPipe flag | Default | YuNet flag | Default |
| --- | ---: | --- | ---: |
| `--mp-width` | 640 | `--yunet-width` | 640 |
| `--mp-confidence` | 0.50 | `--yunet-confidence` | 0.90 |
| `--mp-max-faces` | 50 | `--yunet-nms` | 0.30 |
| `--mp-tile-grid` | 4 | `--yunet-top-k` | 5000 |
| `--mp-adaptive` / `--no-mp-adaptive` | On | | |
| `--mp-scan-every` | 20 | | |

Set a width to `0` to use native resolution. Use `--no-mp-adaptive` with `--mp-tile-grid 1`. The MediaPipe confidence applies to detection, presence, and tracking. The YuNet confidence applies to face detection.

```powershell
.\Start-Compare.cmd --mode both --clip "D:\clips\classroom.mp4" --mp-confidence 0.55 --yunet-confidence 0.70
.\Start-Compare.cmd --mode yunet --camera 1 --yunet-width 960 --yunet-confidence 0.70
```

To save a preset, create `classroom.params` beside `Start-Compare.cmd`, or use an absolute path. Put one argument per line in plain text, using `=` for values. Copy the preset to the other PC to reuse it.

```text
--mp-width=640
--mp-max-faces=30
--mp-confidence=0.55
--yunet-width=640
--yunet-confidence=0.70
```

```powershell
.\Start-Compare.cmd "@classroom.params" --mode both --image "D:\clips\classroom.jpg"
.\Start-Compare.cmd "@classroom.params"
```

The second command opens the file and camera picker with the preset loaded. Add a flag after the preset to override that setting for one run.

The launcher reuses the sibling prototype's MediaPipe code, tracker, and `models/face_landmarker.task`. If the Google download is blocked and you already have the model, copy it into `../face-tracking-prototype/models/` and retry.

- [Attentify prototype](../face-tracking-prototype/)
- [OpenCV Zoo YuNet source](https://github.com/opencv/opencv_zoo/tree/48b10fa505aeb51fbd127fc60a5c660040e823b7/models/face_detection_yunet)
