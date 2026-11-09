# Attentify Detector Compare

1. Clone or download the Attentify repo on your Windows PC and install 64-bit Python 3.13 if needed. Keep the default Tcl/Tk and pip components.
2. Open `computer-vision/detector-comparison` and double-click **Start-Compare.cmd**. The first setup installs the prototype's dependencies. The first preview downloads any missing model assets and the official Zoo wrapper.
3. Choose **Side by side**, **Attentify MediaPipe**, or **OpenCV Zoo YuNet**.
4. Select a camera index and click **Start camera**, or **Browse** to a clip or image and click **Open file**. Start with camera 0. Try 1 for another connected camera.
5. In the preview, press **Space** to pause or resume a camera or clip, and **Q** or **Esc** to return to the launcher. A photo stays open until you close it.

Side by side uses one camera, clip, or photo at a time. Both detectors receive a copy of the same pixels. The preview processes frames as quickly as inference allows. Only the MediaPipe panel shows Attentify's temporary track IDs. Camera frames, clips, and photos are processed locally.

Both panels default to a maximum input width of 640 px. MediaPipe uses Attentify's adaptive 4-by-4 classroom configuration with a 50-face limit and its existing tracker. YuNet uses the official Zoo detector wrapper, the 2023mar FP32 weights, and the Zoo demo's 0.90 score threshold.

The launcher prints average processing time per detector after each run. Timing includes inference, tracking if enabled, and box drawing. Use individual runs on the same clip to compare times without the other pipeline active.

## Run from PowerShell

From the Attentify repo root, open the comparison folder and list the available options:

```powershell
cd computer-vision\detector-comparison
.\Start-Compare.cmd --help
```

Replace the sample paths below with your own files.


```powershell
.\Start-Compare.cmd --mode both --camera 0
.\Start-Compare.cmd --mode yunet --camera 1
.\Start-Compare.cmd --mode mediapipe --clip "D:\clips\classroom.mp4"
.\Start-Compare.cmd --mode both --image "D:\clips\classroom.jpg"
.\Start-Compare.cmd --mode both --clip "D:\clips\classroom.mp4" --headless --limit 100
```

## Launcher flags

| Flag | Accepted values and behavior |
| --- | --- |
| `--mode` | `both` (default), `mediapipe`, or `yunet`. Applies when you supply a source on the command line. The picker has its own mode selector. |
| `--camera` | Camera index, starting at `0`. Try `1` for a second camera. |
| `--clip` | Path to a saved video. Quote paths that contain spaces. |
| `--image` | Path to a still image. Quote paths that contain spaces. |
| `--headless` | Process the input and print timing and detection totals without opening a preview. Requires a source. |
| `--limit` | Positive maximum number of frames to process. Useful for testing a short portion of a clip. Requires a source. |
| `-h`, `--help` | Show all options and their defaults. |

Choose one source per command. To open the picker, omit the source, `--headless`, and `--limit`.

## Tune each detector

The same flags work in individual and side-by-side runs. Each detector can use a different input width while receiving the same source frame. The side-by-side view scales the two previews to the same display size and labels each with its actual input width.

| MediaPipe flag | Default | Accepted values and effect |
| --- | ---: | --- |
| `--mp-width` | 640 | Maximum input width in pixels. Use `0` for native resolution or an integer of at least `320`. |
| `--mp-confidence` | 0.50 | `0` to `1`. Sets the detection, face presence, and tracking thresholds. |
| `--mp-max-faces` | 50 | Positive integer. Caps the number of faces. |
| `--mp-tile-grid` | 4 | `1` through `4`. A value of `4` uses a 4×4 grid. Use `--mp-tile-grid 1 --no-mp-adaptive` to disable tiling. |
| `--mp-adaptive` / `--no-mp-adaptive` | On | Turns adaptive detection on or off. |
| `--mp-scan-every` | 20 | Positive integer. Sets how many frames pass between adaptive scans. |

| YuNet flag | Default | Accepted values and effect |
| --- | ---: | --- |
| `--yunet-width` | 640 | Maximum input width in pixels. Use `0` for native resolution or an integer of at least `320`. |
| `--yunet-confidence` | 0.90 | `0` to `1`. Sets the minimum face detection score. |
| `--yunet-nms` | 0.30 | `0` to `1`. Sets the overlap threshold used to suppress duplicate boxes. |
| `--yunet-top-k` | 5000 | Positive integer. Caps the candidates considered before duplicate suppression. |

```powershell
.\Start-Compare.cmd --mode both --clip "D:\clips\classroom.mp4" --mp-confidence 0.55 --yunet-confidence 0.70
.\Start-Compare.cmd --mode yunet --camera 1 --yunet-width 960 --yunet-confidence 0.70
```

## Save and reuse a preset

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

```powershell
.\Start-Compare.cmd "@classroom.params" --mode yunet --camera 1 --yunet-confidence 0.65
```

## Model files

The launcher reuses the sibling prototype's MediaPipe code, tracker, and `models/face_landmarker.task`. If the Google download is blocked and you already have the model, copy it into `../face-tracking-prototype/models/` and retry.

- [Attentify prototype](../face-tracking-prototype/)
- [OpenCV Zoo YuNet source](https://github.com/opencv/opencv_zoo/tree/48b10fa505aeb51fbd127fc60a5c660040e823b7/models/face_detection_yunet)
