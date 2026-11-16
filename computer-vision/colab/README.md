# Portable Attentify workspace

For component/resolution/output-cost experiments, use the separate [scaling benchmark](SCALING.md). It does not replace this tester or change its default model sizes.

[Open the testing notebook in Colab](https://colab.research.google.com/github/NKD52/Attentify/blob/feature/portable-workspace/computer-vision/colab/Attentify_Colab_Testing.ipynb)

This source-only workspace runs YuNet faces, the fixed refined SCB YOLOv7 action model, or both. It does not estimate attention or identify students across classes. The head-pose handoff is recorded as pending; it is not silently loaded.

## Colab: start here

1. Open the notebook above. Select a GPU runtime for CUDA action inference, or explicitly select CPU in section 1.
2. In section 1 select `FEATURE_MODE`: `faces`, `actions` or `combined`, then run it to load the selected GitHub branch/tag/commit and dependencies.
3. In section 2 connect Drive, or use a runtime path/upload. Replace the generic `MODEL_SOURCE` placeholder with your own private model folder path in your private notebook copy.
4. Run section 3. Only required assets are checked and loaded; weights are automatically copied from persistent storage into a runtime cache. No repeated manual model upload is needed with Drive.
5. Run section 4 for an image/video or section 5 for browser camera. Download the uniquely named results ZIP.

Store the registered baseline pack privately as two transport segments, each below the connector's 100 MB upload limit. The loader joins and verifies them automatically. Select the folder, not one segment. A full model ZIP or unpacked model folder also works. Each teammate needs authorized storage access or their own private copy; private sharing links and actual storage paths are deliberately absent from this public repository. A shared folder can appear under a different Drive path; set `MODEL_SOURCE` privately to its mounted location. The old private Windows tester ZIP is also accepted as an asset source. Uploaded scripts are never executed.

Datasets can live on Drive, on the PC running locally, or in temporary Colab storage. Set `SOURCE_PATH` for a selected image/clip, or use upload mode. The tester is not a dataset training/evaluation runner. Do not merge train/validation/test groups or call repeated development reviews a held-out accuracy test.

## Local image/video use

Use a clean Python 3.11 environment. From this directory:

```powershell
python -m venv .venv
.venv\Scripts\python -m pip install -r requirements-cpu.txt
.venv\Scripts\python run_workspace.py --models "PATH_TO_PRIVATE_MODEL_ZIP" --source "PATH_TO_CLIP.mp4" --features combined --backend cpu
```

For Linux/macOS replace `.venv\Scripts\python` with `.venv/bin/python`. Local CUDA requires a compatible NVIDIA CUDA/cuDNN environment and `onnxruntime-gpu` instead of the CPU package. Never install both ONNX Runtime packages into the same environment. The older Windows team tester remains separate for Intel/DirectML live-camera testing; this adapter does not replace its drivers or environment. Browser-camera mode runs in Colab, not through `cv2.VideoCapture(0)` on the remote VM.

Override paths using CLI flags or `configs/default.json`; relative configuration paths are resolved from the working directory. Default settings: confidence 0.50, NMS 0.45, action input long edge 640, every fifth video frame, at most 300 processed frames. Review interval 10 saves one review frame per 10 processed frames. Keep a copy of the configuration when comparing experiments.

## Results and human review

Each run produces a separate directory and `SEND-THIS-...zip`. It includes `config.json` with actual code/model hashes, `source.json` with selected media fingerprint, `observations.jsonl`, `summary.json`, annotated media and optional `review/` images/page. Anonymous temporary track IDs are not cross-class identities. Unknown action ownership stays unassigned or ambiguous.

Extract the ZIP and open `review/index.html` in a browser. Compare the raw and numbered-proposal images, choose visible actions and judge each proposed box; describe missing students or boundary problems. Frames without predictions are included. Answers attempt browser autosave using the run ID; export review JSON before closing, moving folders or switching browsers. Send the exported review JSON alongside the original run ZIP. This sampled review package does not calculate precision/recall without complete, trustworthy labels.

Image/video mode writes annotated media. `review_every=0` disables sampled review images. Browser camera saves only numeric observations by default; explicitly enabling camera review images adds identifiable snapshots. Camera frames are still transferred to Google even when not saved. No audio is requested. No result is automatically uploaded to GitHub or sent to another person. Choose a private Drive `RESULTS_ROOT` if persistent output is wanted; otherwise download before the runtime expires.

## Iteration and versions

- Code updates: develop on a separate branch; set `CODE_REF` to that branch/tag/commit and rerun setup plus engine initialization. Setup refuses to overwrite modified source checkouts. Results record the real checked-out commit and file hashes, not just a branch name.
- Parameter updates: change the notebook settings or JSON config and rerun initialization/testing. These do not retrain the model.
- Weight updates: register a new model ID/checksum and compatible adapter; preserve the previous weights/configuration. Only `scb-refined-onnx-v1` is currently executable as an action model. Original PyTorch and head entries are historical/pending, not menu choices pretending to work.
- Team features: keep standalone tests before combined tests. Connect outputs by source/frame/time/track with uncertainty; do not infer attentive/inattentive from one cue.
- Model binaries and private datasets/results belong in private storage, not this public repository.

## Validation and limitations

Tests: `python -m unittest discover -s tests -v`. Notebook generation: `python build_notebook.py`.

Local validation covers asset reconstruction and checksum rejection, feature isolation, CPU inference, media outputs, diagnostic failures, review packaging and a copied source-tree trial. Hosted Colab GPU/dependency setup, browser camera permission/display and mobile compatibility still require a Colab trial. YuNet runs on CPU; CUDA may place unsupported action operators on CPU. Processing times exclude media decoding/encoding and browser transfer/display. Camera request-cycle timing includes transfer/inference but excludes returned-image display. These figures are not local classroom hardware requirements.

Upstream: [SCB dataset/model source](https://huggingface.co/datasets/wintonYF/SCB-Dataset), [YuNet/OpenCV Zoo](https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet). Existing upstream terms remain applicable. This workspace does not grant a new license to refined model weights.
