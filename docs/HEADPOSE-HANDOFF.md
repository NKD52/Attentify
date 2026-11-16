# Head-pose handoff inspection — 15 November 2026

## What was actually supplied

The supplied project has three scripts, a processed demonstration video and a Windows virtual environment. It uses `sixdrepnet==0.1.6`, importing its pretrained `SixDRepNet` wrapper. No separate custom-trained architecture or fine-tuned checkpoint is referenced by the scripts. This is a pretrained-model prototype, which is consistent with the project's intended approach.

The demo scripts detect faces with OpenCV's frontal Haar cascade, crop each face, and call `model.predict(crop)`. They force CPU with `gpu_id=-1`. Prediction output order is **pitch, yaw, roll**. The installed wrapper converts BGR crops to RGB, resizes the shorter edge to 224, center crops to 224, converts to tensor and applies ImageNet normalization. It returns Euler angles in degrees. The exact signs, mirroring behavior and camera coordinate interpretation still require a known-motion check; do not assign left/right categories before that check.

`download_model.py` constructs the wrapper; it does not define or train a new model. With an empty `dict_path`, the installed wrapper downloads `6DRepNet_300W_LP_AFLW2000.pth` from the upstream host into the Torch cache. No standalone weight file was found in the supplied project outside its environment, and the standard current-user checkpoint path was absent. The friend's original cache may not have been transferred. Do not describe this as a verified portable model bundle yet.

The video script has a hardcoded input path and fixed output filename. The webcam script estimates every third frame but draws cached results on later live frames without a timestamp or reassociation. It can therefore display stale positions. Neither script exports structured angle observations or validated pose error measurements. No scripts, video, environment or model binaries from the handoff have been published or executed during this inspection.

## Integration decision

Use YuNet once for shared face detection; pass its source-frame crops to an optional 6DRepNet adapter. Preserve the existing action-only pipeline as a baseline. First evaluate this crop substitution, because changing Haar to YuNet changes crop boundaries and may change pose estimates.

The head adapter consumes a source frame plus its timestamp and face tracks. Each observation must contain source frame ID, source timestamp and basis, temporary track ID, face box, model/checkpoint version, pitch/yaw/roll degrees, validity and processing time. Preserve `student_id=null` until cross-class identity is implemented. Head pose does not supply a calibrated confidence score: store that as null; YuNet confidence remains a separate detector score.

Never infer attention from a head angle alone. Never join unmatched/null tracks. Missing or unusable crops remain unknown rather than forward-facing. Old angle estimates must stay attached to their source frame; display an age or render the matching snapshot.

## How later updates work

Every head version needs an immutable weight hash, package/code version, preprocessing definition, angle convention and evaluation record. A configuration selects the active version. A weights-only update retains the adapter if interfaces/preprocessing match. Changed preprocessing, output axes or architecture require an adapter version too. Compare the new version independently and inside the combined pipeline before changing the default. Preserve previous weights and run results.

The copied `.venv` is not the deployment package. Reconstruct dependencies in a clean environment, using compatible PyTorch/torchvision builds for the target device. DirectML/OpenVINO acceleration of the action ONNX model does not automatically accelerate this PyTorch head model. Head inference is per face and must be scheduled/batched and measured at representative face counts.

## Next executable milestone

1. Obtain the upstream checkpoint explicitly in a controlled setup step, hash it and register it privately. No silent model downloads during inference.
2. Implement an independently selectable head adapter with explicit CPU/CUDA settings, without loading it when disabled.
3. Validate pitch/yaw/roll order, finite outputs, invalid crops and source-frame/track association; perform a known-motion trial to establish signs.
4. Compare the supplied crop/prediction path against YuNet crops on consented local inputs; measure per-face and combined latency.
5. Enable the head option in the shared tester only after those checks. Keep camera preview rate, inference rate and observation age separate.

## Portability inventory status

A local, ignored `outputs/portable-inventory/local-register.json` records hashes for eight selected code/model/contract artifacts and installed head dependency metadata. It is a starting register, not a complete migration of historical experiments or reviews. Private absolute paths stay out of public documentation. Original and refined action versions, earlier reviews and grouped dataset splits still need a complete register before claiming the whole FYP is reproducible from GitHub/Drive.
