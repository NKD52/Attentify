# Colab pilot validation — 15 November 2026

Passed locally on Windows with the existing private tester environment:

- Every generated notebook Python code cell parsed successfully.
- Existing v3 private handoff ZIP reconstructed the expected SCB model and matched both SCB and YuNet SHA-256 hashes.
- CPU adapter completed inference with input shape [1, 3, 480, 640].
- A CUDA request was rejected on this machine, which has no CUDA provider.
- Synthetic image run saved one observation and annotated image in a valid ZIP.
- Synthetic six-frame video sampled every two frames saved three observations and annotated video in a valid ZIP.
- Separate run folders/ZIPs were produced, without overwriting prior runs.

Portable workspace extension:

- 13 workspace tests pass: persistent folders, flat/segmented ZIPs, bad hashes, duplicate archive entries, disabled-feature isolation, explicit action-only ownership, failed-source diagnostics, review retention/escaping and camera bridge fingerprints.
- All 9 existing Windows tester tests pass.
- All 21 existing face-tracking tests pass with their expected working directory and a writable workspace temporary directory. Initial runs with the wrong import directory and the sandbox's inaccessible default temp directory failed before those test fixtures could execute; these were corrected in the test invocation without changing the legacy implementation.
- Real CPU image runs passed in faces/actions/combined modes, including source/model/code provenance and offline review ZIPs.
- A real six-frame synthetic video produced three sampled inference/review records.
- A copied, source-only directory executed the local CLI successfully using the private model pack and an image. It reused the installed test Python environment; this is not a fresh dependency-installation test.
- The actual two transport parts reconstructed the complete ZIP and both expected ONNX hashes in a fresh temporary cache. The assembled pack SHA-256 is `5d2d148fe08d2a2ab1c8b74faef7c04eb3ee208aa3ff9d58ae2393adf7307c99`.
- Independent code review identified unnecessary action/CUDA initialization in face-only mode and omitted camera-code provenance; both were corrected and covered by the workspace tests.

The Drive connector's full-file upload hit its 100 MB limit. Two smaller transport segments were uploaded instead. This is transport segmentation of the same verified pack, not a new model version.

No friend footage was used or uploaded. No hosted Colab runtime, CUDA inference, browser permission flow, camera display, mobile browser or clean Colab dependency installation was exercised here. Those remain trial requirements. The notebook is a testing pilot rather than a validated classroom deployment.
