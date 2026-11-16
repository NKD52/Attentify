# Scaling benchmark validation - 16 November 2026

- 24 workspace tests pass, including the existing 13 and 11 scaling checks.
- Action preprocessing at 640 is bit-identical to the prior baseline on a non-square image.
- Default 640 YuNet/action predictions exactly match the prior baseline on a provided multi-face frame; the frame and results remain local.
- Actual-model CPU smoke trials completed at 640 and 320 input sizes, with smaller encoded preview output and deferred review generation. These synthetic trials verify execution, not model quality or hardware suitability.
- Additional actual ONNX inference at 960 and 1280 completed with input shapes `[1,3,544,960]` and `[1,3,736,1280]`. These are shape/execution checks, not speed/quality comparisons on classroom crowds.
- Preview tests verify source FPS/frame counts, separate current/raw versus inferred snapshot imagery, and encoded output dimensions.
- Fault tests verify failed model initialization, continuation to later cases, review-generation failure retention, and unavailable/unmatched GPU process-memory measurements remaining null.
- Generated notebook code cells parse and have no saved outputs; its refresh clears both benchmark modules to avoid stale code executing under new file hashes.
- Independent review identified stale benchmark-module reloads, an uncaught optional-review failure, and missing GPU memory being recorded as zero. All were reproduced with failing tests and corrected.

Hosted CUDA matrix runs, varied classroom student counts and precision/quality comparisons are not performed by these local checks. The benchmark records process memory as sampled maxima and explicitly does not measure live camera latency or establish minimum hardware requirements.

The existing ordinary tester remains on its prior branch and retains its default size settings. New size/phase options are backward-compatible defaults in the separate benchmark branch.
