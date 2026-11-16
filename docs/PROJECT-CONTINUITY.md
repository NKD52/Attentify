# Attentify project continuity and body-posture scope

Updated 16 November 2026. This is a checkpoint for continuing work across chats,
machines and teammates. It separates user decisions, measured findings and open
questions. It is not evidence that unimplemented features work.

## Current user decision

Following new supervisor guidance, pause active handraise/reading/writing feature
refinement and combined-pipeline optimization. The immediate work is to compare
existing models for body posture, initially independently in Colab:

1. Sitting upright.
2. Leaning forward.
3. Leaning backward.
4. Bending over desk.
5. Head resting on hand or desk.
6. Slouching.

The user and a teammate are trying different models. LSP-YOLO is the first named
candidate. Preserve previous work and results; no existing model, notebook or
private evidence should be deleted or silently replaced. Do not assume this
change cancels the whole attention-monitoring project or the other teammates'
modules. The longer-term goals remain persistent student history and periodic
teacher feedback, unless subsequently changed by the user.

## Existing implementation preserved

- YuNet FP32 face detection, temporary tracking, experimental refined SCB YOLOv7
  ONNX handraise/reading/writing proposals, optional 6DRepNet head pose.
- Head pose reuses detected face crops and returns pitch/yaw/roll. It is not eye
  gaze, emotion or attention classification. Action-to-student geometric
  association is provisional. Student IDs are not persistent across classes.
- Portable testing and scaling notebooks load shared GitHub Python code and
  registered private weights. Current notebooks perform inference/evaluation,
  not automatic model training or learning from new uploaded footage.
- PRs 8, 9, 10 and 11 are merged. PR 10 added head integration; PR 11 added
  isolated profiling. Main after PR 11: f590225c93c6b83e5b95394038f37edac9555163.
- Owner-specific Drive paths are saved in the user's existing Drive notebook
  copies. Public templates retain generic storage configuration. Preserve
  saved settings when updating notebooks; Drive authorization can still be
  required by Google.
- Benchmark profile mode uses a bounded 250-source-frame window. Ordinary
  benchmark mode remains separate. Profiling is diagnostic, not an FPS or
  accuracy certification. The personal notebook's normal benchmark frame cap
  was preserved at 20000; ordinary testing cap was 10000.
- Models, footage, friend camera trials and review answers remain private.
  A public source repository is not a backup of every historical dataset/run.

## Historical scientific conclusions

- Mendeley Random Forest: reported accuracy 92.07% on the tabular dataset;
  this cannot establish physical-classroom camera accuracy.
- DIPSER partial group 03: approximately 60 GB local collection. Training was
  possible without the full dataset. Subject-separated evaluation matters.
  Small CNN test MAE 0.631 vs constant-median baseline 0.607; frozen
  MobileNetV3 Small + ridge validation MAE 0.757 vs baseline 0.651. Neither
  established a successful attention estimator.
- Local detector benchmark selected YuNet provisionally at width 640,
  approximately 11.82 median pipeline FPS on the tested classroom clip.
- SCB behavior experiments and human reviews exposed face/hair touching false
  raises, missed raises, merged/wrong-person boxes, ambiguous activities and
  missing/incorrect annotations. Repeatedly reviewed footage is development
  evidence; do not reuse it as an untouched test set.
- Decreasing loss, a changed threshold, successful model execution and produced
  predictions do not themselves establish improved accuracy.

## GPU investigation: measured findings to retain

The latest diagnostic used a Tesla T4, 1080p input at 25 FPS, combined face/action/
head inference at 640 settings and stride 5. It processed 250 source frames,
50 inference updates and 10 seconds of footage in 10.182 seconds in its normal
no-export baseline. Mean detected faces: 3.76, not true student count.

| Baseline work | Total elapsed stage time | Approximate share |
| --- | ---: | ---: |
| Video decode | 3.899 s | 38% |
| YuNet and tracking | 2.203 s | 22% |
| Head pose including preparation/transfers | 2.650 s | 26% |
| Action session execution | 1.178 s | 12% |
| Other preparation, postprocessing and loop work | 0.252 s | 2% |

Remote device-wide sampled GPU busy percentage had median 27.5%, peak 75%,
including initialization/warmup. Allocated GPU memory reached 953 MiB; memory
allocation is not compute utilization. The sampling process itself adds some
overhead, so this is not a controlled regression against earlier runs.

ONNX trace confirmed CUDA convolution execution. CPU provider events were small
Concat/Gather/Unsqueeze operations, not proof the action network ran entirely
on CPU. The PyTorch trace confirmed CUDA head convolutions. All 188 detected-face
observations in that baseline produced head estimates; angle correctness was
not measured. Profiler label/event totals overlap and must not be summed into
utilization percentages.

The evidence supports investigating serial CPU/decode work and scheduling rather
than immediately buying stronger hardware. Decode/inference overlap, head batch
sizes and independent feature scheduling were proposed, not implemented speedups.
They are now paused behind the body-posture model comparison.

Previous 13.9-second 1080p benchmark: all 20 trials completed. Combined-head
averaged 97.4 ms of trial median model time, with approximately 6.3 offline
updates per wall second. A 3840-wide exported preview added substantial cost.
Stride 5 provides five new predictions per source second at 25 FPS, not live
25-FPS inference. None of these results certify a full crowded classroom.

## LSP-YOLO candidate: verified research, not installed

Primary paper: https://arxiv.org/html/2511.14322v1

The relevant paper is *LSP-YOLO: A Lightweight Single-Stage Network for Sitting
Posture Recognition on Embedded Devices*. An unrelated power-line detector has
a similar name; do not confuse them. The posture paper combines keypoint and
posture prediction and describes a custom 5000-image dataset from 15 people,
with 11 upper-body keypoints, person boxes and posture labels.

| Paper class | Relationship to supervisor requirements |
| --- | --- |
| Correct | Candidate match to sitting upright; verify examples |
| LeanLeft / LeanRight | Extra classes; not forward/backward leaning |
| ChinSupport | Partial candidate for head resting on hand |
| OnDesk | Inspect whether examples represent bending or actual head/desk support |
| HeadDown | Not an interchangeable label for slouching or forward torso lean |

The paper does not define separate forward-lean, backward-lean and slouch classes.
Paper searches and GitHub/Hugging Face searches on this date did not establish
an official runnable code release, pretrained checkpoint or downloadable image
dataset. This is an unresolved availability check, not proof no release exists.
The user's exact found model/repository/weights link is the next missing input.
Do not present ordinary YOLO11-Pose weights as pretrained LSP-YOLO, or create a
notebook pretending to run this unavailable checkpoint. Reported paper speed
and accuracy are author results, not Attentify benchmark measurements.

## Dataset leads checked

- LSP custom images: relevant upper-body labels but incomplete category coverage;
  download/release and reuse terms are unverified.
- Multi-view sitting-posture dataset linked by LGCSPNet paper:
  https://github.com/LKW123800/multi-view-sitting-posture-dataset . Its README
  currently directs downloads to Baidu. Do not send the user back through Baidu
  without an accessible alternative; this was an earlier constraint.
  Paper: https://www.sciencedirect.com/science/article/pii/S0957417425021050 .
- MultiPosture: https://zenodo.org/records/14230872 . The record offers a 9.3 MB
  CSV of MediaPipe skeletal coordinates from 13 participants, including upright,
  forward/backward and lateral trunk labels. Raw images/video are explicitly
  excluded. Useful for keypoint-classifier research; not RGB inference evaluation
  of LSP-YOLO. Skeleton conventions must match before any transfer.
- USSP / MSBN-SPose: https://www.mdpi.com/2079-9292/14/19/3889 . Related classroom
  research; its stated four classes do not cover all six requested postures.
  The data availability section directs inquiries to the author; a ready public
  image/weights download was not established here.
- Existing SCB/DIPSER images may provide real-classroom qualitative footage.
  Their behavior/attention labels must not be relabeled automatically as the new
  six body-posture targets. A new target-specific evaluation requires suitable
  labels or a small explicitly reviewed evaluation sample, not complete manual
  relabeling of hundreds of gigabytes.

## Next execution milestone

1. Resolve the user's LSP model URL and verify code, trained weights, input
   requirements, labels, license and dependency compatibility.
2. If runnable, build a separate posture-only Colab experiment using original
   weights first. Preserve the current benchmark and fixed model registry.
3. Evaluate that model and the teammate's alternatives on the same selected
   inputs, including all six requirements, occlusion and an unknown/unclear
   option. Preserve native labels; document partial/uncovered categories.
4. Measure model processing, total throughput, person coverage and per-class
   errors separately. Existing generic pose models output joints; they do not
   automatically supply these six semantic posture classes.
5. Define boundaries between forward lean, desk bend and slouch before scoring.
   Resting on a hand can coexist with a torso lean, so decide a multi-label
   policy or an explicit precedence rule; do not force ambiguous examples.
6. Select a model based on target coverage and held-out evaluation, then revisit
   integration and speed. Body posture remains an observable cue, not direct
   ground truth of attention or engagement.

