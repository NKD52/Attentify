# Existing Attentify experiments and migration status

This is a register of selected existing work, not a new model evaluation or a claim that every historical artifact has been migrated. Large files, private inputs and review answers remain outside the public repository. Exact runnable weights are listed in `computer-vision/colab/model-register.json`.

| Work | Preserved conclusion | Portable-workspace status |
| --- | --- | --- |
| Mendeley tabular Random Forest | Reported 92.07% accuracy is a baseline for that tabular/online-setting dataset, not physical-classroom performance | Historical experiment; not part of current visual inference |
| DIPSER small CNN regression | Previously reported test MAE 0.631 vs constant training-median baseline 0.607; the CNN did not beat that baseline | Keep results and grouped splits; do not promote as attention engine |
| Frozen MobileNetV3 Small + ridge | Previously reported validation MAE 0.757 vs median baseline 0.651 | Historical unsuccessful baseline; preserve it |
| SCB SSDLite exercise | Training loss fell; early BowHead/TurnHead detection metrics remained weak | Historical experiment; separate from current action model |
| Original SCB YOLOv7 | Original weights remain the comparison baseline | Hash registered; PyTorch/ONNX portable adapter not yet added |
| Refined SCB YOLOv7 | Existing three-epoch detection-head refinement, followed by threshold selection, has trade-offs; face touching and missed actions remain | Current experimental action model is the exact registered ONNX export, not an attention classifier |
| YuNet | Selected face detector from local detector comparisons | Runnable alone or alongside actions |
| Human SCB/Mustafa/DIPSER reviews | Review evidence identifies missing labels, wrong cues and boundary errors; repeatedly reviewed footage is development data | Earlier review files remain local; new run review format is portable |
| Friend camera trials | Useful hardware/qualitative checks, not labelled classroom accuracy | Private files remain local; no friend footage is included here |
| 6DRepNet handoff | Pretrained face-crop head estimation; original demo uses Haar and CPU | Registered optional adapter and verified upstream checkpoint; CPU parity/execution checked, classroom angle accuracy pending |

## Continuing on another PC

Get the repository code, the registered private model pack and only the input data needed for that run. Select the model/feature settings, run inference, and keep the resulting ZIP and exported review JSON. Model use does not require repeating training or downloading the whole original dataset. Reproducing an old training experiment additionally requires its exact data manifest, split/grouping policy, source code, environment, seeds and training configuration.

## Remaining historical migration

The current model pack preserves the executable face/action baseline. It does not contain every training checkpoint, the original SCB PyTorch weights, all historical runs or all review exports. A selected-artifact local register has hashes and locations, but those references are not remote backups. Migrate these deliberately into private storage before retiring the original PC; do not delete local history on the assumption that GitHub contains it.

New runs record actual source and code/model fingerprints. New weight/architecture versions must be separately registered and evaluated. A threshold change is a configuration change, not training; a review correction is evidence until intentionally converted into an approved training/evaluation manifest.
