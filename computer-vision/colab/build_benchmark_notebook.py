"""Generate a separate benchmark notebook; reuse the baseline's setup without replacing it."""
import json
from pathlib import Path
HERE=Path(__file__).resolve().parent
baseline=json.loads((HERE/'Attentify_Colab_Testing.ipynb').read_text())
setup=next(''.join(c['source']) for c in baseline['cells'] if c['cell_type']=='code')
setup=setup.replace("CODE_REF='feature/portable-workspace'","CODE_REF='main'")
setup += "\nfor module_name in ['benchmark','benchmark_resources','profile_pipeline']:\n    sys.modules.pop(module_name,None)\n"
setup += "\nsubprocess.check_call([sys.executable,'-m','pip','install','-q','psutil>=6,<8'])\n"
cells=[]
def cell(kind,text):
    c=dict(cell_type=kind,id=f'scaling-{len(cells):03}',metadata={},source=text.splitlines(True))
    if kind=='code':c.update(execution_count=None,outputs=[])
    cells.append(c)

cell('markdown','''# Attentify - scaling benchmark
This is separate from the ordinary tester. It compares **faces only**, **actions only**, **combined**, independent face/action input sizes, and output disabled versus smaller/larger preview encoding. It uses the same fixed weights. No training, minimum-hardware certification or attention accuracy is claimed.

Use the same clip on different machines to compare hardware. Use several genuine classroom clips with different student counts to investigate crowd scaling. **Detected face count is not true student count.** A larger inference size cannot restore details that were absent in the camera recording.

The baseline has eight cases; enabling head pose adds two cases, giving 20 trials per clip with two repeats, processing at most the first 250 source frames each. It may take several minutes on GPU and much longer on CPU. Every case uses the same frame window and stride; even repeats reverse the case order to reduce ordering bias. Start with one short clip. Head cases use bounded crop batches and report their own timing.
''')
cell('markdown','## 1. Load benchmark code and dependencies\nSelect a GPU runtime for CUDA. This notebook loads the merged main branch; choose another CODE_REF for an isolated experiment. Restart the runtime if switching ONNX Runtime package/backend after it has already been imported.')
cell('code',setup)
cell('markdown','## 2. Private model storage\nSet your own private model folder path. Drive model segments are joined and verified automatically; no repeated manual ZIP upload is needed. Connecting Drive grants the notebook access to it. Keep private paths and outputs out of GitHub commits.')
cell('code', '''from google.colab import drive
drive.mount('/content/drive')
MODEL_SOURCE='/content/drive/MyDrive/YOUR_MODEL_FOLDER'
from getpass import getpass
MODEL_SOURCE=getpass('Private model folder path (not saved in notebook code): ').strip()
RESULTS_ROOT='/content/attentify-benchmarks' # Or a private writable Drive path
INCLUDE_HEAD_CASES=True
from workspace_storage import ensure_assets,ensure_head_assets
assets=ensure_assets(MODEL_SOURCE,'/content/attentify-model-cache','combined')
if INCLUDE_HEAD_CASES:ensure_head_assets(MODEL_SOURCE,assets)
''')
cell('markdown','## 3. Select recorded clips\nUpload one or more clips, or provide paths already in Colab/Drive. Visible student counts are optional human-supplied metadata, not labels generated from face detection. Leave null/None if unknown. Files uploaded to Colab disappear when the runtime is deleted.')
cell('code', '''INPUT_MODE='upload' # 'upload' or 'path'
CLIPS=[{'path':'/content/your-clip.mp4','label':'classroom-1','visible_students':None}]
if INPUT_MODE=='upload':
    from google.colab import files
    uploaded=files.upload()
    CLIPS=[{'path':name,'label':name,'visible_students':None} for name in uploaded]
    del uploaded
elif INPUT_MODE!='path':raise ValueError('Choose upload or path')
print(CLIPS)
# You can edit student counts in CLIPS before the next cell.
''')
cell('markdown','## 4. Run the controlled matrix or focused profiler\nSet RUN_MODE to profile for a short 250-source-frame diagnostic on one clip. It runs a no-export baseline with sampled remote GPU busy percentage, then a separate PyTorch CPU/CUDA trace and ONNX provider trace. Profile timings contain overhead and are not normal FPS results. No images or movies are exported in profile mode.\n\nSizes affect face detection and action inference independently. Output tests hold both model sizes at 640. Width 960 creates a small two-panel preview; width 3840 compares a larger encoding workload, not higher-resolution inference. The actual encoded dimensions are recorded. Preview movies retain the source frame rate: current raw source on the left, the matching dated inference snapshot on the right. Old boxes are never overlaid on current students.\n\nGPU memory is a sampled maximum for the whole Python process, including initialization/warm-up and any retained context; it is not exact peak/model-exclusive VRAM. Missing measurements stay null. Review samples, if enabled, are generated after timing and contain identifiable images. Set REVIEW_EVERY=0 for speed-only runs. No outputs are published or sent automatically.')
cell('code', '''from benchmark import make_cases,run_suite
RUN_MODE='benchmark' # 'profile' runs one short no-export baseline plus diagnostic traces
ACTION_SIZES=(640,960) # Add 1280 later if needed; dynamic input sizes must be multiples of 32
FACE_WIDTHS=(640,960)
OUTPUT_WIDTHS=(960,3840) # Use (960,) to omit the more expensive large output case
REPEATS=2
MAX_SOURCE_FRAMES=250
INFERENCE_STRIDE=5
WARMUP_INFERENCES=3
REVIEW_EVERY=10 # One raw/numbered review sample per ten inference updates; 0 disables images
cases=make_cases(ACTION_SIZES,FACE_WIDTHS,OUTPUT_WIDTHS,include_head=INCLUDE_HEAD_CASES)
print('Trials:',len(CLIPS)*len(cases)*REPEATS)
if RUN_MODE=='profile':
    if len(CLIPS)!=1:raise ValueError('Select one clip for focused profiling')
    from profile_pipeline import run_profile
    result_zip=run_profile(CLIPS[0],assets,backend=BACKEND,enable_head=INCLUDE_HEAD_CASES,
        max_source_frames=250,stride=INFERENCE_STRIDE,results_root=RESULTS_ROOT)
elif RUN_MODE=='benchmark':
    result_zip=run_suite(CLIPS,assets,cases=cases,backend=BACKEND,results_root=RESULTS_ROOT,
        stride=INFERENCE_STRIDE,max_source_frames=MAX_SOURCE_FRAMES,repeats=REPEATS,
        warmup=WARMUP_INFERENCES,review_every=REVIEW_EVERY)
else:raise ValueError('Choose benchmark or profile')
print('Saved:',result_zip)
''')
cell('markdown','## 5. Inspect and download results\nCompare decode, face processing, action preprocessing/runtime/postprocessing, rendering and encoding separately. Pipeline throughput includes decoding every source frame, including skipped inference frames, and encoder finalization. Model initialization and warm-up are reported separately. The timed pipeline excludes final review generation, JSON/CSV export, ZIP creation and upload/download.\n\n**This is offline:** source-time snapshot gaps describe the scheduling gap in the output, not camera exposure-to-screen latency. Failed cases remain failed and are not silently retried with smaller inputs or CPU. Compare quality using the exported reviews before choosing a faster configuration. A test with few inference updates is only a smoke test.')
cell('code', '''import pandas as pd,zipfile,json
with zipfile.ZipFile(result_zip) as z:
    if RUN_MODE=='profile':
        name=next(n for n in z.namelist() if n.endswith('profile-summary.json'))
        report=json.loads(z.read(name));print(json.dumps(report,indent=2))
        table=pd.DataFrame([report['baseline']])
    else:
        name=next(n for n in z.namelist() if n.endswith('comparison.csv'))
        with z.open(name) as f:table=pd.read_csv(f)
columns=['source_label','repeat','case_name','status','median_model_ms','pipeline_elapsed_s',
         'decoded_source_frames_per_wall_s','inference_updates_per_wall_s','realtime_factor',
         'median_yunet_tracking_ms','median_action_runtime_ms','median_head_pose_ms',
         'mean_detected_faces','sampled_max_process_gpu_memory_mib']
display(table[[c for c in columns if c in table]])
from google.colab import files
files.download(str(result_zip))
''')
cell('markdown','''Extract the ZIP. `comparison.csv` and `comparison.json` summarize all cases; each case folder contains settings, `frames.csv`, predictions and optional preview/review files. Open a case's `review/index.html` and export answers before switching browsers. Send the run ZIP plus exported review JSONs for analysis. Compare identical sources/code/model versions and repeated trials; do not infer minimum hardware or classroom accuracy from a single clip.

Next: collect genuine clips with larger classes, review resolution/quality trade-offs, and only then choose an optimization. Smaller models, lower precision, temporal scheduling remain future experiments, not changes silently applied here.
''')
nb=dict(nbformat=4,nbformat_minor=5,metadata=dict(colab=dict(name='Attentify_Scaling_Benchmark.ipynb',private_outputs=True),kernelspec=dict(name='python3',display_name='Python 3'),accelerator='GPU'),cells=cells)
(HERE/'Attentify_Scaling_Benchmark.ipynb').write_text(json.dumps(nb,indent=1),encoding='utf-8')
print('Built',HERE/'Attentify_Scaling_Benchmark.ipynb')
