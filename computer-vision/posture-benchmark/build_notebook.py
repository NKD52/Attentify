"""Generate the isolated pose comparison notebook; no private paths or footage."""
import json
from pathlib import Path
cells=[]
def cell(kind,text):
    item=dict(cell_type=kind,id=f'posture-{len(cells):03}',metadata={},source=text.splitlines(True))
    if kind=='code':item.update(execution_count=None,outputs=[])
    cells.append(item)

cell('markdown','''# Attentify — body-pose model comparison
Compare original pretrained **YOLO11s-Pose** with **YOLOX-m + RTMPose-m** on identical recorded footage. This is a separate experiment: your face/action/head notebooks are unchanged.

**These models produce person boxes and COCO17 joints, not the six requested posture labels.** This first comparison checks usable skeletons, missed people and execution cost before selecting a posture-classification component. Accuracy stays unknown without reference labels. No training occurs.

Use a hosted GPU runtime. Your laptop's GPU does not matter; Colab's actual device is recorded. A CPU run can be selected explicitly, but will remain labeled CPU. Hosted notebook webcam transfer/display is deliberately excluded: upload a recorded clip for a repeatable inference benchmark. Public weights download automatically and are cached; no SCB model ZIP is needed.

Run cells in order. First use one representative classroom clip plus a separate six-posture demonstration. The demonstration is a functional check, not classroom accuracy or a final untouched test set.
''')
cell('markdown','## 1. Setup — choose Runtime → Change runtime type → T4 GPU\nIf installation requests a restart, restart the runtime and rerun this cell. This installs pinned adapters; it retains Colab’s supplied PyTorch/CUDA and records their actual versions. No fallback to a CPU-only run is allowed when BACKEND is cuda.')
cell('code', '''import sys,subprocess,json
from pathlib import Path
BACKEND='cuda' # Change explicitly to 'cpu' only for a separate CPU experiment
CODE_REF='feature/posture-model-benchmark' # Isolated experiment, never changes the old pipeline
if BACKEND not in ('cuda','cpu'):raise ValueError('Choose cuda or cpu')
import torch
if BACKEND=='cuda' and not torch.cuda.is_available():
    raise RuntimeError('No hosted CUDA GPU. Select a GPU runtime before proceeding; your laptop GPU is not used.')
alternate='onnxruntime' if BACKEND=='cuda' else 'onnxruntime-gpu'
subprocess.check_call([sys.executable,'-m','pip','uninstall','-y',alternate],stdout=subprocess.DEVNULL)
runtime='onnxruntime-gpu==1.23.2' if BACKEND=='cuda' else 'onnxruntime==1.23.2'
subprocess.check_call([sys.executable,'-m','pip','install','-q',runtime,'numpy==2.2.6',
    'opencv-python-headless==4.12.0.88','psutil>=6,<8','polars>=1,<2','tqdm','pyyaml','requests','scipy','matplotlib','pillow','pandas'])
subprocess.check_call([sys.executable,'-m','pip','install','-q','--no-deps',
    'ultralytics==8.3.200','rtmlib==0.0.14','ultralytics-thop==2.0.17'])
REPO=Path('/content/attentify-posture-source')
if not REPO.exists():subprocess.check_call(['git','clone','https://github.com/NKD52/Attentify.git',str(REPO)])
if subprocess.check_output(['git','status','--porcelain'],cwd=REPO,text=True).strip():raise RuntimeError('Save your source edits before updating, or start a fresh runtime')
subprocess.check_call(['git','fetch','origin',CODE_REF],cwd=REPO)
subprocess.check_call(['git','checkout','--detach','FETCH_HEAD'],cwd=REPO)
CODE=REPO/'computer-vision/posture-benchmark'
sys.path.insert(0,str(CODE))
for name in ['posture_benchmark','posture_models']:sys.modules.pop(name,None)
import cv2,onnxruntime as ort,numpy as np
from importlib.metadata import version
if np.__version__!=version('numpy'):
    raise RuntimeError('NumPy was already loaded before installation. Restart the runtime and rerun setup.')
cv2.setNumThreads(2);torch.set_num_threads(2)
if hasattr(ort,'preload_dlls'):ort.preload_dlls()
if BACKEND=='cuda' and 'CUDAExecutionProvider' not in ort.get_available_providers():raise RuntimeError('ONNX CUDA provider unavailable; restart after installation or fix the runtime')
print('Selected device:',torch.cuda.get_device_name(0) if BACKEND=='cuda' else 'CPU')
print('Source commit:',subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip())
''')
cell('markdown','## 2. Storage and footage\nUpload images/videos or use paths already accessible inside Colab. A Windows D: path is not accessible to the hosted runtime. Optional Drive storage preserves downloaded public weights and results between sessions. Mounting Drive requires Google authorization; it does not publish footage. Review samples contain identifiable images; disable them in the next cell if unwanted.')
cell('code', '''USE_DRIVE=False # Your personal copy can save Drive cache/results paths
CACHE_ROOT='/content/attentify-public-pose-models'
RESULTS_ROOT='/content/attentify-posture-results'
if USE_DRIVE:
    from google.colab import drive
    drive.mount('/content/drive')
    CACHE_ROOT='/content/drive/MyDrive/Attentify-Pose-Experiments/models'
    RESULTS_ROOT='/content/drive/MyDrive/Attentify-Pose-Experiments/results'
INPUT_MODE='upload' # 'upload' or 'path'
SOURCES=['/content/your-classroom-clip.mp4']
if INPUT_MODE=='upload':
    from google.colab import files
    uploaded=files.upload();SOURCES=list(uploaded);del uploaded
elif INPUT_MODE!='path':raise ValueError('Choose upload or path')
print('Selected sources:',[Path(p).name for p in SOURCES])
''')
cell('markdown','''## 3. Run the paired comparison
Default: both model pipelines, two repeats, same first 750 source frames, inference every fifth frame. At 25 FPS this covers 30 seconds with five predictions per source second. It is **not a 5-FPS cap on processing capacity**. Shorter clips end naturally; change MAX_SOURCE_FRAMES for longer tests.

Even repeats reverse the model order. Inference timing uses synchronization and includes preprocessing, execution, output transfers and conversion. Total processing also includes decoding skipped source frames. Model download/initialization, warmup, review generation, packaging, browser display and uploads are excluded from timed throughput. RTMPose includes its person detector and per-person processing; neither native confidence scores nor detector thresholds are calibrated to one another. This is an original-pipeline comparison, not a controlled backbone-only ablation.

Resource sampling is optional and has overhead. GPU memory is sampled usage for the whole process, not exact/model-exclusive memory. Missing readings remain missing. For quality, both models export the same evenly selected inference frames with raw and joint overlays, including frames with no detections. No automatic posture or attention labels are invented.
''')
cell('code', '''from posture_benchmark import run_comparison
MODELS=['yolo11s-pose','rtmpose-m']
REPEATS=2
MAX_SOURCE_FRAMES=750
STRIDE=5
WARMUP=5
REVIEW_SAMPLES=6 # Per case, after timing; 0 exports no images
SAMPLE_RESOURCES=True
print('Trials:',len(SOURCES)*len(MODELS)*REPEATS)
result_zip=run_comparison(SOURCES,RESULTS_ROOT,cache=CACHE_ROOT,backend=BACKEND,
    models=MODELS,repeats=REPEATS,max_source_frames=MAX_SOURCE_FRAMES,stride=STRIDE,
    warmup=WARMUP,review_samples=REVIEW_SAMPLES,sample_resources=SAMPLE_RESOURCES)
print('Saved:',result_zip)
''')
cell('markdown','## 4. Inspect and download\nSend the result ZIP for timing analysis. Open each case’s review folder to inspect joints on matching frames. More detected people or higher native confidence alone does not mean better accuracy. A failed case stays failed: inspect its error rather than interpreting it as a fast result. Do not use GPU timings as your laptop’s minimum requirements.')
cell('code', '''import pandas as pd,zipfile,json
with zipfile.ZipFile(result_zip) as z:
    suite=json.loads(z.read(next(n for n in z.namelist() if n.endswith('/suite.json'))))
    rows=json.loads(z.read(next(n for n in z.namelist() if n.endswith('/comparison.json'))))
print('Actual runtime:',suite['environment'])
table=pd.DataFrame(rows)
columns=['source_filename','repeat','model','status','median_inference_ms','p95_inference_ms',
    'pipeline_elapsed_s','updates_per_wall_s','realtime_factor','mean_person_proposals',
    'sampled_max_process_gpu_memory_mib','review_status','error']
display(table[[c for c in columns if c in table]])
if suite['failed_cases']:print('FAILED cases:',suite['failed_cases'],'— see their error.txt files')
from google.colab import files
files.download(str(result_zip))
''')
cell('markdown','''References: [YOLO11 pretrained pose models](https://docs.ultralytics.com/models/yolo11/), [RTMPose project](https://github.com/open-mmlab/mmpose/tree/main/projects/rtmpose), [rtmlib](https://github.com/Tau-J/rtmlib). Model use follows the respective projects' and training datasets' terms. This notebook exports no weights and sends/publishes no results automatically. Controlled classroom accuracy, subject-separated evaluation and a six-posture classifier remain future work.
''')
nb=dict(nbformat=4,nbformat_minor=5,metadata=dict(colab=dict(name='Attentify_Posture_Comparison.ipynb',private_outputs=True),
    kernelspec=dict(name='python3',display_name='Python 3'),accelerator='GPU'),cells=cells)
target=Path(__file__).with_name('Attentify_Posture_Comparison.ipynb')
target.write_text(json.dumps(nb,indent=1),encoding='utf-8')
print('Built',target.name)
