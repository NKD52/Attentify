"""Generate the GitHub-backed notebook. No weights, footage or implementation copies."""
import json
from pathlib import Path
HERE=Path(__file__).resolve().parent
cells=[]
def cell(kind,text):
    c=dict(cell_type=kind,id=f'cell-{len(cells):03}',metadata={},source=text.splitlines(True))
    if kind=='code':c.update(execution_count=None,outputs=[])
    cells.append(c)

cell('markdown', '''# Attentify - portable testing workspace
Use the same GitHub code and registered private weights in Colab or on a new PC. Choose **faces**, **actions**, or **combined**. Current action model: experimental refined YOLOv7 (handraise/reading/writing). Head pose is registered as pending, not integrated. No training, attention prediction or permanent student identity is provided.

Run sections 1-3, then section 4 (image/video) OR section 5 (camera). Change the visible settings and rerun the affected cells. For code updates, rerun setup to fetch the chosen GitHub revision. Every result records the actual revision, source fingerprint, model hashes, settings and timings.

Keep the registered model pack in a private Drive folder. Each teammate needs a private copy or approved access; this notebook does not grant folder access. Set MODEL_SOURCE to your own mounted folder in your private notebook copy. Models can instead live in temporary Colab storage. No repeated upload is needed when loading from Drive. Never commit private storage links, credentials or camera outputs to GitHub. Clear outputs before sharing a notebook.
''')
cell('markdown','## 1. Load dependencies and the chosen GitHub code version\nSelect Runtime > Change runtime type > GPU for CUDA. CPU is also supported. If Colab requests a restart after installation, restart and rerun this cell. The GPU belongs to Colab, not your laptop. Your local working files are never overwritten by this setup.')
cell('code', '''import subprocess,sys,json
from pathlib import Path
CODE_REF='feature/portable-workspace' # Branch, tag or immutable commit SHA
FEATURE_MODE='combined' # 'faces', 'actions', 'combined'
BACKEND='cuda' # 'cpu' explicitly selects CPU action inference
if FEATURE_MODE not in ('faces','actions','combined'):raise ValueError('Choose faces, actions or combined')
dependencies=['opencv-python-headless==4.12.0.88','numpy==2.2.6']
if FEATURE_MODE!='faces':
    package='onnxruntime-gpu==1.23.2' if BACKEND=='cuda' else 'onnxruntime==1.23.2'
    # Both packages provide the same module; remove the alternate distribution.
    alternate='onnxruntime' if BACKEND=='cuda' else 'onnxruntime-gpu'
    subprocess.check_call([sys.executable,'-m','pip','uninstall','-y',alternate],stdout=subprocess.DEVNULL)
    dependencies.append(package)
subprocess.check_call([sys.executable,'-m','pip','install','-q',*dependencies])
if FEATURE_MODE!='faces' and BACKEND=='cuda':
    import torch # Load Colab's CUDA/cuDNN libraries before ONNX Runtime
REPO=Path('/content/attentify-source')
if not REPO.exists():subprocess.check_call(['git','clone','https://github.com/NKD52/Attentify.git',str(REPO)])
dirty=subprocess.check_output(['git','status','--porcelain'],cwd=REPO,text=True).strip()
if dirty:raise RuntimeError('Source checkout has local changes. Save them before updating or use a new runtime.')
subprocess.check_call(['git','fetch','origin',CODE_REF],cwd=REPO)
subprocess.check_call(['git','checkout','--detach','FETCH_HEAD'],cwd=REPO)
CODE=REPO/'computer-vision/colab'
sys.path[:0]=[str(CODE),str(REPO/'computer-vision/team-pilot')]
# Reload after fetching an update; existing engine objects are recreated in section 3.
for name in ['browser_camera','colab_runtime','workspace_storage','review_package','core','tracker','association']:
    sys.modules.pop(name,None)
print('Code commit:',subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip())
if FEATURE_MODE!='faces' and BACKEND=='cuda':print('Remote GPU:',torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none')
else:print('Selected inference uses CPU')
''')
cell('markdown','## 2. Storage and threshold settings\nDrive mode connects your Google Drive and reuses the private model folder. The distributed model pack has two parts; loading and verification are automatic. Your first run must authorize the connection. A local path here means a path inside the Colab runtime, not D: on your laptop. Upload mode is optional for a full ZIP. Results can stay temporarily in Colab or be saved under a private writable Drive folder. Drive mounting grants notebook code access to your Drive; use only trusted code. Choose FEATURE_MODE in section 1.')
cell('code', '''STORAGE='drive' # 'drive', 'path', or 'upload'
ACTION_MODEL_ID='scb-refined-onnx-v1' # Only this action version has a portable adapter currently
CONFIDENCE=0.50
NMS_IOU=0.45
MODEL_SOURCE='/content/drive/MyDrive/YOUR_MODEL_FOLDER' # Set this privately; do not commit your private path
RESULTS_ROOT='/content/attentify-results' # Change to your private Drive path for persistent results
if STORAGE=='drive':
    from google.colab import drive
    drive.mount('/content/drive')
elif STORAGE=='upload':
    from google.colab import files
    uploaded=files.upload()
    if len(uploaded)!=1:raise ValueError('Upload exactly one private model ZIP')
    MODEL_SOURCE=next(iter(uploaded));del uploaded
elif STORAGE!='path':raise ValueError('Choose drive, path or upload')
print('Mode:',FEATURE_MODE,'| Models:',MODEL_SOURCE,'| Results:',RESULTS_ROOT)
''')
cell('markdown','## 3. Initialize only the selected features\nModel checksums must match the register. The model ZIP is copied into the temporary runtime cache to avoid repeated small Drive reads. This cache copy is automatic on a fresh session; you do not reupload the ZIP. A requested CUDA action run fails if CUDA is unavailable. YuNet uses CPU even in combined CUDA mode.')
cell('code', '''from workspace_storage import ensure_assets
from colab_runtime import Engine,process_media
assets=ensure_assets(MODEL_SOURCE,'/content/attentify-model-cache',FEATURE_MODE)
engine=Engine(assets,BACKEND,CONFIDENCE,NMS_IOU,FEATURE_MODE,ACTION_MODEL_ID)
print(json.dumps(engine.config,indent=2))
''')
cell('markdown','## 4. Test an image/video and download a review package\nUse an existing Drive/runtime path or upload one file. Default: every fifth video frame, up to 300 processed frames, with a review sample every 10 processed frames. Review samples include frames with no action detections so you can report misses. Extract the returned ZIP and open review/index.html, then export your answers as JSON. Send that review JSON with the run ZIP. This diagnostic sampling is not an accuracy test.\n\nThe ZIP contains identifiable annotated media and sampled raw review images. Set REVIEW_EVERY=0 to disable review images; image/video mode still writes the annotated output. Use only permitted footage. Three warm-ups are excluded from processing timings; decoding, encoding and network delays are excluded too.')
cell('code', '''INPUT_MODE='upload' # 'upload' or 'path'
SOURCE_PATH='/content/drive/MyDrive/your-selected-clip.mp4'
STRIDE=5
MAX_PROCESSED_FRAMES=300
REVIEW_EVERY=10 # 0 disables sampled review images
from google.colab import files
if INPUT_MODE=='upload':
    uploaded=files.upload()
    if len(uploaded)!=1:raise ValueError('Choose exactly one image or clip')
    SOURCE_PATH=next(iter(uploaded));del uploaded
elif INPUT_MODE!='path':raise ValueError('Choose upload or path')
result_zip=process_media(engine,SOURCE_PATH,STRIDE,MAX_PROCESSED_FRAMES,RESULTS_ROOT,REVIEW_EVERY)
print('Saved:',result_zip)
files.download(str(result_zip))
''')
cell('markdown','## 5. Experimental browser camera\nRun instead of section 4. Browser permission is required. Frames go to Google for processing. No audio is requested. One frame is requested at a time, so this is sampled inference, not a guaranteed live video rate. Stop with the button or the time limit. Camera review images are OFF by default; enable only if you want those identifiable images exported. Processing time and request-cycle time are separate. Request-cycle time excludes returned-image display, so it is not full camera-to-screen latency. Mobile compatibility and hosted GPU operation still need a real Colab trial.')
cell('code', '''from browser_camera import run_camera
from google.colab import files
CAMERA_SECONDS=60
CAMERA_REVIEW_EVERY=0 # Set e.g. 10 only to save review images explicitly
result_zip=run_camera(engine,CAMERA_SECONDS,RESULTS_ROOT,CAMERA_REVIEW_EVERY)
print('Saved:',result_zip)
files.download(str(result_zip))
''')
cell('markdown','## Updating and teamwork\nKeep working models intact. Change code on a branch, open this notebook, choose CODE_REF and rerun setup plus initialization. Keep downloaded run ZIPs and exported review JSONs. Dataset train/validation/test groups must stay separate; repeatedly reviewed examples are development evidence. New trained weights require a new registered model ID and checksum before they can be selected. The original PyTorch action model and the head handoff are preserved in the register as unsupported pending adapters; they are not silently substituted.\n\nEach teammate uses their own runtime and authorized private storage. Code is shared through GitHub; model files and data are not publicly distributed by this notebook. A single-person camera trial does not establish classroom accuracy or minimum hardware.\n\nReferences: [Colab FAQ](https://research.google.com/colaboratory/faq.html), [Google camera snippet](https://colab.research.google.com/notebooks/snippets/advanced_outputs.ipynb), [ONNX CUDA requirements](https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html).')
nb=dict(nbformat=4,nbformat_minor=5,metadata=dict(colab=dict(name='Attentify_Colab_Testing.ipynb'),kernelspec=dict(name='python3',display_name='Python 3'),accelerator='GPU'),cells=cells)
(HERE/'Attentify_Colab_Testing.ipynb').write_text(json.dumps(nb,indent=1),encoding='utf-8')
print('Built',HERE/'Attentify_Colab_Testing.ipynb')
