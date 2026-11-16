"""Colab adapter for the fixed private SCB ONNX model and YuNet. No uploads here."""
import json, platform, time, zipfile, sys,subprocess
from pathlib import Path
import cv2
import numpy as np
from core import prepare, decode, digest, assemble, create_run_directory, package_run, COLORS
from tracker import FaceTracker
from association import associate
from workspace_storage import ensure_assets
from review_package import ReviewPackage

EXPECTED_ACTION = 'a79c61b137e6c3ed125d27289ad615387925eb7b838a591704143fc33d30af72'
EXPECTED_FACE = '8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4'

def load_assets(bundle, destination):
    """Backward-compatible explicit ZIP loader; prefer ensure_assets for persistent paths."""
    return ensure_assets(bundle,destination)/'scb-refined.onnx'

class Engine:
    def __init__(self, assets, backend='cuda', confidence=.5, nms=.45,feature_mode='combined',action_model_id='scb-refined-onnx-v1',action_size=640,face_width=640):
        if not 0 < confidence < 1 or not 0 < nms < 1:
            raise ValueError('Confidence and NMS IoU must be between zero and one')
        if backend not in ('cuda', 'cpu'):
            raise ValueError('Choose cuda or cpu')
        if feature_mode not in ('faces','actions','combined'):raise ValueError('Choose faces, actions or combined')
        if not isinstance(action_size,int) or action_size<32 or action_size%32:raise ValueError('Action size must be a positive multiple of 32')
        if not isinstance(face_width,int) or face_width<32:raise ValueError('Face width must be an integer >=32')
        if action_model_id!='scb-refined-onnx-v1':raise ValueError('This model version has no portable adapter yet; see model-register.json')
        self.feature_mode=feature_mode
        self.action_size=action_size;self.face_width=face_width
        self.session=None;self.face=None
        ort=None
        if feature_mode!='faces':
            import onnxruntime as ort
            ort.disable_telemetry_events()
        if backend == 'cuda' and feature_mode!='faces':
            if hasattr(ort, 'preload_dlls'): ort.preload_dlls()
            if 'CUDAExecutionProvider' not in ort.get_available_providers():
                raise RuntimeError('No CUDA provider. Select a GPU runtime, rerun setup, or explicitly choose cpu')
        assets = Path(assets)
        if feature_mode!='faces':
            if digest(assets/'scb-refined.onnx')!=EXPECTED_ACTION:raise ValueError('Unexpected action model checksum')
            options = ort.SessionOptions(); options.intra_op_num_threads = 2
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if backend == 'cuda' else ['CPUExecutionProvider']
            self.session = ort.InferenceSession(str(assets/'scb-refined.onnx'), options, providers=providers)
            if backend == 'cuda' and 'CUDAExecutionProvider' not in self.session.get_providers():
                raise RuntimeError('CUDA initialization failed; refusing a silently CPU-only run')
            self.session.disable_fallback()
        if feature_mode!='actions':
            if digest(assets/'yunet.onnx')!=EXPECTED_FACE:raise ValueError('Unexpected face model checksum')
            self.face = cv2.FaceDetectorYN.create(str(assets/'yunet.onnx'), '', (640,480), .6, .3, 5000)
        self.confidence, self.nms = confidence, nms
        model_ids=(['yunet-fp32'] if feature_mode!='actions' else [])+([action_model_id] if feature_mode!='faces' else [])
        self.config = dict(version='portable-pilot-2',feature_mode=feature_mode,model_ids=model_ids,
            model_sha256=EXPECTED_ACTION if self.session else None,yunet_sha256=EXPECTED_FACE if self.face else None,
            confidence=confidence, nms_iou=nms, action_long_edge=action_size, face_max_width=face_width,
            providers=self.session.get_providers() if self.session else ['CPU (YuNet)'], provider_note='CUDA can place unsupported operators on CPU; YuNet runs on CPU',
            python=platform.python_version(), ort=ort.__version__ if ort else None, opencv=cv2.__version__,
            attention_model=False, ownership='Unvalidated geometric candidates; temporary track IDs only')
        torch = sys.modules.get('torch')
        if torch is not None and feature_mode!='faces':
            self.config['torch_version'] = torch.__version__
            self.config['torch_cuda_version'] = torch.version.cuda
            self.config['remote_gpu'] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
            self.config['remote_gpu_vram_bytes'] = torch.cuda.get_device_properties(0).total_memory if torch.cuda.is_available() else None
        self.reset()

    def reset(self): self.tracker = FaceTracker(1.5,.6)

    def infer(self, frame, seconds):
        start = time.perf_counter(); h,w = frame.shape[:2]; fw = min(w,self.face_width); fh = max(1,round(h*fw/w))
        found=None
        if self.face is not None:
            self.face.setInputSize((fw,fh)); found = self.face.detect(cv2.resize(frame,(fw,fh)))[1]
        faces = []
        if found is not None:
            for f in found:
                x,y,bw,bh = f[:4]
                faces.append(dict(box=[float(x*w/fw),float(y*h/fh),float((x+bw)*w/fw),float((y+bh)*h/fh)], score=float(f[-1])))
        for f,t in zip(faces,self.tracker.update([tuple(f['box']) for f in faces],seconds)):
            f.update(track_id=t.track_id, student_id=None)
        face_ms = (time.perf_counter()-start)*1000
        actions=[];tensor=None;phases={'yunet_tracking_ms':face_ms,'action_preprocess_ms':0.,'action_runtime_ms':0.,'action_postprocess_ms':0.,'association_ms':0.}
        if self.session is not None:
            tick=time.perf_counter();tensor, meta = prepare(frame,self.action_size)
            phases['action_preprocess_ms']=(time.perf_counter()-tick)*1000
            tick=time.perf_counter()
            raw = self.session.run(None, {self.session.get_inputs()[0].name:tensor})[0]
            phases['action_runtime_ms']=(time.perf_counter()-tick)*1000
            tick=time.perf_counter()
            actions = decode(raw,meta,self.confidence,self.nms)
            phases['action_postprocess_ms']=(time.perf_counter()-tick)*1000
        tick=time.perf_counter()
        for a in actions: a.update(associate(a['box'],faces), student_id=None)
        phases['association_ms']=(time.perf_counter()-tick)*1000
        elapsed = (time.perf_counter()-start)*1000
        return dict(source_seconds=seconds, faces=faces, actions=actions, attention=None,
                    input_shape=list(tensor.shape) if tensor is not None else None,face_input_shape=[fh,fw] if self.face is not None else None,
                    phases_ms=phases,combined_ms=elapsed, yunet_tracking_ms=face_ms, actions_ms=elapsed-face_ms)

def draw(frame, result):
    out = frame.copy()
    for f in result['faces']:
        x,y,r,b = map(int,f['box']); cv2.rectangle(out,(x,y),(r,b),(60,220,70),1)
        cv2.putText(out,f"T{f['track_id']}",(x,max(20,y)),0,.5,(60,220,70),1)
    for a in result['actions']:
        x,y,r,b = map(int,a['box']); color = COLORS[a['class_id']]
        cv2.rectangle(out,(x,y),(r,b),color,2)
        cv2.putText(out,f"{a['cue']}? {a['score']:.2f}",(x,max(20,y+18)),0,.55,color,2)
    return out

class Run:
    def __init__(self, engine, mode, extra=None, root='results',review_every=0):
        self.path = create_run_directory(root); self.rows = []; self.engine = engine
        self.review=ReviewPackage(self.path,review_every)
        engine.reset()
        config = dict(engine.config, mode=mode, settings=extra or {})
        config['code_sha256'] = {n+'.py':digest(sys.modules[n].__file__) for n in ['colab_runtime','core','tracker','association','workspace_storage','review_package']}
        if mode=='browser_camera':config['code_sha256']['browser_camera.py']=digest(Path(__file__).with_name('browser_camera.py'))
        config['model_register_sha256']=digest(Path(__file__).with_name('model-register.json'))
        config['run_id']=self.path.name
        config['review_every_processed_frames']=review_every
        try:
            revision=subprocess.run(['git','rev-parse','HEAD'],cwd=Path(__file__).parent,capture_output=True,text=True,timeout=5,check=True).stdout.strip()
            config['git_commit']=revision
            config['git_dirty']=bool(subprocess.run(['git','status','--porcelain','--untracked-files=no'],cwd=Path(__file__).parent,capture_output=True,text=True,timeout=5,check=True).stdout.strip())
        except (OSError,subprocess.SubprocessError):config['git_commit']=None;config['git_dirty']=None
        (self.path/'config.json').write_text(json.dumps(config,indent=2))
    def add(self, result,frame=None):
        result.update(run_id=self.path.name,timestamp_basis='video_relative_seconds' if (self.path/'source.json').exists() else 'python_request_relative_seconds')
        self.rows.append(result)
        with (self.path/'observations.jsonl').open('a') as f: f.write(json.dumps(result)+'\n')
        if frame is not None:self.review.add(frame,result)
    def finish(self, error=None):
        self.review.finish()
        times = [r['combined_ms'] for r in self.rows]
        summary = dict(processed_frames=len(times), median_processing_ms=float(np.median(times)) if times else None,
                       p95_processing_ms=float(np.percentile(times,95)) if times else None,
                       error=error, limits='No accuracy measured; no attention prediction; processing time excludes browser transfer/display')
        (self.path/'summary.json').write_text(json.dumps(summary,indent=2))
        return package_run(self.path)

def process_media(engine, source, stride=5, max_frames=300,results_root='results',review_every=10):
    if stride < 1 or max_frames < 1: raise ValueError('Positive stride and frame limit required')
    source = Path(source)
    image = cv2.imread(str(source)) if source.suffix.lower() in ('.jpg','.jpeg','.png','.webp') else None
    run = Run(engine,'image' if image is not None else 'video',dict(stride=stride,max_frames=max_frames),root=results_root,review_every=review_every)
    cap = writer = None
    try:
        (run.path/'source.json').write_text(json.dumps(dict(filename=source.name,sha256=digest(source),bytes=source.stat().st_size),indent=2))
        if image is not None:
            for _ in range(3): engine.infer(image,0)
            engine.reset(); result=engine.infer(image,0);result['source_frame_id']=0;run.add(result,image)
            cv2.imwrite(str(run.path/'annotated.jpg'),draw(image,result))
        else:
            cap=cv2.VideoCapture(str(source)); fps=cap.get(cv2.CAP_PROP_FPS)
            if not cap.isOpened() or not np.isfinite(fps) or fps<=0: raise ValueError('Cannot decode video or invalid frame rate')
            i=0
            while len(run.rows)<max_frames:
                ok,frame=cap.read()
                if not ok: break
                if i%stride==0:
                    if not run.rows:
                        for _ in range(3): engine.infer(frame,0)
                        engine.reset()
                    result=engine.infer(frame,i/fps); result['source_frame_id']=i; run.add(result,frame)
                    if writer is None:
                        h,w=frame.shape[:2]
                        writer=cv2.VideoWriter(str(run.path/'annotated.mp4'),cv2.VideoWriter_fourcc(*'mp4v'),fps/stride,(w,h))
                        if not writer.isOpened(): raise RuntimeError('Video encoder unavailable')
                    writer.write(draw(frame,result))
                i+=1
            if not run.rows: raise ValueError('No decodable frames')
    except Exception as exc:
        if writer is not None: writer.release(); writer=None
        diagnostic=run.finish(str(exc))
        raise RuntimeError(f'Failed; partial diagnostic ZIP: {diagnostic}') from exc
    finally:
        if cap is not None: cap.release()
        if writer is not None: writer.release()
    return run.finish()
