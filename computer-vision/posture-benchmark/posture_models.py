"""Original pretrained pose models, not six-class posture classifiers."""
import gc,time,urllib.request,uuid,zipfile
from pathlib import Path
import numpy as np

URLS={
 'yolo11s-pose.pt':'https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11s-pose.pt',
 'yolox-m.onnx':'https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/yolox_m_8xb8-300e_humanart-c2c7a14a.zip',
 'rtmpose-m.onnx':'https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/rtmpose-m_simcc-body7_pt-body7_420e-256x192-e48f03d0_20230504.zip',
}

def validate_checkpoint(path):
    try:
        with zipfile.ZipFile(path) as archive:
            if not archive.namelist() or archive.testzip() is not None:
                raise ValueError('Incomplete or corrupt PyTorch checkpoint')
    except (zipfile.BadZipFile,EOFError) as exc:
        raise ValueError('Incomplete or corrupt PyTorch checkpoint') from exc

def download_asset(name,cache):
    cache=Path(cache);cache.mkdir(parents=True,exist_ok=True);target=cache/name
    if target.is_file():
        if target.suffix=='.pt':
            try:validate_checkpoint(target)
            except ValueError:
                target.replace(cache/(name+'.invalid-'+uuid.uuid4().hex))
                print('Preserved incomplete checkpoint; downloading a fresh copy',flush=True)
            else:return target
        else:return target
    temp=cache/(name+'.'+uuid.uuid4().hex+'.download')
    payload=temp.with_suffix('.payload')
    try:
        print('Downloading public pretrained asset:',name,flush=True)
        request=urllib.request.Request(URLS[name],headers={'User-Agent':'Attentify-research/1'})
        with urllib.request.urlopen(request,timeout=60) as response,temp.open('wb') as out:
            if response.status!=200 or response.headers.get('Content-Range'):
                raise ValueError('Server returned partial model bytes; refusing an incomplete download')
            expected=int(response.headers['Content-Length']) if response.headers.get('Content-Length') else None
            received=0
            while chunk:=response.read(1024*1024):out.write(chunk);received+=len(chunk)
            if expected is not None and received!=expected:raise ValueError('Incomplete model download')
        if URLS[name].endswith('.zip'):
            with zipfile.ZipFile(temp) as archive:
                entries=[n for n in archive.namelist() if n.endswith('.onnx')]
                if len(entries)!=1:raise ValueError('Expected exactly one ONNX model in official archive')
                with archive.open(entries[0]) as src,payload.open('wb') as out:
                    while chunk:=src.read(1024*1024):out.write(chunk)
            payload.replace(target)
        else:validate_checkpoint(temp);temp.replace(target)
    finally:temp.unlink(missing_ok=True);payload.unlink(missing_ok=True)
    return target

def pose_for_boxes(model,frame,boxes):
    if len(boxes)==0:return np.empty((0,17,2),np.float32),np.empty((0,17),np.float32)
    return model(frame,bboxes=boxes)

class PoseEngine:
    def __init__(self,name,cache,backend='cuda'):
        import torch
        from importlib.metadata import version
        import sys
        sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'team-pilot'))
        from core import digest
        if backend not in ('cpu','cuda'):raise ValueError('Choose cpu or cuda')
        if backend=='cuda' and not torch.cuda.is_available():raise RuntimeError('CUDA requested; select a Colab GPU runtime. No CPU substitution.')
        self.name=name;self.backend=backend;self.torch=torch;self.paths={}
        if name=='yolo11s-pose':
            from ultralytics import YOLO
            path=download_asset('yolo11s-pose.pt',cache);self.paths['pose']=path
            self.model=YOLO(str(path),task='pose')
            self.model.to('cuda:0' if backend=='cuda' else 'cpu')
            self.config=dict(implementation='ultralytics',version=version('ultralytics'),input_long_edge=640,
                detection_confidence=.25,nms_iou=.7,precision='float32',device=str(self.model.device),
                architecture='single-stage person detection and 17-keypoint pose')
        elif name=='rtmpose-m':
            import onnxruntime as ort
            if hasattr(ort,'preload_dlls'):ort.preload_dlls()
            from rtmlib import YOLOX,RTMPose
            det=download_asset('yolox-m.onnx',cache);pose=download_asset('rtmpose-m.onnx',cache)
            self.paths={'detector':det,'pose':pose}
            self.detector=YOLOX(str(det),model_input_size=(640,640),backend='onnxruntime',device=backend)
            self.pose=RTMPose(str(pose),model_input_size=(192,256),to_openpose=False,backend='onnxruntime',device=backend)
            providers={}
            for label,tool in [('detector',self.detector),('pose',self.pose)]:
                # Recreate with fixed CPU thread count; verify actual provider registration.
                options=ort.SessionOptions();options.intra_op_num_threads=2
                requested=['CUDAExecutionProvider','CPUExecutionProvider'] if backend=='cuda' else ['CPUExecutionProvider']
                tool.session=ort.InferenceSession(str(self.paths[label]),options,providers=requested)
                providers[label]=tool.session.get_providers()
                if backend=='cuda' and 'CUDAExecutionProvider' not in providers[label]:raise RuntimeError('RTM '+label+' CUDA initialization failed')
                tool.session.disable_fallback()
            self.config=dict(implementation='rtmlib',version=version('rtmlib'),onnxruntime=ort.__version__,
                detector='YOLOX-m HumanArt',detector_input=[640,640],pose_input=[192,256],
                wrapper_score_threshold=getattr(self.detector,'score_thr',None),
                wrapper_nms_iou=getattr(self.detector,'nms_thr',None),precision='float32',providers=providers,
                detector_output_shape=self.detector.session.get_outputs()[0].shape,
                threshold_note='Official detector includes NMS; rtmlib applies a final score >0.3 filter. Wrapper score/NMS settings are not necessarily embedded graph thresholds.',
                architecture='person detector then sequential RTMPose per detected person',
                provider_note='CUDA registration allows CPU operators; not a claim every operator runs on GPU')
        else:raise ValueError('Unknown pose model')
        self.config.update(model=name,model_sha256={k:digest(v) for k,v in self.paths.items()},
            source_urls={p.name:URLS[p.name] for p in self.paths.values()},backend=backend,
            keypoint_schema='COCO 17; original image pixels; native confidence scores',posture_classifier=False,
            confidence_note='Native scores are not calibrated across models; counts are proposals, not true people')

    def sync(self):
        if self.backend=='cuda':self.torch.cuda.synchronize(0)

    def infer(self,frame):
        self.sync();phases={}
        if self.name=='yolo11s-pose':
            result=self.model.predict(frame,imgsz=640,conf=.25,iou=.7,device=0 if self.backend=='cuda' else 'cpu',
                                      half=False,verbose=False)[0]
            self.sync()
            points=result.keypoints.xy.cpu().numpy() if result.keypoints is not None else np.empty((0,17,2))
            scores=result.keypoints.conf.cpu().numpy() if result.keypoints is not None else np.empty((0,17))
            boxes=result.boxes.xyxy.cpu().numpy();box_scores=result.boxes.conf.cpu().numpy()
            phases={'library_'+k+'_ms':float(v) for k,v in result.speed.items()}
        else:
            tick=time.perf_counter();boxes=self.detector(frame);self.sync();phases['detector_wall_ms']=(time.perf_counter()-tick)*1000
            tick=time.perf_counter();points,scores=pose_for_boxes(self.pose,frame,boxes);self.sync();phases['pose_wall_ms']=(time.perf_counter()-tick)*1000
            box_scores=[None]*len(boxes)
        people=[dict(proposal_index=i,box=np.asarray(box).astype(float).tolist(),
            box_score=None if score is None else float(score),keypoints=np.asarray(p).astype(float).tolist(),
            keypoint_scores=np.asarray(s).astype(float).tolist(),posture=None)
            for i,(box,score,p,s) in enumerate(zip(boxes,box_scores,points,scores))]
        self.sync()
        return dict(people=people,phases_ms=phases)

    def close(self):
        for name in ('model','detector','pose'):
            if hasattr(self,name):delattr(self,name)
        gc.collect()
        if self.backend=='cuda':self.torch.cuda.empty_cache()
