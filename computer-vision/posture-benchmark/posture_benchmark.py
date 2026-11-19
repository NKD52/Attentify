"""Recorded-media comparison of pose pipelines, not posture/attention classification."""
import csv,gc,json,os,platform,subprocess,sys,time,traceback
from importlib.metadata import PackageNotFoundError,version
from pathlib import Path
import cv2
import numpy as np

HERE=Path(__file__).resolve().parent
sys.path.extend([str(HERE.parent/'team-pilot'),str(HERE.parent/'colab')])
from core import create_run_directory,digest,package_run
from benchmark_resources import Resources
from posture_models import PoseEngine

MODELS=('yolo11s-pose','rtmpose-m')
JOINTS=['nose','left_eye','right_eye','left_ear','right_ear','left_shoulder','right_shoulder',
        'left_elbow','right_elbow','left_wrist','right_wrist','left_hip','right_hip',
        'left_knee','right_knee','left_ankle','right_ankle']
EDGES=[(5,6),(5,7),(7,9),(6,8),(8,10),(5,11),(6,12),(11,12),(11,13),(13,15),(12,14),(14,16)]

def environment(backend):
    packages={}
    for name in ['torch','torchvision','ultralytics','rtmlib','onnxruntime-gpu','onnxruntime','numpy','opencv-python','opencv-python-headless']:
        try:packages[name]=version(name)
        except PackageNotFoundError:pass
    result=dict(python=platform.python_version(),platform=platform.platform(),logical_cpus=os.cpu_count(),
                loaded_numpy=np.__version__,loaded_opencv=cv2.__version__,
                opencv_threads=cv2.getNumThreads(),packages=packages,backend_requested=backend,
                source_sha256={p.name:digest(p) for p in [Path(__file__),HERE/'posture_models.py']})
    try:
        import torch
        result.update(torch_threads=torch.get_num_threads(),cuda_available=torch.cuda.is_available(),
            torch_cuda=torch.version.cuda,cudnn=torch.backends.cudnn.version())
        if torch.cuda.is_available():
            props=torch.cuda.get_device_properties(0);result.update(gpu_name=props.name,gpu_total_vram_bytes=props.total_memory)
        if backend=='cuda' and not torch.cuda.is_available():raise RuntimeError('Select Runtime > Change runtime type > GPU. CUDA is required for this run.')
    except ImportError:
        if backend=='cuda':raise RuntimeError('PyTorch CUDA is required for a GPU benchmark')
    try:result['git_commit']=subprocess.check_output(['git','rev-parse','HEAD'],cwd=HERE,text=True,timeout=5).strip()
    except (OSError,subprocess.SubprocessError):result['git_commit']=None
    return result

def render(frame,observation):
    out=frame.copy()
    for person in observation['people']:
        box=np.asarray(person['box'],float)
        if np.isfinite(box).all():
            x1,y1,x2,y2=map(int,box);cv2.rectangle(out,(x1,y1),(x2,y2),(0,180,255),2)
        points=np.asarray(person['keypoints']);scores=np.asarray(person['keypoint_scores'])
        usable=np.isfinite(points).all(axis=1)&np.isfinite(scores)&(scores>=.3)
        for a,b in EDGES:
            if usable[a] and usable[b]:cv2.line(out,tuple(points[a].astype(int)),tuple(points[b].astype(int)),(0,220,0),2)
        for i,(point,ok) in enumerate(zip(points,usable)):
            if ok:cv2.circle(out,tuple(point.astype(int)),3,(0,100,255),-1)
    cv2.putText(out,f"Source frame {observation['source_frame_id']} | joints only; no posture classification",(8,22),0,.5,(255,255,255),1)
    return out

def reviews(source,image,observations,folder,count):
    if not count or not observations:return
    chosen=np.unique(np.linspace(0,len(observations)-1,min(count,len(observations)),dtype=int))
    targets={observations[i]['source_frame_id']:observations[i] for i in chosen}
    out=folder/'review';out.mkdir();cap=None
    try:
        if image is not None:frames=[(0,image)]
        else:
            cap=cv2.VideoCapture(str(source))
            def stream():
                for i in range(max(targets)+1):
                    ok,frame=cap.read()
                    if not ok:break
                    if i in targets:yield i,frame
            frames=stream()
        for i,frame in frames:
            cv2.imwrite(str(out/f'frame-{i:06}-raw.jpg'),frame)
            cv2.imwrite(str(out/f'frame-{i:06}-joints.jpg'),render(frame,targets[i]))
        (out/'README.txt').write_text('Compare raw and matching joints images across models. Check correct person, shoulders/hips/wrists, missed people and occlusion. Drawing threshold 0.3 is visualization only and not a shared calibrated confidence. These are proposals, not ground truth. No posture accuracy is calculated.\n')
    finally:
        if cap is not None:cap.release()

def trial(source,model,cache,folder,backend,stride,max_source_frames,warmup,review_samples,engine_factory,sample_resources):
    folder.mkdir();cap=engine=None;rows=[];observations=[];status='failed';error=None;settings={}
    elapsed=init_ms=warmup_ms=None;fps=None;image=None;resources=Resources(sample_resources)
    totals=dict(decode_ms=0.,inference_ms=0.)
    try:
        # Extension check avoids trying to decode videos as still images.
        if source.suffix.lower() in ('.jpg','.jpeg','.png','.bmp','.webp'):image=cv2.imread(str(source))
        if image is not None:first=image;fps=None
        else:
            cap=cv2.VideoCapture(str(source));ok,first=cap.read();fps=cap.get(cv2.CAP_PROP_FPS)
            if not ok or not np.isfinite(fps) or fps<=0:raise ValueError('Invalid/undecodable source or FPS')
        settings.update(source_width=first.shape[1],source_height=first.shape[0],source_fps=fps,kind='image' if image is not None else 'video')
        resources.start();tick=time.perf_counter();engine=engine_factory(model,cache,backend=backend);init_ms=(time.perf_counter()-tick)*1000
        settings['engine']=engine.config
        tick=time.perf_counter()
        for _ in range(warmup):engine.infer(first)
        warmup_ms=(time.perf_counter()-tick)*1000
        if cap is not None:cap.release();cap=cv2.VideoCapture(str(source))
        start=time.perf_counter()
        for index in range(1 if image is not None else max_source_frames):
            tick=time.perf_counter()
            if image is not None:ok,frame=True,image
            else:ok,frame=cap.read()
            decode_ms=(time.perf_counter()-tick)*1000;totals['decode_ms']+=decode_ms
            if not ok:break
            row=dict(source_frame_id=index,decode_ms=decode_ms,inferred=False,inference_ms=0.)
            if index%stride==0:
                tick=time.perf_counter();result=engine.infer(frame);cost=(time.perf_counter()-tick)*1000
                row.update(inferred=True,inference_ms=cost);totals['inference_ms']+=cost
                result.update(source_frame_id=index,source_seconds=index/fps if fps else None,model=model,
                              timestamp_basis='video_relative_seconds' if fps else 'still_image',inference_wall_ms=cost)
                observations.append(result)
            rows.append(row)
        elapsed=time.perf_counter()-start
        if not observations:raise RuntimeError('No inference updates')
        status='completed'
    except Exception as exc:
        error=type(exc).__name__+': '+str(exc);(folder/'error.txt').write_text(traceback.format_exc())
    finally:
        if cap is not None:cap.release()
        memory=resources.stop()
        if engine is not None and hasattr(engine,'close'):engine.close()
        del engine;gc.collect()
    summary=dict(model=model,status=status,error=error,decoded_frames=len(rows),inference_updates=len(observations),
        initialization_ms=init_ms,warmup_ms=warmup_ms,pipeline_elapsed_s=elapsed,stage_totals_ms=totals,
        updates_per_wall_s=len(observations)/elapsed if elapsed else None,
        realtime_factor=len(rows)/fps/elapsed if fps and elapsed else None,
        median_inference_ms=float(np.median([r['inference_wall_ms'] for r in observations])) if observations else None,
        p95_inference_ms=float(np.percentile([r['inference_wall_ms'] for r in observations],95)) if observations else None,
        mean_person_proposals=float(np.mean([len(r['people']) for r in observations])) if observations else None,
        phase_medians_ms={k:float(np.median([r['phases_ms'][k] for r in observations if k in r['phases_ms']])) for k in {k for r in observations for k in r['phases_ms']}},
        posture_accuracy=None,keypoint_accuracy=None,posture_classifier=False,**settings,**memory)
    (folder/'settings.json').write_text(json.dumps(dict(stride=stride,max_source_frames=max_source_frames,warmup=warmup,backend=backend,**settings),indent=2))
    (folder/'observations.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in observations))
    if rows:
        with (folder/'frames.csv').open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    summary['review_status']='disabled' if not review_samples else 'not_generated'
    if review_samples and status=='completed':
        try:reviews(source,image,observations,folder,review_samples);summary['review_status']='completed'
        except Exception as exc:summary.update(review_status='failed',review_error=str(exc))
    (folder/'summary.json').write_text(json.dumps(summary,indent=2));return summary

def run_comparison(sources,results_root,cache=None,backend='cuda',models=MODELS,repeats=2,
                   max_source_frames=750,stride=5,warmup=3,review_samples=6,
                   engine_factory=PoseEngine,sample_resources=True):
    if backend not in ('cuda','cpu'):raise ValueError('Choose cuda or cpu')
    if not sources or not models or len(set(models))!=len(models) or any(m not in MODELS for m in models):raise ValueError('Select sources and supported unique models')
    for value,minimum in [(repeats,1),(max_source_frames,1),(stride,1),(warmup,0),(review_samples,0)]:
        if not isinstance(value,int) or value<minimum:raise ValueError('Invalid run limits')
    paths=[Path(p).expanduser() for p in sources]
    if not all(p.is_file() for p in paths):raise FileNotFoundError('An input source does not exist')
    env=environment(backend);root=create_run_directory(results_root);cache=cache or root.parent/'model-cache'
    manifest=dict(version='posture-benchmark-1',environment=env,models=list(models),repeats=repeats,stride=stride,
        max_source_frames=max_source_frames,warmup=warmup,keypoint_names=JOINTS,review_samples=review_samples,
        sources=[dict(filename=p.name,sha256=digest(p),bytes=p.stat().st_size) for p in paths],
        limits='Offline pose-estimator pipelines only; no six-class posture or attention classifier. No accuracy without reference labels. Native detectors/thresholds differ. CUDA requested fails instead of silent CPU-only substitution. Initialization, downloads, warmup and review generation excluded from timed loop. Images have no real-time factor; GPU samples are whole-process, not model-exclusive.')
    (root/'suite.json').write_text(json.dumps(manifest,indent=2));results=[]
    for source_index,path in enumerate(paths):
        for repeat in range(repeats):
            for model in (models if repeat%2==0 else list(reversed(models))):
                name=f'case-{len(results):04}-{model}';print(f'Source {source_index+1}, repeat {repeat+1}: {model}',flush=True)
                row=trial(path,model,cache,root/name,backend,stride,max_source_frames,warmup,review_samples,engine_factory,sample_resources)
                row.update(source_index=source_index,source_filename=path.name,repeat=repeat+1,case_directory=name);results.append(row)
    (root/'comparison.json').write_text(json.dumps(results,indent=2))
    flat=[{k:v for k,v in r.items() if not isinstance(v,(dict,list))} for r in results]
    with (root/'comparison.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(dict.fromkeys(k for r in flat for k in r)));writer.writeheader();writer.writerows(flat)
    manifest.update(failed_cases=sum(r['status']=='failed' for r in results),failed_reviews=sum(r['review_status']=='failed' for r in results))
    (root/'suite.json').write_text(json.dumps(manifest,indent=2))
    (root/'README.txt').write_text('Read comparison.csv/json and suite.json. Each case has original-coordinate COCO17 proposals, frame timings, model hashes and optional matching raw/joint review images. All model predictions are proposals, not ground truth. Compare paired source frames. No trained posture classifier or attention output exists here. Timed inference is synchronized host wall time including preparation, model execution, transfers and output conversion; library subphase timers are not comparable across frameworks. Source decoding is counted even on skipped inference frames. Live Colab browser-camera latency is not measured. Weights and footage are never published automatically.\n')
    return package_run(root)
