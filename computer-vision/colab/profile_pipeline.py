"""Bounded diagnostics, separate from the normal benchmark. Never exports source images."""
import gc
import json
import subprocess
import threading
import time
from contextlib import contextmanager
from pathlib import Path

import cv2
from benchmark import Case, Engine, code_provenance, run_case, source_info
from core import create_run_directory, digest, package_run


def provider_summary(events):
    """ORT provider durations are event sums, not GPU occupancy or wall time."""
    out={}
    for event in events:
        provider=event.get('args',{}).get('provider')
        if event.get('cat')!='Node' or not event.get('name','').endswith('_kernel_time') or not provider:
            continue
        row=out.setdefault(provider,dict(events=0,duration_us=0.))
        row['events']+=1;row['duration_us']+=float(event.get('dur',0))
    return out


class GpuSamples:
    """Remote device-wide busy percentage, distinct from VRAM. Missing stays missing."""
    def __init__(self,enabled):
        self.enabled=enabled;self.rows=[];self.error=None;self.stop_event=threading.Event();self.thread=None

    def start(self):
        if self.enabled:
            self.thread=threading.Thread(target=self._poll,daemon=True);self.thread.start()

    def _poll(self):
        start=time.perf_counter()
        while not self.stop_event.is_set():
            try:
                result=subprocess.run(['nvidia-smi','--id=0','--query-gpu=utilization.gpu,utilization.memory,memory.used',
                                       '--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=3,check=True)
                busy,bandwidth,memory=map(float,result.stdout.strip().split(','))
                self.rows.append(dict(elapsed_s=time.perf_counter()-start,gpu_busy_percent=busy,
                                      memory_busy_percent=bandwidth,memory_used_mib=memory))
            except Exception as exc:
                self.error=type(exc).__name__+': '+str(exc);break
            self.stop_event.wait(.2)

    def stop(self):
        self.stop_event.set()
        if self.thread:self.thread.join(timeout=4)
        return dict(samples=self.rows,error=self.error,scope='Device 0, device-wide sampled busy %, not process-exclusive or exact occupancy')


@contextmanager
def head_regions(head,record_function):
    """Add trace labels only to this diagnostic instance; restore even after failure."""
    if head is None:
        yield;return
    transform,forward=head.transforms,head.model.forward
    def wrapped_transform(*args,**kwargs):
        with record_function('attentify.head_cpu_crop_transform'):return transform(*args,**kwargs)
    def wrapped_forward(*args,**kwargs):
        with record_function('attentify.head_network'):return forward(*args,**kwargs)
    head.transforms=wrapped_transform;head.model.forward=wrapped_forward
    try:yield
    finally:head.transforms=transform;head.model.forward=forward


def run_profile(clip,assets,backend='cuda',enable_head=True,max_source_frames=250,stride=5,
                results_root='results',engine_factory=Engine):
    """One no-export baseline plus a separate overhead-bearing CPU/CUDA trace pass."""
    if not isinstance(max_source_frames,int) or not 1<=max_source_frames<=300:
        raise ValueError('Profile a short window: max_source_frames must be 1..300')
    if not isinstance(stride,int) or stride<1:raise ValueError('Stride must be positive')
    if backend not in ('cpu','cuda'):raise ValueError('Choose cpu or cuda')
    import torch
    if backend=='cuda' and not torch.cuda.is_available():raise RuntimeError('CUDA profiling requested but unavailable')
    info=source_info(clip);root=create_run_directory(results_root)
    report=dict(version='pipeline-profile-1',source=info,provenance=code_provenance(),
                profiler_sha256=digest(__file__),backend=backend,enable_head=enable_head,
                max_source_frames=max_source_frames,stride=stride,
                limits='Profiler overhead changes timings. Compare normal baseline separately; no video export, live latency or accuracy test. ORT trace includes warmup. CUDA event sums can overlap and are not utilization percentages.')
    samples=GpuSamples(backend=='cuda');samples.start()
    try:
        report['baseline']=run_case(clip,info,assets,Case('profile-baseline',enable_head=enable_head),
                                    root/'baseline',backend,stride,max_source_frames,3,engine_factory,False)
    finally:report['baseline_gpu_samples']=samples.stop()
    if report['baseline']['status']!='completed':
        (root/'profile-summary.json').write_text(json.dumps(report,indent=2))
        raise RuntimeError('Baseline failed; inspect '+str(root/'baseline/error.txt'))
    gc.collect()
    engine=cap=None;ort_trace=None;updates=decoded=0
    try:
        engine=engine_factory(assets,backend=backend,feature_mode='combined',enable_head=enable_head)
        # ORT tracing must be enabled when constructing a session. The diagnostic
        # instance uses the same fixed model/provider/thread settings as Engine.
        if engine.session is not None:
            import onnxruntime as ort
            options=ort.SessionOptions();options.intra_op_num_threads=2;options.enable_profiling=True
            options.profile_file_prefix=str(root/'ort-trace')
            providers=['CUDAExecutionProvider','CPUExecutionProvider'] if backend=='cuda' else ['CPUExecutionProvider']
            engine.session=ort.InferenceSession(str(Path(assets)/'scb-refined.onnx'),options,providers=providers)
            if backend=='cuda' and 'CUDAExecutionProvider' not in engine.session.get_providers():
                raise RuntimeError('Diagnostic CUDA session failed to initialize')
            engine.session.disable_fallback()
        cap=cv2.VideoCapture(str(clip['path']));ok,first=cap.read();fps=cap.get(cv2.CAP_PROP_FPS)
        if not ok or not fps>0:raise ValueError('No decodable frames or invalid FPS')
        for _ in range(3):engine.infer(first,0)
        engine.reset();cap.release();cap=cv2.VideoCapture(str(clip['path']))
        activities=[torch.profiler.ProfilerActivity.CPU]
        if backend=='cuda':activities.append(torch.profiler.ProfilerActivity.CUDA)
        with torch.profiler.profile(activities=activities,record_shapes=False,with_stack=False,profile_memory=False) as prof:
            with head_regions(engine.head,torch.profiler.record_function):
                for index in range(max_source_frames):
                    with torch.profiler.record_function('attentify.decode'):
                        ok,frame=cap.read()
                    if not ok:break
                    decoded+=1
                    if index%stride==0:
                        with torch.profiler.record_function('attentify.combined_inference'):
                            engine.infer(frame,index/fps)
                        updates+=1
                    prof.step()
        prof.export_chrome_trace(str(root/'torch-trace.json'))
        averages=prof.key_averages()
        (root/'torch-cpu.txt').write_text(averages.table(sort_by='self_cpu_time_total',row_limit=40))
        if backend=='cuda':
            (root/'torch-device.txt').write_text(averages.table(sort_by='self_device_time_total',row_limit=40))
        report.update(profiled_inference_updates=updates,profiled_decoded_frames=decoded,engine=engine.config)
        if engine.session is not None:
            ort_trace=Path(engine.session.end_profiling())
            report['ort_provider_events']=provider_summary(json.loads(ort_trace.read_text()))
        report['status']='completed'
    except Exception as exc:
        report.update(status='failed',error=type(exc).__name__+': '+str(exc))
        raise
    finally:
        if cap is not None:cap.release()
        if engine is not None and engine.session is not None and ort_trace is None:
            try:engine.session.end_profiling()
            except Exception:pass
        (root/'profile-summary.json').write_text(json.dumps(report,indent=2))
        del engine;gc.collect()
    (root/'README.txt').write_text('Read profile-summary.json first. baseline/ timings exclude video export and profiler traces. '
        'baseline_gpu_samples separates GPU busy % from memory use; samples cover the baseline initialization/warmup/loop. '
        'torch-trace.json captures PyTorch CPU/CUDA operations; head CPU transforms and network have named regions. '
        'ORT has a separate trace/provider summary for action operators; PyTorch alone does not establish ORT GPU placement. '
        'ORT event durations are host/runtime measurements, not pure kernel execution times. '
        'Traces may include machine paths and operation metadata; keep the ZIP private. No frames or weights are exported.\n')
    return package_run(root)
