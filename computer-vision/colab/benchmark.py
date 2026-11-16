"""Controlled recorded-video scaling experiments. This does not measure live camera latency."""
import argparse,csv,gc,json,platform,re,subprocess,time,traceback,uuid
from dataclasses import asdict,dataclass
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent.parent/'team-pilot'))
import cv2,numpy as np
from colab_runtime import Engine,draw
from core import create_run_directory,digest,package_run
from review_package import ReviewPackage
from benchmark_resources import Resources

@dataclass(frozen=True)
class Case:
    name:str
    mode:str='combined'
    action_size:int=640
    face_width:int=640
    output_width:int=0
    enable_head:bool=False

    def __post_init__(self):
        if not re.fullmatch(r'[A-Za-z0-9_-]{1,80}',self.name):raise ValueError('Case name must use ASCII letters/digits/dashes/underscores')
        if self.mode not in ('faces','actions','combined'):raise ValueError('Invalid feature mode')
        if self.enable_head and self.mode=='actions':raise ValueError('Head pose requires faces')
        if not isinstance(self.action_size,int) or self.action_size<32 or self.action_size%32:raise ValueError('Action size must be a multiple of 32')
        if not isinstance(self.face_width,int) or self.face_width<32:raise ValueError('Face width must be >=32')
        if not isinstance(self.output_width,int) or (self.output_width and (self.output_width<128 or self.output_width%2)):raise ValueError('Output width must be zero or an even integer >=128')

def make_cases(action_sizes=(640,960),face_widths=(640,960),output_widths=(960,3840),include_head=False):
    cases=[Case('faces-640','faces'),Case('actions-640','actions'),Case('combined-640')]
    for size in dict.fromkeys(action_sizes):
        if size!=640:cases.append(Case(f'action-{size}','combined',size,640))
    for width in dict.fromkeys(face_widths):
        if width!=640:cases.append(Case(f'faces-{width}','combined',640,width))
    if action_sizes and face_widths and (max(action_sizes),max(face_widths))!=(640,640):
        cases.append(Case('both-larger','combined',max(action_sizes),max(face_widths)))
    for width in dict.fromkeys(output_widths):cases.append(Case(f'output-{width}','combined',640,640,width))
    if include_head:cases.extend([Case('faces-head','faces',enable_head=True),Case('combined-head',enable_head=True)])
    return cases

def preview_frame(current,snapshot,result,seconds,width):
    """Smooth raw source on left; dated matching inference snapshot on right, never current-person claims."""
    h,w=current.shape[:2];pw=width//2;ph=max(1,round(h*pw/w));height=max(64,ph+36)
    height+=height%2;canvas=np.zeros((height,width,3),np.uint8)
    canvas[36:36+ph,:pw]=cv2.resize(current,(pw,ph))
    cv2.putText(canvas,f'Raw source {seconds:.2f}s',(8,23),0,.5,(200,240,240),1)
    age=None if result is None else max(0,seconds-result['source_seconds'])
    if result is not None and snapshot is not None and age<=2:
        scaled=cv2.resize(snapshot,(pw,ph));sx=pw/snapshot.shape[1];sy=ph/snapshot.shape[0]
        view={'faces':[],'actions':[]}
        for kind in view:
            for box in result[kind]:
                item=dict(box);item['box']=(np.array(box['box'])*[sx,sy,sx,sy]).tolist();view[kind].append(item)
        view['head_poses']=[dict(p,face_box=(np.array(p['face_box'])*[sx,sy,sx,sy]).tolist()) for p in result.get('head_poses',[])]
        canvas[36:36+ph,pw:]=draw(scaled,view)
        text=f'Snapshot {result["source_seconds"]:.2f}s | source gap {age:.2f}s'
    else:text='No fresh inference snapshot'
    cv2.putText(canvas,text,(pw+8,23),0,.45,(0,220,255),1)
    return canvas

def stats(values):
    return {'median':float(np.median(values)),'p95':float(np.percentile(values,95))} if values else {'median':None,'p95':None}

def source_info(clip):
    path=Path(clip['path']).expanduser()
    if not path.is_file():raise FileNotFoundError('Select an existing recorded clip: '+str(path))
    count=clip.get('visible_students')
    if count is not None and (not isinstance(count,int) or isinstance(count,bool) or count<0):raise ValueError('visible_students must be a nonnegative human-supplied integer or null')
    return {'filename':path.name,'sha256':digest(path),'bytes':path.stat().st_size,'label':clip.get('label',path.stem),
            'visible_students':count,'student_count_basis':'user-supplied, not independently verified' if count is not None else 'unknown'}

def code_provenance():
    here=Path(__file__).parent
    paths=[here/n for n in ['benchmark.py','benchmark_resources.py','colab_runtime.py','review_package.py','workspace_storage.py','model-register.json']]
    paths.append(here/'head_pose.py')
    paths += [here.parent/'team-pilot'/n for n in ['core.py','tracker.py','association.py']]
    report={'file_sha256':{str(p.relative_to(here.parent)):digest(p) for p in paths},'python':platform.python_version(),'opencv':cv2.__version__,'numpy':np.__version__,'cpu_logical_count':__import__('os').cpu_count(),'cpu_description':platform.processor(),'opencv_threads':cv2.getNumThreads()}
    try:
        report['git_commit']=subprocess.check_output(['git','rev-parse','HEAD'],cwd=here,text=True,timeout=5).strip()
        report['git_dirty']=bool(subprocess.check_output(['git','status','--porcelain'],cwd=here,text=True,timeout=5).strip())
    except (OSError,subprocess.SubprocessError):report.update(git_commit=None,git_dirty=None)
    return report

def run_case(clip,info,assets,case,folder,backend,stride,max_source_frames,warmup,engine_factory,sample_resources):
    folder.mkdir();resources=Resources(sample_resources);resources.start()
    cap=writer=engine=None;records=[];observations=[];error=None;pipeline_s=None;init_ms=None;warmup_ms=None;fps=None
    totals={'decode_ms':0.,'inference_ms':0.,'render_ms':0.,'encode_ms':0.};engine_config={};trial_start=time.perf_counter()
    settings={'case':asdict(case),'source':info,'stride':stride,'max_source_frames':max_source_frames,'warmup_inferences':warmup,'backend':backend}
    (folder/'case.json').write_text(json.dumps(settings,indent=2))
    try:
        cap=cv2.VideoCapture(str(clip['path']));ok,first=cap.read();fps=cap.get(cv2.CAP_PROP_FPS)
        if not ok or not np.isfinite(fps) or fps<=0:raise ValueError('No decodable frame or invalid source FPS')
        settings.update(source_fps=fps,source_width=first.shape[1],source_height=first.shape[0])
        options={'enable_head':True} if case.enable_head else {}
        tick=time.perf_counter();engine=engine_factory(assets,backend=backend,feature_mode=case.mode,action_size=case.action_size,face_width=case.face_width,**options)
        init_ms=(time.perf_counter()-tick)*1000;engine_config=engine.config
        tick=time.perf_counter()
        for _ in range(warmup):engine.infer(first,0)
        warmup_ms=(time.perf_counter()-tick)*1000;engine.reset();cap.release();cap=cv2.VideoCapture(str(clip['path']))
        last_result=last_frame=None;start=time.perf_counter()
        for index in range(max_source_frames):
            tick=time.perf_counter();ok,frame=cap.read();decode_ms=(time.perf_counter()-tick)*1000
            totals['decode_ms']+=decode_ms
            if not ok:break
            seconds=index/fps;row={'source_frame_id':index,'source_seconds':seconds,'decode_ms':decode_ms,'inference_ms':0.,'render_ms':0.,'encode_ms':0.,'inferred':False}
            if index%stride==0:
                tick=time.perf_counter();result=engine.infer(frame,seconds);row['inference_ms']=(time.perf_counter()-tick)*1000;row['inferred']=True
                result.update(source_frame_id=index,run_id=folder.name,timestamp_basis='video_relative_seconds')
                for head in result.get('head_poses',[]):head.update(source_frame_id=index,run_id=folder.name,timestamp_basis='video_relative_seconds')
                observations.append(result);last_result=result;last_frame=frame
                totals['inference_ms']+=row['inference_ms']
            row['snapshot_source_gap_ms']=None if last_result is None else (seconds-last_result['source_seconds'])*1000
            if case.output_width:
                tick=time.perf_counter();preview=preview_frame(frame,last_frame,last_result,seconds,case.output_width);row['render_ms']=(time.perf_counter()-tick)*1000
                tick=time.perf_counter()
                if writer is None:
                    writer=cv2.VideoWriter(str(folder/'preview.mp4'),cv2.VideoWriter_fourcc(*'mp4v'),fps,(preview.shape[1],preview.shape[0]))
                    if not writer.isOpened():raise RuntimeError('Preview encoder unavailable')
                    settings.update(output_width=preview.shape[1],output_height=preview.shape[0],output_fps=fps)
                writer.write(preview);row['encode_ms']=(time.perf_counter()-tick)*1000
                totals['render_ms']+=row['render_ms'];totals['encode_ms']+=row['encode_ms']
            records.append(row)
        if writer is not None:
            tick=time.perf_counter();writer.release();writer=None;totals['encode_ms']+=(time.perf_counter()-tick)*1000
        pipeline_s=time.perf_counter()-start
        if not records or not observations:raise RuntimeError('No timed inference observations')
    except Exception as exc:
        error=str(exc);(folder/'error.txt').write_text(traceback.format_exc())
    finally:
        if cap is not None:cap.release()
        if writer is not None:writer.release()
        memory=resources.stop()
        del engine;gc.collect()
    (folder/'case.json').write_text(json.dumps(dict(settings,engine=engine_config),indent=2))
    (folder/'observations.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in observations))
    with (folder/'frames.csv').open('w',newline='') as f:
        if records:
            output=csv.DictWriter(f,fieldnames=list(records[0]));output.writeheader();output.writerows(records)
    model=stats([r['combined_ms'] for r in observations]);phase_keys={k for r in observations for k in r.get('phases_ms',{})}
    n=len(records);updates=len(observations)
    summary=dict(case_name=case.name,case_directory=folder.name,feature_mode=case.mode,enable_head=case.enable_head,action_size=case.action_size,face_width=case.face_width,output_width=case.output_width,
        visible_students=info['visible_students'],source_label=info['label'],status='failed' if error else 'completed',error=error,
        decoded_frames=n,inference_updates=updates,source_fps=fps,source_width=settings.get('source_width'),source_height=settings.get('source_height'),
        initialization_ms=init_ms,warmup_ms=warmup_ms,pipeline_elapsed_s=pipeline_s,trial_wall_s=time.perf_counter()-trial_start,
        decoded_source_frames_per_wall_s=n/pipeline_s if pipeline_s else None,inference_updates_per_wall_s=updates/pipeline_s if pipeline_s else None,
        realtime_factor=(n/fps)/pipeline_s if fps and pipeline_s else None,
        median_model_ms=model['median'],p95_model_ms=model['p95'],stage_totals_ms=totals,
        median_model_phases_ms={k:stats([r['phases_ms'][k] for r in observations if k in r.get('phases_ms',{})])['median'] for k in phase_keys},
        mean_detected_faces=float(np.mean([len(r['faces']) for r in observations])) if observations and case.mode!='actions' else None,
        mean_action_proposals=float(np.mean([len(r['actions']) for r in observations])) if observations and case.mode!='faces' else None,
        median_snapshot_source_gap_ms=stats([r['snapshot_source_gap_ms'] for r in records if r['snapshot_source_gap_ms'] is not None])['median'],
        live_camera_latency_ms=None,**memory)
    summary['other_pipeline_ms']=max(0.,pipeline_s*1000-sum(totals.values())) if pipeline_s is not None else None
    (folder/'summary.json').write_text(json.dumps(summary,indent=2))
    return summary

def deferred_review(folder,source,every):
    if not every:return
    rows=[json.loads(l) for l in (folder/'observations.jsonl').read_text().splitlines() if l.strip()][::every]
    targets={r['source_frame_id']:r for r in rows};review=ReviewPackage(folder,every=1);cap=cv2.VideoCapture(str(source))
    try:
        for i in range(max(targets,default=-1)+1):
            ok,frame=cap.read()
            if not ok:break
            if i in targets:review.add(frame,targets[i])
    finally:cap.release()
    review.finish()

def run_suite(clips,assets,cases=None,backend='cuda',results_root='results',stride=5,max_source_frames=250,repeats=2,warmup=3,engine_factory=Engine,sample_resources=True,review_every=0):
    if not clips:raise ValueError('Select at least one clip')
    for value,name,minimum in [(stride,'stride',1),(max_source_frames,'max_source_frames',1),(repeats,'repeats',1),(warmup,'warmup',0),(review_every,'review_every',0)]:
        if not isinstance(value,int) or value<minimum:raise ValueError('Invalid '+name)
    cases=list(cases if cases is not None else make_cases())
    if not cases or len({c.name for c in cases})!=len(cases):raise ValueError('Provide uniquely named cases')
    root=create_run_directory(results_root);infos=[source_info(c) for c in clips];summaries=[];start=time.perf_counter()
    config=dict(version='scaling-benchmark-1',provenance=code_provenance(),sources=infos,cases=[asdict(c) for c in cases],
                backend=backend,stride=stride,max_source_frames=max_source_frames,repeats=repeats,warmup=warmup,review_every=review_every,
                case_order='forward on odd repeats, reversed on even repeats',limits='Recorded-video benchmark, not live latency or accuracy. Student counts are user-supplied. No synthetic duplication is treated as a crowd.')
    (root/'suite.json').write_text(json.dumps(config,indent=2))
    for clip_index,(clip,info) in enumerate(zip(clips,infos)):
        for repeat in range(repeats):
            for case in (cases if repeat%2==0 else list(reversed(cases))):
                folder=root/f'case-{len(summaries):04}-{uuid.uuid4().hex[:8]}'
                print(f'Clip {clip_index+1}, repeat {repeat+1}: {case.name}',flush=True)
                result=run_case(clip,info,assets,case,folder,backend,stride,max_source_frames,warmup,engine_factory,sample_resources)
                result.update(clip_index=clip_index,repeat=repeat+1);summaries.append(result)
                result.update(review_status='disabled' if not review_every else 'not_generated',review_error=None)
                if review_every and result['status']=='completed':
                    try:
                        deferred_review(folder,clip['path'],review_every);result['review_status']='completed'
                    except Exception as exc:
                        result.update(review_status='failed',review_error=str(exc));(folder/'review-error.txt').write_text(traceback.format_exc())
                (folder/'summary.json').write_text(json.dumps(result,indent=2))
    (root/'comparison.json').write_text(json.dumps(summaries,indent=2))
    flat=[]
    for row in summaries:
        entry={k:v for k,v in row.items() if not isinstance(v,(dict,list))}
        entry.update({'total_'+k:v for k,v in row['stage_totals_ms'].items()})
        entry.update({'median_'+k:v for k,v in row['median_model_phases_ms'].items()})
        flat.append(entry)
    with (root/'comparison.csv').open('w',newline='') as f:
        fields=list(dict.fromkeys(k for row in flat for k in row))
        output=csv.DictWriter(f,fieldnames=fields);output.writeheader();output.writerows(flat)
    config['suite_wall_s_before_zip']=time.perf_counter()-start;config['failed_cases']=sum(r['status']=='failed' for r in summaries)
    config['failed_reviews']=sum(r['review_status']=='failed' for r in summaries)
    (root/'suite.json').write_text(json.dumps(config,indent=2))
    (root/'README.txt').write_text('Open comparison.csv or comparison.json. Each case contains settings, observations and timing CSV. Output videos preserve source FPS, with current raw source left and a dated inference snapshot right. Source-time gaps are not measured capture-to-display delay. GPU memory is sampled whole-process usage. Speed does not establish accuracy or minimum hardware.\n')
    return package_run(root)

if __name__=='__main__':
    import sys
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--clips',required=True,help='JSON list of clip path/label/visible_students objects');p.add_argument('--assets',required=True);p.add_argument('--backend',choices=['cpu','cuda'],default='cpu');p.add_argument('--results',default='results');p.add_argument('--frames',type=int,default=250);p.add_argument('--repeats',type=int,default=2)
    a=p.parse_args();print(run_suite(json.loads(Path(a.clips).read_text()),a.assets,backend=a.backend,results_root=a.results,max_source_frames=a.frames,repeats=a.repeats))
