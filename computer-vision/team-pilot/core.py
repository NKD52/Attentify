"""Small, framework-independent fixed-model inference and result packaging helpers."""
from pathlib import Path
import hashlib,json,zipfile
import uuid
from datetime import datetime
import cv2
import numpy as np

NAMES=['handraise','reading','writing']
COLORS=[(0,170,255),(255,220,0),(230,90,220)]

def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()

def assemble(folder,manifest,target):
    folder,target=Path(folder),Path(target)
    if target.exists() and digest(target)==manifest['sha256']:return target
    target.parent.mkdir(parents=True,exist_ok=True)
    temp=target.with_suffix('.assembling')
    total=hashlib.sha256()
    try:
        with temp.open('wb') as out:
            for part in manifest['parts']:
                source=folder/part['name']
                if digest(source)!=part['sha256']:raise ValueError('Corrupt/incomplete model part: '+part['name'])
                with source.open('rb') as f:
                    for chunk in iter(lambda:f.read(1024*1024),b''):out.write(chunk);total.update(chunk)
        if total.hexdigest()!=manifest['sha256']:raise ValueError('Reconstructed model hash mismatch')
        temp.replace(target)
    finally:temp.unlink(missing_ok=True)
    return target

def prepare(frame,long_edge=640):
    if not isinstance(long_edge,int) or long_edge<32 or long_edge%32:
        raise ValueError('Action size must be a positive integer multiple of 32')
    h,w=frame.shape[:2];r=min(long_edge/h,long_edge/w)
    nw,nh=round(w*r),round(h*r)
    dw,dh=(long_edge-nw)%32/2,(long_edge-nh)%32/2
    image=cv2.resize(frame,(nw,nh)) if (w,h)!=(nw,nh) else frame
    image=cv2.copyMakeBorder(image,round(dh-.1),round(dh+.1),round(dw-.1),round(dw+.1),cv2.BORDER_CONSTANT,value=(114,114,114))
    th,tw=image.shape[:2];gain=min(th/h,tw/w)
    meta={'shape':(h,w),'gain':gain,'pad':((tw-w*gain)/2,(th-h*gain)/2)}
    return np.ascontiguousarray(image[:,:,::-1].transpose(2,0,1)[None],dtype=np.float32)/255,meta

def decode(raw,meta,confidence=.5,nms=.45):
    p=np.asarray(raw)[0]
    p=p[(p[:,4]>confidence)&np.isfinite(p).all(axis=1)]
    if not len(p):return []
    scores=p[:,5:]*p[:,4:5];classes=scores.argmax(axis=1);scores=scores.max(axis=1)
    keep=scores>confidence;p,classes,scores=p[keep],classes[keep],scores[keep]
    boxes=np.concatenate((p[:,:2]-p[:,2:4]/2,p[:,:2]+p[:,2:4]/2),axis=1)
    selected=[]
    for cls in np.unique(classes):
        indexes=np.flatnonzero(classes==cls)
        indexes=indexes[np.argsort(-scores[indexes],kind='stable')]
        while len(indexes):
            i=int(indexes[0]);selected.append(i);rest=indexes[1:]
            if not len(rest):break
            b=boxes[rest];a=boxes[i]
            inter=np.maximum(0,np.minimum(a[2:],b[:,2:])-np.maximum(a[:2],b[:,:2])).prod(axis=1)
            areas=np.maximum(0,b[:,2:]-b[:,:2]).prod(axis=1)
            area=np.maximum(0,a[2:]-a[:2]).prod()
            iou=inter/np.maximum(area+areas-inter,1e-9)
            indexes=rest[iou<=nms]
    selected=sorted(selected,key=lambda i:-float(scores[i]))[:300]
    out=[];h,w=meta['shape'];px,py=meta['pad']
    for i in selected:
        box=(boxes[i]-np.array([px,py,px,py]))/meta['gain']
        box=np.clip(box,[0,0,0,0],[w,h,w,h])
        out.append({'class_id':int(classes[i]),'cue':NAMES[int(classes[i])],
                    'score':float(scores[i]),'box':box.tolist(),'state':'candidate'})
    return out

def create_run_directory(root,timestamp=None):
    root=Path(root);root.mkdir(parents=True,exist_ok=True)
    stamp=timestamp or datetime.now().strftime('%Y%m%d-%H%M%S-%f')
    for _ in range(100):
        candidate=root/('run-'+stamp+'-'+uuid.uuid4().hex[:12])
        try:candidate.mkdir();return candidate
        except FileExistsError:continue
    raise RuntimeError('Could not reserve a new run folder; no existing result was changed')

def package_run(run):
    run=Path(run).resolve();number=1
    while True:
        suffix='' if number==1 else f'-export-{number}'
        target=run.parent/('SEND-THIS-'+run.name+suffix+'.zip')
        try:
            archive=zipfile.ZipFile(target,'x',zipfile.ZIP_DEFLATED)
            break
        except FileExistsError:number+=1
    with archive as z:
        for f in sorted(run.rglob('*')):
            if f.is_file() and not f.is_symlink() and f.resolve().is_relative_to(run):
                z.write(f,str(Path(run.name)/f.relative_to(run)))
    return target
