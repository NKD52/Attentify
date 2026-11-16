"""Versioned 6DRepNet face-crop adapter. Import is light; inference never downloads weights."""
import json
from pathlib import Path
import numpy as np
from core import digest

MODEL_ID='head-6drepnet-v1'
SHA256='9bdcd0e072df0d21e92288b40ea1d1b3cf7173c44728ab9c45c4d673f361e018'
FILENAME='head-sixdrepnet-v1.pth'

def crop_box(box,shape,minimum=16):
    values=np.asarray(box,dtype=float)
    if values.shape!=(4,) or not np.isfinite(values).all():return None
    h,w=shape[:2];x1,y1=np.floor(values[:2]).astype(int);x2,y2=np.ceil(values[2:]).astype(int)
    x1,y1=max(0,min(w,x1)),max(0,min(h,y1));x2,y2=max(0,min(w,x2)),max(0,min(h,y2))
    return list(map(int,[x1,y1,x2,y2])) if min(x2-x1,y2-y1)>=minimum else None

def rotation_to_angles(rotation):
    r=np.asarray(rotation,dtype=np.float64)
    if r.ndim!=3 or r.shape[1:]!=(3,3):raise ValueError('Expected batch of 3x3 rotation matrices')
    sy=np.sqrt(r[:,0,0]**2+r[:,1,0]**2);singular=sy<1e-6
    pitch=np.where(singular,np.arctan2(-r[:,1,2],r[:,1,1]),np.arctan2(r[:,2,1],r[:,2,2]))
    yaw=np.arctan2(-r[:,2,0],sy)
    roll=np.where(singular,0,np.arctan2(r[:,1,0],r[:,0,0]))
    return np.rad2deg(np.stack([pitch,yaw,roll],axis=1))

class HeadPose:
    def __init__(self,weights,backend='cpu',batch_size=4,min_crop=16):
        weights=Path(weights)
        if not weights.is_file():raise FileNotFoundError('Head checkpoint missing; load registered head assets during setup')
        if digest(weights)!=SHA256:raise ValueError('Head checkpoint checksum mismatch')
        if backend not in ('cpu','cuda'):raise ValueError('Head backend must be cpu or cuda')
        if not isinstance(batch_size,int) or batch_size<1:raise ValueError('Head batch size must be positive')
        if not isinstance(min_crop,int) or min_crop<1:raise ValueError('Minimum crop size must be positive')
        import torch
        from torchvision import transforms
        from sixdrepnet.model import SixDRepNet
        from importlib.metadata import version
        if version('sixdrepnet')!='0.1.6':raise RuntimeError('Use the registered sixdrepnet==0.1.6 adapter version')
        if backend=='cuda' and not torch.cuda.is_available():raise RuntimeError('Head CUDA requested but unavailable; choose CPU explicitly')
        self.torch=torch;self.device=torch.device(backend);self.batch_size=batch_size;self.min_crop=min_crop
        self.model=SixDRepNet(backbone_name='RepVGG-B1g2',backbone_file='',deploy=True,pretrained=False)
        state=torch.load(str(weights),map_location='cpu',weights_only=True)
        self.model.load_state_dict(state,strict=True);self.model.eval().to(self.device)
        self.transforms=transforms.Compose([transforms.Resize(224),transforms.CenterCrop(224),transforms.ToTensor(),transforms.Normalize(mean=[.485,.456,.406],std=[.229,.224,.225])])
        self.config=dict(model_id=MODEL_ID,sha256=SHA256,backend=backend,torch=torch.__version__,torchvision=version('torchvision'),pillow=version('Pillow'),sixdrepnet='0.1.6',precision='float32',preprocess='BGR to RGB; PIL shorter-edge resize224/center crop224; ImageNet normalization',batch_size=batch_size,min_crop_pixels=min_crop,angle_units='degrees',angle_order=['pitch','yaw','roll'],confidence=None,convention='upstream Euler Rz(roll) Ry(yaw) Rx(pitch); camera mirror/direction categories uncalibrated')

    def predict_faces(self,frame,faces,seconds):
        from PIL import Image
        rows=[];tensors=[];indexes=[]
        for i,face in enumerate(faces):
            box=crop_box(face['box'],frame.shape,self.min_crop)
            row=dict(face_index=i,track_id=face.get('track_id'),student_id=None,source_seconds=seconds,model_id=MODEL_ID,
                     face_box=face['box'],crop_box=box,face_detector_score=face.get('score'),confidence=None,
                     pitch=None,yaw=None,roll=None,units='degrees',status='unknown',reason='unusable_or_small_crop')
            rows.append(row)
            if box is None:continue
            x1,y1,x2,y2=box
            image=Image.fromarray(np.ascontiguousarray(frame[y1:y2,x1:x2,::-1]))
            tensors.append(self.transforms(image));indexes.append(i)
        with self.torch.inference_mode():
            for start in range(0,len(tensors),self.batch_size):
                batch=self.torch.stack(tensors[start:start+self.batch_size]).to(self.device)
                matrices=self.model(batch).detach().cpu().numpy()
                angles=rotation_to_angles(matrices)
                for index,values in zip(indexes[start:start+self.batch_size],angles):
                    if np.isfinite(values).all():rows[index].update(pitch=float(values[0]),yaw=float(values[1]),roll=float(values[2]),status='estimated',reason=None)
                    else:rows[index]['reason']='nonfinite_prediction'
        return rows
