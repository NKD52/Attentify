import json,sys,tempfile,unittest,zipfile
from pathlib import Path
import cv2
import numpy as np
HERE=Path(__file__).resolve().parents[1];sys.path.insert(0,str(HERE))

class BenchmarkTests(unittest.TestCase):
    def test_incomplete_checkpoint_is_rejected_before_loading(self):
        from posture_models import validate_checkpoint
        folder=HERE/'.tmp';folder.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=folder) as d:
            p=Path(d)/'partial.pt';p.write_bytes(b'PK\x03\x04interrupted-download')
            with self.assertRaisesRegex(ValueError,'Incomplete'):
                validate_checkpoint(p)

    def test_same_source_frames_reverse_order_and_no_accuracy_claim(self):
        from posture_benchmark import run_comparison
        calls=[]
        class Fake:
            def __init__(self,name,*args,**kwargs):self.config={'model':name};calls.append(name)
            def infer(self,frame):return {'people':[],'phases_ms':{}}
        folder=HERE/'.tmp';folder.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=folder) as d:
            d=Path(d);clip=d/'clip.mp4';w=cv2.VideoWriter(str(clip),cv2.VideoWriter_fourcc(*'mp4v'),25,(160,120))
            for _ in range(12):w.write(np.zeros((120,160,3),np.uint8))
            w.release()
            zpath=run_comparison([str(clip)],d,backend='cpu',models=['yolo11s-pose','rtmpose-m'],repeats=2,
                max_source_frames=10,stride=2,warmup=0,review_samples=0,engine_factory=Fake,sample_resources=False)
            self.assertEqual(calls,['yolo11s-pose','rtmpose-m','rtmpose-m','yolo11s-pose'])
            with zipfile.ZipFile(zpath) as z:
                rows=json.loads(z.read(next(n for n in z.namelist() if n.endswith('/comparison.json'))))
                self.assertEqual(len(rows),4)
                for r in rows:
                    self.assertEqual(r['status'],'completed');self.assertEqual(r['decoded_frames'],10)
                    self.assertEqual(r['inference_updates'],5);self.assertIsNone(r['posture_accuracy'])
                observations=[n for n in z.namelist() if n.endswith('/observations.jsonl')]
                for n in observations:
                    self.assertEqual([json.loads(l)['source_frame_id'] for l in z.read(n).decode().splitlines()],[0,2,4,6,8])

    def test_failed_model_stays_failed_and_other_model_runs(self):
        from posture_benchmark import run_comparison
        class Fake:
            config={}
            def __init__(self,name,*args,**kwargs):
                if name=='yolo11s-pose':raise RuntimeError('intentional unavailable model')
            def infer(self,frame):return {'people':[],'phases_ms':{}}
        folder=HERE/'.tmp';folder.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=folder) as d:
            d=Path(d);image=d/'image.jpg';cv2.imwrite(str(image),np.zeros((120,160,3),np.uint8))
            zpath=run_comparison([str(image)],d,backend='cpu',repeats=1,warmup=0,review_samples=0,engine_factory=Fake,sample_resources=False)
            with zipfile.ZipFile(zpath) as z:
                rows=json.loads(z.read(next(n for n in z.namelist() if n.endswith('/comparison.json'))))
                self.assertEqual([r['status'] for r in rows],['failed','completed'])
                self.assertIn('intentional',rows[0]['error'])

    def test_missing_people_never_become_whole_frame_pose(self):
        from posture_models import pose_for_boxes
        def forbidden(*args,**kwargs):raise AssertionError('Pose must not run without detections')
        points,scores=pose_for_boxes(forbidden,np.zeros((20,20,3),np.uint8),[])
        self.assertEqual(points.shape,(0,17,2));self.assertEqual(scores.shape,(0,17))

    def test_bad_settings_fail_before_model_construction(self):
        from posture_benchmark import run_comparison
        for kwargs in [{'stride':0},{'repeats':0},{'max_source_frames':0},{'models':['unknown']}]:
            with self.assertRaises(ValueError):run_comparison(['missing.mp4'],'.',backend='cpu',**kwargs)

    def test_gpu_request_without_gpu_fails_instead_of_substituting_cpu(self):
        from posture_benchmark import run_comparison
        from unittest.mock import patch
        folder=HERE/'.tmp';folder.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=folder) as d:
            image=Path(d)/'image.jpg';cv2.imwrite(str(image),np.zeros((120,160,3),np.uint8))
            def forbidden(*args,**kwargs):raise AssertionError('Must not initialize a CPU substitute')
            with patch('torch.cuda.is_available',return_value=False):
                with self.assertRaisesRegex(RuntimeError,'GPU'):
                    run_comparison([str(image)],d,backend='cuda',engine_factory=forbidden)

if __name__=='__main__':unittest.main()
