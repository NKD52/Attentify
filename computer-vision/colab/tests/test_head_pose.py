import sys,tempfile,unittest
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parents[1];sys.path[:0]=[str(HERE),str(HERE.parent/'team-pilot')]
TEMP=HERE/'.tmp';TEMP.mkdir(exist_ok=True)

class GeometryTests(unittest.TestCase):
    def test_euler_order_and_signs_match_rz_ry_rx_convention(self):
        from head_pose import rotation_to_angles
        x,y,z=np.deg2rad([-20,30,15])
        rx=np.array([[1,0,0],[0,np.cos(x),-np.sin(x)],[0,np.sin(x),np.cos(x)]])
        ry=np.array([[np.cos(y),0,np.sin(y)],[0,1,0],[-np.sin(y),0,np.cos(y)]])
        rz=np.array([[np.cos(z),-np.sin(z),0],[np.sin(z),np.cos(z),0],[0,0,1]])
        np.testing.assert_allclose(rotation_to_angles((rz@ry@rx)[None])[0],[-20,30,15],atol=1e-5)

    def test_crop_clipping_and_invalid_boxes(self):
        from head_pose import crop_box
        self.assertEqual(crop_box([-5,2,30,40],(100,100,3),16),[0,2,30,40])
        for box in ([1,2,3,4],[40,40,20,20],[np.nan,2,30,40],[200,2,230,40]):
            self.assertIsNone(crop_box(box,(100,100,3),16))

    def test_crop_coordinates_are_json_serializable_python_integers(self):
        from head_pose import crop_box
        import json
        box=crop_box([5.2,7.9,40.3,50.7],(100,100,3),16)
        self.assertEqual(json.loads(json.dumps(box)),[5,7,41,51])

    def test_head_storage_accepts_a_zip_or_folder_with_registered_assets(self):
        from workspace_storage import ensure_head_assets
        from unittest.mock import patch
        import json,hashlib,zipfile
        with tempfile.TemporaryDirectory(dir=TEMP) as d:
            d=Path(d);source=d/'pack.zip';reg=d/'registry.json'
            spec={'filename':'head.pth','sha256':hashlib.sha256(b'head').hexdigest(),'parts':[]}
            reg.write_text(json.dumps({'head-6drepnet-v1':spec}))
            with zipfile.ZipFile(source,'w') as z:z.writestr('head.pth',b'head')
            with patch('workspace_storage.REGISTER',reg):
                result=ensure_head_assets(source,d/'cache')
            self.assertEqual(result.read_bytes(),b'head')

    def test_missing_or_unexpected_weights_fail_before_model_import(self):
        from head_pose import HeadPose
        with tempfile.TemporaryDirectory(dir=TEMP) as d:
            with self.assertRaises(FileNotFoundError):HeadPose(Path(d)/'missing.pth')

class IntegrationTests(unittest.TestCase):
    def test_both_head_benchmark_cases_initialize_and_export(self):
        from benchmark import Case,run_suite
        import json,cv2,zipfile
        class Fake:
            config={}
            def __init__(self,*args,enable_head=False,**kwargs):
                if not enable_head:raise AssertionError('Head flag missing')
            def reset(self):pass
            def infer(self,frame,seconds):return {'source_seconds':seconds,'faces':[],'actions':[],'head_poses':[],'combined_ms':1.}
        with tempfile.TemporaryDirectory(dir=TEMP) as d:
            d=Path(d);source=d/'clip.mp4';w=cv2.VideoWriter(str(source),cv2.VideoWriter_fourcc(*'mp4v'),10,(160,120));w.write(np.zeros((120,160,3),np.uint8));w.release()
            archive=run_suite([{'path':str(source)}],'.',cases=[Case('faces-head','faces',enable_head=True),Case('combined-head',enable_head=True)],results_root=d/'results',backend='cpu',repeats=1,warmup=0,engine_factory=Fake,sample_resources=False)
            with zipfile.ZipFile(archive) as z:
                rows=json.loads(z.read(next(n for n in z.namelist() if n.endswith('comparison.json'))))
                self.assertTrue(all(r['status']=='completed' for r in rows),rows)
    def test_head_results_follow_the_shared_face_track_and_frame(self):
        from unittest.mock import patch,MagicMock
        from colab_runtime import Engine,Run,EXPECTED_FACE
        class Head:
            config={'model_id':'head-test'}
            def __init__(self,*a,**k):pass
            def predict_faces(self,frame,faces,seconds):return [{'face_box':f['box'],'track_id':f['track_id'],'source_seconds':seconds,'status':'estimated','pitch':0.,'yaw':1.,'roll':2.} for f in faces]
        detector=MagicMock();detector.detect.return_value=(1,np.array([[10,10,40,40]+[0]*10+[.9]],np.float32))
        with tempfile.TemporaryDirectory(dir=TEMP) as d:
            with patch('colab_runtime.digest',return_value=EXPECTED_FACE),patch('colab_runtime.cv2.FaceDetectorYN.create',return_value=detector),patch('head_pose.HeadPose',Head):
                e=Engine(d,'cpu',feature_mode='faces',enable_head=True)
                run=Run(e,'video',root=Path(d)/'results')
                result=e.infer(np.zeros((100,100,3),np.uint8),1.25);result['source_frame_id']=25;run.add(result)
                self.assertEqual(result['head_poses'][0]['track_id'],result['faces'][0]['track_id'])
                self.assertEqual(result['head_poses'][0]['source_frame_id'],25)
                self.assertEqual(result['head_poses'][0]['run_id'],run.path.name)
                self.assertIn('head_pose_ms',result['phases_ms'])

    def test_head_module_stays_uninitialized_when_disabled(self):
        from unittest.mock import patch,MagicMock
        from colab_runtime import Engine,EXPECTED_FACE
        module=MagicMock();module.HeadPose.side_effect=AssertionError('disabled module initialized')
        with patch.dict(sys.modules,{'head_pose':module}),patch('colab_runtime.digest',return_value=EXPECTED_FACE),patch('colab_runtime.cv2.FaceDetectorYN.create'):
            e=Engine('unused','cpu',feature_mode='faces',enable_head=False)
            self.assertIsNone(e.head)

    def test_head_requires_shared_faces_instead_of_running_unassigned(self):
        from colab_runtime import Engine
        with self.assertRaisesRegex(ValueError,'face'):
            Engine('unused','cpu',feature_mode='actions',enable_head=True)

if __name__=='__main__':unittest.main()
