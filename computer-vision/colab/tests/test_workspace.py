import hashlib,json,sys,tempfile,unittest,zipfile
from pathlib import Path

HERE=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(HERE),str(HERE.parent/'team-pilot')]
TEMP=HERE/'.tmp';TEMP.mkdir(exist_ok=True)

class AssetTests(unittest.TestCase):
    def test_drive_folder_of_pack_parts_is_reassembled_automatically(self):
        import workspace_storage as s
        with tempfile.TemporaryDirectory(dir=TEMP) as d:
            d=Path(d);source=d/'persistent';source.mkdir();pack=d/'pack.zip'
            with zipfile.ZipFile(pack,'w') as z:z.writestr('yunet.onnx',b'face')
            data=pack.read_bytes();chunks=[data[:30],data[30:]];parts=[]
            for i,chunk in enumerate(chunks):
                name=f'pack.zip.part{i:02}';(source/name).write_bytes(chunk)
                parts.append({'name':name,'sha256':hashlib.sha256(chunk).hexdigest()})
            register={'yunet-fp32':{'filename':'yunet.onnx','sha256':hashlib.sha256(b'face').hexdigest()},'asset_pack':{'filename':'pack.zip','sha256':hashlib.sha256(data).hexdigest(),'parts':parts}}
            result=s.ensure_assets(source,d/'cache','faces',register=register)
            self.assertEqual((result/'yunet.onnx').read_bytes(),b'face')

    def test_face_only_resolves_a_persistent_folder_without_action_weights(self):
        import workspace_storage as s
        with tempfile.TemporaryDirectory(dir=TEMP) as d:
            d=Path(d); source=d/'persistent';source.mkdir()
            (source/'yunet.onnx').write_bytes(b'face')
            register={'yunet-fp32':{'filename':'yunet.onnx','sha256':hashlib.sha256(b'face').hexdigest()}}
            result=s.ensure_assets(source,d/'cache','faces',register=register)
            self.assertEqual((result/'yunet.onnx').read_bytes(),b'face')
            self.assertFalse((result/'scb-refined.onnx').exists())

    def test_flat_zip_accepts_only_matching_registered_assets(self):
        import workspace_storage as s
        with tempfile.TemporaryDirectory(dir=TEMP) as d:
            d=Path(d);zpath=d/'pack.zip'
            with zipfile.ZipFile(zpath,'w') as z:
                z.writestr('yunet.onnx',b'face');z.writestr('../escape.txt','do not extract')
            register={'yunet-fp32':{'filename':'yunet.onnx','sha256':hashlib.sha256(b'face').hexdigest()}}
            s.ensure_assets(zpath,d/'cache','faces',register=register)
            self.assertFalse((d/'escape.txt').exists())

    def test_wrong_hash_rejects_weights_and_keeps_previous_cache(self):
        import workspace_storage as s
        with tempfile.TemporaryDirectory(dir=TEMP) as d:
            d=Path(d); source=d/'persistent';source.mkdir();cache=d/'cache';cache.mkdir()
            (source/'yunet.onnx').write_bytes(b'wrong');(cache/'yunet.onnx').write_bytes(b'old')
            register={'yunet-fp32':{'filename':'yunet.onnx','sha256':hashlib.sha256(b'face').hexdigest()}}
            with self.assertRaises(ValueError): s.ensure_assets(source,cache,'faces',register=register)
            self.assertEqual((cache/'yunet.onnx').read_bytes(),b'old')

    def test_duplicate_archive_members_are_rejected(self):
        import workspace_storage as s
        with tempfile.TemporaryDirectory(dir=TEMP) as d:
            d=Path(d);pack=d/'pack.zip'
            with zipfile.ZipFile(pack,'w') as z:
                z.writestr('yunet.onnx',b'face');z.writestr('another/assets/yunet.onnx',b'face')
            register={'yunet-fp32':{'filename':'yunet.onnx','sha256':hashlib.sha256(b'face').hexdigest()}}
            with self.assertRaises(ValueError):s.ensure_assets(pack,d/'cache','faces',register=register)

class FeatureTests(unittest.TestCase):
    def test_faces_mode_never_constructs_an_action_session(self):
        from unittest.mock import patch
        from colab_runtime import Engine
        with tempfile.TemporaryDirectory(dir=TEMP) as d:
            with patch('colab_runtime.digest',return_value='8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4'),patch('colab_runtime.cv2.FaceDetectorYN.create'),patch('onnxruntime.InferenceSession',side_effect=AssertionError('Action session must stay disabled')):
                e=Engine(d,'cuda',feature_mode='faces')
                self.assertIsNone(e.session)
                self.assertEqual(e.config['model_ids'],['yunet-fp32'])

    def test_actions_mode_has_no_face_detector_and_leaves_ownership_unassigned(self):
        from unittest.mock import patch,MagicMock
        import numpy as np
        from colab_runtime import Engine,EXPECTED_ACTION
        session=MagicMock();session.run.return_value=[np.array([[[100,100,60,80,.9,.9,.1,.1]]],np.float32)]
        with patch('colab_runtime.digest',return_value=EXPECTED_ACTION),patch('colab_runtime.cv2.FaceDetectorYN.create',side_effect=AssertionError('Face detector disabled')),patch('onnxruntime.InferenceSession',return_value=session):
            e=Engine('unused','cpu',feature_mode='actions');r=e.infer(np.zeros((480,640,3),np.uint8),0)
            self.assertEqual(r['faces'],[])
            self.assertIsNone(r['actions'][0]['track_id'])
            self.assertEqual(r['actions'][0]['association_status'],'unassigned')

    def test_faces_mode_does_not_probe_torch_cuda_or_configure_action_runtime(self):
        from unittest.mock import MagicMock,patch
        from colab_runtime import Engine,EXPECTED_FACE
        torch=MagicMock();torch.cuda.is_available.side_effect=AssertionError('Face mode must not probe CUDA')
        with patch.dict(sys.modules,{'torch':torch}),patch('colab_runtime.digest',return_value=EXPECTED_FACE),patch('colab_runtime.cv2.FaceDetectorYN.create'),patch('onnxruntime.disable_telemetry_events',side_effect=AssertionError('Unused action runtime configured')):
            e=Engine('unused','cuda',feature_mode='faces')
            self.assertIsNone(e.config['ort'])

    def test_failed_source_exports_diagnostics_instead_of_succeeding_empty(self):
        from unittest.mock import MagicMock
        from colab_runtime import process_media
        with tempfile.TemporaryDirectory(dir=TEMP) as d:
            d=Path(d);engine=MagicMock();engine.config={'feature_mode':'faces'}
            with self.assertRaisesRegex(RuntimeError,'diagnostic ZIP'):
                process_media(engine,d/'missing.mp4',results_root=d/'results',review_every=0)
            zpath=next((d/'results').glob('*.zip'))
            with zipfile.ZipFile(zpath) as z:
                summary=json.loads(z.read(next(n for n in z.namelist() if n.endswith('summary.json'))))
                self.assertEqual(summary['processed_frames'],0)
                self.assertTrue(summary['error'])

    def test_camera_run_fingerprints_the_camera_bridge(self):
        from unittest.mock import MagicMock
        from colab_runtime import Run
        with tempfile.TemporaryDirectory(dir=TEMP) as d:
            engine=MagicMock();engine.config={}
            run=Run(engine,'browser_camera',root=d)
            config=json.loads((run.path/'config.json').read_text())
            expected=hashlib.sha256((HERE/'browser_camera.py').read_bytes()).hexdigest()
            self.assertEqual(config['code_sha256']['browser_camera.py'],expected)

class ReviewTests(unittest.TestCase):
    def test_empty_detection_frame_still_has_reviewable_evidence(self):
        from review_package import ReviewPackage
        import numpy as np
        with tempfile.TemporaryDirectory(dir=TEMP) as d:
            run=Path(d)/'run-example';run.mkdir()
            review=ReviewPackage(run,every=1)
            review.add(np.zeros((40,60,3),np.uint8),{'source_frame_id':12,'source_seconds':1.2,'actions':[],'faces':[]})
            review.finish()
            records=json.loads((run/'review/frames.json').read_text())
            self.assertEqual(records['run_id'],'run-example')
            self.assertEqual(records['frames'][0]['source_frame_id'],12)
            self.assertTrue((run/'review'/records['frames'][0]['raw_image']).is_file())
            self.assertIn('No action proposals',(run/'review/index.html').read_text())

    def test_disabled_reviews_write_no_identifiable_images(self):
        from review_package import ReviewPackage
        import numpy as np
        with tempfile.TemporaryDirectory(dir=TEMP) as d:
            run=Path(d);review=ReviewPackage(run,every=0)
            review.add(np.zeros((40,60,3),np.uint8),{'actions':[],'faces':[]})
            review.finish()
            self.assertFalse((run/'review').exists())

    def test_review_page_does_not_execute_a_cue_string(self):
        from review_package import ReviewPackage
        import numpy as np
        with tempfile.TemporaryDirectory(dir=TEMP) as d:
            run=Path(d);review=ReviewPackage(run,every=1)
            review.add(np.zeros((40,60,3),np.uint8),{'source_frame_id':0,'source_seconds':0,'actions':[{'cue':'</script><script>alert(1)</script>','score':.8,'box':[0,0,10,10]}],'faces':[]})
            review.finish()
            page=(run/'review/index.html').read_text()
            self.assertNotIn('</script><script>alert(1)</script>',page)

if __name__=='__main__':unittest.main()
