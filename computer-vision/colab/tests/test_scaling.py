import sys,unittest,tempfile
from pathlib import Path
import numpy as np

HERE=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(HERE),str(HERE.parent/'team-pilot')]
TEMP=HERE/'.tmp';TEMP.mkdir(exist_ok=True)

class SizeTests(unittest.TestCase):
    def test_size_change_keeps_boxes_in_original_image_coordinates(self):
        from core import prepare,decode
        frame=np.zeros((1080,1920,3),np.uint8)
        for size in (320,640,960,1280):
            tensor,meta=prepare(frame,long_edge=size)
            self.assertEqual(tensor.shape[3],size)
            x1,y1,x2,y2=100,200,300,500
            gain=meta['gain'];px,py=meta['pad']
            raw=np.array([[[(x1+x2)/2*gain+px,(y1+y2)/2*gain+py,(x2-x1)*gain,(y2-y1)*gain,.9,.9,.1,.1]]],np.float32)
            np.testing.assert_allclose(decode(raw,meta)[0]['box'],[x1,y1,x2,y2],atol=.001)

    def test_sizes_must_be_positive_multiples_of_32(self):
        from core import prepare
        for size in (0,-32,650,640.5):
            with self.assertRaises(ValueError):prepare(np.zeros((100,100,3),np.uint8),long_edge=size)

class BenchmarkTests(unittest.TestCase):
    def test_matrix_isolates_sizes_and_has_output_control(self):
        from benchmark import make_cases
        cases=make_cases(action_sizes=(640,960),face_widths=(640,960),output_widths=(960,))
        settings={(c.mode,c.action_size,c.face_width,c.output_width) for c in cases}
        self.assertTrue({('faces',640,640,0),('actions',640,640,0),('combined',640,640,0),('combined',960,640,0),('combined',640,960,0),('combined',640,640,960)}.issubset(settings))
        self.assertEqual(len({c.name for c in cases}),len(cases))

    def test_preview_uses_current_raw_and_a_dated_snapshot(self):
        from benchmark import preview_frame
        current=np.zeros((120,160,3),np.uint8);current[:,:,2]=200
        snapshot=np.zeros_like(current);snapshot[:,:,1]=180
        result={'source_seconds':0,'faces':[],'actions':[]}
        out=preview_frame(current,snapshot,result,.4,320)
        self.assertEqual(out.shape[1],320)
        self.assertEqual(out.shape[0]%2,0)
        self.assertGreater(out[80,40,2],100)
        self.assertGreater(out[80,240,1],100)
        self.assertEqual(result['source_seconds'],0)

    def test_suite_preserves_source_fps_and_never_calls_face_counts_ground_truth(self):
        from benchmark import Case,run_suite
        import cv2,json,zipfile
        class FakeEngine:
            def __init__(self,*args,**kwargs):self.config={'feature_mode':'combined'}
            def reset(self):pass
            def infer(self,frame,seconds):return {'source_seconds':seconds,'faces':[{'box':[10,10,40,40],'track_id':1}],'actions':[],'combined_ms':1.,'phases_ms':{'yunet_tracking_ms':1.}}
        with tempfile.TemporaryDirectory(dir=TEMP) as d:
            d=Path(d);source=d/'clip.mp4'
            w=cv2.VideoWriter(str(source),cv2.VideoWriter_fourcc(*'mp4v'),10,(160,120))
            for _ in range(10):w.write(np.zeros((120,160,3),np.uint8))
            w.release()
            archive=run_suite([{'path':str(source),'visible_students':6}],'.',cases=[Case('preview','combined',640,640,320)],backend='cpu',results_root=d/'results',stride=2,max_source_frames=10,repeats=1,warmup=0,engine_factory=FakeEngine,sample_resources=False)
            with zipfile.ZipFile(archive) as z:
                summaries=json.loads(z.read(next(n for n in z.namelist() if n.endswith('comparison.json'))))
                self.assertEqual(summaries[0]['decoded_frames'],10)
                self.assertEqual(summaries[0]['inference_updates'],5)
                self.assertEqual(summaries[0]['visible_students'],6)
                self.assertEqual(summaries[0]['mean_detected_faces'],1)
                name=next(n for n in z.namelist() if n.endswith('preview.mp4'));z.extract(name,d)
                cap=cv2.VideoCapture(str(d/name))
                self.assertEqual(round(cap.get(cv2.CAP_PROP_FRAME_COUNT)),10)
                self.assertAlmostEqual(cap.get(cv2.CAP_PROP_FPS),10,places=2);cap.release()

    def test_initialization_failure_is_a_failed_case_not_a_successful_empty_test(self):
        from benchmark import Case,run_suite
        import cv2,json,zipfile
        def broken(*a,**k):raise RuntimeError('simulated unavailable provider')
        with tempfile.TemporaryDirectory(dir=TEMP) as d:
            d=Path(d);source=d/'clip.mp4';w=cv2.VideoWriter(str(source),cv2.VideoWriter_fourcc(*'mp4v'),10,(160,120));w.write(np.zeros((120,160,3),np.uint8));w.release()
            archive=run_suite([{'path':str(source)}],'.',cases=[Case('failure','combined')],backend='cpu',results_root=d/'results',repeats=1,engine_factory=broken,sample_resources=False)
            with zipfile.ZipFile(archive) as z:
                result=json.loads(z.read(next(n for n in z.namelist() if n.endswith('comparison.json'))))[0]
                self.assertEqual(result['status'],'failed')
                self.assertIn('unavailable provider',result['error'])

    def test_a_failed_case_does_not_prevent_later_results_or_csv_export(self):
        from benchmark import Case,run_suite
        import cv2,json,zipfile
        class Sometimes:
            def __init__(self,*a,**k):
                if k['feature_mode']=='faces':raise RuntimeError('failed case')
                self.config={}
            def reset(self):pass
            def infer(self,frame,seconds):return {'source_seconds':seconds,'faces':[],'actions':[],'combined_ms':1.}
        with tempfile.TemporaryDirectory(dir=TEMP) as d:
            d=Path(d);source=d/'clip.mp4';w=cv2.VideoWriter(str(source),cv2.VideoWriter_fourcc(*'mp4v'),10,(160,120));w.write(np.zeros((120,160,3),np.uint8));w.release()
            archive=run_suite([{'path':str(source)}],'.',cases=[Case('bad','faces'),Case('good','actions')],results_root=d/'results',backend='cpu',repeats=1,warmup=0,engine_factory=Sometimes,sample_resources=False)
            with zipfile.ZipFile(archive) as z:
                rows=json.loads(z.read(next(n for n in z.namelist() if n.endswith('comparison.json'))))
                self.assertEqual([r['status'] for r in rows],['failed','completed'])
                self.assertTrue(any(n.endswith('comparison.csv') for n in z.namelist()))

    def test_review_failure_keeps_timing_results_and_later_trials(self):
        from benchmark import Case,run_suite
        from unittest.mock import patch
        import cv2,json,zipfile
        class Fake:
            def __init__(self,*a,**k):self.config={}
            def reset(self):pass
            def infer(self,frame,seconds):return {'source_seconds':seconds,'faces':[],'actions':[],'combined_ms':1.}
        with tempfile.TemporaryDirectory(dir=TEMP) as d:
            d=Path(d);source=d/'clip.mp4';w=cv2.VideoWriter(str(source),cv2.VideoWriter_fourcc(*'mp4v'),10,(160,120));w.write(np.zeros((120,160,3),np.uint8));w.release()
            with patch('benchmark.deferred_review',side_effect=RuntimeError('Cannot save review frame')):
                archive=run_suite([{'path':str(source)}],'.',cases=[Case('one'),Case('two')],results_root=d/'results',backend='cpu',repeats=1,warmup=0,engine_factory=Fake,sample_resources=False,review_every=1)
            with zipfile.ZipFile(archive) as z:
                rows=json.loads(z.read(next(n for n in z.namelist() if n.endswith('comparison.json'))))
                self.assertEqual(len(rows),2)
                self.assertTrue(all(r['status']=='completed' and r['review_error']=='Cannot save review frame' for r in rows))

class ResourceTests(unittest.TestCase):
    def test_gpu_memory_stays_unknown_for_unmatched_or_unavailable_process(self):
        from benchmark_resources import Resources
        from unittest.mock import patch
        from types import SimpleNamespace
        import os
        for stdout in (f'{os.getpid()+1000000}, 120\n','',f'{os.getpid()}, [N/A]\n'):
            r=Resources();r.smi='nvidia-smi'
            with patch('benchmark_resources.subprocess.run',return_value=SimpleNamespace(stdout=stdout)):r.sample(gpu=True)
            self.assertIsNone(r.gpu)

    def test_gpu_memory_records_matching_numeric_pid(self):
        from benchmark_resources import Resources
        from unittest.mock import patch
        from types import SimpleNamespace
        import os
        r=Resources();r.smi='nvidia-smi'
        with patch('benchmark_resources.subprocess.run',return_value=SimpleNamespace(stdout=f'{os.getpid()}, 123\n')):r.sample(gpu=True)
        self.assertEqual(r.gpu,123)

class NotebookTests(unittest.TestCase):
    def test_refresh_evicts_benchmark_modules_before_reimport(self):
        import ast,json
        notebook=json.loads((HERE/'Attentify_Scaling_Benchmark.ipynb').read_text())
        setup=next(''.join(c['source']) for c in notebook['cells'] if c['cell_type']=='code')
        names=[]
        for node in ast.walk(ast.parse(setup)):
            if isinstance(node,ast.For) and isinstance(node.iter,ast.List):
                try:names.extend(ast.literal_eval(node.iter))
                except (ValueError,TypeError):pass
        self.assertTrue({'benchmark','benchmark_resources'}.issubset(names))

if __name__=='__main__':unittest.main()
