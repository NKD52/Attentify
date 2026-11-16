import json, sys, tempfile, unittest, zipfile
from pathlib import Path
import cv2
import numpy as np

HERE=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(HERE),str(HERE.parent/'team-pilot')]

class ProfilingTests(unittest.TestCase):
    def test_provider_totals_only_count_kernel_events(self):
        from profile_pipeline import provider_summary
        events=[{'cat':'Node','name':'conv_kernel_time','dur':20,'args':{'provider':'CUDAExecutionProvider'}},
                {'cat':'Node','name':'shape_kernel_time','dur':5,'args':{'provider':'CPUExecutionProvider'}},
                {'cat':'Session','name':'model_run','dur':50},
                {'cat':'Node','name':'conv_fence_before','dur':99,'args':{'provider':'CUDAExecutionProvider'}}]
        self.assertEqual(provider_summary(events),{'CUDAExecutionProvider':{'events':1,'duration_us':20.0},'CPUExecutionProvider':{'events':1,'duration_us':5.0}})

    def test_profile_is_bounded_exports_no_media_and_keeps_baseline_separate(self):
        from profile_pipeline import run_profile
        class Fake:
            config={'fake':True}
            session=head=None
            def __init__(self,*args,**kwargs):pass
            def reset(self):pass
            def infer(self,frame,seconds):
                return dict(source_seconds=seconds,faces=[],actions=[],head_poses=[],combined_ms=1.,phases_ms={})
        tmp=HERE/'.tmp';tmp.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=tmp) as d:
            d=Path(d);clip=d/'clip.mp4'
            w=cv2.VideoWriter(str(clip),cv2.VideoWriter_fourcc(*'mp4v'),25,(160,120))
            for _ in range(12):w.write(np.zeros((120,160,3),np.uint8))
            w.release()
            result=run_profile({'path':str(clip)},'.',backend='cpu',enable_head=False,
                               max_source_frames=10,stride=2,results_root=d,engine_factory=Fake)
            with zipfile.ZipFile(result) as z:
                names=z.namelist();report=json.loads(z.read(next(n for n in names if n.endswith('/profile-summary.json'))))
                self.assertEqual(report['profiled_inference_updates'],5)
                self.assertEqual(report['baseline']['decoded_frames'],10)
                self.assertEqual(report['baseline']['inference_updates'],5)
                self.assertTrue(any(n.endswith('/torch-trace.json') for n in names))
                self.assertFalse(any(n.endswith(('.jpg','.png','.mp4','.pth','.onnx')) for n in names))
                self.assertIn('overhead',report['limits'])

    def test_rejects_unbounded_or_invalid_runs_before_loading_models(self):
        from profile_pipeline import run_profile
        for frames,stride in [(0,5),(301,5),(10,0)]:
            with self.assertRaises(ValueError):run_profile({'path':'missing.mp4'},'.',max_source_frames=frames,stride=stride)

if __name__=='__main__':unittest.main()
