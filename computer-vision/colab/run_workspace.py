"""Local image/video entry point; paths and settings can come from JSON or CLI."""
import argparse,json,sys
from pathlib import Path

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'team-pilot'))

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',type=Path,default=HERE/'configs/default.json')
    p.add_argument('--models',help='Model folder or ZIP path; overrides model_source')
    p.add_argument('--source',required=True,help='Image/video path on this machine')
    p.add_argument('--features',choices=['faces','actions','combined'])
    p.add_argument('--backend',choices=['cpu','cuda'])
    p.add_argument('--results',help='Output directory; overrides results_root')
    a=p.parse_args();config=json.loads(a.config.read_text())
    from workspace_storage import ensure_assets
    from colab_runtime import Engine,process_media
    mode=a.features or config['feature_mode']
    assets=ensure_assets(a.models or config['model_source'],config['asset_cache'],mode)
    e=Engine(assets,a.backend or config['backend'],config['confidence'],config['nms_iou'],mode,config['action_model_id'])
    result=process_media(e,a.source,config['stride'],config['max_processed_frames'],a.results or config['results_root'],config['review_every'])
    print('Results:',result)

if __name__=='__main__':main()
