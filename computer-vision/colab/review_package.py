"""Offline frame review: no network requests, no labels inferred from predictions."""
import html,json
from pathlib import Path
import cv2

class ReviewPackage:
    def __init__(self,run,every=10):
        if every<0:raise ValueError('Review interval must be zero (disabled) or positive')
        self.run=Path(run);self.every=every;self.count=0;self.frames=[]

    def add(self,raw,result):
        index=self.count;self.count+=1
        if not self.every or index%self.every:return
        folder=self.run/'review';folder.mkdir(exist_ok=True)
        name=f'frame-{index:05}';annotated=raw.copy()
        for i,a in enumerate(result.get('actions',[])):
            x,y,r,b=map(int,a['box']);cv2.rectangle(annotated,(x,y),(r,b),(0,150,255),2)
            cv2.putText(annotated,str(i+1),(x,max(18,y+18)),0,.6,(0,150,255),2)
        for i,p in enumerate(result.get('head_poses',[])):
            x,y,r,b=map(int,p['face_box']);cv2.rectangle(annotated,(x,y),(r,b),(80,230,100),1)
            text=f"H{i+1} P{p['pitch']:+.0f} Y{p['yaw']:+.0f} R{p['roll']:+.0f}" if p['status']=='estimated' else f'H{i+1} unknown'
            cv2.putText(annotated,text,(x,max(18,y)),0,.5,(80,230,100),1)
        for suffix,image in [('raw',raw),('proposals',annotated)]:
            if not cv2.imwrite(str(folder/f'{name}-{suffix}.jpg'),image):raise RuntimeError('Cannot save review frame')
        self.frames.append(dict(review_frame_id=name,source_frame_id=result.get('source_frame_id'),
            source_seconds=result.get('source_seconds'),actions=result.get('actions',[]),head_poses=result.get('head_poses',[]),
            raw_image=f'{name}-raw.jpg',proposal_image=f'{name}-proposals.jpg'))

    def finish(self):
        if not self.every:return
        folder=self.run/'review';folder.mkdir(exist_ok=True)
        payload=dict(schema_version=1,run_id=self.run.name,frames=self.frames,
                     scope='Sampled diagnostic review, not a complete ground-truth accuracy test')
        (folder/'frames.json').write_text(json.dumps(payload,indent=2),encoding='utf-8')
        cards=[]
        for f in self.frames:
            key=f['review_frame_id'];rows=[]
            for j,a in enumerate(f['actions']):
                cue=html.escape(str(a['cue']))
                rows.append(f'<label>Box {j+1}: {cue}, score {a["score"]:.2f} '+
                    f'<select data-key="{key}:box:{j}"><option value="">Unreviewed</option>'+''.join(f'<option>{s}</option>' for s in ['Correct visible cue','Wrong cue / non-action','Wrong student / merged people','Unclear'])+'</select></label>')
            for j,p in enumerate(f['head_poses']):
                values=f"pitch {p['pitch']:.1f}, yaw {p['yaw']:.1f}, roll {p['roll']:.1f}" if p['status']=='estimated' else 'unknown crop/estimate'
                rows.append(f'<label>Head H{j+1}: {values} degrees <select data-key="{key}:head:{j}"><option value="">Unreviewed</option><option>Estimate plausible</option><option>Wrong face/crop</option><option>Implausible estimate</option><option>Unclear; needs known-motion check</option></select></label>')
            visible=''.join(f'<label>Visible {cue}: <select data-key="{key}:visible:{cue}"><option value="">Unreviewed</option><option>Yes</option><option>No</option><option>Unclear</option></select></label>' for cue in ['handraise','reading','writing'])
            cards.append(f'<section><h2>{key} � source frame {f["source_frame_id"]} � {f["source_seconds"]} s</h2>'+
                f'<div class="images"><figure><img src="{f["raw_image"]}"><figcaption>Raw source</figcaption></figure><figure><img src="{f["proposal_image"]}"><figcaption>Numbered action proposals</figcaption></figure></div>'+visible+
                (''.join(rows) or '<p>No action proposals. Still check for missed visible actions.</p>')+
                f'<label>Missed students, boundaries or uncertainty:<textarea data-key="{key}:notes"></textarea></label></section>')
        safe_payload=json.dumps(payload).replace('<','\\u003c')
        page='''<!doctype html><html><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Attentify review</title>
<style>body{font:16px system-ui;max-width:1100px;margin:auto;padding:18px;background:#141920;color:#eee}header{position:sticky;top:0;background:#141920;padding:12px;border-bottom:1px solid #678}section{padding:16px;margin:18px 0;background:#202936;border-radius:8px}.images{display:flex;flex-wrap:wrap}figure{flex:1;min-width:250px;margin:5px}img{width:100%}label{display:block;margin:12px 0}select,textarea,button{font:inherit;padding:8px}textarea{display:block;width:95%}button{cursor:pointer}</style>
<header><h1>Attentify diagnostic review</h1><button id="export">Export review JSON</button><span id="status"></span></header>
<p>Judge visible activity, not mental attention. Orange boxes are proposals, not ground truth. A head-down pose alone does not establish reading or writing. Screen reading can count when visible evidence supports it; choose Unclear when hidden. Review raw images for missed actions too. Export before switching browsers or locations.</p>
'''+''.join(cards)+'''<script>
const evidence=PAYLOAD, key='attentify-review:'+evidence.run_id;
let answers={};const status=document.querySelector('#status');
try{answers=JSON.parse(localStorage.getItem(key)||'{}');}catch(e){status.textContent=' Storage unavailable; export before closing.';}
document.querySelectorAll('[data-key]').forEach(el=>{
 el.value=answers[el.dataset.key]||'';
 el.oninput=()=>{answers[el.dataset.key]=el.value;try{localStorage.setItem(key,JSON.stringify(answers));status.textContent=' Saved in this browser.';}catch(e){status.textContent=' Not saved in browser: export JSON now.';}};
});
document.querySelector('#export').onclick=()=>{
 const report={schema_version:1,run_id:evidence.run_id,reviewed_at:new Date().toISOString(),evidence,answers};
 const url=URL.createObjectURL(new Blob([JSON.stringify(report,null,2)],{type:'application/json'}));
 const a=document.createElement('a');a.href=url;a.download=evidence.run_id+'-reviews.json';a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);
};
</script></html>'''.replace('PAYLOAD',safe_payload)
        (folder/'index.html').write_text(page,encoding='utf-8')
