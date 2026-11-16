"""Colab browser-only camera bridge. Importing this module never opens a camera."""
import base64,time
import cv2,numpy as np
from colab_runtime import Run,draw

START_JS="""
window.attentifyStop?.();
window.attentifyStart=async()=>{
 const box=document.createElement('div'),v=document.createElement('video'),stop=document.createElement('button');
 v.autoplay=true;v.muted=true;v.playsInline=true;v.style.maxWidth='640px';stop.textContent='Stop camera';
 box.append(v,stop);document.body.append(box);let stream;
 const cleanup=()=>{window.attentifyStopped=true;stream?.getTracks().forEach(t=>t.stop());box.remove();};
 window.attentifyStop=cleanup;window.attentifyStopped=false;stop.onclick=cleanup;
 try{
  stream=await navigator.mediaDevices.getUserMedia({video:{width:640,height:480},audio:false});
  v.srcObject=stream;await v.play();
  window.attentifyFrame=()=>{
   if(window.attentifyStopped)return null;
   if(!v.videoWidth)throw Error('Camera has no frame');
   const c=document.createElement('canvas');c.width=v.videoWidth;c.height=v.videoHeight;
   c.getContext('2d').drawImage(v,0,0);return c.toDataURL('image/jpeg',0.85);
  };return true;
 }catch(e){cleanup();throw e;}
};
"""

def run_camera(engine,seconds=60,results_root='results',review_every=0):
    """Camera images stay out of exports unless review_every is explicitly positive."""
    if seconds<=0:raise ValueError('Camera time limit must be positive')
    from IPython.display import display,Image,Javascript
    from google.colab.output import eval_js
    display(Javascript(START_JS))
    run=Run(engine,'browser_camera',dict(seconds=seconds,frame_time='Python request time, not exposure time'),root=results_root,review_every=review_every)
    view=display('Starting camera...',display_id=True);error=None
    try:
        eval_js('attentifyStart()');start=time.perf_counter()
        while time.perf_counter()-start<seconds:
            requested=time.perf_counter();encoded=eval_js('attentifyFrame()')
            if encoded is None:break
            frame=cv2.imdecode(np.frombuffer(base64.b64decode(encoded.split(',')[1]),np.uint8),cv2.IMREAD_COLOR)
            if frame is None:raise ValueError('Invalid camera frame')
            result=engine.infer(frame,requested-start)
            result.update(source_frame_id=len(run.rows),request_cycle_ms=(time.perf_counter()-requested)*1000,
                          source_time_note='Python request time; not camera exposure time')
            run.add(result,frame)
            image=draw(frame,result)
            cv2.putText(image,f"Processing {result['combined_ms']:.0f} ms | request cycle {result['request_cycle_ms']:.0f} ms",(10,25),0,.5,(0,255,255),1)
            ok,jpeg=cv2.imencode('.jpg',image)
            if ok:view.update(Image(data=jpeg.tobytes()))
    except Exception as exc:error=str(exc);print('Camera stopped:',error)
    finally:
        try:eval_js('window.attentifyStop && window.attentifyStop()')
        except Exception:pass
        archive=run.finish(error)
    return archive
