"""Optional sampled process resources; never substitutes PyTorch allocator stats for ORT VRAM."""
import os,shutil,subprocess,threading,time,math

class Resources:
    def __init__(self,enabled=True):
        self.enabled=enabled;self.stop_event=threading.Event();self.thread=None
        self.rss=None;self.gpu=None;self.smi=shutil.which('nvidia-smi') if enabled else None
        try:
            import psutil
            self.process=psutil.Process() if enabled else None
        except ImportError:self.process=None

    def sample(self,gpu=False):
        if self.process is not None:
            try:self.rss=max(self.rss or 0,self.process.memory_info().rss)
            except Exception:pass
        if gpu and self.smi:
            try:
                run=subprocess.run([self.smi,'--query-compute-apps=pid,used_gpu_memory','--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=2,check=True)
                matches=[]
                for line in run.stdout.splitlines():
                    try:
                        pid,memory=[s.strip() for s in line.split(',')]
                        if int(pid)!=os.getpid():continue
                        value=float(memory)
                        if math.isfinite(value) and value>=0:matches.append(value)
                    except ValueError:continue
                if matches:self.gpu=max(self.gpu or 0.,sum(matches))
            except (ValueError,OSError,subprocess.SubprocessError):pass

    def start(self):
        if not self.enabled:return
        def watch():
            next_gpu=0
            while not self.stop_event.is_set():
                now=time.perf_counter();query=now>=next_gpu
                self.sample(query)
                if query:next_gpu=now+1
                self.stop_event.wait(.1)
        self.thread=threading.Thread(target=watch,daemon=True);self.thread.start()

    def stop(self):
        self.stop_event.set()
        if self.thread:self.thread.join(timeout=3)
        if self.enabled:self.sample(gpu=True)
        return {'sampled_max_process_rss_bytes':self.rss,'sampled_max_process_gpu_memory_mib':self.gpu,
                'memory_scope':'Whole Python process during initialization, warmup and timed loop; sampled maxima, not exact peaks or model-exclusive memory',
                'resource_sampling_enabled':self.enabled}
