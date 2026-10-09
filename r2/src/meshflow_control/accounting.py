"""Durable reserve-before-call budget; one GPU owner only."""
from pathlib import Path
import json,os,time,subprocess
LIMITS=dict(train_forward=24000,train_backward=24000,optimizer_update=3000,
    preflight_forward=64,preflight_backward=24,rollout_attempt=320,sampling_forward=16000)
class Budget:
    def __init__(self,root):
        self.root=Path(root);self.root.mkdir(parents=True,exist_ok=True)
        self.path=self.root/"calls.jsonl";self.counts={k:0 for k in LIMITS};self.by_arm={}
        if self.path.exists():
            for line in self.path.read_text().splitlines():
                row=json.loads(line)
                if row["event"]=="RESERVE":
                    k=row["kind"];self.counts[k]+=1
                    if k=="optimizer_update":
                        arm=row.get("context",{}).get("arm","unknown");self.by_arm[arm]=self.by_arm.get(arm,0)+1
    def event(self,event,**fields):
        row=dict(time_unix=time.time(),event=event,**fields)
        with self.path.open("a",encoding="utf-8") as f:
            f.write(json.dumps(row,allow_nan=False)+"\n");f.flush();os.fsync(f.fileno())
        return row
    def reserve(self,kind,context=None):
        from .runtime import check_gpu_deadline
        check_gpu_deadline()
        if kind not in LIMITS: raise ValueError("Unknown GPU budget category "+kind)
        if self.counts[kind]>=LIMITS[kind]: raise RuntimeError("GPU budget exhausted: "+kind)
        ctx=context or {}
        if kind=="optimizer_update" and self.by_arm.get(ctx.get("arm","unknown"),0)>=1000:
            raise RuntimeError("Per-arm update budget exhausted")
        self.event("RESERVE",kind=kind,context=ctx,ordinal=self.counts[kind]+1);self.counts[kind]+=1
        if kind=="optimizer_update":
            arm=ctx.get("arm","unknown");self.by_arm[arm]=self.by_arm.get(arm,0)+1
    def call(self,kind,fn,context=None):
        self.reserve(kind,context);tick=time.perf_counter()
        try: result=fn()
        except BaseException as exc:
            self.event("ERROR",kind=kind,context=context or {},error=repr(exc));raise
        self.event("RETURN",kind=kind,context=context or {},seconds=time.perf_counter()-tick)
        return result
    def snapshot(self):
        return dict(counts=dict(self.counts),limits=LIMITS,updates_by_arm=dict(self.by_arm))
def resource_snapshot():
    def run(args):
        p=subprocess.run(args,text=True,capture_output=True);return dict(returncode=p.returncode,stdout=p.stdout,stderr=p.stderr)
    mem={}
    if Path("/proc/meminfo").exists():
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.split(":")[0] in ("MemTotal","MemAvailable"): mem[line.split(":")[0]]=line.split(":")[1].strip()
    return dict(time_unix=time.time(),cpu_cores=os.cpu_count(),memory=mem,
        gpu=run(["nvidia-smi","--query-gpu=name,memory.total,memory.used,memory.free,utilization.gpu","--format=csv"]),
        gpu_processes=run(["nvidia-smi","--query-compute-apps=pid,process_name,used_memory","--format=csv"]),
        cpu_load=run(["ps","-eo","pid,pcpu,pmem,args","--sort=-pcpu"]))
