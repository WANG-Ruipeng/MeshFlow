"""Diagnose sampler arithmetic without changing the frozen production grid."""
import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG",":4096:8")
import json,time,resource
from pathlib import Path
import torch
from torch.nn.attention import SDPBackend,sdpa_kernel
from .official_wrapper import load_official
from .sample import save_raw

def run():
    root=Path(__file__).resolve().parents[2];out=root/"experiments/mf_h1_local0/outputs/run_20260928_01"
    destination=out/"sampler_diagnosis.json"
    if destination.exists():raise FileExistsError(destination)
    torch.set_num_threads(8);torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    base,_=load_official("/mnt/c/Users/warpwang/Downloads/last.pt",root/"configs/snet/base-120m-ot-v-chair.yaml","cuda")
    counts={"actual_backbone_forwards":0,"complete_rollouts":0}
    def count(m,a):counts["actual_backbone_forwards"]+=1
    base.register_forward_pre_hook(count)
    z=torch.randn((1,512,9),generator=torch.Generator().manual_seed(4101)).cuda()
    y=torch.tensor([512],device="cuda");mask=torch.ones((1,512),device="cuda",dtype=torch.bool)
    grid=torch.linspace(0.,1.,51).to(z)
    directgrid=torch.tensor([k/50 for k in range(51)],device="cuda")
    report=dict(status="RUNNING",purpose="reference implementation parity at exactly equal floating-point inputs; frozen constant-dt protocol is unchanged",
        grid_max_abs=float((grid-directgrid).abs().max()),
        grid_differing_nodes=(grid!=directgrid).nonzero().flatten().tolist(),
        dt_max_abs_from_constant=float(((grid[1:]-grid[:-1])-1/50).abs().max()),
        threshold=dict(atol=1e-4,rtol=1e-4),counts=counts)
    from flow_matching import create_transport
    fm=create_transport(prediction="velocity");fm.method="euler";fm.atol=1e-6;fm.rtol=5e-4
    start=time.perf_counter()
    with torch.no_grad(),sdpa_kernel(SDPBackend.MATH):
        captured=[]
        def traced(x,t,**kw):
            result=base(x,t,**kw)
            captured.append((x.clone(),t.clone(),result.clone()))
            return result
        ref=fm.sample(z,traced,num_steps=51,model_kwargs=dict(y=y,mask=mask))[-1];counts["complete_rollouts"]+=1
        x=z.clone();steps=[]
        for k in range(50):
            t=torch.full((1,),float(grid[k]),device="cuda")
            v=base(x,t,y,mask)
            xr,tr,vr=captured[k]
            steps.append(dict(k=k,t=float(t.item()),dt=float((grid[k+1]-grid[k]).item()),
                input_max_abs=float((x-xr).abs().max()),velocity_max_abs=float((v-vr).abs().max())))
            x=x+(grid[k+1]-grid[k])*v
        counts["complete_rollouts"]+=1
    save_raw(ref[0],out/"raw"/"diagnostic_reference_seed4101",z[0],seed=4101)
    save_raw(x[0],out/"raw"/"diagnostic_matched_grid_seed4101",z[0],seed=4101)
    report.update(status="PASS" if torch.equal(x,ref) and all(s["input_max_abs"]==0 and s["velocity_max_abs"]==0 for s in steps) else "FAIL",
        reference_actual_nfe=len(captured),matched_grid_bitwise_equal=bool(torch.equal(x,ref)),
        matched_grid_max_abs=float((x-ref).abs().max()),steps=steps,seconds=time.perf_counter()-start,
        cpu_peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
        conclusion="The explicit constant 1/50 path is separately preserved; equality claim applies only to identical floating-point nodal dt and model times.")
    destination.write_text(json.dumps(report,indent=2),encoding="utf-8")
    print(json.dumps({k:v for k,v in report.items() if k!="steps"}),flush=True)
    raise SystemExit(0 if report["status"]=="PASS" else 1)
if __name__=="__main__":run()

