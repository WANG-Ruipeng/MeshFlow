"""Exact official-transport reference for the frozen constant-dt path."""
import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG",":4096:8")
from pathlib import Path
import json
import numpy as np
import torch
from torch.nn.attention import SDPBackend,sdpa_kernel
from .official_wrapper import load_official
from .sample import save_raw
from flow_matching import create_transport

def run():
    root=Path(__file__).resolve().parents[2];out=root/"experiments/mf_h1_local0/outputs/run_20260928_01"
    path=out/"sampler_integer_clock_reference.json"
    if path.exists():raise FileExistsError(path)
    torch.set_num_threads(8);torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    base,_=load_official("/mnt/c/Users/warpwang/Downloads/last.pt",root/"configs/snet/base-120m-ot-v-chair.yaml","cuda")
    z=torch.randn((1,512,9),generator=torch.Generator().manual_seed(4101)).cuda()
    y=torch.tensor([512],device="cuda");mask=torch.ones((1,512),device="cuda",dtype=torch.bool)
    count=[0];times=[]
    def integer_clock(x,s,**kwargs):
        count[0]+=1
        t=(s.double()/50).to(x.dtype)
        times.append(float(t[0]))
        return base(x,t,**kwargs).float()*(1/50)
    transport=create_transport(prediction="velocity");transport.method="euler";transport.atol=1e-6;transport.rtol=5e-4
    with torch.no_grad(),sdpa_kernel(SDPBackend.MATH):
        ref=transport.sample(z,integer_clock,t0=0,t1=50,num_steps=51,model_kwargs=dict(y=y,mask=mask))[-1]
    direct=torch.from_numpy(np.load(out/"raw/base_seed4101.npz")["tokens"]).to("cuda")[None]
    equal=bool(torch.equal(ref,direct))
    save_raw(ref[0],out/"raw"/"official_integer_clock_reference_seed4101",z[0],seed=4101)
    report=dict(status="PASS" if equal and count[0]==50 else "FAIL",bitwise_equal=equal,
        max_abs=float((ref-direct).abs().max()),atol=1e-4,rtol=1e-4,nfe=count[0],complete_rollouts=1,
        original_transport_unchanged=True,time_reparameterization="s=50*t; dx/ds=v(x,s/50)/50",
        solver_nodes=list(range(51)),model_times=times,
        model_time_grid="t_k=k/50, k=0..49",model_time_step=1/50,
        original_float32_linspace_difference_is_retained=True,
        note="Independent official torchdiffeq integrator uses integer s to reproduce the exact authorized constant dt arithmetic; no change to the production sampler.")
    path.write_text(json.dumps(report,indent=2),encoding="utf-8")
    print(json.dumps(report),flush=True)
    raise SystemExit(0 if report["status"]=="PASS" else 1)
if __name__=="__main__":run()

