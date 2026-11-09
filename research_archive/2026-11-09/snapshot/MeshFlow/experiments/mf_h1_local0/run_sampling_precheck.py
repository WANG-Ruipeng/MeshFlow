"""Phase 0 full fixed-noise generation and sampler reconciliation."""
import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG",":4096:8")
import argparse,json,time,resource,traceback
from pathlib import Path
import torch
from torch.nn.attention import SDPBackend,sdpa_kernel
from .official_wrapper import load_official,OfficialWrapper,state_sha256
from .adapters import ControlAdapter
from .sample import euler_sample,save_raw
from .prompt_dataset import file_sha256

def render_grid(tokens,path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    import numpy as np
    fig=plt.figure(figsize=(10,10))
    for i,t in enumerate(tokens):
        ax=fig.add_subplot(2,2,i+1,projection="3d")
        tri=t.detach().cpu().numpy().reshape(-1,3,3)
        poly=Poly3DCollection(tri,facecolor="#90a8c5",edgecolor="#30445a",linewidth=.12,alpha=1)
        ax.add_collection3d(poly)
        low,high=tri.min((0,1)),tri.max((0,1))
        center=(low+high)/2;radius=max((high-low).max()/2,1e-3)
        ax.set_xlim(center[0]-radius,center[0]+radius);ax.set_ylim(center[1]-radius,center[1]+radius);ax.set_zlim(center[2]-radius,center[2]+radius)
        ax.set_box_aspect((1,1,1));ax.view_init(elev=18,azim=-60,vertical_axis="y")
        ax.set_title(f"Official chair EMA | seed {4101+i} | raw 512")
        ax.set_axis_off()
    fig.tight_layout();path.parent.mkdir(exist_ok=True,parents=True);fig.savefig(path,dpi=160);plt.close(fig)

def run(checkpoint,output):
    output=Path(output);dest=output/"sampling_precheck.json"
    if dest.exists():raise FileExistsError(dest)
    pre=json.loads((output/"precheck_execution.json").read_text())
    if pre["status"]!="PASS":raise RuntimeError("Official precheck gate failed")
    report=dict(status="RUNNING",actual_backbone_forwards=0,actual_backbone_examples=0,
                complete_rollouts=0,complete_sampler_calls=0,phase2_updates=0,phase3="NOT_AUTHORIZED",runs=[])
    start=time.perf_counter();wrapper=None
    def write():
        report["cpu_peak_rss_bytes"]=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024
        dest.write_text(json.dumps(report,indent=2),encoding="utf-8")
    try:
        torch.set_num_threads(8)
        if file_sha256(checkpoint)!=pre["checkpoint"]["sha256"]:raise ValueError("Checkpoint changed")
        base,audit=load_official(checkpoint,Path(__file__).resolve().parents[2]/"configs/snet/base-120m-ot-v-chair.yaml","cuda")
        def count(module,args):
            report["actual_backbone_forwards"]+=1;report["actual_backbone_examples"]+=len(args[0])
        base.register_forward_pre_hook(count)
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        torch.use_deterministic_algorithms(True)
        y=torch.tensor([512],device="cuda");mask=torch.ones((1,512),device="cuda",dtype=torch.bool)
        outputs=[]
        with sdpa_kernel(SDPBackend.MATH):
            for seed in (4101,4102,4103,4104):
                z=torch.randn((1,512,9),generator=torch.Generator().manual_seed(seed)).cuda()
                if seed==4101:
                    with torch.no_grad():base(z[:,:16].contiguous(),torch.zeros(1,device="cuda"),y,mask[:,:16])
                result,stats=euler_sample(base,z,y,mask,precision="fp32")
                outputs.append(result[0].cpu())
                save_raw(result[0],output/"raw"/f"base_seed{seed}",z[0],seed=seed)
                report["runs"].append(dict(label=f"base_seed{seed}",**stats))
                report["complete_rollouts"]+=1;report["complete_sampler_calls"]+=1;write()
                print(json.dumps(dict(stage="baseline",seed=seed,**stats)),flush=True)
            z=torch.randn((1,512,9),generator=torch.Generator().manual_seed(4101)).cuda()
            torch.manual_seed(10)
            wrapper=OfficialWrapper(base,ControlAdapter("Q_center").cuda()).eval()
            off,stats=euler_sample(wrapper,z,y,mask,precision="fp32")
            save_raw(off[0],output/"raw"/"wrapper_off_seed4101",z[0],seed=4101)
            direct=outputs[0].to("cuda")[None]
            report["off_identity"]=dict(status="PASS" if torch.equal(off,direct) else "FAIL",
                bitwise_equal=bool(torch.equal(off,direct)),max_abs=float((off-direct).abs().max()),atol=1e-6,rtol=1e-5)
            report["runs"].append(dict(label="wrapper_off",**stats))
            report["complete_rollouts"]+=1;report["complete_sampler_calls"]+=1
            report["wrapper_counts"]=dict(wrapper.counts)
            wrapper.close();wrapper=None
            from flow_matching import create_transport
            transport=create_transport(prediction="velocity",use_lognorm=True)
            transport.method="euler";transport.atol=1e-6;transport.rtol=5e-4
            before=report["actual_backbone_forwards"]
            with torch.no_grad():
                reference=transport.sample(z,base,num_steps=51,model_kwargs=dict(y=y,mask=mask))[-1]
            save_raw(reference[0],output/"raw"/"official_reference_seed4101",z[0],seed=4101)
            report["complete_rollouts"]+=1;report["complete_sampler_calls"]+=1
            nfe=report["actual_backbone_forwards"]-before
            report["reference_reconciliation"]=dict(status="PASS" if nfe==50 and torch.allclose(reference,direct,atol=1e-4,rtol=1e-4) else "FAIL",
                nfe=nfe,t_eval_nodes=51,face_cfg=1.0,max_abs=float((reference-direct).abs().max()),atol=1e-4,rtol=1e-4,
                note="Predeclared separate rollout tolerance; reference adjacent float32 grid dt vs explicit constant dt=1/50.")
        report["base_hash_unchanged"]=state_sha256(base)==pre["base_state_sha256"]
        render_grid(outputs,output/"figures"/"official_baseline.png")
        report["status"]="PASS" if report["off_identity"]["status"]=="PASS" and report["reference_reconciliation"]["status"]=="PASS" and report["base_hash_unchanged"] and all(s["status"]=="PASS" for s in report["runs"]) else "FAIL"
    except BaseException as error:
        report.update(status="FAIL",error=repr(error),traceback=traceback.format_exc());print(traceback.format_exc(),flush=True)
    finally:
        if wrapper:wrapper.close()
        report["seconds"]=time.perf_counter()-start;write()
    print(json.dumps(dict(stage="sampling_precheck",status=report["status"],counts={k:report[k] for k in ("actual_backbone_forwards","actual_backbone_examples","complete_rollouts")})),flush=True)
    return report
if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--checkpoint",required=True);p.add_argument("--output",required=True)
    a=p.parse_args();r=run(a.checkpoint,a.output);raise SystemExit(0 if r["status"]=="PASS" else 1)

