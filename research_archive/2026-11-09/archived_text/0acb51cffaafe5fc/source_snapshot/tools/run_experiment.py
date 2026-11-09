"""Single-worker resumable fixed experiment. No adaptive arm or checkpoint selection."""
import argparse,json,os,subprocess,sys,time
from pathlib import Path
from meshflow_control.artifacts import atomic_json,file_sha256
from meshflow_control.accounting import Budget,resource_snapshot
from validate_binding import validate

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--artifacts",required=True);p.add_argument("--start",required=True)
    p.add_argument("--preflight",required=True);p.add_argument("--resume",action="store_true")
    a=p.parse_args();root=Path(a.artifacts).resolve();repo=Path(__file__).resolve().parents[1]
    ledger=root/"audit"/"budget";folder=root/"orchestration";folder.mkdir(exist_ok=True,parents=True)
    status=folder/"status.json"
    if status.exists() and not a.resume:raise RuntimeError("Existing run requires explicit --resume")
    gate=json.loads(Path(a.preflight).read_text())
    binding=validate(root,a.start,a.preflight)
    if status.exists():
        old=json.loads(status.read_text())
        if old.get("binding",binding)!=binding:raise RuntimeError("Prior experiment identity differs")
    env=os.environ.copy();env.pop("PYTHONPATH",None)
    env.update(PYTHONDONTWRITEBYTECODE="1",OMP_NUM_THREADS="4",MKL_NUM_THREADS="4",OPENBLAS_NUM_THREADS="1")
    state=dict(status="RUNNING",binding=binding,start_unix=time.time(),start_checkpoint_sha256=file_sha256(a.start),preflight_sha256=file_sha256(a.preflight),completed=[])
    def call(name,argv):
        state["phase"]=name;atomic_json(status,state)
        stamp=str(time.time_ns());logpath=folder/(name+"_"+stamp+".log")
        atomic_json(folder/(name+"_"+stamp+"_resources.json"),resource_snapshot())
        with logpath.open("w") as log:
            proc=subprocess.Popen([sys.executable,"-B",*map(str,argv)],cwd=root,env=env,stdout=log,stderr=subprocess.STDOUT)
            with (folder/"processes.jsonl").open("a") as f:
                f.write(json.dumps(dict(event="START",phase=name,pid=proc.pid,time_unix=time.time(),argv=list(map(str,argv)),log=str(logpath)))+"\n");f.flush()
            while proc.poll() is None:
                q=subprocess.run(["nvidia-smi","--query-gpu=timestamp,memory.used,memory.free,utilization.gpu","--format=csv,noheader,nounits"],capture_output=True,text=True)
                with (folder/"gpu_monitor.jsonl").open("a") as f:
                    f.write(json.dumps(dict(phase=name,pid=proc.pid,time_unix=time.time(),gpu=q.stdout.strip(),returncode=q.returncode))+"\n")
                time.sleep(10)
        if proc.returncode:raise RuntimeError(name+" exited "+str(proc.returncode)+"; preserve "+str(logpath))
        state["completed"].append(name);atomic_json(status,state);print(name,"COMPLETE",flush=True)
    try:
        for arm,cfg in (("A","a_baseline.yaml"),("B","b_mixed_readout.yaml"),("C","c_clean_readout.yaml")):
            sp=root/"training"/arm/"status.json";resume=None
            if sp.exists():
                prior=json.loads(sp.read_text())
                if prior["status"]=="COMPLETE" and prior["step"]==1000:
                    ck=prior["checkpoint"]
                    if file_sha256(ck["path"])!=ck["sha256"]:raise RuntimeError("Completed checkpoint changed")
                    state["completed"].append("train_"+arm+"_reused");continue
                if prior["status"]!="INTERRUPTED_SAFE" and not (prior["status"]=="COMPLETE" and prior["step"]==500):raise RuntimeError("Incomplete training requires reviewed recovery: "+arm)
                resume=prior["checkpoint"]["path"]
                if file_sha256(resume)!=prior["checkpoint"]["sha256"]:raise RuntimeError("Resume checkpoint changed")
            args=["-m","meshflow_control","train","--config",repo/"configs"/cfg,"--data-manifest",root/"data/train_manifest.json",
                "--stream-root",root/"training_stream","--start-checkpoint",a.start,"--out",root,"--ledger",ledger]
            if resume:args+=["--resume",resume]
            call("train_"+arm,args)
        for label in ("START","A500","B500","C500","A1000","B1000","C1000"):
            checkpoint=a.start if label=="START" else root/"checkpoints"/(label[0]+"_step"+label[1:].zfill(4)+".pt")
            call("sample_"+label,[repo/"tools/sample_panel.py","--checkpoint",checkpoint,"--label",label,
                "--requests",root/"data/evaluation/requests.json","--out",root/"outputs","--ledger",ledger])
        rows=[]
        for label in ("START","A500","B500","C500","A1000","B1000","C1000"):
            rows+=json.loads((root/"outputs"/(label+"_completed.json")).read_text())["rows"]
        if len(rows)!=288:raise RuntimeError("Incomplete registered output matrix")
        manifest=root/"outputs/all_completed.json"
        atomic_json(manifest,dict(schema="direct_context_completed_outputs_v1",rows=rows))
        call("evaluate",["-m","meshflow_control","evaluate","--manifest",manifest,"--out",root/"evaluation","--workers","4"])
        call("render",["-m","meshflow_control","render","--manifest",manifest,"--out",root/"gallery"])
        state.update(status="COMPLETE",budget=Budget(ledger).snapshot(),finish_unix=time.time())
    except BaseException as exc:
        state.update(status="BLOCKED",error=repr(exc),budget=Budget(ledger).snapshot());raise
    finally:atomic_json(status,state)
if __name__=="__main__":main()
