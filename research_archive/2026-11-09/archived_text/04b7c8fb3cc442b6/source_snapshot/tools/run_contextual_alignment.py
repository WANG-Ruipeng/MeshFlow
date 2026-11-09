"""Durable controller for the one authorized contextual-alignment pilot."""
from pathlib import Path
import argparse,json,os,subprocess,sys,time
from meshflow_control.artifacts import atomic_json,file_sha256
from meshflow_control.data.io import freeze_json
from meshflow_control.training.context_budget import Budget
from meshflow_control.accounting import resource_snapshot

def run(args):
    root=Path(args.out).resolve();repo=Path(__file__).resolve().parents[1]
    protocol=root/"protocol.json"
    if not protocol.exists():raise RuntimeError("Run prepare_contextual_alignment.py to freeze protocol first")
    registration=json.loads(protocol.read_text())
    if registration["start"]["sha256"]!=file_sha256(args.start):raise ValueError("START identity differs")
    if registration["data_manifest"]["sha256"]!=file_sha256(Path(args.data)/"train_manifest.json"):raise ValueError("Training data changed")
    folder=root/"orchestration";folder.mkdir(exist_ok=True);status=folder/"status.json"
    if status.exists() and not args.resume:raise RuntimeError("Existing controller requires --resume")
    old=json.loads(status.read_text()) if status.exists() else {}
    if old and old["protocol_sha256"]!=file_sha256(protocol):raise ValueError("Frozen protocol changed")
    state=dict(status="RUNNING",protocol_sha256=file_sha256(protocol),started_unix=time.time(),stage=args.stage,completed=old.get("completed",[]))
    env=os.environ.copy();env.pop("PYTHONPATH",None)
    env.update(PYTHONDONTWRITEBYTECODE="1",OMP_NUM_THREADS="4",MKL_NUM_THREADS="4",OPENBLAS_NUM_THREADS="1")
    def call(name,argv,python=None):
        state["phase"]=name;atomic_json(status,state);stamp=str(time.time_ns());lp=folder/f"{name}_{stamp}.log"
        atomic_json(folder/f"{name}_{stamp}_resources.json",resource_snapshot())
        child_env=env.copy()
        if python:
            child_env.update(PYTHONPATH=str(repo/"src"),OMP_NUM_THREADS="2",MKL_NUM_THREADS="2",OPENBLAS_NUM_THREADS="2")
        with lp.open("w") as log:
            proc=subprocess.Popen([str(python or sys.executable),"-B",*map(str,argv)],env=child_env,cwd=root,stdout=log,stderr=subprocess.STDOUT)
            with (folder/"processes.jsonl").open("a") as f:
                f.write(json.dumps(dict(event="START",phase=name,pid=proc.pid,time_unix=time.time(),argv=list(map(str,argv)),log=str(lp)))+"\n");f.flush()
            while proc.poll() is None:
                q=subprocess.run(["nvidia-smi","--query-gpu=timestamp,memory.total,memory.used,memory.free,utilization.gpu","--format=csv,noheader,nounits"],capture_output=True,text=True)
                with (folder/"gpu_monitor.jsonl").open("a") as f:
                    f.write(json.dumps(dict(phase=name,pid=proc.pid,time_unix=time.time(),gpu=q.stdout.strip(),returncode=q.returncode))+"\n")
                time.sleep(5)
        if proc.returncode:raise RuntimeError(f"{name} exited {proc.returncode}; inspect {lp}")
        if name not in state["completed"]:state["completed"].append(name)
        atomic_json(status,state);print(name,"COMPLETE",flush=True)
    try:
        if args.stage in ("all","teacher"):
            if not (root/"teacher_manifest.json").exists():
                if not args.teacher_python:raise ValueError("Explicit isolated teacher Python required")
                call("teacher",["-m","meshflow_control.teachers.siglip_targets","--stage","encode","--out",root,"--ledger",root/"audit/budget"],args.teacher_python)
            else:state["completed"].append("teacher_verified_cache")
        if args.stage in ("all","preflight"):
            gate=root/"preflight/preflight.json"
            if not gate.exists():
                call("preflight",["-m","meshflow_control.training.context_preflight","--start",args.start,"--data",args.data,
                    "--stream-root",root/"training_stream","--teacher-manifest",root/"teacher_manifest.json",
                    "--out",root/"preflight","--ledger",root/"audit/budget"])
            if json.loads(gate.read_text())["status"]!="PASS":raise RuntimeError("Preflight not PASS")
        if args.stage in ("all","run"):
            gate=root/"preflight/preflight.json"
            if not gate.exists() or json.loads(gate.read_text())["status"]!="PASS":raise RuntimeError("Real preflight required")
            if not (root/"training_stream/precompute_receipt.json").exists():raise RuntimeError("Shared OT stream missing")
            for arm in "ABC":
                sp=root/"training"/arm/"status.json";resume=None
                if sp.exists():
                    prior=json.loads(sp.read_text())
                    if prior["status"]=="COMPLETE" and prior["step"]==1000:
                        if file_sha256(prior["checkpoint"]["path"])!=prior["checkpoint"]["sha256"]:raise RuntimeError("Endpoint changed")
                        continue
                    if prior["status"]!="INTERRUPTED_SAFE":raise RuntimeError("Training needs reviewed engineering recovery: "+arm)
                    resume=prior["checkpoint"]["path"]
                    if file_sha256(resume)!=prior["checkpoint"]["sha256"]:raise RuntimeError("Resume changed")
                argv=[repo/"tools/contextual_alignment_worker.py","--stage","train","--out",root,"--arm",arm,"--start",args.start,"--data",args.data]
                if resume:argv+=["--resume",resume]
                call("train_"+arm,argv)
            for arm in "ABC":
                call("export_"+arm,[repo/"tools/contextual_alignment_worker.py","--stage","export","--out",root,"--arm",arm])
                call("check_export_"+arm,[repo/"tools/contextual_alignment_worker.py","--stage","check-export","--out",root,"--arm",arm,"--data",args.data])
            labels=("START","A_FM_1000","B_FREE_ALIGN_1000","C_KNOWN_ALIGN_1000")
            for label in labels:
                checkpoint=args.start if label=="START" else root/"inference"/f"{label[0]}_generator_step1000.pt"
                call("sample_"+label,[repo/"tools/sample_contextual_alignment.py","--out",root,"--label",label,"--checkpoint",checkpoint])
            rows=[]
            for label in labels:rows+=json.loads((root/"outputs"/f"{label}_completed.json").read_text())["rows"]
            if len(rows)!=192 or len({(r["label"],r["request_id"]) for r in rows})!=192:raise RuntimeError("Output matrix incomplete")
            freeze_json(root/"outputs/manifest.json",dict(schema="contextual_alignment_complete_v1",rows=rows))
        if args.stage in ("all","evaluate"):
            call("score",[repo/"tools/contextual_alignment_evaluation.py","--stage","score","--out",root])
            call("aggregate",[repo/"tools/contextual_alignment_evaluation.py","--stage","aggregate","--out",root])
            call("render",[repo/"tools/contextual_alignment_evaluation.py","--stage","render","--out",root])
            result=json.loads((root/"analysis/results.json").read_text())
            if result.get("status")!="COMPLETE" or result.get("scored")!=192:
                raise RuntimeError("All 192 outputs must have completed scoring; explicit missing/blocked rows remain in results")
            div=result.get("diversity",{})
            if div.get("complete_pairs")!=288 or div.get("expected_pairs")!=288:
                raise RuntimeError("All 288 registered diversity pairs must complete")
            state["evaluation_result"]=result
        state.update(status="COMPUTE_COMPLETE",finished_unix=time.time(),budget=Budget(root/"audit/budget").snapshot(),
                     visual_review="REQUIRES_ACTUAL_INSPECTION",scientific_report="SEPARATE_FINAL_REVIEW")
    except BaseException as exc:
        state.update(status="BLOCKED",error=repr(exc),budget=Budget(root/"audit/budget").snapshot());raise
    finally:atomic_json(status,state)

if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--out",required=True);p.add_argument("--data",required=True);p.add_argument("--start",required=True)
    p.add_argument("--teacher-python");p.add_argument("--stage",choices=["all","teacher","preflight","run","evaluate"],default="all")
    p.add_argument("--resume",action="store_true");run(p.parse_args())
