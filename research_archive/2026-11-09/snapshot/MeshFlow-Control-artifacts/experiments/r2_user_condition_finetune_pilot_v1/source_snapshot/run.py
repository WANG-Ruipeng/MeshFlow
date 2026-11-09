"""Recoverable stage entrypoint; GPU concurrency is assigned by measured dispatch receipts."""
from pathlib import Path
import argparse,json,subprocess,sys,time
from control import OUT,DATA,CODE,resources,event,sha,write
STAGES=("status","prepare-data","prepare-eval","preflight","frozen","train-old","train-user","sample-old","sample-user","score","render","verify-protected")
def status():
 paths=["data/precompute_receipt.json","preflight/completed.json","preflight/receipt.json","evaluation/registration.json",
  "training/R2_CONT_OLD/status.json","training/R2_CONT_USER/status.json","evaluation/raw_index.json","evaluation/scores_all.json","gallery/manifest.json"]
 values={}
 for name in paths:
  p=OUT/name
  if p.exists():
   r=json.loads(p.read_text());values[name]={k:r[k] for k in ("status","finetune_step","global_step","counts","model_outputs","PNG_count","wall_seconds","calls") if k in r}
 print(json.dumps(values,indent=2))
def main():
 p=argparse.ArgumentParser();p.add_argument("--stage",required=True,choices=STAGES);p.add_argument("--attempt",default="initial")
 p.add_argument("--dispatch-receipt",type=Path);a=p.parse_args()
 if a.stage=="status": status();return
 commands={
  "prepare-data":["data.py","--root",str(OUT),"--workers","2"],
  "prepare-eval":["evaluation.py","--stage","prepare","--out",str(OUT)],
  "preflight":["preflight.py","--out",str(OUT),"--registration",str(OUT/"registration.json"),"--attempt-id",a.attempt],
  "frozen":["evaluation.py","--stage","sample","--out",str(OUT),"--label","R2_FROZEN","--attempt",a.attempt,"--controller-owned"],
  "score":["evaluation.py","--stage","score","--out",str(OUT)],
  "render":["evaluation.py","--stage","render","--out",str(OUT)],
  "verify-protected":["protected.py","after"]}
 for suffix,arm in (("old","R2_CONT_OLD"),("user","R2_CONT_USER")):
  commands["train-"+suffix]=["trainer.py","--out",str(OUT),"--registration",str(OUT/"registration.json"),"--arm",arm,"--attempt-id",a.attempt]
  commands["sample-"+suffix]=["evaluation.py","--stage","sample","--out",str(OUT),"--label",arm,"--attempt",a.attempt,"--controller-owned"]
 gpu=a.stage in ("preflight","frozen","train-old","train-user","sample-old","sample-user")
 if gpu:
  if a.dispatch_receipt is None: p.error("GPU stage needs a measured --dispatch-receipt; see resources and registered scheduling policy")
  d=json.loads(a.dispatch_receipt.read_text())
  if d.get("stage")!=a.stage or d.get("authorized") is not True:raise ValueError("Dispatch receipt mismatch")
  if time.time()-d["created_unix"]>600:raise ValueError("Dispatch observation stale")
  resources("before_"+a.stage)
 args=commands[a.stage]
 if a.stage.startswith("sample-"):
  arm="R2_CONT_OLD" if a.stage=="sample-old" else "R2_CONT_USER"
  subprocess.run([sys.executable,"-B",str(CODE/"evaluation.py"),"--stage","bind","--out",str(OUT),"--label",arm,
    "--checkpoint",str(OUT/"inference"/(arm+"_generator_step7000.pt"))],check=True)
 cmd=[sys.executable,"-B",str(CODE/args[0]),*args[1:]]
 event("STAGE_BEGIN",stage=a.stage,attempt=a.attempt,command=cmd,dispatch=None if not gpu else str(a.dispatch_receipt))
 start=time.perf_counter();r=subprocess.run(cmd)
 event("STAGE_END",stage=a.stage,attempt=a.attempt,returncode=r.returncode,seconds=time.perf_counter()-start)
 if gpu:resources("after_"+a.stage)
 raise SystemExit(r.returncode)
if __name__=="__main__":main()
