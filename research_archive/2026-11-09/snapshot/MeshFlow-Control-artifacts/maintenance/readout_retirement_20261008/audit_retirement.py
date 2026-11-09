"""Read-only inventory/hash auditor for explicitly authorized old READOUT retirement."""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import argparse,hashlib,json,os,time
ROOT=Path("/mnt/e/MeshFlow-Control-artifacts")
REPO=Path("/mnt/e/MeshFlow-Control")
AUDIT=ROOT/"maintenance/readout_retirement_20261008"
OLD_CACHE=ROOT/"training_stream/ot_cache"
CURRENT=ROOT/"experiments/chair_contextual_alignment_sie_v1"
ORIGINALS=[
 Path("/mnt/e/MeshFlow/native_t1/checkpoints/chair_hybrid_start_total5000.pt"),
 Path("/mnt/e/MeshFlow/native_t1/runs/chair_domain_control_v1/checkpoints/Geo_phase3000_total4000.pt"),
]
def sha(p):
 h=hashlib.sha256()
 with open(p,"rb") as f:
  for chunk in iter(lambda:f.read(8<<20),b""):h.update(chunk)
 return h.hexdigest()
def js(p):return json.loads(p.read_text(encoding="utf-8-sig"))
def write(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+"\n",encoding="utf8")
def win(p):return "E:\\"+str(p).removeprefix("/mnt/e/").replace("/","\\")
def record(p):
 s=p.lstat()
 if p.is_symlink():return dict(path=str(p),windows_path=win(p),kind="symlink",target=os.readlink(p),bytes=0)
 assert p.is_file(),str(p)
 return dict(path=str(p),windows_path=win(p),kind="file",bytes=s.st_size,sha256=sha(p))
def retired_paths():
 result=[ROOT/"checkpoints"/f"{arm}_step1000.pt" for arm in "ABC"]
 npzs=sorted(OLD_CACHE.glob("*.npz"))
 assert len(npzs)==8000,len(npzs)
 for p in result+npzs:
  assert p.is_file() and not p.is_symlink(),str(p)
  assert p.resolve()==p,str(p)
  assert p.parent in (ROOT/"checkpoints",OLD_CACHE)
 return result+npzs
def inventory(excluded):
 result=[]
 for base in (ROOT,REPO):
  for directory,dirs,files in os.walk(base,followlinks=False):
   d=Path(directory)
   kept=[]
   for name in dirs:
    q=d/name
    if q==AUDIT or (base==ROOT and q==ROOT/"venv") or q==CURRENT/"software":continue
    if base==REPO and name in (".git","__pycache__", ".pytest_cache"):continue
    if q.is_symlink():result.append(q)
    else:kept.append(name)
   dirs[:]=kept
   for name in files:
    p=d/name
    if p not in excluded:result.append(p)
 result.extend(ORIGINALS)
 return sorted(set(result))
def each_hash(paths,out):
 with out.open("w",encoding="utf8") as f,ThreadPoolExecutor(max_workers=4) as pool:
  for i,row in enumerate(pool.map(record,paths),1):
   f.write(json.dumps(row,ensure_ascii=False)+"\n")
   if i%5000==0: print(json.dumps(dict(hashed=i,total=len(paths))),flush=True)
def check_frozen():
 rows=js(CURRENT/"audit/code_frozen_before_training.json")
 assert len(rows)==94
 for r in rows:assert sha(REPO/r["Path"])==r["Sha256"],r["Path"]
 return 94
def check_active():
 hits=[]
 for d in Path("/proc").iterdir():
  if not d.name.isdigit() or int(d.name)==os.getpid():continue
  try:
   parts=(d/"cmdline").read_bytes().split(b"\0")
   decoded=[p.decode("utf8","replace") for p in parts if p]
   if not decoded or "python" not in Path(decoded[0]).name:continue
   if any(Path(a).name in {"run_experiment.py","run_readout_precision.py","sample_panel.py","train.py","run_contextual_alignment.py","train_contextual_alignment.py","sample_contextual_alignment.py"} for a in decoded[1:]):
    hits.append(dict(pid=d.name,command=decoded))
  except (OSError,PermissionError):pass
 assert not hits,hits
 return hits
def main():
 ap=argparse.ArgumentParser();ap.add_argument("stage",choices=("prepare","verify"));a=ap.parse_args()
 started=time.time();AUDIT.mkdir(parents=True,exist_ok=True)
 if a.stage=="prepare":
  assert not (AUDIT/"prepared.json").exists(),"Already prepared; inspect instead of resnapshotting"
  check_active();assert js(CURRENT/"FINAL_STATUS.json")["status"]=="STOPPED_FIXED_TASK_COMPLETE"
  old=js(ROOT/"training_stream/precompute_receipt.json")
  assert old["namespace"]=="NATIVE_MESH_DIRECT_CONTEXT_READOUT_V1/train/v1"
  assert js(CURRENT/"audit/execution_binding.json")["files"].get(str(CURRENT/"training_stream/train_plan.json"))
  paths=retired_paths();records=[record(p) for p in paths]
  old_expected={"A_step1000.pt":"2ab9635f9c91d862ac798915d76079f93ce8d58d8e60852f145f7b79a7cd49df","B_step1000.pt":"3f63481c782d367e9ce2d4d650b3e29bab7da9e23aebdfac80f1a2a3a3017fdf","C_step1000.pt":"cb47f9e14e105a4058590f82d7d674cdc3bd8aef7209669393ba93166ed634a4"}
  for r in records[:3]:assert r["sha256"]==old_expected[Path(r["path"]).name],r["path"]
  write(AUDIT/"delete_manifest.json",dict(schema="exact_old_readout_files_v1",status="PREPARED",files=records,count=len(records),bytes=sum(r["bytes"] for r in records)))
  protected=inventory(set(paths));each_hash(protected,AUDIT/"protected_before.jsonl")
  receipt=dict(status="PREPARED",utc=datetime.now(timezone.utc).isoformat(),delete_files=len(paths),delete_bytes=sum(r["bytes"] for r in records),protected_entries=len(protected),frozen_source_files=check_frozen(),workers=4,scope="All research artifacts except software environments and this maintenance folder; source repo except .git/bytecode caches; original START and Geo4000",no_active_experiment_workers=True,model_calls=0,seconds=time.time()-started)
  write(AUDIT/"prepared.json",receipt);print(json.dumps(receipt),flush=True)
 else:
  prepared=js(AUDIT/"prepared.json");deletion=js(AUDIT/"delete_manifest.json")
  deleted={Path(r["path"]) for r in deletion["files"]}
  assert not [str(p) for p in deleted if p.exists()],"Deletion incomplete"
  rows=[json.loads(s) for s in (AUDIT/"protected_before.jsonl").read_text(encoding="utf8").splitlines()]
  issues=[]
  def verify(r):
   try:
    actual=record(Path(r["path"]))
    if actual!=r:return dict(path=r["path"],before=r,after=actual)
   except OSError as e:return dict(path=r["path"],error=str(e))
  with ThreadPoolExecutor(max_workers=4) as pool:
   for i,issue in enumerate(pool.map(verify,rows),1):
    if issue:issues.append(issue)
    if i%5000==0:print(json.dumps(dict(verified=i,total=len(rows))),flush=True)
  actual=set(inventory(deleted));expected={Path(r["path"]) for r in rows}
  missing=sorted(map(str,expected-actual));added=sorted(map(str,actual-expected))
  frozen=check_frozen()
  for r in js(CURRENT/"checkpoint_manifest.json")["checkpoints"]:
   path=Path(r["path"]);assert path.is_file(),str(path)
   assert sha(path)==r["sha256"],str(path)
  final=js(CURRENT/"FINAL_STATUS.json");assert sha(CURRENT/"EXPERIMENT_REPORT.md")==final["experiment_report_sha256"]
  receipt=dict(status="PASS" if not issues and not missing and not added else "FAIL",utc=datetime.now(timezone.utc).isoformat(),deleted_files=len(deleted),deleted_bytes=deletion["bytes"],protected_entries=len(rows),issues=issues,missing_unexpected=missing,added_unexpected=added,frozen_source_files=frozen,current_candidate_checkpoints_verified=9,original_START_and_Geo4000_unchanged=True,all_existing_RAW_and_reports_preserved=True,model_calls=0,training_updates=0,generation=0,geometry_scoring=0,seconds=time.time()-started)
  write(AUDIT/"verification.json",receipt);print(json.dumps(receipt),flush=True)
  assert receipt["status"]=="PASS"
if __name__=="__main__":main()
