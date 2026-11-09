"""Verify delivery completeness without loading weights, scoring, or running models."""
import csv, hashlib, json, re, time
from pathlib import Path
from urllib.parse import unquote
OUT=Path(__file__).resolve().parents[1]
CODE=Path("/mnt/e/MeshFlow/r2/pilots/user_condition_finetune_v1")
def read(p): return json.loads(p.read_text(encoding="utf-8-sig"))
def digest(p):
 h=hashlib.sha256()
 with p.open("rb") as f:
  for b in iter(lambda:f.read(1<<20),b""):h.update(b)
 return h.hexdigest()
def csvread(p):
 with p.open(encoding="utf-8-sig",newline="") as f:return list(csv.DictReader(f))
def write(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+"\n",encoding="utf8")
checks=[]
def check(name,value,detail=None):
 checks.append(dict(name=name,pass_=bool(value),detail=detail))
 if not value: raise RuntimeError(name+": "+str(detail))
audit=read(OUT/"audit/execution_audit.json")
coverage=read(OUT/"review/review_coverage.json")
reg=read(OUT/"evaluation/registration.json")
scores=read(OUT/"evaluation/scores_all.json")
raw=read(OUT/"evaluation/raw_index.json")
gallery=read(OUT/"gallery/manifest.json")
review=read(OUT/"review/per_output_ai_proxy.json")["rows"]
paired=read(OUT/"review/paired_ai_proxy.json")["rows"]
labels=("R2_FROZEN","R2_CONT_OLD","R2_CONT_USER")
keys={(label,q["request_id"]) for label in labels for q in reg["requests"]}
check("execution_audit",audit["status"]=="PASS" and not audit["failures"],len(audit["checks"]))
check("protected_before_after",audit["protected_changed_or_missing"]==0,audit["protected_file_count"])
check("all_scores",scores["status"]=="COMPLETE" and {(r["label"],r["request_id"]) for r in scores["rows"]}==keys and len(scores["rows"])==84)
check("all_raw",raw["status"]=="COMPLETE" and len(raw["rows"])==84)
check("all_output_reviews",{(r["label"],r["request_id"]) for r in review}==keys and len(review)==84)
check("all_request_reviews",len(paired)==28 and {r["request_id"] for r in paired}=={q["request_id"] for q in reg["requests"]})
check("coverage",coverage["status"]=="PASS" and coverage["fully_covered"] and coverage["original_png_count"]==56)
check("render_complete",gallery["status"]=="COMPLETE" and gallery["PNG_count"]==56)
for rel,count in {
"evaluation/per_output.csv":84,"evaluation/per_condition.csv":42,
"evaluation/paired_noise.csv":84,"evaluation/paired_conditions.csv":42,
"evaluation/summary_equal_parent.csv":6,"evaluation/diversity.csv":42,
"review/per_output_ai_proxy.csv":84,"review/paired_ai_proxy.csv":28}.items():
 check("rows:"+rel,len(csvread(OUT/rel))==count,count)
manifest=read(OUT/"code_manifest.json")
for p,h in manifest["source_hashes"].items():
 check("frozen_pilot:"+Path(p).name,digest(Path(p))==h)
required=[
"R2_USER_CONDITION_FINETUNE_REPORT.md","DELIVERY.md","gallery/index.html",
"gallery/index.html.receipt.json","audit/runtime_summary.json","audit/runtime_summary.csv",
"audit/runtime_weight_inventory.csv","audit/numeric_interpretation.md",
"audit/numeric_interpretation.json","audit/ledger_summary.json","code.diff",
"review/summary_ai_proxy.json","evaluation/geometry_distributions.csv",
"evaluation/paired_direction_counts.csv","evaluation/descriptive_summary.json"]
for rel in required:check("exists:"+rel,(OUT/rel).is_file())
for item in audit["endpoints"]:
 p=Path(item["path"])
 check("endpoint_exists:"+p.name,p.is_file() and p.stat().st_size>100000000)
for rel in ("R2_USER_CONDITION_FINETUNE_REPORT.md","DELIVERY.md"):
 text=(OUT/rel).read_text(encoding="utf8")
 check("utf8:"+rel,sum(ord(c)>127 for c in text)>150 and "????" not in text)
 for target in re.findall(r"\]\(([^)]+)\)",text):
  if target.startswith(("http","file:")):continue
  target=unquote(target.split("#")[0])
  if target in ("completion.json","delivery_hashes.csv"):continue
  check("link:"+target,(OUT/target).exists(),rel)
rows=[]
excluded={"completion.json","delivery_hashes.csv"}
exts={".py",".md",".json",".jsonl",".csv",".diff",".html",".png",".txt"}
for p in sorted(OUT.rglob("*")):
 if p.is_file() and p.suffix.lower() in exts and p.name not in excluded:
  rows.append(dict(path=p.relative_to(OUT).as_posix(),bytes=p.stat().st_size,sha256=digest(p),scope="actual_read_at_delivery"))
with (OUT/"delivery_hashes.csv").open("w",encoding="utf-8-sig",newline="") as f:
 w=csv.DictWriter(f,fieldnames=["path","bytes","sha256","scope"]);w.writeheader();w.writerows(rows)
check("delivery_hash_inventory",len(rows)>100,len(rows))
checks_path=OUT/"audit/delivery_checks.json"
resource_files=sorted((OUT/"resources").glob("*_final_delivery_no_GPU_work.json"))
check("final_resource_receipt",bool(resource_files))
resource=read(resource_files[-1])
gpu_stdout=resource.get("gpu_processes",{}).get("stdout","")
write(checks_path,dict(status="PASS",checks=checks,model_calls=0,geometry_calls=0,weight_loads=0,
 small_text_and_image_files_hashed=len(rows),binary_hash_evidence="Prior successful execution audit and endpoint/raw/input receipts; no duplicate weight read"))
completion=dict(schema="r2_user_condition_finetune_delivery_complete_v1",status="COMPLETE",time_unix=time.time(),
 report="R2_USER_CONDITION_FINETUNE_REPORT.md",report_sha256=digest(OUT/"R2_USER_CONDITION_FINETUNE_REPORT.md"),
 gallery="gallery/index.html",gallery_sha256=digest(OUT/"gallery/index.html"),
 delivery_hashes_sha256=digest(OUT/"delivery_hashes.csv"),
 delivery_checks_sha256=digest(checks_path),
 fixed_budget_fulfilled=True,stop_requested_by_contract=True,additional_model_work_started=False,
 default_model_changed=False,git_commit_or_push=False,automatic_weight_deletion=False,
 model_recommendation="Do not promote R2_CONT_USER; retain main R2 step6000 as research baseline",
 train_updates=2000,train_forwards=16000,train_backwards=16000,
 preflight_forwards=8,preflight_backwards=8,preflight_optimizer_updates=0,
 new_rollouts=68,sampling_forwards=3400,reused_historical_RAW=16,
 scored_outputs=84,reviewed_outputs=84,reviewed_requests=28,PNG_count=56,
 training_sampling_failed_calls=0,OOM=0,retries=0,execution_audit_checks=len(audit["checks"]),
 protected_files_unchanged=audit["protected_file_count"],
 final_resource_receipt=str(resource_files[-1]),final_GPU_compute_process_stdout=gpu_stdout,
 qualitative="AI_PROXY; labels visible; not human evaluation; root last NEWDEV condition had aggregate numeric hint",
 binary_hashes="audit/execution_endpoints.csv plus original checkpoint, raw, trajectory and input receipts",
 helper_fixes="CPU JSON int64 serialization and Chinese text encoding; preserved evidence; original metrics and model code unchanged")
write(OUT/"completion.json",completion)
event=dict(event="DELIVERY_COMPLETE",time_unix=time.time(),completion_path="completion.json",
 report_sha256=completion["report_sha256"],model_calls=0,stop=True)
with (OUT/"controller_events.jsonl").open("a",encoding="utf8") as f:f.write(json.dumps(event)+"\n")
# The append above intentionally finalizes the controller log; refresh just its inventory row.
for row in rows:
 if row["path"]=="controller_events.jsonl":
  p=OUT/row["path"];row["bytes"]=p.stat().st_size;row["sha256"]=digest(p)
# Include the delivery checks receipt created during finalization.
rows=[row for row in rows if row["path"]!="audit/delivery_checks.json"]
rows.append(dict(path="audit/delivery_checks.json",bytes=checks_path.stat().st_size,sha256=digest(checks_path),scope="actual_read_at_delivery"))
rows.sort(key=lambda r:r["path"])
with (OUT/"delivery_hashes.csv").open("w",encoding="utf-8-sig",newline="") as f:
 w=csv.DictWriter(f,fieldnames=["path","bytes","sha256","scope"]);w.writeheader();w.writerows(rows)
completion["delivery_hashes_sha256"]=digest(OUT/"delivery_hashes.csv")
write(OUT/"completion.json",completion)
print(json.dumps(dict(status="COMPLETE",checks=len(checks),small_artifacts=len(rows),outputs=84,
 report_sha256=completion["report_sha256"],completion_sha256=digest(OUT/"completion.json"),
 final_gpu_processes=gpu_stdout),ensure_ascii=False),flush=True)
