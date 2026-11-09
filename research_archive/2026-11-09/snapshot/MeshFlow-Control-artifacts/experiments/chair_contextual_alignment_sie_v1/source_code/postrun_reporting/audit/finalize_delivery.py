"""Close the fixed experiment after actual reviews/report, with small delivery hashes."""
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,csv,shutil
R=Path(__file__).resolve().parents[1];repo=Path("/mnt/e/MeshFlow-Control")
def js(p):return json.loads(Path(p).read_text(encoding="utf-8-sig"))
def sha(p):
 h=hashlib.sha256()
 with Path(p).open("rb") as f:
  for b in iter(lambda:f.read(8<<20),b""):h.update(b)
 return h.hexdigest()
def write(p,x):Path(p).write_text(json.dumps(x,indent=2,ensure_ascii=False)+"\n",encoding="utf8")
controller=js(R/"orchestration/status.json");assert controller["status"]=="COMPUTE_COMPLETE"
assert js(R/"audit/call_closure_final.json")["status"]=="PASS"
assert js(R/"audit/training_endpoint_independent.json")["status"]=="PASS"
assert js(R/"visual_review/final_review_receipt.json")["status"]=="PASS"
required=["EXPERIMENT_REPORT.md","METHOD_AND_DEVIATIONS.md","protocol.json","teacher_manifest.json","teacher_render_contact_sheet.png","checkpoint_manifest.json","training_logs/manifest.json","per_output_metrics.csv","paired_parent_metrics.csv","paired_noise_metrics.csv","sie_summary.csv","visual_review.csv","OPEN_GALLERY.html","code.diff","README_RESUME.md"]
for p in required:assert (R/p).is_file() and (R/p).stat().st_size>0,p
with (R/"visual_review.csv").open(encoding="utf-8-sig",newline="") as f:reviews=list(csv.DictReader(f))
assert len(reviews)==192 and len({(r["request_id"],r["label"]) for r in reviews})==192
assert all(r["human_usability"]=="NOT_RUN" for r in reviews)
code_checks=[]
for row in js(R/"audit/code_frozen_before_training.json"):
 actual=sha(repo/row["Path"]);assert actual==row["Sha256"],row["Path"]
 code_checks.append(dict(path=row["Path"],sha256=actual,unchanged=True))
assert len(code_checks)==94
write(R/"audit/code_frozen_final_verification.json",dict(status="PASS",files=code_checks,model_calls=0))
D=R/"source_code/postrun_reporting";D.mkdir(parents=True,exist_ok=True);copies=[]
for p in sorted(list((R/"audit").glob("*.py"))+list((R/"visual_review").glob("*.py"))):
 rel=p.relative_to(R);dest=D/rel;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,dest)
 assert sha(p)==sha(dest)
 copies.append(dict(original=str(rel),snapshot=str(dest.relative_to(R)),sha256=sha(p),bytes=p.stat().st_size))
write(D/"manifest.json",dict(status="PASS",scope="Artifact-side CPU reporting/auditing and review recording scripts; supplements, does not alter original frozen source snapshot",files=copies))
account=js(R/"audit/final_accounting.json");assert not account["errors"] and not account["gpu_pending_owners"] and account["gpu_owner_ids_unique"] and all(x["unchanged"] for x in account["execution_binding"])
paths=set(required+["TASK_REQUEST.txt","audit/call_closure_final.json","audit/training_endpoint_independent.json","audit/final_accounting.json","audit/code_frozen_final_verification.json","audit/final_report_fact_review.md","audit/git_status_final.json","audit/budget/calls.jsonl","audit/budget/gpu_occupancy.jsonl","analysis/results.json","analysis/comparison_summary.csv","analysis/leave_one_parent_out.csv","analysis/distance_tails/receipt.json","outputs/manifest.json","gallery/gallery_manifest.json","gallery/camera_manifest.json","gallery/method_mapping.json","source_code/source_inventory.json","source_code/postrun_reporting/manifest.json","visual_review/source_hidden_freeze.json","visual_review/final_review_receipt.json","visual_review/visual_summary.md","visual_review/blind_review_combined.csv","visual_review/reference_review_combined.csv"])
paths.update(str(p.relative_to(R)) for p in (R/"analysis/distance_tails").glob("*.csv"))
paths.update(str(p.relative_to(R)) for p in (R/"visual_review").glob("*.csv"))
records=[dict(path=p,bytes=(R/p).stat().st_size,sha256=sha(R/p)) for p in sorted(paths)]
manifest=dict(schema="contextual_alignment_delivery_v1",status="COMPLETE",root=str(R),files=records,raw_policy="192 output identities and original/raw paths bound in outputs/manifest.json and scoring receipts; no RAW deletion or recopy",checkpoints_policy="6 training +3 generator exports retained outside git; hashes in checkpoint_manifest.json",third_party_weights_repacked=False,source_snapshots="Original source_code/source_inventory.json plus source_code/postrun_reporting/manifest.json",new_model_calls=0)
write(R/"DELIVERY_MANIFEST.json",manifest)
final=dict(schema="contextual_alignment_final_status_v1",status="STOPPED_FIXED_TASK_COMPLETE",finished_utc=datetime.now(timezone.utc).isoformat(),training_updates=3000,new_Euler50=160,cached_START=32,scored_outputs=192,diversity_pairs=288,actual_AI_source_hidden_pages=48,actual_AI_reference_pages=32,human_usability="NOT_RUN",screen_B="FAIL",screen_C="FAIL",default_checkpoint="START_UNCHANGED",automatic_new_experiment=False,git_commit_or_push=False,delivery_manifest_sha256=sha(R/"DELIVERY_MANIFEST.json"),experiment_report_sha256=sha(R/"EXPERIMENT_REPORT.md"),visual_review_sha256=sha(R/"visual_review.csv"),budget_calls_sha256=sha(R/"audit/budget/calls.jsonl"),gpu_ownership_seconds=account["gpu_ownership_seconds"])
write(R/"FINAL_STATUS.json",final)
print(json.dumps(final,indent=2))
