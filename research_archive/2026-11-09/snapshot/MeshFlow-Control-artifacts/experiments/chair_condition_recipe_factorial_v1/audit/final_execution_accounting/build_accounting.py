"""Aggregate completed execution receipts; never load weights, metrics or aliases."""
from pathlib import Path
from collections import Counter
import csv,hashlib,json,time
RUN=Path("/mnt/e/MeshFlow-Control-artifacts/experiments/chair_condition_recipe_factorial_v1")
REPO=Path("/mnt/e/MeshFlow-Control")
OUT=Path(__file__).resolve().parent
VERSION="recipe_final_execution_accounting_v1"
started=time.time();sources=[];checks=[]
def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding="utf-8-sig"))
def record(p,kind):
 p=Path(p);item=dict(path=str(p),kind=kind,sha256=digest(p),bytes=p.stat().st_size);sources.append(item);return item

def document(relative,kind):
 p=RUN/relative;record(p,kind);return read(p)
def check(name,ok,detail=None):checks.append(dict(item=name,status="PASS" if ok else "FAIL",detail=detail))
def csv_write(path,rows):
 if not rows:path.write_text("");return
 fields=list(dict.fromkeys(k for row in rows for k in row))
 with path.open("w",newline="",encoding="utf-8-sig") as f:
  writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader();writer.writerows(rows)
def output(name,value):(OUT/name).write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False)+"\n",encoding="utf8")
if (OUT/"accounting.json").exists():raise FileExistsError("Final accounting is immutable; use a new version")
training=[];sampling=[];workers=[];stage_rows=[];audits={};waves={}
train_specs=[("main","training_accounting","main_training"),("course","curriculum_training_accounting","curriculum_training"),("confirm","confirmation_training_accounting","confirmation_training_input_repeat1")]
sample_specs=[("main","sampling_accounting","main_sampling"),("course","curriculum_sampling_accounting","curriculum_sampling"),("confirm","confirmation_sampling_accounting","confirmation_sampling_input_repeat1")]
for phase,folder,wave in train_specs:
 d=document("audit/"+folder+"/accounting.json","completed_training_audit");audits[phase+"_training"]=d
 check(phase+"_training_audit_PASS",d["status"]=="PASS")
 for arm in d["arms"]:
  attempts=arm["attempts"]
  row=dict(phase=phase,recipe=arm["recipe"],unique_updates=arm["unique_successful_updates"],physical_updates=arm["physical_successful_optimizer_updates"],
   replayed_updates=arm["successful_replayed_updates"],attempts=len(attempts),failed_attempts=arm["failed_attempts"],
   forward_calls=sum(a["successful_forward_calls"] for a in attempts),backward_calls=sum(a["successful_backward_calls"] for a in attempts),
   failed_calls=sum(a["failed_calls"] for a in attempts),unfinished_calls=sum(a["unfinished_calls"] for a in attempts),resumes=sum(a["resume_count"] for a in attempts),
   pytorch_peak_allocated_bytes=max(a["cuda_peak_allocated_bytes"] for a in attempts),pytorch_peak_reserved_bytes=max(a["cuda_peak_reserved_bytes"] for a in attempts))
  training.append(row)
  check(phase+"/"+arm["recipe"]+"_6000_unique_and_physical",row["unique_updates"]==row["physical_updates"]==6000)
 check(phase+"_unique_and_physical_totals_match_receipt",sum(x["unique_updates"] for x in training if x["phase"]==phase)==d["new_unique_logical_updates"] and sum(x["physical_updates"] for x in training if x["phase"]==phase)==d["new_physical_optimizer_updates"])
for phase,folder,wave in sample_specs:
 d=document("audit/"+folder+"/accounting.json","completed_sampling_audit");audits[phase+"_sampling"]=d
 check(phase+"_sampling_audit_PASS",d["status"]=="PASS")
 for label in d["per_label"]:
  sampling.append(dict(phase=phase,**label))
 check(phase+"_128_or384_unique_cells_and_actual_counts",d["outputs"]==d["new_outputs"]+d["reused_outputs"] and d["actual_rollouts_reserved"]==d["actual_rollouts_completed"]==d["new_outputs"] and d["actual_new_sampling_forwards_reserved"]==d["actual_new_sampling_forwards_succeeded"]==50*d["new_outputs"])
 check(phase+"_all_hard_output_contracts",d["all_actual_output_contracts_passed"])
for phase,folder,wave in train_specs+sample_specs:
 kind="training" if (phase,folder,wave) in train_specs else "sampling"
 d=document("audit/dispatch_"+wave+".json","completed_dispatch_receipt");waves[wave]=d
 check(wave+"_COMPLETE_no_failure",d["status"]=="COMPLETE" and not d["remaining"] and not d["failed"] and d["oom_downgrades"]==0)
 registration=document("registrations/"+({"main_training":"main_training_jobs.json","curriculum_training":"curriculum_training_jobs.json","confirmation_training_input_repeat1":"confirmation_training_jobs.json","main_sampling":"main_sampling_jobs.json","curriculum_sampling":"curriculum_sampling_jobs.json","confirmation_sampling_input_repeat1":"confirmation_sampling_jobs.json"}[wave]),"frozen_dispatch_jobs")
 check(wave+"_job_count_matches_completed",len(registration["jobs"])==len(d["completed"]))
 stage_rows.append(dict(phase=phase,kind=kind,wave=wave,status=d["status"],wall_seconds=d["wall_seconds"],jobs=len(registration["jobs"]),registered_max_workers=registration["max_workers"]))
# Read only past complete lines from the live controller; do NOT hash that hot file.
controller=RUN/"audit/controller.jsonl";raw=controller.read_bytes();last=raw.rfind(b"\n")+1
all_events=[(line,json.loads(line)) for line in raw[:last].splitlines() if line]
opened={};intervals=[]
for line,e in all_events:
 if e.get("wave") not in waves:continue
 key=(e["wave"],e.get("name"),e.get("attempt"))
 if e.get("event")=="WORKER_START":opened[key]=e
 elif e.get("event")=="WORKER_END":
  start=opened.pop(key,None)
  check("worker_start_end_pair",start is not None,str(key))
  if start:
   intervals.append(dict(wave=e["wave"],label=e["name"],attempt=e["attempt"],kind=e["phase"],start=start["time_unix"],end=e["time_unix"],wall_seconds=e["wall_seconds"],workers_at_start=start["worker_count"],returncode=e["returncode"]))
check("no_unclosed_GPU_worker",not opened,str(opened))
selected=[];resources=[];cpu_events=[]
for line,e in all_events:
 choose=e.get("wave") in waves and e.get("event") in ("WORKER_START","WORKER_END","OOM_DOWNGRADE","RECOVERY_ACCOUNTING")
 if e.get("event")=="RESOURCES":
  matched=[w for w in waves if any(i["wave"]==w and i["start"]<=e["time_unix"]<=i["end"] for i in intervals)]
  for wave in matched:
   snap=e["snapshot"];g=snap.get("gpu") or {}
   resources.append(dict(wave=wave,time_unix=e["time_unix"],gpu_name=g.get("name"),gpu_total_mib=g.get("total_mib"),gpu_used_mib=g.get("used_mib"),gpu_free_mib=g.get("free_mib"),gpu_utilization=g.get("utilization"),cpu_cores=snap.get("cpu_cores"),available_ram_bytes=snap.get("ram",{}).get("MemAvailable"),loadavg1=snap.get("loadavg",[None])[0]))
  choose|=bool(matched)
 if e.get("event")=="STAGE_RETURN" and any(x in e.get("command",[]) for x in ("score","render")):
  cpu_events.append(e);choose=True
 if choose:selected.append(line+b"\n")
# This newly frozen subset may be hashed. It never asserts a whole-hot-log hash.
snapshot=OUT/"completed_stage_event_snapshot.jsonl";snapshot.write_bytes(b"".join(selected));record(snapshot,"immutable_past_event_subset")
controller_provenance=dict(path=str(controller),read_complete_prefix_bytes=last,whole_source_hash="NOT_COMPUTED_LIVE_FILE",selection="Only completed registered GPU-wave worker intervals and CPU stage returns; copied original complete lines",snapshot_path=str(snapshot),snapshot_sha256=digest(snapshot))
for stage in stage_rows:
 rr=[r for r in resources if r["wave"]==stage["wave"] and r["gpu_used_mib"] is not None];ww=[i for i in intervals if i["wave"]==stage["wave"]]
 stage.update(actual_worker_start_counts=[i["workers_at_start"] for i in ww],max_observed_workers=max((i["workers_at_start"] for i in ww),default=0),resource_samples=len(rr),
  gpu_name=rr[0]["gpu_name"] if rr else None,device_peak_used_mib=max((r["gpu_used_mib"] for r in rr),default=None),device_min_free_mib=min((r["gpu_free_mib"] for r in rr),default=None),maximum_gpu_utilization=max((r["gpu_utilization"] for r in rr),default=None),cpu_cores=max((r["cpu_cores"] for r in rr),default=None),minimum_available_ram_bytes=min((r["available_ram_bytes"] for r in rr),default=None))
 check(stage["wave"]+"_at_most_two_workers",stage["max_observed_workers"]<=stage["registered_max_workers"]==2)
 check(stage["wave"]+"_zero_return_codes",all(i["returncode"]==0 for i in ww))
# Source registries are checked separately by version; no merged rewrite.
registry_rows=[];source_rows=[];distinct={}
registry_names=["source_hashes.json","curriculum_additional_source_hashes.json","confirmation_additional_source_hashes.json","confirmation_sampling_adapter_sources.json","all_baselines_analysis_sources.json","all_baselines_analysis_v2_sources.json"]
for name in registry_names:
 d=document("registrations/"+name,"versioned_source_registry")
 if name=="source_hashes.json":entries=d;schema="original97_frozen_scientific_sources"
 elif "sources" in d:entries=d["sources"];schema=d.get("schema")
 else:
  entries={"tools/compare_recipe_all_baselines.py":d["tool_sha256"],"tests/test_compare_recipe_all_baselines.py":d["test_sha256"]};schema=d["schema"]
 failures=[]
 for relative,expected in entries.items():
  path=(REPO/relative).resolve()
  if not path.is_relative_to(REPO):raise ValueError("Registered source escapes repository")
  actual=digest(path);ok=actual==expected
  source_rows.append(dict(registry=name,schema=schema,relative_path=relative,expected_sha256=expected,actual_sha256=actual,status="PASS" if ok else "FAIL"))
  if not ok:failures.append(relative)
  if relative in distinct:check("cross_registry_same_source_identity",distinct[relative]==expected,relative)
  distinct[relative]=expected
 registry_rows.append(dict(registry=name,schema=schema,files=len(entries),status="PASS" if not failures else "FAIL",failures=failures))
 check(name+"_source_hashes",not failures,failures)
 if name=="source_hashes.json":check("exactly97_original_sources",len(entries)==97)
# Recheck the actual training authorizations, including transitive new training dependencies.
for name in ("stage1_training.json","curriculum_training.json","confirmation_training.json"):
 auth=document("registrations/"+name,"training_authorization")
 for relative,expected in auth["source_hashes"].items():check(name+"_source_"+relative,digest(REPO/relative)==expected)
# Read only completed score/gallery receipts, never score tables, quality metrics or alias maps.
cpu=[];galleries=[];registrations=[]
for phase in ("main","course","confirm"):
 reg=document("phases/"+phase+"/registration.json","phase_budget_registration");registrations.append(reg)
 path=RUN/"phases"/phase/"evaluation/scoring_receipt.json"
 if path.exists() and read(path).get("status")=="COMPLETE":
  rec=read(path);record(path,"completed_scoring_metadata_only")
  cpu.append(dict(phase=phase,kind="scoring",status="COMPLETE",registered=rec["registered"],scored=rec["scored"],workers=rec["workers"],threads_per_worker=rec["cpu_threads_per_worker"],wall_seconds=rec["wall_seconds"],model_forwards=rec["model_forwards"],auxiliary_blocked=rec["auxiliary_blocked"]))
  check(phase+"_all_registered_outputs_scored",rec["registered"]==rec["scored"]==reg["expected_outputs"])
 else:cpu.append(dict(phase=phase,kind="scoring",status="PENDING_NOT_HASHED"))
 relative="comparisons/curriculum/gallery/gallery.json" if phase=="course" else "phases/"+phase+"/gallery/gallery.json"
 path=RUN/relative
 if path.exists() and read(path).get("status")=="COMPLETE":
  rec=read(path);record(path,"completed_gallery_metadata_only")
  galleries.append(dict(phase=phase,status="COMPLETE",primary_pngs=rec.get("primary_page_count",rec.get("page_count")),supplemental_pngs=rec.get("supplemental_page_count",0),metadata_path=str(path),wall_seconds=None,wall_time_evidence="No separate duration in gallery receipt; existing CPU stage returns retained without inference"))
 else:galleries.append(dict(phase=phase,status="PENDING_NOT_HASHED",metadata_path=str(path)))
unique=sum(r["unique_updates"] for r in training);physical=sum(r["physical_updates"] for r in training)
forward=sum(r["forward_calls"] for r in training);backward=sum(r["backward_calls"] for r in training)
new=sum(r["new_outputs"] for r in sampling);reused=sum(r["reused_outputs"] for r in sampling)
sample_forward=sum(r["actual_successful_new_forwards"] for r in sampling)
registered=sum(r["expected_outputs"] for r in registrations)
failures=dict(training_failed_attempts=sum(r["failed_attempts"] for r in training),training_failed_calls=sum(r["failed_calls"] for r in training),training_unfinished_calls=sum(r["unfinished_calls"] for r in training),training_resumes=sum(r["resumes"] for r in training),training_replayed_updates=sum(r["replayed_updates"] for r in training),sampling_failed_calls=sum(audits[p+"_sampling"]["failed_call_events"] for p in ("main","course","confirm")),sampling_unfinished_forward_calls=sum(audits[p+"_sampling"]["unfinished_forward_calls"] for p in ("main","course","confirm")),sampling_unfinished_rollouts=sum(audits[p+"_sampling"]["unfinished_rollouts"] for p in ("main","course","confirm")),sampling_cell_retries=sum(audits[p+"_sampling"]["cell_retry_count"] for p in ("main","course","confirm")),sampling_dispatch_retry_markers=sum(audits[p+"_sampling"]["dispatch_retry_markers"] for p in ("main","course","confirm")),sampling_failure_markers=sum(audits[p+"_sampling"]["failure_markers"] for p in ("main","course","confirm")),global_oom_downgrades=max(d["oom_downgrades"] for d in waves.values()))
check("training_total42000_derived_from_seven_actual_arms",unique==physical==42000)
check("training_F_B_each336000_from_actual_ledgers",forward==backward==8*physical==336000)
check("output_budget640_includes72_reused_cells",new+reused==registered==640 and new==568 and reused==72)
check("sampling_forward28400_from_actual_ledgers",sample_forward==50*new==28400)
check("all_failure_replay_OOM_counters_zero",not any(failures.values()),failures)
limits=dict(max_new_unique_updates=42000,max_registered_output_cells=640,max_new_rollouts=640-72,max_new_sampling_forwards=(640-72)*50)
check("no_budget_exceeded",unique<=limits["max_new_unique_updates"] and registered<=limits["max_registered_output_cells"] and new<=limits["max_new_rollouts"] and sample_forward<=limits["max_new_sampling_forwards"])
totals=dict(new_unique_updates=unique,new_physical_updates=physical,training_forward_calls=forward,training_backward_calls=backward,new_rollouts=new,reused_RAW_cells=reused,registered_distinct_cells=registered,actual_new_sampling_forwards=sample_forward,scored_cells=sum(r.get("scored",0) for r in cpu))
result=dict(schema=VERSION,status="PASS" if all(c["status"]=="PASS" for c in checks) else "FAIL",totals=totals,budget_limits=limits,failures=failures,
 training=training,sampling=sampling,gpu_stages=stage_rows,cpu_scoring=cpu,gallery_metadata=galleries,cpu_stage_return_events=cpu_events,
 source_registries=registry_rows,distinct_registered_source_files=len(distinct),controller_provenance=controller_provenance,
 source_inputs=sources,checks=checks,model_calls_by_this_aggregation=0,scoring_calls_by_this_aggregation=0,render_calls_by_this_aggregation=0,
 quality_tables_read=False,private_mapping_read=False,weights_rehashed=False,live_files_hashed=False,
 default_status=dict(observation="No default-changing action appears among registered experiment jobs; original registered config/source SHA unchanged",scope_limit="No independent global default/runtime preference audit; this aggregation did not change defaults"),
 git_status=dict(observation="This aggregation executed no Git operation; registered experiment jobs invoke no Git command",scope_limit="Not a proof that no unrelated/manual Git operation occurred"),
 timing_limits="Per-wave elapsed walls and resource samples include concurrency. Their sum is active-wave accounting, not end-to-end project elapsed time or an isolated algorithm benchmark.",
 memory_limits="Device-used includes desktop/context/other GPU processes; PyTorch allocated/reserved is a separate narrower scope.",
 versions_note="Original97 sources, curriculum additions, confirmation additions, sampling adapter freeze, and v1/v2 analysis corrections remain independent registries; no original result/source was overwritten.",
 elapsed_seconds=time.time()-started)
csv_write(OUT/"training_arms.csv",training);csv_write(OUT/"sampling_models.csv",sampling);csv_write(OUT/"gpu_stages.csv",stage_rows);csv_write(OUT/"gpu_workers.csv",intervals);csv_write(OUT/"resource_samples.csv",resources);csv_write(OUT/"source_verification.csv",source_rows);csv_write(OUT/"checks.csv",checks);csv_write(OUT/"cpu_scoring.csv",cpu)
record(Path(__file__),"aggregation_source")
output("accounting.json",result)
lines=["# Final execution accounting", "", "Status: "+result["status"], "", "Counts are derived from completed per-phase ledgers and cross-checked against frozen budgets.", "",
 "| Item | Actual |", "|---|---:|"]+["| "+k+" | "+str(v)+" |" for k,v in totals.items()]
lines += ["", "Training forward and backward counts are separate. Reused RAW cells incur no new rollout/forward here.", "", "| Phase | Kind | Wall seconds | Max workers | Peak device-used MiB | Min free MiB |", "|---|---|---:|---:|---:|---:|"]
lines += ["| {phase} | {kind} | {wall_seconds:.3f} | {max_observed_workers} | {device_peak_used_mib} | {device_min_free_mib} |".format(**x) for x in stage_rows]
lines += ["", "All recorded failure, unfinished call, replay, retry and OOM counters: "+str(failures), "", "Source registries: "+str(registry_rows), "", result["timing_limits"], result["memory_limits"], "", result["versions_note"], "", "No quality tables, aliases, weights, model operations, scoring or rendering were used by this aggregation.", "", "Default/Git statements are scoped observations, not global absence proofs."]
(OUT/"ACCOUNTING.md").write_text("\n".join(lines)+"\n",encoding="utf8")
manifest={str(p.relative_to(OUT)):digest(p) for p in sorted(OUT.iterdir()) if p.is_file()}
output("artifact_hashes.json",manifest)
print(json.dumps(dict(status=result["status"],totals=totals,checks=len(checks),failures=[c for c in checks if c["status"]!="PASS"],sources=registry_rows,accounting_sha256=digest(OUT/"accounting.json")),ensure_ascii=False))
