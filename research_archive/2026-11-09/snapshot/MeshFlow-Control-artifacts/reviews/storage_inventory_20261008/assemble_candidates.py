from pathlib import Path
import json,csv
D=Path("/mnt/e/MeshFlow-Control-artifacts/reviews/storage_inventory_20261008")
rows=json.loads((D/"weights_inventory.json").read_text(encoding="utf8"))
by={r["linux_path"]:r for r in rows}
R=Path("/mnt/e/MeshFlow/native_t1/runs")
groups=[]
def group(name,paths,reason,loss,priority):
 found=[]
 for p in paths:
  r=by[str(p)]
  assert r["links"]==1,str(p)
  found.append(dict(path=r["path"],linux_path=r["linux_path"],bytes=r["bytes"],modified_utc=r["modified_utc"],device=r["device"],inode=r["inode"],links=r["links"],status="CANDIDATE_ONLY_NOT_DELETED"))
 groups.append(dict(name=name,priority=priority,count=len(found),logical_bytes=sum(r["bytes"] for r in found),reason=reason,consequence=loss,files=found))
group("P1_hybrid_stale_800",[R/"chair_hybrid_coupling_v1/checkpoints"/f"{a}_latest.pt" for a in ("OT_BASE","OT_HYBRID")],"Both metadata pilot800/total4800; corresponding pilot1000 endpoints retained","Cannot restore or newly sample the step800 state",1)
group("P2_historical_curve_checkpoints",[R/"chair_domain_control_v1/checkpoints"/f"{a}_phase{phase}_total{total}.pt" for a in ("T1","Geo","Full") for phase,total in ((1000,2000),(2500,3500))]+[R/"control_mvp48_budget_single_v1/checkpoints"/f"{a}_step{s}.pt" for a in ("T1","Geo","Full") for s in (2000,3500)],"Historical fixed intermediate evaluation weights; final endpoints, RAW, curves and CSV retained","Cannot rerun these 12 intermediate model states; original reports and outputs remain",2)
group("P3_old_two_parent_fit_probe",[R/"control_mvp48_budget_single_v1/checkpoints"/f"T1_fit2_probe_step{s}.pt" for s in (500,1000)],"Completed old two-parent fit diagnostic; outside current B continuation","Cannot rerun the two old probe checkpoints",2)
adapter=[Path(r["linux_path"]) for r in rows if "/experiments/mf_h1_local0/outputs/" in r["linux_path"] and ("adapter_" in Path(r["linux_path"]).name or Path(r["linux_path"]).name=="official_probe_resume_step5.pt")]
assert len(adapter)==22,len(adapter)
group("P4_legacy_adapter_probe",adapter,"Early independent adapter/probe routes; not current START/B; explicitly retained historically so separately decidable","Retires old adapter re-sampling/resume ability",3)
cache=dict(pycache=dict(directories=943,files=6504,bytes=81229831,rebuild="Python bytecode regenerated from retained package source"),extension_builds=dict(files=51,bytes=13215804,rebuild="Requires compatible C++/CUDA or CGAL toolchain; lower priority"),current_ot=dict(npz_files=8000,npz_bytes=19314005,json_files=16000,json_bytes=9717324,action="KEEP"),start_hybrid_ot=dict(npz_files=8000,npz_bytes=19304623,json_files=8000,json_bytes=8256145,action="KEEP"),old_ponq_environment=dict(bytes=93425098,action="CONDITIONAL_ARCHIVE_REVIEW",overlap="Includes 25819276 bytes also counted in pycache"),teacher_environment=dict(bytes=127248998,action="KEEP"),teacher_assets=dict(bytes=812871662,action="KEEP"))
result=dict(status="ASSESSMENT_COMPLETE_NO_DELETION",groups=groups,recommended_primary_batch=dict(groups=[g["name"] for g in groups[:3]],files=sum(g["count"] for g in groups[:3]),logical_bytes=sum(g["logical_bytes"] for g in groups[:3])),optional_adapter_batch=dict(files=22,logical_bytes=groups[-1]["logical_bytes"]),cache_inventory=cache,protected=["Official models/data","Both START hardlink names, counted as one file","Geo4000 and Geo1000 prior conditional initialization","All current contextual A/B/C training and inference weights","Historical native T1 and maintained N112 profile dependencies","All RAW, inputs, queries, scores, CSVs, reports, source and audit JSON"],execution_notes=["No deletion has been performed this turn","Read-only metadata only; no model deserialization/forward","Candidates are not claimed byte-identical duplicates or universally failed models","Revalidate absolute paths, file identity, full hash and protected assets before any future deletion","Existing historical retained-checkpoint rules remain in effect until a specific cleanup is authorized"])
(D/"cleanup_candidates.json").write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding="utf8")
csvrows=[dict(group=g["name"],priority=g["priority"],**r,reason=g["reason"],consequence=g["consequence"]) for g in groups for r in g["files"]]
with (D/"cleanup_candidates.csv").open("w",encoding="utf-8-sig",newline="") as f:
 w=csv.DictWriter(f,fieldnames=list(csvrows[0]));w.writeheader();w.writerows(csvrows)
print(json.dumps(dict(groups=[{k:v for k,v in g.items() if k!="files"} for g in groups],recommended_primary_batch=result["recommended_primary_batch"],total_four_groups_bytes=sum(g["logical_bytes"] for g in groups)),ensure_ascii=False,indent=2))
