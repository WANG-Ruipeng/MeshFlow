#!/usr/bin/env python3
"""Assemble complete scalar/code review evidence after the finite experiment finishes.
No model, scoring, sampling, deletion, Git or external upload operations.
"""
from __future__ import annotations
import argparse, hashlib, json, zipfile, difflib, csv
from pathlib import Path

ROOT_NAME="chair_condition_recipe_factorial_v1"
VERSION="recipe_factorial_review_bundle_v3"
def sha(p):
    h=hashlib.sha256()
    with Path(p).open("rb") as f:
        for b in iter(lambda:f.read(1024*1024),b""):h.update(b)
    return h.hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding="utf-8-sig"))

PHASE_LABELS = {
    "main": ("START5000", "HIST_FM6000", "R1_40_H_ALL", "R2_MIX_H_ALL", "R3_40_H_LATE", "R4_MIX_H_LATE"),
    "course": ("C20_40_MIX", "C40_20_MIX"),
    "confirm": ("C20_40_MIX", "R2_MIX_H_ALL"),
}
CORE_MANDATORY = {
    "EXPERIMENT_REPORT.md", "FINAL_STATUS.json", "MODEL_INDEX.md", "analysis/model_index.json",
    "analysis/recipe_factorial_delivery_tables_v1/complete_per_output_640.csv",
    "analysis/recipe_factorial_delivery_tables_v1/reference_A20_per_output_all.csv",
    "analysis/recipe_factorial_delivery_tables_v1/receipt.json",
    "evaluation/requests.json", "evaluation/confirmation_plan.json",
    "source_snapshot/tools/evaluate_recipe_factorial.py",
    "source_snapshot/tools/score_recipe_factorial_conflict.py",
    "source_snapshot/src/meshflow_control/evaluation/control.py",
    "source_snapshot/src/meshflow_control/evaluation/pair_audit.py",
    "additional_source/tools/evaluate_recipe_confirmation.py",
    "additional_source/tools/analyze_recipe_confirmation_results.py",
    "additional_source/tools/merge_recipe_confirmation_review.py",
    "code.diff",
    "source_external_snapshot/third_party/trimesh/LICENSE",
    "additional_source/tools/package_recipe_review_v3.py",
}
for _phase in PHASE_LABELS:
    CORE_MANDATORY.update({
        "phases/"+_phase+"/registration.json",
        "phases/"+_phase+"/evaluation/per_output.csv",
        "phases/"+_phase+"/evaluation/scoring_receipt.json",
        "audit/visual_review/"+_phase+"_complete/per_output.csv",
        "audit/visual_review/"+_phase+"_complete/coverage.json",
    })

def require(ok, message):
    if not ok:
        raise ValueError(message)

def read_csv(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as handle:
        reader=csv.DictReader(handle)
        rows=list(reader)
    require(reader.fieldnames is not None and len(reader.fieldnames)==len(set(reader.fieldnames)), "CSV header missing/duplicated")
    require(not any(None in row or any(value is None for value in row.values()) for row in rows), "Malformed delivery CSV")
    return rows

def selected_file_allowed(archive_path):
    parts=Path(archive_path).parts
    if any("draft" in part.lower() for part in parts):
        return False
    return Path(archive_path).suffix.lower() not in {
        ".pt", ".pth", ".safetensors", ".npz", ".npy", ".ply", ".obj", ".bin", ".pkl"}

def evaluation_metadata_paths(run):
    # Fixed small JSON definitions only; input/query arrays are deliberately local.
    paths=[run/"evaluation/requests.json", run/"evaluation/confirmation_plan.json"]
    paths.extend(sorted((run/"evaluation/query_registrations").glob("*.json")))
    return paths

def phase_table_identity(rows, labels, requests):
    expected={(label,q["request_id"]) for label in labels for q in requests}
    actual=[(r["label"],r["request_id"]) for r in rows]
    require(len(requests)==64 and len({q["request_id"] for q in requests})==64, "Exactly64 registered phase requests")
    require(len(actual)==64*len(labels) and len(actual)==len(set(actual)) and set(actual)==expected,
            "Complete unique model/request cells required")
    return {key:row for key,row in zip(actual,rows)}

def visual_identity(rows,labels,requests,phase):
    expected={(label,q["request_id"]) for label in labels for q in requests}
    actual=[(r["model_label"],r["request_id"]) for r in rows]
    require(len(actual)==len(expected) and len(actual)==len(set(actual)) and set(actual)==expected,
            "Complete unique visual cells required: "+phase)

def validate_final_evidence(run):
    """Read existing final tables/metadata only when packaging is explicitly run."""
    phase_rows={};request_sets={};required=set(CORE_MANDATORY);counts={}
    for phase,labels in PHASE_LABELS.items():
        p=run/"phases"/phase;reg=read(p/"registration.json")
        require(set(reg["labels"])==set(labels) and len(reg["labels"])==len(labels), "Registered phase labels differ")
        request_path=Path(reg["requests_path"])
        require(request_path.is_relative_to(run), "Requests must belong to this local run")
        require(sha(request_path)==reg["requests_sha256"], "Registered phase request hash differs")
        required.add(request_path.relative_to(run).as_posix())
        requests=read(request_path)["rows"];request_sets[phase]=requests
        query_path=run/"evaluation/query_registrations"/(reg["requests_sha256"]+".json")
        query=read(query_path)
        require(query.get("status")=="FROZEN" and query.get("identity",{}).get("requests_sha256")==reg["requests_sha256"]
                and len(query.get("rows",[]))==16, "Registered16 query conditions missing")
        required.add(query_path.relative_to(run).as_posix())
        receipt=read(p/"evaluation/scoring_receipt.json");expected=64*len(labels)
        require(receipt.get("status")=="COMPLETE" and receipt.get("registered")==expected and receipt.get("scored")==expected,
                "Phase scoring incomplete: "+phase)
        rows=read_csv(p/"evaluation/per_output.csv")
        phase_rows[phase]=phase_table_identity(rows,labels,requests);counts[phase]=len(rows)
    combined_path=run/"analysis/recipe_factorial_delivery_tables_v1/complete_per_output_640.csv"
    combined=read_csv(combined_path);seen=set()
    for row in combined:
        phase=row["evidence_phase"];key=(row["label"],row["request_id"]);identity=(phase,*key)
        require(phase in phase_rows and key in phase_rows[phase] and identity not in seen, "Duplicate/unknown combined output")
        seen.add(identity)
        require(row["model_instance_id"]==phase+"/"+row["label"], "Phase/model instance identity differs")
        require(all(row.get(k)==v for k,v in phase_rows[phase][key].items()), "Combined original field string differs")
    require(len(combined)==len(seen)==640 and counts=={"main":384,"course":128,"confirm":128}, "Exactly640 outputs required")
    model_index=read(run/"analysis/model_index.json");instances=model_index.get("instances",[])
    expected_instances={phase+"/"+label for phase,labels in PHASE_LABELS.items() for label in labels}
    require(model_index.get("status")=="COMPLETE" and len(instances)==10
            and {r["instance"] for r in instances}==expected_instances, "Exactly10 evaluated model instances required")
    for r in instances:
        require(r["instance"]==r["phase"]+"/"+r["label"] and r.get("registered_outputs")==64, "Model instance metadata differs")
    visual_counts={}
    for phase in PHASE_LABELS:
        root=run/"audit/visual_review"/(phase+"_complete");coverage=read(root/"coverage.json")
        labels=PHASE_LABELS[phase]+(("R2_MIX_H_ALL",) if phase=="course" else ())
        rows=read_csv(root/"per_output.csv");visual_identity(rows,labels,request_sets[phase],phase)
        if phase=="main":
            c=coverage.get("expected_and_actual",{})
            require(coverage.get("status")=="COMPLETE" and c.get("unique_outputs")==384 and c.get("requests")==64
                    and c.get("original_pngs")==128 and coverage.get("all_128_original_png_hashes")=="PASS",
                    "Main visual coverage incomplete")
        else:
            require(coverage.get("status")=="PASS" and coverage.get("requests")==64
                    and coverage.get("unique_outputs")==len(rows) and coverage.get("primary_pngs")==128
                    and coverage.get("all_image_bytes_sha256_verified") is True, "Visual coverage incomplete: "+phase)
            if phase=="course":
                require(coverage.get("reused_R2_outputs")==64 and coverage.get("new_course_outputs")==128
                        and coverage.get("supplemental_pngs")==56, "Course visual reuse/supplement counts differ")
            else:
                require(coverage.get("supplemental_pngs")==0, "Unexpected unregistered confirmation supplement")
        visual_counts[phase]=len(rows)
    return dict(phase_rows=counts,model_instances=10,combined_rows=640,visual_rows=visual_counts,
                visual_reused_R2_rows=64,required_archive_members=sorted(required))

def verify_archive(target,metadata,generated,mandatory):
    """Validate CRC and every exact ZIP member against its manifest/source bytes."""
    with zipfile.ZipFile(target) as z:
        require(z.testzip() is None, "ZIP CRC failure")
        names=z.namelist()
        require(len(names)==len(set(names)), "Duplicate ZIP member")
        items=metadata["files"];by={item["archive_path"]:item for item in items}
        require(len(by)==len(items), "Duplicate manifest entry")
        require(set(names)==set(by)|set(generated), "ZIP/manifest member set differs")
        require(set(mandatory).issubset(names), "Mandatory review evidence missing")
        for name in names:
            with z.open(name) as handle:
                h=hashlib.sha256();size=0
                for block in iter(lambda:handle.read(1024*1024),b""):
                    h.update(block);size+=len(block)
            if name in by:
                require(h.hexdigest()==by[name]["sha256"] and size==by[name]["bytes"], "ZIP member SHA/size differs: "+name)
            else:
                data=generated[name]
                require(h.hexdigest()==hashlib.sha256(data).hexdigest() and size==len(data), "Generated ZIP member differs: "+name)
        require(json.loads(z.read("BUNDLE_MANIFEST.json"))==metadata, "Embedded bundle manifest differs")
    return dict(all_members_sha256_verified=True,crc_verified=True,verified_members=len(names))

def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument("--run",required=True)
    args=ap.parse_args();run=Path(args.run).resolve()
    if run.name!=ROOT_NAME:raise ValueError("Wrong experiment root")
    final=read(run/"FINAL_STATUS.json")
    if final.get("status")!="COMPLETE":raise ValueError("Final task is not complete")
    if not (run/"EXPERIMENT_REPORT.md").is_file():raise ValueError("Final report missing")
    if final.get("default_changed") or final.get("automatic_git_operation"):raise ValueError("Unexpected default/Git action")
    final_evidence=validate_final_evidence(run)
    source_repo=run.parents[2]/"MeshFlow-Control"
    # artifact root and source checkout are siblings beneath /mnt/e.

    dest=run/"delivery";dest.mkdir(exist_ok=True)
    selected={};excluded=[]
    def add(p,arc=None):
        p=Path(p)
        if not p.is_file():return
        arc=arc or str(p.relative_to(run)).replace("\\","/")
        if not selected_file_allowed(arc):return
        if arc in selected and selected[arc]!=p:raise ValueError("Archive name collision")
        selected[arc]=p
    for p in run.glob("*.md"):add(p)
    for name in ("FINAL_STATUS.json","RUN_STATE.json","REQUEST.txt"):
        add(run/name)
    for folder in ("analysis","comparisons","registrations","source_snapshot","source_external_snapshot"):
        for p in (run/folder).rglob("*"):
            if p.is_file() and p.name!="FINAL_REPORT_DRAFT_ZH.md" and p.suffix.lower() in (".csv",".json",".md",".py",".txt",".diff",".html"):add(p)
    for p in evaluation_metadata_paths(run):add(p)
    add(run/"source_external_snapshot/third_party/trimesh/LICENSE")
    # Keep all per-output metrics, subcomponents and paired tables. RAW/query arrays and weights stay local.
    for phase in ("main","course","confirm"):
        base=run/"phases"/phase
        if not base.exists():continue
        add(base/"registration.json")
        add(base/"preparation_binding.json")
        for p in (base/"checkpoints").glob("*.json"):add(p)
        for p in (base/"evaluation").glob("*"):
            if p.is_file() and p.suffix.lower() in (".csv",".json",".md"):add(p)
        for p in (base/"gallery").glob("*"):
            if p.is_file() and p.suffix.lower() in (".json",".html",".md"):add(p)
    # Audit tables are complete where compact; the large training exposure ledger stays canonical on disk.
    for p in (run/"audit").rglob("*"):
        if not p.is_file() or p.suffix.lower() not in (".csv",".json",".md",".py",".txt"):continue
        if any(x in p.parts for x in ("logs","waves","training_accounting_R2_R3_partial")):continue
        if p.stat().st_size>5_000_000:
            excluded.append(dict(path=str(p),bytes=p.stat().st_size,sha256=sha(p),reason="Large execution ledger retained locally"));continue
        add(p)
    for training in (run/"training",run/"confirmation/training"):
        for pattern in ("*/completed.json","*/initialization_audit.json","*/attempts/*/final_status.json"):
            for p in training.glob(pattern):add(p)
    for name in ("precompute_receipt.json","precompute_progress.json"):
        add(run/"confirmation/streams"/name)
    for p in (run/"confirmation/registrations").glob("*.json"):add(p)
    for p in (run/"confirmation/streams").glob("*.json"):
        if p.name in ("precompute_receipt.json","precompute_progress.json"):continue
        excluded.append(dict(path=str(p),bytes=p.stat().st_size,sha256=sha(p),reason="Full independent training stream retained locally"))
    for p in (run/"confirmation/streams/paired_inputs").glob("*.json"):
        excluded.append(dict(path=str(p),bytes=p.stat().st_size,sha256=sha(p),reason="Full independent paired inputs retained locally"))
    # Include additional adapters used after the original 97-file freeze, with separate live-source provenance.
    extended=[]
    for pattern in ("*recipe*curriculum*.py","*recipe*factorial*.py","*recipe*confirmation*.py","*recipe*all_baselines*.py"):
        for folder in ("tools","tests"):
            for p in (source_repo/folder).glob(pattern):
                arc="additional_source/"+str(p.relative_to(source_repo)).replace("\\","/")
                add(p,arc);extended.append(dict(archive_path=arc,path=str(p),sha256=sha(p)))
    add(source_repo/"tools/package_recipe_review.py","additional_source/tools/package_recipe_review.py")
    add(source_repo/"tools/package_recipe_review_v2.py","additional_source/tools/package_recipe_review_v2.py")
    add(source_repo/"tools/package_recipe_review_v3.py","additional_source/tools/package_recipe_review_v3.py")
    add(source_repo/"tests/test_package_recipe_review_v3.py","additional_source/tests/test_package_recipe_review_v3.py")
    for relative in ("src/meshflow_control/data/recipe_confirmation.py","src/meshflow_control/training/recipe_confirmation.py"):
        add(source_repo/relative,"additional_source/"+relative)
    image_manifest=[]
    for p in sorted(run.glob("**/*.png")):
        if "gallery" not in str(p.parent) and "comparison" not in str(p.parent):continue
        image_manifest.append(dict(path=str(p),bytes=p.stat().st_size,sha256=sha(p)))
    mandatory=set(final_evidence["required_archive_members"])
    require(mandatory.issubset(set(selected)|{"BUNDLE_MANIFEST.json","code.diff","READ_FIRST.txt"}), "Mandatory delivery source/table absent")
    metadata=dict(schema=VERSION,status="COMPLETE",final_evidence=final_evidence,
        content="All scalar metrics, paired parent/noise tables, complete actual visual notes, source code and provenance; no weights or RAW.",
        local_run=str(run),local_only_images=image_manifest,local_only_large_ledgers=excluded,
        additional_sources=extended,all_images_in_zip=False,
        code_diff_definition="Full text additions against /dev/null; not a fabricated diff from an unavailable original Git commit.",
        files=[dict(archive_path=a,bytes=p.stat().st_size,sha256=sha(p)) for a,p in sorted(selected.items())])
    code_parts=[]
    for a,p in sorted(selected.items()):
        if p.suffix==".py" and (a.startswith("source_snapshot/") or a.startswith("additional_source/")):
            text=p.read_text(encoding="utf-8-sig").splitlines(keepends=True)
            code_parts.extend(difflib.unified_diff([],text,fromfile="/dev/null",tofile=a))
    code_diff="".join(code_parts)
    manifest=json.dumps(metadata,ensure_ascii=False,indent=2)
    (dest/"BUNDLE_MANIFEST.json").write_text(manifest,encoding="utf8")
    (dest/"code.diff").write_text(code_diff,encoding="utf8")
    target=dest/"CHAIR_CONDITION_RECIPE_FACTORIAL_REVIEW.zip"
    generated={
        "BUNDLE_MANIFEST.json":manifest.encode("utf8"),
        "code.diff":code_diff.encode("utf8"),
        "READ_FIRST.txt":(
            "Read EXPERIMENT_REPORT.md. All640 per-output CSVs,10 model instances and actual scoring code are included.\n"
            "Visual notes cover384 main +192 course (including64 reused R2) +128 confirmation outputs; these are not704 new samples.\n"
            "Weights, RAW arrays, full gallery PNGs and large exposure ledgers remain at paths/hashes in BUNDLE_MANIFEST.json.\n"
            "Gallery HTML in this review bundle is an index; open the canonical local gallery for actual images.\n"
            "No model default, Git commit or remote repository was changed automatically.\n").encode("utf8")}
    with zipfile.ZipFile(target,"w",compression=zipfile.ZIP_DEFLATED,compresslevel=9) as z:
        for a,p in sorted(selected.items()):z.write(p,a)
        for a,data in generated.items():z.writestr(a,data)
    integrity=verify_archive(target,metadata,generated,mandatory)
    result=dict(status="COMPLETE",zip=str(target),bytes=target.stat().st_size,sha256=sha(target),
        files=len(selected)+3,local_only_images=len(image_manifest),new_model_calls=0,new_scores=0,
        final_evidence=final_evidence,**integrity)
    (dest/"PACKAGING_RECEIPT.json").write_text(json.dumps(result,indent=2),encoding="utf8")
    print(json.dumps(result))
if __name__=="__main__":main()
