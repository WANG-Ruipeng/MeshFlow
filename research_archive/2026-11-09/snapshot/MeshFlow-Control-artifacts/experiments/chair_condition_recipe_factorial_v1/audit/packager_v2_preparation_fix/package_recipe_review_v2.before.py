#!/usr/bin/env python3
"""Assemble complete scalar/code review evidence after the finite experiment finishes.
No model, scoring, sampling, deletion, Git or external upload operations.
"""
from __future__ import annotations
import argparse, hashlib, json, zipfile, difflib
from pathlib import Path

ROOT_NAME="chair_condition_recipe_factorial_v1"
VERSION="recipe_factorial_review_bundle_v2"
def sha(p):
    h=hashlib.sha256()
    with Path(p).open("rb") as f:
        for b in iter(lambda:f.read(1024*1024),b""):h.update(b)
    return h.hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding="utf-8-sig"))
def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument("--run",required=True)
    args=ap.parse_args();run=Path(args.run).resolve()
    if run.name!=ROOT_NAME:raise ValueError("Wrong experiment root")
    final=read(run/"FINAL_STATUS.json")
    if final.get("status")!="COMPLETE":raise ValueError("Final task is not complete")
    if not (run/"EXPERIMENT_REPORT.md").is_file():raise ValueError("Final report missing")
    if final.get("default_changed") or final.get("automatic_git_operation"):raise ValueError("Unexpected default/Git action")
    source_repo=run.parents[2]/"MeshFlow-Control"
    # artifact root and source checkout are siblings beneath /mnt/e.

    dest=run/"delivery";dest.mkdir(exist_ok=True)
    selected={};excluded=[]
    def add(p,arc=None):
        p=Path(p)
        if not p.is_file():return
        arc=arc or str(p.relative_to(run)).replace("\\","/")
        if arc in selected and selected[arc]!=p:raise ValueError("Archive name collision")
        selected[arc]=p
    for p in run.glob("*.md"):add(p)
    for name in ("FINAL_STATUS.json","RUN_STATE.json","REQUEST.txt"):
        add(run/name)
    for folder in ("analysis","comparisons","registrations","source_snapshot","source_external_snapshot"):
        for p in (run/folder).rglob("*"):
            if p.is_file() and p.name!="FINAL_REPORT_DRAFT_ZH.md" and p.suffix.lower() in (".csv",".json",".md",".py",".txt",".diff",".html"):add(p)
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
    for relative in ("src/meshflow_control/data/recipe_confirmation.py","src/meshflow_control/training/recipe_confirmation.py"):
        add(source_repo/relative,"additional_source/"+relative)
    image_manifest=[]
    for p in sorted(run.glob("**/*.png")):
        if "gallery" not in str(p.parent) and "comparison" not in str(p.parent):continue
        image_manifest.append(dict(path=str(p),bytes=p.stat().st_size,sha256=sha(p)))
    metadata=dict(schema=VERSION,status="COMPLETE",
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
    with zipfile.ZipFile(target,"w",compression=zipfile.ZIP_DEFLATED,compresslevel=9) as z:
        for a,p in sorted(selected.items()):z.write(p,a)
        z.writestr("BUNDLE_MANIFEST.json",manifest)
        z.writestr("code.diff",code_diff)
        z.writestr("READ_FIRST.txt","Read EXPERIMENT_REPORT.md. All per-output CSVs and actual scoring code are included.\n"
            "Weights, RAW arrays, full gallery PNGs and large exposure ledgers remain at paths/hashes in BUNDLE_MANIFEST.json.\n"
            "Gallery HTML in this review bundle is an index; open the canonical local gallery for actual images.\n"
            "No model default, Git commit or remote repository was changed automatically.\n")
    with zipfile.ZipFile(target) as z:
        if z.testzip() is not None:raise ValueError("ZIP CRC failure")
        names=set(z.namelist())
        mandatory={"EXPERIMENT_REPORT.md","phases/main/evaluation/per_output.csv","source_snapshot/tools/score_recipe_factorial_conflict.py","code.diff"}
        if not mandatory.issubset(names):raise ValueError("Mandatory review evidence missing")
        for phase in ("course","confirm"):
            if (run/"phases"/phase/"evaluation/per_output.csv").exists() and "phases/"+phase+"/evaluation/per_output.csv" not in names:raise ValueError(phase+" rows missing")
        if final.get("confirmation_status")=="COMPLETE" and "phases/confirm/evaluation/per_output.csv" not in names:
            raise ValueError("Completed confirmation evidence missing")
    result=dict(status="COMPLETE",zip=str(target),bytes=target.stat().st_size,sha256=sha(target),
        files=len(selected)+3,local_only_images=len(image_manifest),new_model_calls=0,new_scores=0)
    (dest/"PACKAGING_RECEIPT.json").write_text(json.dumps(result,indent=2),encoding="utf8")
    print(json.dumps(result))
if __name__=="__main__":main()
