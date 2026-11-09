#!/usr/bin/env python3
"""Create a compact, explicitly scoped cloud review subset of the verified full ZIP.
No model, geometry, score, rendering, extraction, deletion or Git operations.
"""
import argparse, csv, hashlib, io, json, zipfile
from pathlib import Path

def digest(data): return hashlib.sha256(data).hexdigest()
def file_hash(path):
    h=hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda:f.read(1024*1024),b""): h.update(b)
    return h.hexdigest()
def keep(name):
    p=Path(name)
    if len(p.parts)==1:
        return name in {"EXPERIMENT_REPORT.md","FINAL_STATUS.json","RUN_STATE.json","MODEL_INDEX.md",
          "DELIVERY_INDEX.md","PROTOCOL_CLARIFICATIONS.md","REQUEST.txt","code.diff"}
    if p.parts[0] in {"source_snapshot","source_external_snapshot","additional_source"}: return True
    if p.parts[0]=="analysis":
        return p.suffix in {".csv",".md"} or p.name in {
            "model_index.json","receipt.json","provenance.json","binding_receipt.json","latest_v2.json"}
    if p.parts[0]=="evaluation": return p.suffix==".json"
    if p.parts[0]=="registrations": return p.suffix in {".json",".md"}
    if p.parts[0]=="phases":
        return ("/evaluation/" in name and (p.suffix==".csv" or p.name=="scoring_receipt.json")) or (
          "/checkpoints/" in name and p.suffix==".json") or p.name in {"registration.json","preparation_binding.json"}
    if name.startswith("audit/visual_review/"):
        return len(p.parts)>2 and p.parts[2] in {"main_complete","course_complete","confirm_complete"} and p.suffix in {".csv",".json",".md"}
    if name.startswith("audit/final_execution_accounting/"):
        return p.suffix in {".csv",".json",".md"} and p.name!="gpu_events_snapshot.jsonl"
    if name.startswith("audit/final_report_review/"): return p.suffix in {".json",".md"}
    if name.startswith("audit/confirmation_independent_review/"):
        return p.suffix in {".csv",".md"} or p.name in {"audit.json","delivery_receipt.json","audit_scalars.py","table_counts.json"}
    if name.startswith("audit/course_independent_review/"):
        return p.name in {"FINDINGS.md","audit.json","source_hashes.csv"}
    return False

def main():
    ap=argparse.ArgumentParser();ap.add_argument("--run",required=True);args=ap.parse_args()
    run=Path(args.run).resolve();delivery=run/"delivery"
    full=delivery/"CHAIR_CONDITION_RECIPE_FACTORIAL_REVIEW.zip"
    receipt=json.loads((delivery/"PACKAGING_RECEIPT.json").read_text())
    assert receipt["status"]=="COMPLETE" and receipt["all_members_sha256_verified"] and receipt["crc_verified"]
    full_sha=file_hash(full);assert receipt["sha256"]==full_sha
    target=delivery/"CHAIR_CONDITION_RECIPE_FACTORIAL_CLOUD.zip"
    entries={};excluded=[]
    with zipfile.ZipFile(full) as src:
        metadata=json.loads(src.read("BUNDLE_MANIFEST.json"))
        originals={x["archive_path"]:x for x in metadata["files"]}
        required=set(metadata["final_evidence"]["required_archive_members"])
        for name in src.namelist():
            if not keep(name) and name not in required: excluded.append(name);continue
            data=src.read(name)
            if name in originals:
                assert digest(data)==originals[name]["sha256"] and len(data)==originals[name]["bytes"],name
            entries[name]=data
        required=set(metadata["final_evidence"]["required_archive_members"])
        assert required <= set(entries), sorted(required-set(entries))
        full_manifest=src.read("BUNDLE_MANIFEST.json")
    # Explicit output count checks; no metric recomputation.
    combined=list(csv.DictReader(io.StringIO(entries[
      "analysis/recipe_factorial_delivery_tables_v1/complete_per_output_640.csv"].decode("utf-8-sig"))))
    assert len(combined)==640 and len({(r["evidence_phase"],r["label"],r["request_id"]) for r in combined})==640
    for phase,n in (("main",384),("course",192),("confirm",128)):
        rows=list(csv.DictReader(io.StringIO(entries[f"audit/visual_review/{phase}_complete/per_output.csv"].decode("utf-8-sig"))))
        assert len(rows)==n
    license_name="source_external_snapshot/third_party/trimesh/LICENSE"
    assert license_name in entries and len(entries[license_name])>0
    manifest={"schema":"recipe_factorial_cloud_subset_v1","scope":
      "All scalar CSV tables, complete 640 per-output data, full parent/noise/stratum/LOO analysis CSVs, actual scoring/source code and licenses, complete 704 visual output observations, final reports and essential audit. Intermediate duplicate JSON, preparation/test history and detailed execution logs remain in the full bundle. No images, weights or RAW.",
      "full_zip":str(full),"full_zip_sha256":full_sha,"full_zip_bytes":full.stat().st_size,
      "new_model_calls":0,"new_scores":0,"distinct_outputs":640,"visual_rows":704,
      "files":[{"archive_path":n,"bytes":len(b),"sha256":digest(b)} for n,b in sorted(entries.items())],
      "omitted_full_bundle_members":excluded,
      "original_full_manifest":"FULL_BUNDLE_MANIFEST.json"}
    entries["CLOUD_MANIFEST.json"]=json.dumps(manifest,ensure_ascii=False,indent=2).encode()
    entries["FULL_BUNDLE_MANIFEST.json"]=full_manifest
    entries["READ_FIRST.txt"]=(
      "Read EXPERIMENT_REPORT.md. This is a compact review subset, not the complete execution archive.\n"
      "All640 output scalar rows and all analysis CSVs are retained, with actual scoring source and licenses.\n"
      "All704 visual observations are retained; these describe640 distinct outputs and are not independent human ratings.\n"
      "No weights, RAW or PNGs. Full bundle identity and omitted members are explicit in CLOUD_MANIFEST.json.\n"
      "The complete local PNG galleries and large execution evidence remain at the reported local paths.\n").encode()
    entries["CLOUD_PACKAGER.py"]=Path(__file__).read_bytes()
    with zipfile.ZipFile(target,"w",zipfile.ZIP_DEFLATED,compresslevel=9) as z:
        for name,data in sorted(entries.items()): z.writestr(name,data)
    with zipfile.ZipFile(target) as z:
        assert z.testzip() is None and set(z.namelist())==set(entries)
        for name,data in entries.items(): assert digest(z.read(name))==digest(data),name
    result={"status":"COMPLETE","zip":str(target),"bytes":target.stat().st_size,
      "under_10_decimal_MB":target.stat().st_size<10_000_000,"sha256":file_hash(target),
      "members":len(entries),"crc_verified":True,"all_members_sha256_verified":True,
      "full_zip_sha256":full_sha,"distinct_output_rows":640,"visual_output_rows":704,
      "actual_scoring_source":True,"external_license_included":True,"new_model_calls":0,"new_scores":0}
    (delivery/"CLOUD_PACKAGING_RECEIPT.json").write_text(json.dumps(result,indent=2))
    print(json.dumps(result))
if __name__=="__main__":main()
