#!/usr/bin/env python3
"""Copy complete phase tables into explicitly separated delivery tables; no rescoring."""
import argparse,csv,hashlib,io,json
from pathlib import Path
VERSION="recipe_factorial_delivery_tables_v1"
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p): return json.loads(p.read_text(encoding="utf-8-sig"))
def freeze(p,b):
    p.parent.mkdir(parents=True,exist_ok=True)
    if p.exists() and p.read_bytes()!=b: raise ValueError("Refuse overwriting different delivery evidence: "+str(p))
    if not p.exists(): p.write_bytes(b)
def combine(run,name,expected,required):
    fields=None; rows=[]; sources=[]
    for phase,count in expected.items():
        p=run/"phases"/phase/"evaluation"/name
        if not p.exists():
            if required: raise ValueError("Missing required completed table: "+str(p))
            sources.append(dict(phase=phase,status="MISSING",expected=count));continue
        with p.open(encoding="utf-8-sig",newline="") as f:
            reader=csv.DictReader(f); current=list(reader.fieldnames or []); values=list(reader)
        if fields is None:fields=current
        if fields!=current: raise ValueError("Original phase columns differ; do not silently normalize definitions")
        if required and len(values)!=count:raise ValueError("Wrong phase row count")
        if any(None in r or any(v is None for v in r.values()) for r in values):raise ValueError("Malformed original CSV")
        digest=sha(p); seen=set(); ratio_counts={str(ratio):sum(float(r["ratio"])==ratio for r in values) for ratio in (0.2,0.4)}
        for r in values:
            key=(r["label"],r["request_id"])
            if key in seen:raise ValueError("Duplicate original model/request")
            seen.add(key)
            rows.append(dict(evidence_phase=phase,model_instance_id=phase+"/"+r["label"],source_csv=str(p),source_csv_sha256=digest,**r))
        sources.append(dict(phase=phase,status="COMPLETE" if len(values)==count else "PARTIAL",path=str(p),sha256=digest,rows=len(values),expected=count,ratio_counts=ratio_counts))
    if not fields:return None,sources
    extra=["evidence_phase","model_instance_id","source_csv","source_csv_sha256"]
    if set(extra)&set(fields):raise ValueError("Provenance column collision")
    out=io.StringIO(newline="");writer=csv.DictWriter(out,fieldnames=extra+fields);writer.writeheader();writer.writerows(rows)
    return out.getvalue().encode("utf-8-sig"),sources
def main():
    ap=argparse.ArgumentParser();ap.add_argument("--run",required=True);a=ap.parse_args();run=Path(a.run).resolve()
    if run.name!="chair_condition_recipe_factorial_v1":raise ValueError("Wrong run")
    expected={"main":384,"course":128,"confirm":128}
    for phase,count in expected.items():
        receipt=read(run/"phases"/phase/"evaluation/scoring_receipt.json")
        if receipt.get("status")!="COMPLETE" or receipt.get("scored")!=count:raise ValueError("All phase scores must be complete")
    dest=run/"analysis"/VERSION;tables={}
    for name,counts,required,target in [
        ("per_output.csv",expected,True,"complete_per_output_640.csv"),
        ("reference_A20_per_output.csv",expected,True,"reference_A20_per_output_all.csv")]:
        payload,sources=combine(run,name,counts,required)
        if payload is not None:freeze(dest/target,payload)
        tables[target]=dict(sources=sources,bytes=None if payload is None else len(payload),sha256=None if payload is None else hashlib.sha256(payload).hexdigest())
    for t in tables.values():
        for s in t["sources"]:
            if "path" in s and sha(Path(s["path"]))!=s["sha256"]:raise ValueError("Source changed during pure table copy")
    receipt=dict(version=VERSION,status="COMPLETE",tables=tables,model_calls=0,new_scores=0,
        note="Original strings and fields preserved. model_instance_id includes phase: same recipe label in main/course versus confirm is a different training endpoint. No cross-stage same-noise pairing. Primary canonical-scale metrics and source-L0 A20 diagnostics remain separate. Reference table retains all 640 rows: 320 A20 diagnostic-domain rows and 320 A40 explicit NA rows; per-row status is authoritative.",
        implementation_sha256=sha(Path(__file__)))
    freeze(dest/"receipt.json",(json.dumps(receipt,ensure_ascii=False,indent=2)+"\n").encode())
    print(json.dumps({"status":"COMPLETE","path":str(dest),"primary_rows":640,"model_calls":0,"new_scores":0}))
if __name__=="__main__":main()
