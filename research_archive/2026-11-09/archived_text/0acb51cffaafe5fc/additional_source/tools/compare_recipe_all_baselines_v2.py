#!/usr/bin/env python3
"""V2 identity guard around frozen eight-label table aggregation; no rescoring."""
from __future__ import annotations
import argparse
import importlib.util
import json
from pathlib import Path

_spec=importlib.util.spec_from_file_location("_all_baselines_frozen_v1",Path(__file__).with_name("compare_recipe_all_baselines.py"))
v1=importlib.util.module_from_spec(_spec);_spec.loader.exec_module(v1)
base=v1.base
VERSION="recipe_all_baselines_registered_binding_v2"
V1_SHA="54edc6ee7224aa81b697222a8c5effc2e6badbb2dcffc6537ede1b95e2ee43d1"


def registered_bindings(rows, requests, query_registration, requests_sha256):
    if query_registration.get("status")!="FROZEN":
        raise ValueError("Query registration is not frozen")
    if query_registration.get("identity",{}).get("requests_sha256")!=requests_sha256:
        raise ValueError("Query registration request identity differs")
    qby={q["request_id"]:q for q in requests}
    if len(qby)!=64 or len(requests)!=64:
        raise ValueError("Exactly64 unique registered requests required")
    query_rows=query_registration["rows"]
    queries={r["condition_id"]:r for r in query_rows}
    if len(query_rows)!=16 or len(queries)!=16 or set(queries)!={q["condition_id"] for q in requests}:
        raise ValueError("Exactly16 registered query conditions required")
    expected={(label,rid) for label in v1.LABELS for rid in qby}
    seen=set()
    for row in rows:
        key=row["label"],row["request_id"]
        if key in seen or key not in expected:
            raise ValueError("Duplicate/unregistered score binding")
        seen.add(key)
        q=qby[key[1]];query=queries[q["condition_id"]]
        for field in ("condition_id","parent_id","ratio","C_sha256"):
            if query[field]!=q[field]:
                raise ValueError("Query condition binding differs: "+field)
        if query.get("query_count")!=2048 or query.get("all_four_noises_share_same_bank") is not True:
            raise ValueError("Registered query count/noise sharing differs")
        identity=row["identity"]
        if identity.get("input_sha256")!=q["input_sha256"]:
            raise ValueError("Scored input hash differs from registered request")
        if identity.get("request_id")!=q["request_id"] or identity.get("label")!=row["label"]:
            raise ValueError("Inner score request/label identity differs")
        if identity.get("query_sha256")!=query["query_sha256"] or row.get("query_sha256")!=query["query_sha256"]:
            raise ValueError("Inner/outer query hash differs from registered query")
        if row.get("query_weight_denominator")!=query["weight_denominator"]:
            raise ValueError("Scored denominator differs from registered query")
        if row.get("registered_C_queries")!=query["query_count"] or row.get("L0")!=query["identity"]["L0"]:
            raise ValueError("Scored query count/scale differs from registration")
        if row.get("registered_free_faces")!=q["N"]-q["K"]:
            raise ValueError("Free denominator differs from registered request")
    if seen!=expected or len(rows)!=512:
        raise ValueError("All512 score bindings required")
    return dict(registered_outputs=512,registered_requests=64,registered_query_conditions=16,
                input_query_binding="PASS",queries_regenerated=0)


def analyze(out=base.DEFAULT_OUT):
    out=base.resolve(out).resolve()
    if out.name!=base.ROOT_NAME or out.parent.name!="experiments":
        raise ValueError("Wrong experiment root")
    if base.digest(v1.__file__)!=V1_SHA:
        raise ValueError("Frozen v1 comparison source changed")
    files={}
    for phase in ("main","course"):
        p=out/"phases"/phase
        files[phase+"_registration"]=p/"registration.json"
        files[phase+"_scores"]=p/"evaluation/per_output.json"
        files[phase+"_receipt"]=p/"evaluation/scoring_receipt.json"
    def pending():
        return dict(status="PENDING",missing_files=[str(p) for p in files.values() if not p.is_file()],
                    new_model_calls=0,new_scoring_calls=0,automatic_selection=False)
    if any(not p.is_file() for p in files.values()):return pending()
    regs={p:base.read_json(files[p+"_registration"]) for p in ("main","course")}
    request_sha=regs["main"]["requests_sha256"]
    if regs["course"]["requests_sha256"]!=request_sha:
        raise ValueError("Cross-phase request hash differs")
    for phase in regs:
        files[phase+"_requests"]=base.resolve(regs[phase]["requests_path"])
    files["query_registration"]=out/"evaluation/query_registrations"/(request_sha+".json")
    if any(not p.is_file() for p in files.values()):return pending()
    for phase in regs:
        if base.digest(files[phase+"_requests"])!=request_sha:
            raise ValueError("Actual request file hash differs")
    query_registration=base.read_json(files["query_registration"])
    for i,query in enumerate(query_registration["rows"]):
        path=base.resolve(query["query_path"]);files["query_bank_"+str(i)]=path
        if not path.is_file():return pending()
        if base.digest(path)!=query["query_sha256"]:
            raise ValueError("Actual registered query file hash differs")
    sources={str(Path(p).resolve()):base.digest(p) for p in (
        __file__,v1.__file__,v1.course.__file__,base.__file__)}
    provenance=dict(version=VERSION,
        input_files={k:dict(path=str(p),sha256=base.digest(p)) for k,p in files.items()},
        source_files=sources,operations="Registered table identities and file hashes only")
    identity=base.canonical_hash(provenance)
    dest=out/"analysis/all_baselines"/(VERSION+"_"+identity[:16])
    requests=base.read_json(files["main_requests"])["rows"]
    main_rows=base.read_json(files["main_scores"])["rows"]
    course_rows=base.read_json(files["course_scores"])["rows"]
    # Full frozen metadata/status/pair validation remains authoritative.
    rows=v1.validate_rows(regs["main"],regs["course"],requests,main_rows,course_rows,
        base.read_json(files["main_receipt"]),base.read_json(files["course_receipt"]))
    facts=registered_bindings(rows,requests,query_registration,request_sha)
    # The frozen v1 implements unchanged numerical aggregation and output checks.
    result=v1.analyze(out)
    if result["status"]!="COMPLETE":
        raise ValueError("Validated complete panel did not produce complete aggregation")
    for item in provenance["input_files"].values():
        if base.digest(item["path"])!=item["sha256"]:
            raise ValueError("Registered input changed during guarded aggregation")
    for path,sha in sources.items():
        if base.digest(path)!=sha:
            raise ValueError("Guarded analysis source changed during execution")
    receipt=dict(status="COMPLETE",version=VERSION,input_identity=identity,provenance=provenance,
        binding=facts,analysis_dir=result["analysis_dir"],
        frozen_v1_analysis_receipt_sha256=base.digest(Path(result["analysis_dir"])/"analysis_receipt.json"),
        preserved_outputs=512,new_model_calls=0,new_scoring_calls=0,automatic_selection=False,
        composite_score=None,changed_metric_definitions=False,changed_numerical_aggregation=False)
    rp=dest/"binding_receipt.json"
    if rp.exists():
        if base.read_json(rp)!=receipt:
            raise ValueError("Immutable v2 binding receipt differs")
    else:
        base.write_json(rp,receipt)
    base.write_json(out/"analysis/all_baselines/latest_v2.json",dict(
        binding_receipt_path=str(rp),binding_receipt_sha256=base.digest(rp),
        analysis_dir=result["analysis_dir"],input_identity=identity))
    return dict(result,binding_version=VERSION,binding_receipt_path=str(rp),
                binding_receipt_sha256=base.digest(rp))


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("--out",default=base.DEFAULT_OUT)
    result=analyze(parser.parse_args(argv).out);print(json.dumps(result,ensure_ascii=False),flush=True)
    return 0 if result["status"]=="COMPLETE" else 2


if __name__=="__main__":raise SystemExit(main())
