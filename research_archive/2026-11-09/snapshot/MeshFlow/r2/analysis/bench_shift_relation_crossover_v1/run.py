"""Aggregate completed crossover collector output, without models or scorers.

The established 177 definitions and noise/condition/parent aggregation are
imported read-only. No old source files are copied or modified.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path

from ..bench_joint_shift_v1 import run as core

ROOT=core.ROOT
DEFAULT=ROOT/"experiments/r2_bench_shift_relation_crossover_v1"
DEST="analysis_crossover_v1"
SCHEMA="BENCH_SHIFT_RELATION_CROSSOVER_ANALYSIS_V1"
COLLECTOR_SCHEMA="BENCH_SHIFT_RELATION_CROSSOVER_COLLECTION_V1"
CANDIDATE="SHIFT_RELATION_T050"
REFERENCES=("RELATION_V1_S1500","JOINT_SHIFT_S1500")
LABELS=(*REFERENCES,CANDIDATE)
STAGES=core.STAGES
PRIOR=ROOT/"experiments/r2_bench_joint_shift_v1"
FIXED_PATH=Path(__file__).with_name("fixed_sets.json")


def verify_hashes(values):
    for p,h in values.items():
        if core.sha(core.resolve(p))!=h:
            raise ValueError("Protected source changed: "+str(p))


def fixed_sets(path=FIXED_PATH):
    d=core.read(path)
    assert d["schema"]=="CROSSOVER_PRE_REGISTERED_ADOPTION_SETS_V1"
    verify_hashes({d["source_path"]:d["source_sha256"]})
    sets={k:{r["request_id"] for r in d[k]} for k in ("gain9","original_positive8")}
    assert len(sets["gain9"])==9 and len(sets["original_positive8"])==8
    assert not sets["gain9"] & sets["original_positive8"]
    return d,sets


def load_inputs(exp):
    exp=core.resolve(exp).resolve()
    required=[exp/p for p in ("results/completion.json","results/per_output.csv",
                              "registration.json","eval_manifest.json")]
    missing=[str(p) for p in required if not p.exists()]
    if missing:
        return dict(status="WAITING_FOR_COMPLETED_COLLECTOR",missing=missing)
    done=core.read(required[0])
    if done.get("status")!="COMPLETE":
        return dict(status="WAITING_FOR_COMPLETED_COLLECTOR",collector_status=done.get("status"))
    assert done["schema"]==COLLECTOR_SCHEMA
    assert done["whole_mesh_boundary_edges_preserved"] is True
    assert done["input_C_boundary_field"]=="C_boundary_edges"
    hashes={str(p):core.sha(p) for p in required}
    output=required[1]
    expect=next((h for p,h in done["files"].items() if core.resolve(p).resolve()==output),None)
    assert expect==core.sha(output)
    assert done["eval_manifest_sha256"]==core.sha(required[3])
    assert done["registration_sha256"]==core.sha(required[2])
    metrics=done["metric_fields"]
    assert len(metrics)==len(set(metrics))==177
    assert "boundary_edges" in metrics and "C_boundary_edges" not in metrics
    queries=core.read(required[3])["requests"]
    old_queries=core.read(PRIOR/"eval_manifest.json")["requests"]
    assert len(queries)==len(old_queries)==48
    # New prefix/cache metadata may be added; every inherited field is unchanged.
    for q,old in zip(queries,old_queries):
        for k,v in old.items():
            assert q[k]==v,(q["request_id"],k)
    qmap={q["request_id"]:q for q in queries}
    membership_path=exp/"data/evaluation_membership.json"
    if not membership_path.exists():
        membership_path=PRIOR/"data/evaluation_membership.json"
    membership={r["request_id"]:r for r in core.read(membership_path)}
    hashes[str(membership_path)]=core.sha(membership_path)
    hashes[str(PRIOR/"eval_manifest.json")]=core.sha(PRIOR/"eval_manifest.json")
    fdoc,fsets=fixed_sets()
    hashes[str(FIXED_PATH)]=core.sha(FIXED_PATH)
    hashes[fdoc["source_path"]]=fdoc["source_sha256"]
    assert all(s<=set(qmap) for s in fsets.values())
    rows=core.csv_rows(output)
    assert len(rows)==done["rows"]==432
    expected={(l,s,r) for l in LABELS for s in STAGES for r in qmap}
    seen=set()
    for r in rows:
        key=(r["label"],r["stage"],r["request_id"])
        assert key not in seen and key in expected
        seen.add(key)
        q=qmap[r["request_id"]]
        for k in ("condition_id","parent_id","split","part_type"):
            assert r[k]==q[k],(key,k)
        for k in ("N","K","noise_index"):
            r[k]=int(r[k]); assert r[k]==int(q[k])
        for m in metrics:
            assert m in r,(key,m)
            r[m]=None if r[m]=="" else float(r[m])
            assert r[m] is None or math.isfinite(r[m]),(key,m)
        assert r["I_status"]==r["ConflictCov_C_status"]
        for k in ("in_current_training_parent_pool","exact_current_training_condition"):
            r[k]=membership[r["request_id"]][k]
        for name,ids in fsets.items():
            r["prior_"+name]=r["request_id"] in ids
    assert seen==expected
    return dict(status="READY",rows=rows,metric_fields=metrics,source_hashes=hashes,
                fixed_sets=fsets,fixed_set_document=fdoc)


def pair_summaries(pairs,metrics):
    # Old helper computes values generically but stamps its old candidate name.
    # Correct only that descriptive field; do not mutate imported module globals.
    result=core.pair_summaries(pairs,metrics)
    for row in result:
        row["candidate"]=CANDIDATE
    return result


def aggregate(rows,metrics,sets):
    summary,key,support=core.summarize(rows,metrics)
    pairs=core.pair_rows(rows,metrics,candidate=CANDIDATE,references=REFERENCES)
    ps=pair_summaries(pairs,metrics)
    parents,pp=core.parent_tables(rows,pairs,metrics)
    fs,fp=[],[]
    for name,ids in sets.items():
        for label in LABELS:
            for stage in STAGES:
                rs=[r for r in rows if r["label"]==label and r["stage"]==stage and r["request_id"] in ids]
                for m in metrics:
                    domains=[("ALL_RECORDED_SUPPORT",rs)]
                    if m.startswith("ConflictCov_C"):
                        domains.append(("PASS_ONLY",[r for r in rs if r["I_status"]=="PASS"]))
                    for domain,group in domains:
                        fs.append(dict(fixed_set=name,label=label,stage=stage,support=domain,
                                       **core.statistics_for(group,m)))
        selected=[r for r in pairs if r["request_id"] in ids]
        for row in pair_summaries(selected,metrics):
            fp.append(dict(fixed_set=name,**row))
    return dict(all_metrics_summary=summary,key_metrics=key,I_support_summary=support,
                paired_per_output=pairs,paired_metric_summary=ps,
                per_parent_all_metrics=parents,paired_per_parent_all_metrics=pp,
                fixed_set_metrics=fs,fixed_set_paired_metrics=fp,per_output_with_membership=rows)


def run(exp=DEFAULT,check_inputs=False):
    exp=core.resolve(exp).resolve()
    d=load_inputs(exp)
    if d["status"]!="READY": return d
    if check_inputs:
        return dict(status="READY",rows=len(d["rows"]),numeric_fields=len(d["metric_fields"]),model_calls=0,scoring_calls=0)
    dest=exp/DEST; completion=dest/"completion.json"
    code={str(p):core.sha(p) for p in (Path(__file__),Path(core.__file__),Path(core.__file__).with_name("stream_audit.py"))}
    if completion.exists():
        old=core.read(completion)
        assert old["collector_input_hashes"]==d["source_hashes"] and old["analysis_code_hashes"]==code
        verify_hashes(old["files"]); verify_hashes(d["source_hashes"])
        return old
    tables=aggregate(d["rows"],d["metric_fields"],d["fixed_sets"])
    for name,table in tables.items(): core.write_csv(dest/(name+".csv"),table)
    core.write_json(dest/"definitions.json",dict(schema=SCHEMA,metric_fields=d["metric_fields"],
        labels=LABELS,stages=STAGES,training_updates=0,
        primary_weighting="noise -> condition -> parent equal weighting; ALL directly over 12 parents",
        units="Unchanged original units; fractions 0..1, no source GT ranking",
        I="PASS_ONLY parent macro and BOTH_PASS paired domains; LOW_SUPPORT never filled with zero",
        references="JOINT_SHIFT adoption-retention reference; RELATION late-stage quality reference; both preserved",
        fixed_sets="9 prior DEV adoption gains + 8 shared prior DEV positives, frozen before candidate; not new metric",
        boundary="boundary_edges is whole mesh metric; C_boundary_edges is input metadata",
        absolute_costs="FF/CF pair counts, actual/deleted faces and area retained alongside fractions",
        visual_review="Fresh actual review required; never infer labels from scores or reuse old labels",
        no_best_checkpoint_or_extra_samples=True))
    core.write_json(dest/"fixed_sets.json",d["fixed_set_document"])
    files={str(p):core.sha(p) for p in sorted(dest.iterdir()) if p.is_file() and p.name!="completion.json"}
    verify_hashes(d["source_hashes"])
    result=dict(status="COMPLETE",schema=SCHEMA,completed_utc=datetime.now(timezone.utc).isoformat(),
        rows=432,original_numeric_fields=177,paired_rows=len(tables["paired_per_output"]),
        collector_input_hashes=d["source_hashes"],analysis_code_hashes=code,files=files,
        new_model_calls=0,new_scoring_calls=0,original_scores_modified=False,
        visual_complete_claim=False,experiment_complete_claim=False)
    core.write_json(completion,result)
    return result


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--experiment",default=str(DEFAULT)); p.add_argument("--check-inputs",action="store_true")
    a=p.parse_args(); result=run(a.experiment,a.check_inputs)
    print(json.dumps({k:v for k,v in result.items() if k not in {"files","collector_input_hashes","analysis_code_hashes"}},indent=2))
