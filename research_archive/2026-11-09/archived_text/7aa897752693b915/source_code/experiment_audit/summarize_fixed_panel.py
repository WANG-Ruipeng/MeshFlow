#!/usr/bin/env python3
"""Arithmetic-only complete-panel summary of already computed schedule40 scores.

Frozen field list; no geometry calls, model imports, scoring, statistical tests,
confidence intervals, favorable-field selection, or changes to source tables.
"""
from __future__ import annotations
import argparse
from collections import Counter
import csv
import hashlib
from itertools import combinations
import json
import math
from pathlib import Path
import statistics
import time

LABELS = ("START5000","HIST_FM6000","HIST_FREE_ALIGN6000","FM_ONLY","EARLY","LATE")
ROOT_NAME = "chair_early_alignment_schedule40_v1"
STRATA = ("overall8","existing_A40_6","expanded_C_2")
NUMERIC_FIELDS = (
    ("CF_proper_pair_count","main","CF"),
    ("CF_coplanar_pair_count","main","CF"),
    ("CF_proper_affected_free_fraction","main","CF"),
    ("CF_unresolved_or_degenerate_pair_count","main","CF"),
    ("FF_proper_pair_count","main","FF"),
    ("FF_coplanar_pair_count","main","FF"),
    ("FF_proper_affected_free_fraction","main","FF"),
    ("FF_unresolved_or_degenerate_pair_count","main","FF"),
    ("FF_illegal_pair_count","main","FF"),
    ("FF_illegal_affected_free_fraction","main","FF"),
    ("free_minimum_angle_lt5_fraction","main","shape"),
    ("free_angle_p05_degrees","main","shape"),
    ("free_q_gt10_fraction","main","shape"),
    ("free_q_mean","main","finite_subset_shape"),
    ("free_near_degenerate_count","main","shape"),
    ("free_near_degenerate_fraction","main","shape"),
    ("free_exact_zero_area_count","main","shape"),
    ("free_exact_zero_area_fraction","main","shape"),
    ("free_total_surface_area","main","descriptive_area"),
    ("EXACT_COORD_vertices","main","descriptive_topology"),
    ("EXACT_COORD_edges","main","descriptive_topology"),
    ("EXACT_COORD_boundary_edges","main","descriptive_topology"),
    ("EXACT_COORD_nonmanifold_edges","main","descriptive_topology"),
    ("EXACT_COORD_same_direction_two_face_edges","main","descriptive_topology"),
    ("EXACT_COORD_face_components","main","descriptive_topology"),
    ("EXACT_COORD_CF_shared_edges","main","descriptive_topology"),
    ("free_exact_duplicate_excess_fraction","main","exact_duplicate"),
    ("whole_total_surface_area","main","descriptive_area"),
    ("whole_near_degenerate_count","main","shape"),
    ("whole_minimum_angle_lt5_fraction","main","shape"),
    ("ConflictCov_C","conflict","I"),
    ("ConflictCov_C_upper","conflict","I"),
    ("ConflictCov_C_cross","conflict","I"),
    ("ConflictCov_C_dup","conflict","I"),
    ("ConflictCov_C_cross_and_dup","conflict","I"),
    ("ConflictCov_C_uncertain_fraction","conflict","I"),
    ("ConflictCov_C_unresolved_pairs","conflict","I_diagnostic"),
    ("ConflictCov_C_coplanar_illegal_pairs","conflict","I_diagnostic"),
    ("ConflictCov_C_proper_pairs","conflict","I_diagnostic"),
    ("ConflictCov_C_segment_extraction_failures","conflict","I_diagnostic"),
)
EXCLUDED_NUMERIC = {
    "main": ("C_total_surface_area","C_near_degenerate_count","C_exact_zero_area_count",
             "RAW_SOUP_face_components","RAW_SOUP_CF_shared_edges"),
    "conflict": ("ConflictCov_C_interior_samples","ConflictCov_C_interior_weight_fraction"),
}
EXCLUDED_BOOLEAN = ("EXACT_COORD_closed_two_faces_per_edge",)
METRIC_BY_NAME = {name:dict(name=name,origin=origin,family=family) for name,origin,family in NUMERIC_FIELDS}


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def sha(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda:handle.read(1024*1024),b""): h.update(block)
    return h.hexdigest()


def save_json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+".tmp")
    tmp.write_text(json.dumps(value,sort_keys=True,indent=2,allow_nan=False)+"\n",encoding="utf-8")
    tmp.replace(path)


def save_csv(path,rows):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    rows=list(rows);fields=list(dict.fromkeys(k for row in rows for k in row))
    tmp=path.with_name(path.name+".tmp")
    with tmp.open("w",newline="",encoding="utf-8-sig") as handle:
        writer=csv.DictWriter(handle,fieldnames=fields);writer.writeheader()
        for row in rows:
            writer.writerow({k:json.dumps(v,sort_keys=True) if isinstance(v,(dict,list,tuple)) else v for k,v in row.items()})
    tmp.replace(path)


def finite_numeric(value):
    return isinstance(value,(int,float)) and not isinstance(value,bool) and math.isfinite(value)


def observation(row,metric,parent):
    name,origin,family=metric
    status=None if row is None else row.get("status")
    metric_status=None if row is None else row.get("metric_status")
    metrics={} if row is None else row.get("metrics",{})
    value=metrics.get(name)
    execution_ok=status==("PASS" if origin=="main" else "SCORED")
    available=execution_ok and finite_numeric(value)
    partial=origin=="conflict" and metric_status=="PARTIAL"
    low_support=origin=="conflict" and metric_status=="LOW_SUPPORT"
    unknown=False;unknown_reason=[]
    if family=="CF" and metrics.get("CF_unresolved_or_degenerate_pair_count",0)>0:
        unknown=True;unknown_reason.append("CF_UNRESOLVED_OR_DEGENERATE_PAIRS_RECORDED")
    if family=="FF" and metrics.get("FF_unresolved_or_degenerate_pair_count",0)>0:
        unknown=True;unknown_reason.append("FF_UNRESOLVED_OR_DEGENERATE_PAIRS_RECORDED")
    if origin=="conflict":
        if metrics.get("ConflictCov_C_uncertain_fraction",0)>0:
            unknown=True;unknown_reason.append("ORIGINAL_I_UNKNOWN_COVERAGE_POSITIVE")
        if metrics.get("ConflictCov_C_segment_extraction_failures",0)>0:
            unknown=True;unknown_reason.append("ORIGINAL_I_PROPER_WITNESS_EXTRACTION_FAILURE")
    finite_subset=family=="finite_subset_shape" and metrics.get("free_exact_zero_area_count",0)>0
    reason=None
    if not available:
        reason=("MISSING_SOURCE_ROW" if row is None else
                "SOURCE_EXECUTION_"+str(status)+": "+str(row.get("reason","")) if not execution_ok else
                "MISSING_METRIC_FIELD" if name not in metrics else
                "NULL_OR_NONFINITE_METRIC")
    return dict(parent_id=parent,metric=name,origin=origin,family=family,
        available=available,value=value if available else None,
        source_status=status,source_metric_status=metric_status if origin=="conflict" else "NOT_SEPARATELY_PROVIDED",
        missing_reason=reason,partial=partial,low_support=low_support,
        unknown=unknown,unknown_reason=unknown_reason,finite_subset_mean=finite_subset)


def summarize_scalar(items):
    values=[x["value"] for x in items if x["available"]]
    n=len(items);valid=len(values)
    return dict(registered_denominator=n,effective_denominator=valid,
        missing_count=n-valid,partial_count=sum(x["available"] and x["partial"] for x in items),
        low_support_count=sum(x["available"] and x["low_support"] for x in items),
        unknown_count=sum(x["available"] and x["unknown"] for x in items),
        finite_subset_mean_count=sum(x["available"] and x["finite_subset_mean"] for x in items),
        source_status_counts=dict(Counter(str(x["source_status"]) for x in items)),
        source_metric_status_counts=dict(Counter(str(x["source_metric_status"]) for x in items)),
        missing_reasons=dict(Counter(x["missing_reason"] for x in items if not x["available"])),
        missing_parent_ids=[x["parent_id"] for x in items if not x["available"]],
        partial_parent_ids=[x["parent_id"] for x in items if x["available"] and x["partial"]],
        unknown_parent_ids=[x["parent_id"] for x in items if x["available"] and x["unknown"]],
        mean_available=statistics.fmean(values) if values else None,
        median_available=statistics.median(values) if values else None,
        minimum_available=min(values) if values else None,
        maximum_available=max(values) if values else None,
        mean_all_registered=statistics.fmean(values) if valid==n and n else None,
        panel_computation_status="ALL_REGISTERED_VALUES_PRESENT" if valid==n else "PARTIAL_NOT_FULL_REGISTERED_PANEL",
        status_note="Computation completeness does not erase source PARTIAL/LOW_SUPPORT/unknown flags")


def paired(first,second,first_label,second_label):
    available=first["available"] and second["available"]
    delta=first["value"]-second["value"] if available else None
    direction="lower" if available and delta<0 else "higher" if available and delta>0 else "equal" if available else "unavailable"
    return dict(parent_id=first["parent_id"],metric=first["metric"],origin=first["origin"],family=first["family"],
        first=first_label,second=second_label,delta_direction="first minus second",
        first_value=first["value"],second_value=second["value"],delta=delta,
        comparison=direction,available=available,
        first_source_status=first["source_status"],second_source_status=second["source_status"],
        first_metric_status=first["source_metric_status"],second_metric_status=second["source_metric_status"],
        partial=available and (first["partial"] or second["partial"]),
        low_support=available and (first["low_support"] or second["low_support"]),
        unknown=available and (first["unknown"] or second["unknown"]),
        finite_subset_mean=available and (first["finite_subset_mean"] or second["finite_subset_mean"]),
        missing_reason=None if available else dict(first=first["missing_reason"],second=second["missing_reason"]))


def summarize_pairs(items):
    values=[r["delta"] for r in items if r["available"]]
    return dict(registered_pair_denominator=len(items),effective_pair_denominator=len(values),
        missing_pair_count=len(items)-len(values),
        lower_count=sum(r["comparison"]=="lower" for r in items),
        equal_count=sum(r["comparison"]=="equal" for r in items),
        higher_count=sum(r["comparison"]=="higher" for r in items),
        partial_pair_count=sum(r["partial"] for r in items),
        low_support_pair_count=sum(r["low_support"] for r in items),
        unknown_pair_count=sum(r["unknown"] for r in items),
        finite_subset_mean_pair_count=sum(r["finite_subset_mean"] for r in items),
        missing_parent_ids=[r["parent_id"] for r in items if not r["available"]],
        mean_delta_available=statistics.fmean(values) if values else None,
        median_delta_available=statistics.median(values) if values else None,
        minimum_delta_available=min(values) if values else None,
        maximum_delta_available=max(values) if values else None,
        mean_delta_all_registered=statistics.fmean(values) if len(values)==len(items) and items else None,
        equality="Exact equality of saved numeric values; no added tolerance",
        interpretation="lower/equal/higher are numerical directions, not wins or preferred outcomes")


def load_table(path,origin,requests):
    source=read(path)
    by={}
    for row in source["rows"]:
        label,parent=row["label"],row["parent_id"]
        if label not in LABELS or parent not in requests: raise ValueError("Unregistered output in "+str(path))
        key=(label,parent)
        if key in by: raise ValueError("Duplicate registered output in "+str(path))
        request=requests[parent]
        for field in ("request_id","N","K"):
            if row[field]!=request[field]: raise ValueError("Input identity changed: "+field)
        for field,value in row.get("metrics",{}).items():
            if isinstance(value,bool):
                if field not in EXCLUDED_BOOLEAN: raise ValueError("Unregistered boolean field "+field)
            elif isinstance(value,(int,float)) or value is None:
                if field not in METRIC_BY_NAME and field not in EXCLUDED_NUMERIC[origin]:
                    raise ValueError("Unregistered numeric metric; do not silently omit: "+field)
                if field in METRIC_BY_NAME and METRIC_BY_NAME[field]["origin"]!=origin:
                    raise ValueError("Metric source differs: "+field)
        by[key]=row
    return by


def run(out):
    out=Path(out).resolve()
    if out.name!=ROOT_NAME or out.parent.name!="experiments": raise ValueError("Wrong fixed experiment directory")
    preparation=read(out/"audit/summarize_fixed_panel_preparation.json")
    if preparation["script_sha256"]!=sha(__file__) or preparation["actual_main_table_exists"] or preparation["actual_conflict_table_exists"]:
        raise ValueError("Field-list source was not fixed before score tables existed")
    request_path=out/"evaluation/requests.json"
    manifest=read(request_path)
    if tuple(manifest["labels"])!=LABELS: raise ValueError("Registered model labels changed")
    requests={r["parent_id"]:r for r in manifest["rows"]}
    if len(requests)!=8: raise ValueError("Expected eight registered parents")
    existing=[p for p,r in requests.items() if r["condition_construction"]=="EXISTING_A40_CONNECTED_CONDITION"]
    expanded=[p for p,r in requests.items() if r["condition_construction"]=="OLD_C_PLUS_NEAREST_FREE_CENTROIDS_EXACT_COORDINATES"]
    if len(existing)!=6 or len(expanded)!=2: raise ValueError("Expected preregistered6/2 construction strata")
    groups={"overall8":sorted(requests),"existing_A40_6":sorted(existing),"expanded_C_2":sorted(expanded)}
    main_path=out/"evaluation/per_output.json";conflict_path=out/"evaluation/per_output_conflict.json"
    if not main_path.is_file() or not conflict_path.is_file():
        raise FileNotFoundError("Both actual main and conflict per-output tables are required; no scoring is run")
    tables={"main":load_table(main_path,"main",requests),"conflict":load_table(conflict_path,"conflict",requests)}
    observations={}
    for label in LABELS:
        for parent in requests:
            for metric in NUMERIC_FIELDS:
                observations[(label,parent,metric[0])]=observation(tables[metric[1]].get((label,parent)),metric,parent)
    all_per_parent=[]
    model_summaries=[]
    pair_per_parent=[]
    pair_summaries=[]
    for group,parents in groups.items():
        for label in LABELS:
            for name,origin,family in NUMERIC_FIELDS:
                items=[observations[label,parent,name] for parent in parents]
                model_summaries.append(dict(stratum=group,label=label,metric=name,origin=origin,family=family,
                                            **summarize_scalar(items)))
                all_per_parent.extend(dict(stratum=group,label=label,**x) for x in items)
        for first,second in combinations(LABELS,2):
            for name,origin,family in NUMERIC_FIELDS:
                items=[paired(observations[first,parent,name],observations[second,parent,name],first,second) for parent in parents]
                pair_per_parent.extend(dict(stratum=group,**x) for x in items)
                pair_summaries.append(dict(stratum=group,first=first,second=second,metric=name,origin=origin,family=family,
                    delta_direction="first minus second",**summarize_pairs(items)))
    diagnostics=[]
    for parent,request in sorted(requests.items()):
        record=dict(parent_id=parent,N=request["N"],K=request["K"],
                    condition_construction=request["condition_construction"])
        for origin,fields in EXCLUDED_NUMERIC.items():
            for field in fields:
                values={label:tables[origin].get((label,parent),{}).get("metrics",{}).get(field) for label in LABELS}
                available=[v for v in values.values() if finite_numeric(v)]
                record[field+"_by_label"]=values
                record[field+"_equal_among_available"]=not available or all(x==available[0] for x in available)
        diagnostics.append(record)
    destination=out/"audit/fixed_panel_summary"
    catalog=dict(schema="schedule40_fixed_numeric_field_summary_v1",created_unix=time.time(),
        script_path=str(Path(__file__).resolve()),script_sha256=sha(__file__),
        numeric_fields=[dict(name=n,origin=o,family=f) for n,o,f in NUMERIC_FIELDS],
        fields_fixed_before_scores=True,field_count=len(NUMERIC_FIELDS),labels=LABELS,
        strata=groups,input_table_hashes={str(p):sha(p) for p in (request_path,main_path,conflict_path)},
        excluded_input_or_construction_fields=EXCLUDED_NUMERIC,excluded_boolean_flags=EXCLUDED_BOOLEAN,
        statistical_tests=False,confidence_intervals=False,new_geometric_metrics=False,
        model_forwards=0,scorer_calls=0,
        aggregation="Equal parent weight within each explicit construction stratum; no face pooling",
        limitations=[
            "Eight observed DEV parents and one noise each; not a multiple-seed significance study.",
            "Overall8 mixes six originalA40 and two expandedC conditions; inspect6/2 strata and paired-parent rows.",
            "Overall8 is the union of existing6 and expanded2; stratum tables repeat evidence and must not be pooled as16 independent parents.",
            "Missing values remain in registered denominators; complete-panel means are null when any numeric value is missing.",
            "PARTIAL/LOW_SUPPORT/known detector uncertainty are retained; computed lower/upper values are not discarded.",
            "CF/FF unknown counts mark outputs with recorded unresolved/degenerate pairs, not reclassified illegal pairs.",
            "I unknown flags reflect positive unknown coverage or missing proper-segment witnesses, not a statistical confidence interval.",
            "q_mean is the original finite-q subset mean; zero-area outputs receive a finite_subset_mean flag.",
            "Topology counts and areas are descriptive; numerical lower/higher is not a claim of quality direction.",
            "C-only geometry/support and raw-soup construction constants are input diagnostics, not model-quality outcomes.",
            "No re-scoring, geometry processing, field selection by outcome, test or CI is performed."])
    save_json(destination/"summary_manifest.json",catalog)
    save_json(destination/"per_model_by_stratum.json",dict(rows=model_summaries))
    save_csv(destination/"per_model_by_stratum.csv",model_summaries)
    save_csv(destination/"per_output_registered_metrics.csv",all_per_parent)
    save_csv(destination/"paired_parent_differences.csv",pair_per_parent)
    save_csv(destination/"paired_summary_by_stratum.csv",pair_summaries)
    save_json(destination/"input_diagnostics.json",dict(rows=diagnostics))
    receipt=dict(status="COMPLETE_ARITHMETIC_SUMMARY",metrics=40,strata=3,models=6,
        model_summary_rows=len(model_summaries),paired_parent_rows=len(pair_per_parent),
        paired_summary_rows=len(pair_summaries),source_tables_modified=False,
        source_main_present_rows=len(tables["main"]),source_conflict_present_rows=len(tables["conflict"]),
        unknown_and_missing_retained=True,output=str(destination),script_sha256=sha(__file__))
    save_json(destination/"receipt.json",receipt)
    return receipt


def self_test():
    metric=("ConflictCov_C","conflict","I")
    a=observation(dict(status="SCORED",metric_status="PASS",metrics={"ConflictCov_C":.2}),metric,"a")
    b=observation(dict(status="SCORED",metric_status="PARTIAL",
        metrics={"ConflictCov_C":.4,"ConflictCov_C_uncertain_fraction":.1}),metric,"b")
    c=observation(None,metric,"c")
    result=summarize_scalar([a,b,c])
    assert result["registered_denominator"]==3 and result["effective_denominator"]==2
    assert result["missing_count"]==1 and result["partial_count"]==1 and result["unknown_count"]==1
    assert abs(result["mean_available"]-.3)<1e-15 and result["mean_all_registered"] is None
    values=[]
    for i,(left,right) in enumerate(((.1,.2),(.3,.3),(.5,.3))):
        first=observation(dict(status="SCORED",metric_status="PASS",metrics={"ConflictCov_C":left}),metric,str(i))
        second=observation(dict(status="SCORED",metric_status="PASS",metrics={"ConflictCov_C":right}),metric,str(i))
        values.append(paired(first,second,"EARLY","FM_ONLY"))
    values.append(paired(c,a,"EARLY","FM_ONLY"))
    result=summarize_pairs(values)
    assert (result["lower_count"],result["equal_count"],result["higher_count"])==(1,1,1)
    assert result["registered_pair_denominator"]==4 and result["effective_pair_denominator"]==3
    assert result["missing_pair_count"]==1 and result["mean_delta_all_registered"] is None
    cf=observation(dict(status="PASS",metrics={"CF_proper_affected_free_fraction":.1,
        "CF_unresolved_or_degenerate_pair_count":1}),
        ("CF_proper_affected_free_fraction","main","CF"),"p")
    assert cf["available"] and cf["unknown"] and cf["value"]==.1
    assert not finite_numeric(True) and not finite_numeric(float("nan"))
    assert len(NUMERIC_FIELDS)==40 and len(METRIC_BY_NAME)==40
    assert set(groups for _,_,groups in NUMERIC_FIELDS)
    return dict(status="PASS",checks="artificial scalar tables only; registered/effective denominators, partial/unknown retention, all numerical direction counts, missing pairs, boolean/nonfinite exclusion",
                geometry_calls=0,model_forwards=0,actual_results_read=False)


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out",default=str(Path(__file__).resolve().parents[1]))
    parser.add_argument("--self-test",action="store_true")
    args=parser.parse_args(argv)
    result=self_test() if args.self_test else run(args.out)
    print(json.dumps(result),flush=True)


if __name__=="__main__":main()
