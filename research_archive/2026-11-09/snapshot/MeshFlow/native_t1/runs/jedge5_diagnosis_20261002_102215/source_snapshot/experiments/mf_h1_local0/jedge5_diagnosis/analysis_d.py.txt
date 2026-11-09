"""Stage D scalar-only matched raw/EMA analysis; no torch/model/geometry imports."""
from pathlib import Path
import argparse, csv, hashlib, json, math, time
import numpy as np

EXPECTED_RAW={"R_B":"f006c49c8c70efa4bf2df6ca96431c844a25a1bdd38e2db4756115a37a23545b",
              "R_J":"b9c1df5326ad4815ec3c32ed20e08dc34ff6248ff05be2c412e051fcb728aae3"}
D_LABELS=("G0","R_B_raw250","R_J_raw250","R_B_EMA250","R_J_EMA250","R_B_EMA500","R_J_EMA500")
HIGHER={"fixed_R12_coverage_within_1pct_fraction","boundary_coverage_fraction","free_angle_p05_degrees","Delta0","Delta1","B"}
DESCRIPTIVE={"near_C_area_fraction","near_C_boundary_band_area_fraction","near_C_outside_boundary_band_area_fraction",
             "C_surface_area_fraction","exact_C_free_shared_edge_count","exact_C_free_shared_vertex_count",
             "C_inherited_nonmanifold_vertex_count","C_current_winding_conflict_count","shared_edge_component_count","free_total_surface_area"}
FULL_COMPARISONS={"B_EMA_minus_raw":{"R_B_EMA500":1,"B_raw3000":-1},
                  "J_EMA_minus_raw":{"R_J_EMA500":1,"J_raw3000":-1},
                  "J_minus_B_raw":{"J_raw3000":1,"B_raw3000":-1},
                  "J_minus_B_EMA":{"R_J_EMA500":1,"R_B_EMA500":-1},
                  "gap_DiD_EMA_minus_raw":{"R_J_EMA500":1,"R_B_EMA500":-1,"J_raw3000":-1,"B_raw3000":1}}


def _read(path,sources=None):
    path=Path(path);data=path.read_bytes()
    if sources is not None:sources[str(path)]=hashlib.sha256(data).hexdigest()
    return json.loads(data.decode("utf-8-sig"))


def _write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix(path.suffix+".tmp")
    temporary.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+"\n",encoding="utf-8");temporary.replace(path)


def _direction(metric):
    if metric in DESCRIPTIVE:return None
    if metric in HIGHER or (metric.startswith("Gamma_") and "_coverage_" in metric and metric.endswith("_fraction")):return "higher"
    return "lower"


def _identity(row):
    keys=("parent","patch","seed","noise_index","teacher_t","state_id")
    return {key:row[key] for key in keys if key in row}


def _stats(rows,metric,paired=False):
    present=[(r,float(r["scalars"][metric])) for r in rows if r["scalars"].get(metric) is not None]
    if not present:return dict(status="NOT_RUN",count=0,missing=len(rows),direction=_direction(metric))
    values=np.asarray([x for _,x in present],dtype=np.float64)
    if not np.isfinite(values).all():raise ValueError("Nonfinite saved scalar "+metric)
    direction=_direction(metric);lo,hi=float(values.min()),float(values.max())
    worst=lo if direction=="higher" else hi if direction=="lower" else None
    result=dict(status="AVAILABLE",count=len(values),missing=len(rows)-len(values),mean=float(values.mean()),median=float(np.median(values)),
                q25=float(np.quantile(values,.25)),q75=float(np.quantile(values,.75)),minimum=lo,maximum=hi,direction=direction,
                worst=worst,worst_cases=[_identity(r) for r,v in present if v==worst] if worst is not None else [],
                minimum_cases=[_identity(r) for r,v in present if v==lo],maximum_cases=[_identity(r) for r,v in present if v==hi])
    if paired:
        result.update(negative_count=int((values<0).sum()),zero_count=int((values==0).sum()),positive_count=int((values>0).sum()),
                      improved_count=int(((values>0) if direction=="higher" else (values<0)).sum()) if direction else None,
                      worse_count=int(((values<0) if direction=="higher" else (values>0)).sum()) if direction else None,
                      count_policy="Strict signed differences; descriptive metrics have no preferred direction")
    return result


def _groups(rows):
    groups={"all":rows}
    for field in ("parent","patch","seed","noise_index","teacher_t"):
        values=sorted({r[field] for r in rows if field in r})
        for value in values:groups[field+":"+str(value)]=[r for r in rows if r.get(field)==value]
    for parent in sorted({r["parent"] for r in rows if "parent" in r}):
        for patch in sorted({r["patch"] for r in rows if r.get("parent")==parent}):
            groups[f"parent:{parent}/patch:{patch}"]=[r for r in rows if r.get("parent")==parent and r.get("patch")==patch]
    if any("patch" in r for r in rows):
        for name,condition in (("training_C",lambda p:p==0),("other_observed_C",lambda p:p!=0)):
            selected=[r for r in rows if "patch" in r and condition(r["patch"])]
            if selected:groups["context:"+name]=selected
            for parent in sorted({r["parent"] for r in selected if "parent" in r}):
                groups[f"parent:{parent}/context:{name}"]=[r for r in selected if r["parent"]==parent]
    return groups


def summarize(rows,paired=False):
    names=sorted(set().union(*(r["scalars"] for r in rows))) if rows else []
    return {name:dict(count=len(group),metrics={metric:_stats(group,metric,paired) for metric in names}) for name,group in _groups(rows).items() if group}


def _index(rows,keys):
    index={tuple(row[k] for k in keys):row for row in rows}
    if len(index)!=len(rows):raise ValueError("Duplicate saved-case identity")
    return index


def contrast(by_label,weights,keys=("parent","patch","seed")):
    labels=list(weights);indices={name:_index(by_label[name],keys) for name in labels};first=indices[labels[0]]
    if any(set(x)!=set(first) for x in indices.values()):raise ValueError("Unmatched contrast case set")
    result=[]
    for key in sorted(first):
        source=first[key];names=set(source["scalars"])
        if any(set(x[key]["scalars"])!=names for x in indices.values()):raise ValueError("Contrast metric schema differs")
        values={}
        for metric in sorted(names):
            parts=[indices[label][key]["scalars"][metric] for label in labels]
            if any(value is None for value in parts):values[metric]=None
            elif len(labels)==4 and tuple(weights.values())==(1,-1,-1,1):
                values[metric]=(float(parts[0])-float(parts[1]))-(float(parts[2])-float(parts[3]))
            else:values[metric]=sum(float(weights[label])*float(indices[label][key]["scalars"][metric]) for label in labels)
        result.append(dict(_identity(source),scalars=values))
    return result


def bindings(rows):
    indexed=_index(rows,("parent","patch","seed"));out=[]
    for patch,seed in sorted({(r["patch"],r["seed"]) for r in rows}):
        a,b=indexed[0,patch,seed]["cross_remainder"],indexed[1,patch,seed]["cross_remainder"]
        e00,e10=a[0]["symmetric_rms_bbox_pct"],a[1]["symmetric_rms_bbox_pct"]
        e01,e11=b[0]["symmetric_rms_bbox_pct"],b[1]["symmetric_rms_bbox_pct"]
        if any(x is None for x in (e00,e10,e01,e11)):raise ValueError("Incomplete saved binding four-grid")
        d0,d1=e01-e00,e10-e11
        out.append(dict(patch=patch,seed=seed,scalars=dict(E00=e00,E01=e01,E11=e11,E10=e10,Delta0=d0,Delta1=d1,B=(d0+d1)/2),
                        both_directions_positive=d0>0 and d1>0))
    return out


def _assert_case_grid(rows,seeds,small=False):
    expected={(p,j,int(seed)) for p in (0,1) for j in ((0,4) if small else range(5)) for seed in (seeds[:1] if small else seeds)}
    if set(_index(rows,("parent","patch","seed")))!=expected:raise ValueError("Fixed D 4/20 case set differs")


def _score_rows(out,phase,label,sources):
    saved=_read(out/"scores"/phase/"results.json",sources)
    if saved.get("status")!="PASS":raise ValueError("Scoring phase not complete: "+phase)
    rows=saved["rows"]
    if any(r["model"]!=label for r in rows):raise ValueError("Score phase model differs")
    return rows


def _normalized(rows,label):
    return [dict(_identity(r),label=label,scalars=dict(r["scalars"]),**({"cross_remainder":r["cross_remainder"]} if "cross_remainder" in r else {})) for r in rows]


def _raw_teacher(out,model,sources):
    record=_read(out/"stage_c/teacher"/(model+"_bf16.json"),sources)
    if record.get("status")!="PASS" or record.get("phase")!="C" or record.get("precision")!="bf16":raise ValueError("Raw3000 teacher C record not reusable")
    return [r for r in record["rows"] if r["patch"] in (0,4) and r["noise_index"]==0]


def _teacher_rows(record,label):
    expected={(p,j,0,t) for p in (0,1) for j in (0,4) for t in (.1,.3,.5,.75,.9)}
    if len(record)!=20 or {(r["parent"],r["patch"],r["noise_index"],r["teacher_t"]) for r in record}!=expected:
        raise ValueError("Teacher curve differs from its fixed20 states")
    return [dict(_identity(r),label=label,shared_input_sha256=r["shared_input_sha256"],target_mapping_sha256=r["target_mapping_sha256"],
                 FM_coordinate_denominator=r["FM_coordinate_denominator"],scalars=dict(r["losses"],
                    weighted_Edge_actual=r["weighted_Edge_actual"],weighted_Surface_actual=r["weighted_Surface_actual"])) for r in record]


def _curve_comparisons():
    answer={}
    for model,arm in (("B","R_B"),("J","R_J")):
        raw250=arm+"_raw250";ema250=arm+"_EMA250";raw500=model+"_raw3000";ema500=arm+"_EMA500"
        for name,weights in (
            (model+"_raw2750_minus_G0",{raw250:1,"G0":-1}),
            (model+"_raw3000_minus_G0",{raw500:1,"G0":-1}),
            (model+"_raw3000_minus_raw2750",{raw500:1,raw250:-1}),
            (model+"_EMA2750_minus_raw2750",{ema250:1,raw250:-1}),
            (model+"_EMA3000_minus_EMA2750",{ema500:1,ema250:-1}),
            (model+"_EMA3000_minus_raw2750",{ema500:1,raw250:-1}),
            (model+"_EMA3000_minus_G0",{ema500:1,"G0":-1})):
            answer[name]=weights
    answer.update(FULL_COMPARISONS)
    answer.update(J_minus_B_raw2750={"R_J_raw250":1,"R_B_raw250":-1},
                  J_minus_B_EMA2750={"R_J_EMA250":1,"R_B_EMA250":-1})
    return answer


def _section(by_label,comparisons,keys=("parent","patch","seed"),include_binding=False):
    section=dict(models={label:dict(rows=rows,groups=summarize(rows)) for label,rows in by_label.items()},comparisons={})
    for name,weights in comparisons.items():
        rows=contrast(by_label,weights,keys);section["comparisons"][name]=dict(formula=weights,rows=rows,groups=summarize(rows,True))
    if include_binding:
        b={label:bindings(rows) for label,rows in by_label.items()}
        section["binding"]=dict(models={label:dict(rows=rows,groups=summarize(rows),both_directions_positive_count=sum(r["both_directions_positive"] for r in rows)) for label,rows in b.items()},comparisons={})
        for name,weights in comparisons.items():
            rows=contrast(b,weights,keys=("patch","seed"))
            section["binding"]["comparisons"][name]=dict(formula=weights,rows=rows,groups=summarize(rows,True))
        section["binding"]["note"]="Original target-bbox-normalized four-grid, existing cross-remainder scores only. Positive binding is descriptive; no new success threshold."
    return section


def _csv_rows(result):
    for section in ("endpoint20","generation_curve4","teacher_curve20"):
        if section not in result:continue
        block=result[section]
        for kind in ("models","comparisons"):
            for name,item in block[kind].items():
                for row in item["rows"]:yield dict(section=section,kind=kind,name=name,**_identity(row),**row["scalars"])
        for kind,values in block.get("binding",{}).items():
            if kind not in ("models","comparisons"):continue
            for name,item in values.items():
                for row in item["rows"]:yield dict(section=section+"_binding",kind=kind,name=name,**_identity(row),**row["scalars"])


def analyze(out):
    """One pure read/aggregate pass after D scores and teacher curves complete."""
    out=Path(out).resolve();destination=out/"analysis/D.json"
    if destination.exists():raise FileExistsError("Never overwrite a completed D analysis implicitly")
    tick=time.perf_counter();sources={};result=dict(schema="jedge5_diagnosis_D_analysis_v1",status="RUNNING",sources=sources,
        counts=dict(new_model_calls=0,new_geometry_scores=0,new_OT_calls=0,new_autograd_calls=0),
        limitations=["Two observed parent objects and two generation noises; descriptive, no significance or tail-probability claim.",
          "Patch1-4 are observed contexts from seen objects, not unseen shapes.",
          "Per-case differences precede all grouped means/medians; overlapping group summaries are not independent experiments.",
          "The four-case trajectory is fixed before outcomes; it cannot establish convergence or an optimal EMA window.",
          "Five equally listed teacher times are a diagnostic grid, not a training-distribution expectation.",
          "EMA can resemble earlier weights; report explicit same-four-case comparisons, not a causal proof of an averaging benefit.",
          "Signed improvements and quality/coverage changes are kept separately; no aggregate winner score or checkpoint selection."])
    replay_status={}
    for arm,expected in EXPECTED_RAW.items():
        path=out/"stage_d"/arm/"replay.json"
        if not path.is_file():replay_status[arm]=dict(status="MISSING");continue
        saved=_read(path,sources);good=(saved.get("status")=="PASS_EXACT_REPLAY" and saved.get("EMA_evaluation_allowed") is True
                                       and saved.get("raw500_bitwise_reference_equal") is True and saved.get("raw500_state_sha256")==expected)
        replay_status[arm]=dict(status=saved.get("status"),completed_updates=saved.get("completed_updates"),raw500_state_sha256=saved.get("raw500_state_sha256"),
                               expected_raw_state_sha256=expected,EMA_claim_allowed=good,
                               first_observed_training_scalar_mismatch=saved.get("first_observed_training_scalar_mismatch"),
                               first_final_state_difference=saved.get("first_final_state_difference"))
    result["replay"]=replay_status
    if not all(x.get("EMA_claim_allowed") for x in replay_status.values()):
        result.update(status="BLOCKED_NONEXACT_OR_INCOMPLETE_REPLAY",EMA_comparisons="NOT_RUN",reason="Do not attribute a changed raw training trajectory to EMA")
    else:
        seeds=_read(out/"registration/protocol.json",sources)["generation_seeds"]
        if len(seeds)!=2 or len(set(seeds))!=2:raise ValueError("Exactly two registered evaluation seeds")
        full={};curve={};teacher={}
        for model in ("B","J"):
            label=model+"_raw3000"
            rows=[r for r in _score_rows(out,"A_"+model,model,sources) if r["precision"]=="bf16" and r["nfe"]==50]
            _assert_case_grid(rows,seeds);full[label]=_normalized(rows,label)
            teacher[label]=_teacher_rows(_raw_teacher(out,model,sources),label)
        for label in D_LABELS:
            worker=_read(out/"D_evaluation"/label/"results.json",sources)
            if worker.get("status")!="PASS" or worker.get("label")!=label:raise ValueError("D evaluation worker incomplete: "+label)
            rows=_score_rows(out,"D_"+label,label,sources);small=not label.endswith("500")
            _assert_case_grid(rows,seeds,small=small)
            if small:curve[label]=_normalized(rows,label)
            else:full[label]=_normalized(rows,label)
            tp=out/worker["teacher"]["path"];record=_read(tp,sources)
            if sources[str(tp)]!=worker["teacher"]["sha256"] or record.get("status")!="PASS" or record.get("label")!=label or record.get("precision")!="bf16":
                raise ValueError("D teacher file identity differs")
            if record.get("counts",{}).get("teacher_forward_returns")!=20:raise ValueError("D teacher forward count differs")
            teacher[label]=_teacher_rows(record["rows"],label)
        for label,rows in full.items():
            subset=[r for r in rows if r["patch"] in (0,4) and r["seed"]==seeds[0]];_assert_case_grid(subset,seeds,small=True)
            curve[label]=subset
        ref=_index(teacher["G0"],("state_id",))
        for label,rows in teacher.items():
            index=_index(rows,("state_id",))
            if set(index)!=set(ref):raise ValueError("Teacher curve state IDs differ")
            for key,row in index.items():
                if any(row[x]!=ref[key][x] for x in ("shared_input_sha256","target_mapping_sha256","FM_coordinate_denominator")):
                    raise ValueError("Teacher curve target/input/normalization mismatch")
        curve_pairs=_curve_comparisons()
        result.update(status="PASS",seeds=seeds,endpoint20=_section(full,FULL_COMPARISONS,include_binding=True),
                      generation_curve4=_section(curve,curve_pairs,include_binding=True),
                      teacher_curve20=_section(teacher,curve_pairs,keys=("state_id",)),
                      reuse=dict(raw3000_generation="A_B/A_J BF16 Euler50 (40 existing rows, no new rollout)",
                                 raw3000_teacher="C B/J BF16 fixed20-state subset each, no new forward",
                                 new_D_generation_count=60,new_D_teacher_forward_count=140))
    result["analysis_seconds"]=time.perf_counter()-tick
    rows=list(_csv_rows(result));fields=["section","kind","name","parent","patch","seed","noise_index","teacher_t","state_id"]
    fields+=sorted(set().union(*(row.keys() for row in rows))-set(fields)) if rows else []
    destination.parent.mkdir(parents=True,exist_ok=True);csv_path=out/"analysis/D_per_case.csv"
    if csv_path.exists():raise FileExistsError(csv_path)
    with csv_path.open("w",newline="",encoding="utf-8") as stream:
        writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader();writer.writerows(rows)
    result["per_case_csv"]=dict(path=str(csv_path.relative_to(out)),rows=len(rows),sha256=hashlib.sha256(csv_path.read_bytes()).hexdigest())
    _write(destination,result);return result


def run_scalar_fixtures():
    metric="remainder_symmetric_RMS_bbox_pct";checks=0
    def row(seed,value):return dict(parent=0,patch=0,seed=seed,scalars={metric:value,"free_total_surface_area":3.,"exact_C_free_shared_edge_count":None})
    bank={"a":[row(1,1),row(2,100),row(3,2)],"b":[row(1,3),row(2,99),row(3,9)]}
    delta=contrast(bank,{"b":1,"a":-1});stat=_stats(delta,metric,True)
    assert stat["median"]==2. and stat["improved_count"]==1 and stat["worse_count"]==2;checks+=1
    assert stat["median"]!=float(np.median([3,99,9])-np.median([1,100,2]));checks+=1
    assert _stats(delta,"free_total_surface_area",True)["improved_count"] is None;checks+=1
    assert _stats(delta,"exact_C_free_shared_edge_count",True)["status"]=="NOT_RUN";checks+=1
    four={name:[row(1,value)] for name,value in (("je",9),("be",4),("jr",8),("br",1))}
    assert contrast(four,{"je":1,"be":-1,"jr":-1,"br":1})[0]["scalars"][metric]==-2;checks+=1
    try:contrast({"a":bank["a"],"b":bank["b"][:2]},{"b":1,"a":-1})
    except ValueError:checks+=1
    else:raise AssertionError("Partial case pairing accepted")
    a=dict(parent=0,patch=0,seed=1,cross_remainder=[dict(symmetric_rms_bbox_pct=1.),dict(symmetric_rms_bbox_pct=3.)])
    b=dict(parent=1,patch=0,seed=1,cross_remainder=[dict(symmetric_rms_bbox_pct=5.),dict(symmetric_rms_bbox_pct=2.)])
    binding=bindings([a,b])[0];assert binding["scalars"]["B"]==2.5 and binding["both_directions_positive"];checks+=1
    assert _direction("Gamma_to_free_edges_coverage_0.001_fraction")=="higher";checks+=1
    return dict(status="PASS",checks=checks,model_calls=0,geometry_calls=0,OT_calls=0,autograd_calls=0)


def main(argv=None):
    parser=argparse.ArgumentParser();parser.add_argument("--out",required=True)
    args=parser.parse_args(argv);result=analyze(args.out);print(json.dumps(dict(status=result["status"],counts=result["counts"])),flush=True)

if __name__=="__main__":main()
