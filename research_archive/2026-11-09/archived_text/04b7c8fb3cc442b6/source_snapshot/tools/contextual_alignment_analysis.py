"""Saved-score summaries and preregistered conservative SIE screen; no model imports."""
from __future__ import annotations
import itertools,math,statistics,time
from collections import Counter,defaultdict
from pathlib import Path
import numpy as np
import contextual_alignment_evaluation as ce
from meshflow_control.evaluation import registry
from meshflow_control.evaluation.io import read_json,write_json,write_csv,arrays,digest,resolve
from meshflow_control.evaluation.surface import surface_distance_squared

SIE=("SeamCov","ConflictCov_C","ExposedRetain")
CONTRASTS=((ce.LABELS[3],ce.LABELS[1]),(ce.LABELS[3],ce.LABELS[2]),(ce.LABELS[2],ce.LABELS[1]))+tuple((x,"START") for x in ce.LABELS[1:])
DIRECTIONS={"SeamCov":1,"ConflictCov_C":-1,"ExposedRetain":1}
BASIC=SIE+("SeamCov_upper","ConflictCov_C_upper","ConflictCov_C_cross","ConflictCov_C_dup","ConflictCov_C_cross_and_dup","ExposedRetain_upper","ExposedRetain_global_lower","ExposedRetain_global_upper","D_sym2048","GT_to_F_MSE2048_L0","F_to_GT_MSE2048_L0","fixed_query_MSE","fixed_query_p95_L","coverage_fraction","CF_proper_affected_free_fraction","FF_illegal_affected_free_fraction","free_minimum_angle_lt5_fraction","free_near_degenerate_fraction","free_exact_duplicate_excess_fraction","EXACT_COORD_CF_shared_edges","EXACT_COORD_face_components")
def finite(v):return isinstance(v,(int,float)) and not isinstance(v,bool) and math.isfinite(v)
def mean(v):return math.fsum(v)/len(v) if v and all(finite(x) for x in v) else None
def median(v):return statistics.median(v) if v and all(finite(x) for x in v) else None
def scope_rows(rr,scope):return [r for r in rr if scope in ("ALL","OPEN") or r["cohort"]==scope]
def interval(metric,m):
    v=m.get(metric)
    if not finite(v):return None,None
    if metric=="ExposedRetain":return m.get("ExposedRetain_global_lower"),m.get("ExposedRetain_global_upper")
    if metric=="SeamCov":return v,m.get("SeamCov_upper")
    if metric=="ConflictCov_C":return v,m.get("ConflictCov_C_upper")
    if metric.startswith("SeamCov_") and not metric.startswith("SeamCov_upper"):return v,m.get("SeamCov_upper"+metric[len("SeamCov"):])
    return v,v
def val(row,k):return row.get("metrics",{}).get(k) if row.get("status")=="PASS" else None
def bounds(row,k):return interval(k,row.get("metrics",{})) if row.get("status")=="PASS" else (None,None)
def original_status(row,k):
    if row.get("status")!="PASS":return row.get("status","MISSING")
    return row.get("metric_status",{}).get(k,row.get("metrics",{}).get(k+"_status","PASS" if finite(val(row,k)) else "NA"))
def load_scores(out):
    out=ce.root(out);rows=[]
    for q in ce.request_rows(out):
        for label in ce.LABELS:
            p=out/"evaluation/rows"/label/(q["request_id"]+".json")
            if p.exists():
                r=read_json(p)
                if r.get("adapter_identity",{}).get("source_hashes")!=ce.ev.sources():raise ValueError("Score scientific source changed")
                rows.append(r)
            else:
                rows.append(dict(q,label=label,status="PENDING",metrics={},metric_status={},contracts={},files={},L0=read_json(q["reference_path"])["L0"]))
    return rows

def criterion(lo,hi,op,threshold):
    if not finite(lo) or not finite(hi):return "INCOMPLETE"
    if lo>hi:raise ValueError("Reversed conservative interval")
    if op==">=":return "PASS" if lo>=threshold else "FAIL" if hi<threshold else "INCOMPLETE"
    if op==">":return "PASS" if lo>threshold else "FAIL" if hi<=threshold else "INCOMPLETE"
    if op=="<=":return "PASS" if hi<=threshold else "FAIL" if lo>threshold else "INCOMPLETE"
    raise ValueError(op)

def screen(paired,profiles,candidate):
    contrast=candidate+"-A_FM_1000"
    def get(scope,key):
        return [r for r in paired if r["contrast"]==contrast and r["panel"]=="P1" and r["cohort"]==scope and r["metric"]==key]
    allS,newS,I,E=[get(s,k) for s,k in (("ALL","SeamCov"),("NEW","SeamCov"),("ALL","ConflictCov_C"),("ALL","ExposedRetain"))]
    items=[]
    def add(name,point,lo,hi,op,threshold):
        items.append(dict(name=name,point=point,lower=lo,upper=hi,operator=op,threshold=threshold,status=criterion(lo,hi,op,threshold)))
    def vals(rr,key):return [r.get(key) for r in rr] if len(rr)==8 else [None]*8
    add("ALL_S_mean_gain",mean(vals(allS,"delta")),mean(vals(allS,"delta_lower")),mean(vals(allS,"delta_upper")),">=",.01)
    add("ALL_S_parent_median",median(vals(allS,"delta")),median(vals(allS,"delta_lower")),median(vals(allS,"delta_upper")),">",0)
    lo,hi=vals(allS,"delta_lower"),vals(allS,"delta_upper")
    pos=sum(v>0 for v in vals(allS,"delta")) if all(finite(v) for v in vals(allS,"delta")) else None
    guaranteed=sum(v>0 for v in lo) if all(finite(v) for v in lo) else None
    possible=sum(v>0 for v in hi) if all(finite(v) for v in hi) else None
    add("ALL_S_improving_parents",pos,guaranteed,possible,">=",5)
    add("NEW_S_mean_gain",mean(vals(newS,"delta")),mean(vals(newS,"delta_lower")),mean(vals(newS,"delta_upper")),">",0)
    add("ALL_I_mean_cost",mean(vals(I,"delta")),mean(vals(I,"delta_lower")),mean(vals(I,"delta_upper")),"<=",.01)
    add("ALL_E_mean_change",mean(vals(E,"delta")),mean(vals(E,"delta_lower")),mean(vals(E,"delta_upper")),">=",-.01)
    st="FAIL" if any(x["status"]=="FAIL" for x in items) else "PASS" if all(x["status"]=="PASS" for x in items) else "INCOMPLETE"
    return dict(candidate=candidate,reference="A_FM_1000",status=st,criteria=items,parent_denominator=8,scope="P1 ALL4noises; NEW2noises same DEV parents",
        interval_interpretation="Detector conservative bounds, not statistical confidence intervals; E uses original global eligibility bounds",
        screening_only=True,automatic_promotion=False,source_MSE_veto=False,visual_and_other_costs_are_separate=True)

def aggregate(out,compute_diversity=False):
    out=ce.root(out);dest=out/"analysis";rows=load_scores(out)
    metrics=sorted(set(BASIC)|{k for r in rows for k,v in r["metrics"].items() if finite(v) or v is None})
    flat=[];long=[]
    for r in rows:
        d={k:r.get(k) for k in ce.META};d.update(L0=r.get("L0"),raw_path=r.get("raw_path"),source_result_path=r.get("source_result_path"),raw_sha256=r.get("identity",{}).get("raw_sha256"),input_sha256=r.get("identity",{}).get("input_sha256"),reference_sha256=r.get("identity",{}).get("reference_sha256"),scientific_scores_recomputed=r.get("scientific_scores_recomputed"),cpu_adapter_seconds=r.get("cpu_adapter_seconds"),status=r["status"],cache=r.get("cache"),scoring_seconds=r.get("scoring_seconds"),**r["metrics"])
        d.update({k:r.get("contracts",{}).get(k) for k in ce.CONTRACTS})
        for k in SIE:
            d[k+"_status"]=original_status(r,k);d[k+"_lower"],d[k+"_upper_for_screen"]=bounds(r,k)
        d.update(S_scope="REFERENCE_INTERFACE" if r["panel"]=="P1" else "NA_NO_REQUIRED_INTERFACE",E_scope="REFERENCE_EXPOSURE" if r["panel"]=="P1" else "NA_NO_REFERENCE_EXPOSURE")
        flat.append(d)
        if r["status"]=="PASS":long.extend(registry.records(r))
    write_json(out/"evaluation/per_output.json",dict(schema=ce.VERSION,rows=rows))
    write_csv(out/"per_output_metrics.csv",flat);write_csv(dest/"metric_registry_results.csv",long)
    parent=[];profiles=[]
    for label in ce.LABELS:
        for panel in ("P1","P2"):
            rr=[r for r in rows if r["label"]==label and r["panel"]==panel]
            for scope in (("OLD","NEW","ALL") if panel=="P1" else ("OPEN",)):
                subset=scope_rows(rr,scope)
                for pid in sorted({r["parent_id"] for r in subset}):
                    prs=[r for r in subset if r["parent_id"]==pid]
                    for k in metrics:
                        vv=[val(r,k) for r in prs];bb=[bounds(r,k) for r in prs]
                        parent.append(dict(label=label,panel=panel,cohort=scope,parent_id=pid,metric=k,value=mean(vv),lower=mean([x[0] for x in bb]),upper=mean([x[1] for x in bb]),expected_noises=len(prs),finite_noises=sum(finite(v) for v in vv),original_statuses=dict(Counter(original_status(r,k) for r in prs)),status="COMPLETE" if all(finite(v) for v in vv) else "NA_OR_INCOMPLETE"))
                for k in metrics:
                    pr=[r for r in parent if (r["label"],r["panel"],r["cohort"],r["metric"])==(label,panel,scope,k)]
                    vv=[r["value"] for r in pr]
                    profiles.append(dict(label=label,panel=panel,cohort=scope,metric=k,value=mean(vv),lower=mean([r["lower"] for r in pr]),upper=mean([r["upper"] for r in pr]),expected_parents=len(pr),finite_parents=sum(finite(x) for x in vv),parent_median=median(vv),parent_min=min(vv) if all(finite(v) for v in vv) else None,parent_max=max(vv) if all(finite(v) for v in vv) else None,
                        status="COMPLETE" if all(finite(v) for v in vv) else "NA_OR_INCOMPLETE",unit="fraction" if k in SIE else "original_metric_unit",aggregation="within-parent noise mean then equal parents; P2 conditions not independent shape claims"))
    write_csv(dest/"per_parent_metrics.csv",parent);write_csv(dest/"checkpoint_profiles.csv",profiles)
    write_csv(out/"sie_summary.csv",[r for r in profiles if r["metric"] in SIE])
    by={(r["label"],r["request_id"]):r for r in rows};noise=[];paired=[];comparisons=[];lopo=[];tails=[]
    for cand,ref in CONTRASTS:
        name=cand+"-"+ref
        for q in ce.request_rows(out):
            a,b=by[cand,q["request_id"]],by[ref,q["request_id"]]
            for k in metrics:
                av,bv=val(a,k),val(b,k);al,ah=bounds(a,k);bl,bh=bounds(b,k)
                noise.append(dict(contrast=name,candidate_label=cand,reference_label=ref,panel=q["panel"],cohort=q["cohort"],parent_id=q["parent_id"],condition_id=q["condition_id"],request_id=q["request_id"],candidate=q["candidate"],metric=k,candidate_value=av,reference_value=bv,delta=av-bv if finite(av) and finite(bv) else None,delta_lower=al-bh if finite(al) and finite(bh) else None,delta_upper=ah-bl if finite(ah) and finite(bl) else None,candidate_status=original_status(a,k),reference_status=original_status(b,k)))
        for panel in ("P1","P2"):
            for scope in (("OLD","NEW","ALL") if panel=="P1" else ("OPEN",)):
                ns=scope_rows([r for r in noise if r["contrast"]==name and r["panel"]==panel],scope)
                for k in metrics:
                    ms=[r for r in ns if r["metric"]==k];pp=[]
                    for pid in sorted({r["parent_id"] for r in ms}):
                        pr=[r for r in ms if r["parent_id"]==pid];vv=[r["delta"] for r in pr]
                        x=dict(contrast=name,candidate_label=cand,reference_label=ref,panel=panel,cohort=scope,parent_id=pid,metric=k,candidate_value=mean([r["candidate_value"] for r in pr]),reference_value=mean([r["reference_value"] for r in pr]),delta=mean(vv),delta_lower=mean([r["delta_lower"] for r in pr]),delta_upper=mean([r["delta_upper"] for r in pr]),expected_noises=len(pr),finite_noises=sum(finite(v) for v in vv),noise_positive=sum(v>0 for v in vv if finite(v)),noise_negative=sum(v<0 for v in vv if finite(v)),noise_opposed=any(finite(v) and v>0 for v in vv) and any(finite(v) and v<0 for v in vv))
                        paired.append(x);pp.append(x)
                    vv=[r["delta"] for r in pp];complete=bool(vv) and all(finite(v) for v in vv)
                    direction=DIRECTIONS.get(k)
                    cs=dict(contrast=name,candidate_label=cand,reference_label=ref,panel=panel,cohort=scope,metric=k,expected_parents=len(pp),finite_parents=sum(finite(v) for v in vv),mean_delta=mean(vv),median_delta=median(vv),delta_lower=mean([r["delta_lower"] for r in pp]),delta_upper=mean([r["delta_upper"] for r in pp]),positive_parents=sum(v>0 for v in vv) if complete else None,negative_parents=sum(v<0 for v in vv) if complete else None,zero_parents=sum(v==0 for v in vv) if complete else None,improving_parents=sum(direction*v>0 for v in vv) if complete and direction else None,noise_opposed_parents=sum(r["noise_opposed"] for r in pp),min_delta=min(vv) if complete else None,max_delta=max(vv) if complete else None,status="COMPLETE" if complete else "NA_OR_INCOMPLETE")
                    comparisons.append(cs)
                    if panel=="P1":
                        for excluded in pp:
                            remain=[r for r in pp if r is not excluded]
                            lopo.append(dict(contrast=name,cohort=scope,metric=k,excluded_parent=excluded["parent_id"],remaining_parents=len(remain),LOPO_mean=mean([r["delta"] for r in remain]),LOPO_lower=mean([r["delta_lower"] for r in remain]),LOPO_upper=mean([r["delta_upper"] for r in remain]),formal_full_mean=cs["mean_delta"],excluded_mean_contribution=excluded["delta"]/len(pp) if finite(excluded["delta"]) else None,formal_denominator_unchanged=True))
                    if k in SIE:
                        for x in pp:
                            if k not in ("ConflictCov_C","ExposedRetain"):continue
                            cost=x["delta"] if k=="ConflictCov_C" else -x["delta"] if finite(x["delta"]) else None
                            low=x["delta_lower"] if k=="ConflictCov_C" else -x["delta_upper"] if finite(x["delta_upper"]) else None
                            high=x["delta_upper"] if k=="ConflictCov_C" else -x["delta_lower"] if finite(x["delta_lower"]) else None
                            # Include possible costs beyond the registered boundary even if the point estimate does not cross it.
                            if (finite(cost) and cost>.10) or (finite(high) and high>.10):
                                tails.append(dict(x,tail_scope="PARENT_REGISTERED_10PP_DISCLOSURE",cost=cost,cost_lower=low,cost_upper=high,point_beyond_10pp=finite(cost) and cost>.10,confirmed_beyond_10pp=finite(low) and low>.10,possible_beyond_10pp=finite(high) and high>.10))
                        # Preserve both point and interval extrema; no invented per-output severe threshold.
                        valid=[r for r in ms if finite(r["delta"])]
                        if valid:
                            for tag,x in (("MIN_OUTPUT_POINT",min(valid,key=lambda r:r["delta"])),("MAX_OUTPUT_POINT",max(valid,key=lambda r:r["delta"]))):tails.append(dict(x,report_cohort=scope,tail_scope=tag))
                        lo=[r for r in ms if finite(r["delta_lower"])]
                        hi=[r for r in ms if finite(r["delta_upper"])]
                        if lo:tails.append(dict(min(lo,key=lambda r:r["delta_lower"]),report_cohort=scope,tail_scope="MIN_OUTPUT_LOWER_BOUND"))
                        if hi:tails.append(dict(max(hi,key=lambda r:r["delta_upper"]),report_cohort=scope,tail_scope="MAX_OUTPUT_UPPER_BOUND"))
    write_csv(out/"paired_noise_metrics.csv",noise);write_csv(out/"paired_parent_metrics.csv",paired)
    write_csv(dest/"comparison_summary.csv",comparisons);write_csv(dest/"leave_one_parent_out.csv",lopo);write_csv(dest/"control_tails.csv",tails)
    screens=[screen(paired,profiles,c) for c in ce.LABELS[2:]]
    write_json(dest/"sie_screen.json",dict(schema="preregistered_development_SIE_screen_v1",screens=screens,global_geometry_quality_or_promotion_gate=False,
        severe_parent_rule="Every I increase or E decrease >0.10 is disclosed separately; cannot cancel it using other parents",
        NEW="Two new fixed Gaussians on observed DEV parents, not new-parent confirmation",E_bounds="original global lower/upper; PARTIAL alone neither FAIL nor PASS"))
    ds=diversity(rows,out,compute=compute_diversity)
    status_counts=Counter(r["status"] for r in rows)
    result=dict(status="COMPLETE" if status_counts=={"PASS":192} else "INCOMPLETE",expected_outputs=192,status_counts=dict(status_counts),scored=status_counts["PASS"],cached=sum(r.get("cache")=="VERIFIED_START_SCALARS" for r in rows),new_scored=sum(r.get("scientific_scores_recomputed")==1 for r in rows),
        screen_statuses={x["candidate"]:x["status"] for x in screens},screens_numeric_only=True,visual_review="SEPARATE_PENDING_UNTIL_ACTUAL_REVIEW",human_usability="NOT_RUN",
        diversity=ds,source_hashes={str(p.relative_to(out)):digest(p) for p in [out/"per_output_metrics.csv",out/"paired_noise_metrics.csv",out/"paired_parent_metrics.csv",out/"sie_summary.csv",dest/"comparison_summary.csv",dest/"leave_one_parent_out.csv"]})
    write_json(dest/"results.json",result);return result

def diversity(rows,out,compute=False):
    out=ce.root(out);dest=out/"analysis";by={(r["label"],r["request_id"]):r for r in rows};qs=ce.request_rows(out);pairs=[];summary=[];calls=0;start=time.perf_counter()
    for label in ce.LABELS:
        for cid in sorted({r["condition_id"] for r in qs}):
            group=[q for q in qs if q["condition_id"]==cid];panel=group[0]["panel"]
            for qa,qb in itertools.combinations(group,2):
                a,b=by[label,qa["request_id"]],by[label,qb["request_id"]];p=dest/"diversity_cache"/label/f"{cid}__{qa['candidate']}_{qb['candidate']}.json"
                ident=dict(label=label,condition_id=cid,candidate_a=qa["candidate"],candidate_b=qb["candidate"],raw_a=a.get("identity",{}).get("raw_sha256"),raw_b=b.get("identity",{}).get("raw_sha256"),q_a=a.get("files",{}).get(a.get("quality_vectors_path")),q_b=b.get("files",{}).get(b.get("quality_vectors_path")),source_surface=digest(ce.ev.surface_distance_squared.__code__.co_filename))
                if p.exists():
                    v=read_json(p)
                    if v["identity"]!=ident:raise ValueError("Diversity pair identity changed")
                elif compute and a["status"]==b["status"]=="PASS":
                    x=arrays(a["quality_vectors_path"])["Q_F2048"];y=arrays(b["quality_vectors_path"])["Q_F2048"]
                    xx=arrays(a["raw_path"]);yy=arrays(b["raw_path"]);fx=xx.get("raw",xx.get("output")).reshape(-1,3,3)[a["K"]:];fy=yy.get("raw",yy.get("output")).reshape(-1,3,3)[b["K"]:]
                    if a["L0"]!=b["L0"]:raise ValueError("Pair scale mismatch")
                    value=float((surface_distance_squared(x,fy).mean()+surface_distance_squared(y,fx).mean())/(2*a["L0"]**2));calls+=2
                    v=dict(identity=ident,D_free_pair=value,status="PASS",distance="symmetric mean squared fixed Q_F2048 point-to-finite-triangle/L^2",not_original_TMD=True);write_json(p,v,freeze=True)
                else:v=dict(identity=ident,D_free_pair=None,status="PENDING" if not compute else "MISSING_CANDIDATE")
                pairs.append(dict(label=label,panel=panel,condition_id=cid,candidate_a=qa["candidate"],candidate_b=qb["candidate"],cohort_a=qa["cohort"],cohort_b=qb["cohort"],D_free_pair=v["D_free_pair"],status=v["status"]))
            for scope in (("OLD","NEW","ALL") if panel=="P1" else ("OPEN",)):
                pr=[r for r in pairs if r["label"]==label and r["condition_id"]==cid and (scope in ("ALL","OPEN") or r["cohort_a"]==r["cohort_b"]==scope)]
                vv=[r["D_free_pair"] for r in pr];summary.append(dict(label=label,panel=panel,condition_id=cid,cohort=scope,expected_pairs=len(pr),finite_pairs=sum(finite(v) for v in vv),value=mean(vv),status="COMPLETE" if all(finite(v) for v in vv) else "INCOMPLETE",interpretation="Variation, not larger-is-better; seat tessellations not independent shape evidence"))
    write_csv(dest/"diversity_pairs.csv",pairs);write_csv(dest/"diversity_per_condition.csv",summary)
    return dict(computed_this_call=calls,expected_pairs=len(pairs),complete_pairs=sum(r["status"]=="PASS" for r in pairs),wall_seconds=time.perf_counter()-start,model_calls=0)

def self_test():
    assert criterion(.01,.02,">=",.01)=="PASS"
    assert criterion(.009,.011,">=",.01)=="INCOMPLETE"
    assert criterion(-.02,-.011,">=",-.01)=="FAIL"
    assert criterion(-.005,.01,"<=",.01)=="PASS"
    assert criterion(0,0,">",0)=="FAIL"
    assert criterion(0,.02,">",0)=="INCOMPLETE"
    assert mean([.1,None]) is None
    assert interval("ExposedRetain",dict(ExposedRetain=.9,ExposedRetain_upper=.95,ExposedRetain_global_lower=.7,ExposedRetain_global_upper=.98))==(.7,.98)
    fixture=[]
    for scope in ("OLD","NEW","ALL"):
        for i in range(8):
            for metric,value in (("SeamCov",.02),("ConflictCov_C",.005),("ExposedRetain",-.005)):
                fixture.append(dict(contrast="C_KNOWN_ALIGN_1000-A_FM_1000",panel="P1",cohort=scope,parent_id=str(i),metric=metric,delta=value,delta_lower=value,delta_upper=value))
    assert screen(fixture,[],"C_KNOWN_ALIGN_1000")["status"]=="PASS"
    for r in fixture:
        if r["metric"]=="ExposedRetain":r["delta_lower"]=-.02;r["delta_upper"]=0
    assert screen(fixture,[],"C_KNOWN_ALIGN_1000")["status"]=="INCOMPLETE"
    assert "torch" not in ce.sys.modules
    return dict(status="PASS",tests=11,geometry_scoring=0,model_calls=0)
