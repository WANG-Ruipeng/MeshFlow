"""Synthetic saved-table fixtures only; no geometry, model or real run access."""
import copy
import importlib.util
import itertools
import json
from pathlib import Path
import pytest

REPO=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location("curriculum_analysis",REPO/"tools/analyze_recipe_curriculum_results.py")
a=importlib.util.module_from_spec(spec);spec.loader.exec_module(a)
METRIC="CF_proper_affected_free_fraction"

def panel():
    requests=[];main=[];course=[]
    for p in range(8):
        for ratio in (.2,.4):
            for noise in a.base.NOISES:
                q=dict(request_id=f"p{p}_r{ratio}_n{noise}",parent_id=f"p{p}",condition_id=f"p{p}_r{ratio}",
                    ratio=ratio,candidate=noise,condition_cohort="A20" if ratio==.2 else "ORIGINAL6" if p<6 else "EXTENDED2",
                    N=128,K=round(128*ratio))
                requests.append(q)
                for label in (*a.base.LABELS,*a.COURSES):
                    offset={a.BASELINE:10,a.COURSES[0]:8,a.COURSES[1]:12}.get(label,11)
                    row=dict(q,label=label,status="PASS",contracts={"C_bitwise":True},
                        identity={"input_sha256":q["request_id"]},query_sha256=q["condition_id"],
                        registered_free_faces=128-q["K"],registered_C_queries=2048,query_weight_denominator=2048,L0=9.,
                        metrics={METRIC:offset+p+noise/10,"FF_illegal_affected_free_fraction":20-offset+p,
                                 "all_NA":None,"status_field":"PARTIAL" if p==0 else "PASS","boolean_field":False})
                    (course if label in a.COURSES else main).append(row)
    main_reg=dict(kind="main",labels=list(a.base.LABELS),expected_outputs=384,requests_sha256="shared")
    course_reg=dict(kind="course",labels=list(a.COURSES),expected_outputs=128,requests_sha256="shared")
    return main_reg,course_reg,requests,main,course,dict(status="COMPLETE",registered=384,scored=384),dict(status="COMPLETE",registered=128,scored=128)

def selected():
    values=panel();rows=a.validate_panel(*values)
    return values[2],rows

def diversity(rows):
    conditions=[];pairs=[]
    by={(r["label"],r["condition_id"]):r for r in rows}
    for (label,cid),r in by.items():
        conditions.append(dict(label=label,condition_id=cid,parent_id=r["parent_id"],ratio=r["ratio"],
            condition_cohort=r["condition_cohort"],expected_pairs=6,available_pairs=6,D_free_mean=.25))
        for na,nb in itertools.combinations(a.base.NOISES,2):
            pairs.append(dict(label=label,condition_id=cid,parent_id=r["parent_id"],ratio=r["ratio"],
                candidate_a=na,candidate_b=nb,D_free_pair=.25,status="PASS"))
    return conditions,pairs

def references(rows):
    return [dict({k:r[k] for k in ("label","request_id","parent_id","condition_id","ratio","candidate","condition_cohort","N","K")},
                 result=dict(label=r["label"],request_id=r["request_id"],status="PASS" if r["ratio"]==.2 else "NA",metrics={"SeamCov":.5,"D_sym2048":None} if r["ratio"]==.2 else {}))
            for r in rows]

def save_panel(out):
    mr,cr,qs,main,course,ms,cs=panel()
    qp=out/"evaluation/requests.json";a.write_json(qp,dict(rows=qs))
    for phase,reg,rr,receipt in (("main",mr,main,ms),("course",cr,course,cs)):
        reg.update(requests_path=str(qp),requests_sha256=a.digest(qp))
        root=out/"phases"/phase;a.write_json(root/"registration.json",reg)
        ev=root/"evaluation";a.write_json(ev/"per_output.json",dict(rows=rr));a.write_json(ev/"scoring_receipt.json",receipt)
        dc,dp=diversity(rr);a.write_csv(ev/"diversity_per_condition.csv",dc);a.write_csv(ev/"diversity_pairs.csv",dp)
        a.write_json(ev/"reference_A20_per_output.json",dict(rows=references(rr),promotion_gate=False))
    a.write_json(out/"registrations/curriculum_training.json",dict(authorized_recipes=list(a.COURSES),hybrid_schedules={c:"all" for c in a.COURSES}))
    (out/"registrations/CURRICULUM_DECISION.md").write_text("SYNTHETIC prospective H_ALL decision",encoding="utf8")
    (out/"PROTOCOL_CLARIFICATIONS.md").write_text("SYNTHETIC task-pool caveat",encoding="utf8")
    a.write_json(out/"audit/condition_construction_audit.json",dict(synthetic_fixture=True,planned_only=True))
    return out

def test_complete_192_panel_and_no_mutation_of_base_globals():
    labels=a.base.LABELS
    qs,rows=selected()
    assert len(rows)==192 and set(r["label"] for r in rows)==set(a.LABELS)
    assert a.base.LABELS==labels and len(labels)==6

@pytest.mark.parametrize("kind",["missing","duplicate","receipt","input","query","contract","registration"])
def test_rejects_missing_or_mismatched_cells(kind):
    v=list(panel())
    if kind=="missing":v[4]=v[4][:-1]
    elif kind=="duplicate":v[4][-1]=copy.deepcopy(v[4][0])
    elif kind=="receipt":v[6]["scored"]=127
    elif kind=="input":v[4][0]["identity"]["input_sha256"]="different"
    elif kind=="query":v[4][0]["query_sha256"]="different"
    elif kind=="contract":v[4][0]["contracts"]["C_bitwise"]=False
    else:v[1]["requests_sha256"]="different"
    with pytest.raises(ValueError):a.validate_panel(*v)

def test_generic_three_label_pairs_and_both_ratio_parent_strata():
    qs,rows=selected();t=a.tables(rows,qs,a.membership(qs))
    n=t["paired_noise_metrics"];p=t["paired_parent_metrics"];s=t["paired_parent_summary"]
    assert len(n)==3*64*2 and len(p)==3*16*2
    assert set((r["first"],r["second"]) for r in p)==set(a.PAIRS)
    for r in s:
        if r["metric"]==METRIC:
            expected={a.PAIRS[0]:-2,a.PAIRS[1]:2,a.PAIRS[2]:-4}[r["first"],r["second"]]
            assert r["mean"]==pytest.approx(expected)
        assert r["expected"]=={"ALL":8,"ORIGINAL6":6,"EXTENDED2":2}[r["parent_stratum"]]
    assert {r["ratio"] for r in s if r["parent_stratum"]=="EXTENDED2"}=={.2,.4}

def test_metric_NA_preserves_noise_and_parent_denominators_and_status_catalog():
    qs,rows=selected();rows[64]["metrics"][METRIC]=None
    t=a.tables(rows,qs,a.membership(qs))
    p=next(r for r in t["per_parent_metrics"] if r["label"]==a.COURSES[0] and r["parent_id"]=="p0" and r["ratio"]==.2 and r["metric"]==METRIC)
    assert p["mean"] is None and p["available"]==3 and p["expected"]==4
    s=next(r for r in t["paired_parent_summary"] if r["first"]==a.COURSES[0] and r["second"]==a.BASELINE and r["ratio"]==.2 and r["parent_stratum"]=="ALL" and r["metric"]==METRIC)
    assert s["mean"] is None and s["available"]==7 and s["expected"]==8
    catalog={r["metric"]:r for r in t["metric_catalog"]}
    assert catalog["all_NA"]["value_status_counts"]=={"NA_NULL":192}
    assert not catalog["boolean_field"]["numeric_summary"]
    assert catalog["status_field"]["value_status_counts"]['"PARTIAL"']==24

def test_tail_concentration_no_gain_is_NA_and_mean_keeps_adverse_scope():
    qs,rows=selected();t=a.tables(rows,qs,a.membership(qs))
    bad=next(r for r in t["paired_parent_summary"] if r["first"]==a.COURSES[1] and r["second"]==a.BASELINE and r["ratio"]==.4 and r["metric"]==METRIC and r["parent_stratum"]=="EXTENDED2")
    assert bad["mean"]==pytest.approx(2)
    assert bad["top1_share_of_gross_improvement"] is None
    assert bad["leave_one_parent_out_min"]==pytest.approx(2)
    text=a.report(t["model_summary"],t["paired_parent_summary"],len(t["metric_catalog"]))
    assert "C40_20_MIX" in text and "EXTENDED2" in text and "NA" in text

def test_diversity_all_pairs_mean_validation_and_missing_not_shrunk():
    qs,rows=selected();conditions,pairs=diversity(rows)
    t=a.diversity_tables(conditions,pairs,rows,a.membership(qs))
    assert len(t["diversity_paired_parents"])==48
    pairs[0]["D_free_pair"]=.1
    with pytest.raises(ValueError,match="mean differs"):a.diversity_tables(conditions,pairs,rows,a.membership(qs))
    pairs[0].update(status="FAILED",D_free_pair="")
    conditions[0].update(available_pairs=5,D_free_mean="")
    t=a.diversity_tables(conditions,pairs,rows,a.membership(qs))
    first=next(r for r in t["diversity_summary"] if r["label"]==a.BASELINE and r["ratio"]==.2 and r["parent_stratum"]=="ALL")
    assert first["mean"] is None and first["available"]==7 and first["expected"]==8

def test_reference_diagnostics_separate_no_preferred_direction_and_NA_kept():
    qs,rows=selected();raw=references(rows)
    main={"rows":[r for r in raw if r["label"]==a.BASELINE]};course={"rows":[r for r in raw if r["label"] in a.COURSES]}
    rr,rq=a.reference_rows(main,course,qs)
    assert len(rr)==96 and len(rq)==32
    t=a.tables(rr,rq,a.membership(qs),"reference_A20_diagnostic")
    assert all(r["nominal_preferred_direction"]=="reference_diagnostic" for r in t["paired_parent_summary"])
    assert not any("top1_share_of_gross_improvement" in r for r in t["paired_parent_summary"])
    course["rows"].pop()
    with pytest.raises(ValueError,match="96"):a.reference_rows(main,course,qs)

def test_end_to_end_synthetic_hashes_readonly_resume_and_tamper(tmp_path):
    out=save_panel(tmp_path/"experiments"/a.base.ROOT_NAME)
    before={p:a.digest(p) for p in out.rglob("*") if p.is_file()}
    result=a.analyze(out)
    assert result["status"]=="COMPLETE" and result["registered_outputs"]==192
    assert result["reference_A20_outputs"]==96 and not result["automatic_promotion"]
    assert result["new_model_calls"]==result["new_scoring_calls"]==0
    assert all(a.digest(p)==h for p,h in before.items())
    dest=Path(result["analysis_dir"])
    assert len(a.read_json(dest/"all_metrics.json")["rows"])==192
    assert len(a.read_csv(dest/"per_output.csv"))==192
    auxiliary=a.read_json(dest/"reference_A20_diagnostics.json")
    assert len(auxiliary["raw_auxiliary_rows"])==192
    assert sum(r["result"]["status"]=="NA" for r in auxiliary["raw_auxiliary_rows"])==96
    assert a.analyze(out)==result
    (dest/"REPORT.md").write_text("tampered",encoding="utf8")
    with pytest.raises(ValueError,match="Immutable"):a.analyze(out)

def test_absent_course_results_pending_without_writes(tmp_path):
    out=tmp_path/"experiments"/a.base.ROOT_NAME
    result=a.analyze(out)
    assert result["status"]=="PENDING" and result["missing_files"]
    assert not (out/"analysis").exists()


def test_noise_tail_denominator_and_course_request_file_hash(tmp_path):
    qs,rows=selected();t=a.tables(rows,qs,a.membership(qs))
    s=next(r for r in t['paired_noise_summary'] if r['ratio']==.2 and r['parent_stratum']=='EXTENDED2' and r['metric']==METRIC and r['first']==a.COURSES[0] and r['second']==a.BASELINE)
    assert s['expected']==8 and s['mean']==pytest.approx(-2)
    assert s['minimum_request_id'].startswith('p6') or s['minimum_request_id'].startswith('p7')
    out=save_panel(tmp_path/'experiments'/a.base.ROOT_NAME)
    registration=out/'phases/course/registration.json';value=a.read_json(registration)
    fake=out/'evaluation/changed_requests.json';a.write_json(fake,dict(rows=[]))
    value['requests_path']=str(fake);a.write_json(registration,value)
    with pytest.raises(ValueError,match='course requests changed'):a.analyze(out)

